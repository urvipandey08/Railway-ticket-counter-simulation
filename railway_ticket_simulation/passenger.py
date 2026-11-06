"""
passenger.py
Passenger entity and lifecycle process module for Railway Ticket Simulation.
Course: Solving Problems Using Modelling and Simulation (NEC00091)

This module defines the Passenger dataclass and the SimPy passenger process,
modeling arrival, balking decision, queuing, reneging timeout, and counter service.
"""

from dataclasses import dataclass, field
import math
from typing import Optional, Dict, Any
import numpy as np
import simpy
from config import SimulationConfig
from counter import TicketCounterSystem


@dataclass
class Passenger:
    """Represents a passenger in the railway reservation counter queuing system."""
    passenger_id: int
    arrival_time: float
    patience_time: float = float("inf")
    queue_length_at_arrival: int = 0
    system_size_at_arrival: int = 0

    # Lifecycle state: 'SERVED', 'BALKED', 'RENEGED', 'IN_QUEUE', 'IN_SERVICE'
    status: str = "ARRIVED"

    # Event timestamps
    queue_enter_time: Optional[float] = None
    service_start_time: Optional[float] = None
    service_end_time: Optional[float] = None
    departure_time: Optional[float] = None

    # Performance durations
    wait_time: float = 0.0
    service_duration: float = 0.0
    total_system_time: float = 0.0

    # Assigned resources
    assigned_counter_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert passenger metrics to a dictionary for Pandas DataFrame construction."""
        return {
            "passenger_id": self.passenger_id,
            "arrival_time": self.arrival_time,
            "status": self.status,
            "queue_length_at_arrival": self.queue_length_at_arrival,
            "system_size_at_arrival": self.system_size_at_arrival,
            "patience_time": (
                None if math.isinf(self.patience_time) else round(self.patience_time, 4)
            ),
            "queue_enter_time": (
                round(self.queue_enter_time, 4) if self.queue_enter_time is not None else None
            ),
            "service_start_time": (
                round(self.service_start_time, 4) if self.service_start_time is not None else None
            ),
            "service_end_time": (
                round(self.service_end_time, 4) if self.service_end_time is not None else None
            ),
            "departure_time": (
                round(self.departure_time, 4) if self.departure_time is not None else None
            ),
            "wait_time": round(self.wait_time, 4),
            "service_duration": round(self.service_duration, 4),
            "total_system_time": round(self.total_system_time, 4),
            "assigned_counter_id": self.assigned_counter_id,
        }


def decide_balk(
    queue_len: int,
    config: SimulationConfig,
    rng: np.random.Generator
) -> bool:
    """
    Evaluate whether an arriving passenger decides to balk (refuse to enter the queue).

    Balking is a state-dependent behavior where arriving passengers judge the queue length:
    - If queue <= balking_threshold, passenger always joins (P_balk = 0).
    - If queue >= balking_max_queue, passenger always turns away (P_balk = 1).
    - In between, balking probability follows the configured strategy.
    """
    if not config.enable_balking:
        return False

    if queue_len <= config.balking_threshold:
        return False

    if queue_len >= config.balking_max_queue:
        return True

    if config.balking_strategy == "linear":
        prob_balk = (queue_len - config.balking_threshold) / (
            config.balking_max_queue - config.balking_threshold
        )
        return rng.uniform(0.0, 1.0) < prob_balk

    elif config.balking_strategy == "deterministic":
        return queue_len >= config.balking_max_queue

    elif config.balking_strategy == "probabilistic":
        # Nonlinear smooth growth: P(balk) = 1 - exp(-k * delta)
        delta = queue_len - config.balking_threshold
        prob_balk = 1.0 - math.exp(-0.25 * delta)
        return rng.uniform(0.0, 1.0) < prob_balk

    return False


def generate_patience_time(
    config: SimulationConfig,
    rng: np.random.Generator
) -> float:
    """
    Draw a passenger's reneging patience threshold from the configured probability distribution.
    """
    if not config.enable_reneging:
        return float("inf")

    if config.renege_distribution == "exponential":
        return rng.exponential(scale=config.renege_time_mean)
    elif config.renege_distribution == "uniform":
        return rng.uniform(config.renege_time_min, config.renege_time_max)
    elif config.renege_distribution == "constant":
        return config.renege_time_mean
    else:
        return rng.exponential(scale=config.renege_time_mean)


def passenger_process(
    env: simpy.Environment,
    passenger: Passenger,
    ticket_system: TicketCounterSystem,
    config: SimulationConfig,
    rng: np.random.Generator,
    event_logger: Optional[list] = None
):
    """
    SimPy process generator defining the full lifecycle of a railway ticket passenger:
    Arrival -> Balking Decision -> Queue Wait / Reneging Timeout -> Service -> Departure.
    """
    # Check for Balking if passenger is still in initial state
    if passenger.status == "BALKED":
        return

    if passenger.status == "ARRIVED":
        passenger.queue_length_at_arrival = ticket_system.queue_length
        passenger.system_size_at_arrival = ticket_system.total_passengers_in_system

        if decide_balk(passenger.queue_length_at_arrival, config, rng):
            passenger.status = "BALKED"
            passenger.departure_time = env.now
            passenger.wait_time = 0.0
            passenger.service_duration = 0.0
            passenger.total_system_time = 0.0

            if event_logger is not None:
                event_logger.append({
                    "time": env.now,
                    "event": "BALK",
                    "passenger_id": passenger.passenger_id,
                    "queue_length": ticket_system.queue_length,
                    "busy_counters": ticket_system.busy_counters_count,
                })
            return

    # 2. Join Queue
    if passenger.queue_enter_time is None:
        passenger.queue_enter_time = env.now
    passenger.status = "IN_QUEUE"
    passenger.patience_time = generate_patience_time(config, rng)

    if event_logger is not None:
        event_logger.append({
            "time": env.now,
            "event": "QUEUE_ENTER",
            "passenger_id": passenger.passenger_id,
            "queue_length": ticket_system.queue_length + 1,
            "busy_counters": ticket_system.busy_counters_count,
        })

    # 3. Request Counter with potential Reneging Timeout
    req = ticket_system.resource.request()

    if math.isinf(passenger.patience_time):
        # Infinite patience: passenger waits indefinitely until counter becomes free
        yield req
        has_counter = True
    else:
        # Limited patience: race between counter availability and patience expiration
        patience_timer = env.timeout(passenger.patience_time)
        result = yield req | patience_timer

        if req in result:
            has_counter = True
        else:
            # Patience expired: RENEGE
            has_counter = False
            # Cancel outstanding resource request to prevent queue corruption
            req.cancel()

    # 4. Handle Reneging Outcome
    if not has_counter:
        passenger.status = "RENEGED"
        passenger.departure_time = env.now
        passenger.wait_time = env.now - passenger.queue_enter_time
        passenger.service_duration = 0.0
        passenger.total_system_time = passenger.wait_time

        if event_logger is not None:
            event_logger.append({
                "time": env.now,
                "event": "RENEGE",
                "passenger_id": passenger.passenger_id,
                "queue_length": ticket_system.queue_length,
                "busy_counters": ticket_system.busy_counters_count,
            })
        return

    # 5. Service at Assigned Counter
    counter = ticket_system.allocate_idle_counter(passenger.passenger_id)
    passenger.assigned_counter_id = counter.counter_id
    passenger.service_start_time = env.now
    passenger.wait_time = env.now - passenger.queue_enter_time
    passenger.status = "IN_SERVICE"

    if event_logger is not None:
        event_logger.append({
            "time": env.now,
            "event": "SERVICE_START",
            "passenger_id": passenger.passenger_id,
            "counter_id": counter.counter_id,
            "queue_length": ticket_system.queue_length,
            "busy_counters": ticket_system.busy_counters_count,
        })

    # Sample service duration from Exponential distribution (mean = 1 / mu)
    service_duration = rng.exponential(scale=config.mean_service_time)
    yield env.timeout(service_duration)

    # 6. Service Completion and Departure
    ticket_system.release_counter(counter)
    ticket_system.resource.release(req)

    passenger.service_end_time = env.now
    passenger.departure_time = env.now
    passenger.service_duration = service_duration
    passenger.total_system_time = passenger.wait_time + passenger.service_duration
    passenger.status = "SERVED"

    if event_logger is not None:
        event_logger.append({
            "time": env.now,
            "event": "SERVICE_COMPLETE",
            "passenger_id": passenger.passenger_id,
            "counter_id": counter.counter_id,
            "queue_length": ticket_system.queue_length,
            "busy_counters": ticket_system.busy_counters_count,
        })

