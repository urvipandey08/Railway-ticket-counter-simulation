"""
counter.py
Railway Ticket-Counter Management Module.
Course: Solving Problems Using Modelling and Simulation (NEC00091)

This module manages the ticket counters (servers in M/M/c queuing system),
tracking active services, idle/busy states, and utilization metrics.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import simpy
from config import SimulationConfig


@dataclass
class TicketCounter:
    """Represents an individual railway ticket counter (server)."""
    counter_id: int
    is_busy: bool = False
    current_passenger_id: Optional[int] = None
    total_busy_time: float = 0.0
    customers_served: int = 0
    service_start_timestamp: Optional[float] = None
    state_history: List[Dict[str, Any]] = field(default_factory=list)

    def start_service(self, passenger_id: int, current_time: float) -> None:
        """Mark counter as busy serving the specified passenger."""
        self.is_busy = True
        self.current_passenger_id = passenger_id
        self.service_start_timestamp = current_time
        self.state_history.append({
            "timestamp": current_time,
            "status": "BUSY",
            "passenger_id": passenger_id
        })

    def complete_service(self, current_time: float) -> float:
        """Mark counter as idle upon service completion and accumulate busy time."""
        if self.service_start_timestamp is not None:
            duration = current_time - self.service_start_timestamp
            self.total_busy_time += duration
        else:
            duration = 0.0

        self.is_busy = False
        self.current_passenger_id = None
        self.service_start_timestamp = None
        self.customers_served += 1
        self.state_history.append({
            "timestamp": current_time,
            "status": "IDLE",
            "passenger_id": None
        })
        return duration

    def finalize(self, sim_end_time: float) -> None:
        """If counter was still busy at simulation cutoff, account for partial busy time."""
        if self.is_busy and self.service_start_timestamp is not None:
            duration = sim_end_time - self.service_start_timestamp
            self.total_busy_time += duration
            self.state_history.append({
                "timestamp": sim_end_time,
                "status": "CUTOFF_BUSY",
                "passenger_id": self.current_passenger_id
            })


class TicketCounterSystem:
    """
    Manages the multi-server railway ticket booking facility.
    Wraps SimPy's Resource to coordinate the M/M/c queue and tracks
    per-counter performance metrics.
    """

    def __init__(self, env: simpy.Environment, config: SimulationConfig):
        self.env = env
        self.config = config
        self.resource = simpy.Resource(env, capacity=config.num_counters)
        self.counters: List[TicketCounter] = [
            TicketCounter(counter_id=i + 1) for i in range(config.num_counters)
        ]

    @property
    def queue_length(self) -> int:
        """Number of passengers currently waiting in queue (excluding those being served)."""
        return len(self.resource.queue)

    @property
    def busy_counters_count(self) -> int:
        """Number of counters currently serving passengers."""
        return self.resource.count

    @property
    def total_passengers_in_system(self) -> int:
        """Total passengers in system (waiting in queue + currently being served)."""
        return self.queue_length + self.busy_counters_count

    def allocate_idle_counter(self, passenger_id: int) -> TicketCounter:
        """
        Find and allocate an available ticket counter to a passenger.
        Returns the assigned TicketCounter instance.
        """
        for counter in self.counters:
            if not counter.is_busy:
                counter.start_service(passenger_id, self.env.now)
                return counter
        # Fallback (should not happen if SimPy resource capacity matches counters count)
        counter = self.counters[0]
        counter.start_service(passenger_id, self.env.now)
        return counter

    def release_counter(self, counter: TicketCounter) -> float:
        """Release counter after passenger service is complete."""
        return counter.complete_service(self.env.now)

    def finalize(self, sim_end_time: float) -> None:
        """Finalize all counters at the end of the simulation."""
        for counter in self.counters:
            counter.finalize(sim_end_time)

    def get_utilization_summary(self, sim_time: float) -> Dict[str, Any]:
        """
        Compute empirical utilization statistics for each counter and overall system.

        Returns:
            Dict containing per-counter utilization, total busy times, and overall system utilization.
        """
        counter_stats = []
        total_system_busy_time = 0.0

        for counter in self.counters:
            utilization = counter.total_busy_time / sim_time if sim_time > 0 else 0.0
            counter_stats.append({
                "counter_id": counter.counter_id,
                "customers_served": counter.customers_served,
                "total_busy_time": counter.total_busy_time,
                "utilization": utilization,
            })
            total_system_busy_time += counter.total_busy_time

        overall_utilization = (
            total_system_busy_time / (self.config.num_counters * sim_time)
            if sim_time > 0 else 0.0
        )

        return {
            "counters": counter_stats,
            "total_system_busy_time": total_system_busy_time,
            "overall_utilization": overall_utilization,
            "theoretical_rho": self.config.traffic_intensity,
        }

