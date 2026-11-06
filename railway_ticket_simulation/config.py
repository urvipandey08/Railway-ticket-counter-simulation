"""
config.py
Configuration module for Railway Ticket-Counter Simulation.
Course: Solving Problems Using Modelling and Simulation (NEC00091)

This module encapsulates all simulation parameters for an M/M/c queuing model
extended with passenger behavioral mechanisms: balking and reneging.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Any, Optional


@dataclass
class SimulationConfig:
    """Configuration parameters for the Railway Ticket Counter Simulation."""

    # Random Seed for Reproducibility
    random_seed: int = 42

    # Simulation Horizon (in minutes, e.g., 8-hour shift = 480 minutes)
    sim_time: float = 480.0

    # Number of Ticket Counters (c servers)
    num_counters: int = 3

    # Arrival Rate (lambda: passengers arriving per minute, Poisson process)
    # Default: Normal scenario = 0.60 passengers/minute
    arrival_rate: float = 0.60

    # Service Rate (mu: passengers served per minute per counter, Exponential service time)
    # Mean service time = 1 / service_rate (1 / 0.30 = 3.33 minutes)
    service_rate: float = 0.30

    # Simulation Scenario Label: "Normal" (0.60), "Peak" (0.80), "Overloaded" (1.00)
    scenario_name: str = "Normal"

    # Balking Configuration
    # Balking: Passenger arrives, observes queue, and decides not to join if queue is too long.
    enable_balking: bool = True
    balking_threshold: int = 5       # Queue length below which balking probability is 0
    balking_max_queue: int = 15      # Queue length at or above which balking probability is 1.0
    balking_strategy: str = "linear" # "linear", "deterministic", or "probabilistic"

    # Reneging Configuration
    # Reneging: Passenger joins the queue but leaves without service if wait time exceeds patience.
    enable_reneging: bool = True
    renege_distribution: str = "exponential" # "exponential", "uniform", "constant"
    renege_time_mean: float = 10.0   # Mean patience time before reneging (minutes)
    renege_time_min: float = 4.0     # For uniform patience distribution
    renege_time_max: float = 16.0    # For uniform patience distribution

    # Sampling and Monitoring
    queue_sample_interval: float = 0.5 # Interval (minutes) for continuous time-series logging

    # Output settings
    output_dir: str = "outputs"
    save_plots: bool = True
    show_plots: bool = False

    def __post_init__(self):
        """Validate input parameters."""
        if self.num_counters < 1:
            raise ValueError("Number of counters must be at least 1.")
        if self.arrival_rate <= 0:
            raise ValueError("Arrival rate (lambda) must be positive.")
        if self.service_rate <= 0:
            raise ValueError("Service rate (mu) must be positive.")
        if self.sim_time <= 0:
            raise ValueError("Simulation time must be positive.")
        if self.balking_max_queue <= self.balking_threshold:
            raise ValueError("balking_max_queue must be strictly greater than balking_threshold.")

    @property
    def mean_interarrival_time(self) -> float:
        """Mean time between successive arrivals (1 / lambda)."""
        return 1.0 / self.arrival_rate

    @property
    def mean_service_time(self) -> float:
        """Mean service time per passenger (1 / mu)."""
        return 1.0 / self.service_rate

    @property
    def traffic_intensity(self) -> float:
        """Traffic intensity / counter utilization factor (rho = lambda / (c * mu))."""
        return self.arrival_rate / (self.num_counters * self.service_rate)

    @property
    def offered_load(self) -> float:
        """Offered load in Erlangs (a = lambda / mu)."""
        return self.arrival_rate / self.service_rate

    def calculate_theoretical_mmc(self) -> Dict[str, Optional[float]]:
        """
        Calculate analytical theoretical metrics for an ideal infinite-capacity M/M/c queue
        (without balking or reneging) using Kendall's Erlang-C formulation.

        Returns:
            Dict containing theoretical rho, P0, Pq, Lq, Wq, W, L.
        """
        c = self.num_counters
        lam = self.arrival_rate
        mu = self.service_rate
        a = lam / mu
        rho = a / c

        if rho >= 1.0:
            return {
                "traffic_intensity_rho": rho,
                "is_stable": False,
                "status": "UNSTABLE",
                "status_message": f"UNSTABLE (Traffic Intensity ρ = {rho*100:.1f}% ≥ 100%)",
                "unstable_warning": (
                    f"Theoretical Erlang-C is invalid because arrival rate λ ({lam:.2f}) ≥ "
                    f"total capacity c·μ ({c * mu:.2f}). In an infinite queue without abandonment, "
                    "queue length and wait time would grow infinitely. The actual simulation is stabilized "
                    "by passenger balking and reneging."
                ),
                "p0": None,
                "prob_waiting_erlang_c": None,
                "expected_queue_length_Lq": None,
                "expected_wait_time_Wq": None,
                "expected_system_time_W": None,
                "expected_system_size_L": None,
            }

        # Calculate P0 (Idle probability)
        # P0 = [sum_{n=0}^{c-1} (a^n / n!) + (a^c / (c! * (1 - rho)))]^(-1)
        sum_terms = sum((a ** n) / math.factorial(n) for n in range(c))
        erlang_term = (a ** c) / (math.factorial(c) * (1.0 - rho))
        p0 = 1.0 / (sum_terms + erlang_term)

        # Erlang-C formula: P(Wait > 0) = Pq
        erlang_c = erlang_term * p0

        # Lq = Pq * rho / (1 - rho)
        lq = (erlang_c * rho) / (1.0 - rho)

        # Wq = Lq / lambda (Little's Law for queue)
        wq = lq / lam

        # W = Wq + 1 / mu (Total time in system)
        w = wq + (1.0 / mu)

        # L = lambda * W (Little's Law for system)
        l_sys = lam * w

        return {
            "traffic_intensity_rho": rho,
            "is_stable": True,
            "status": "STABLE",
            "status_message": f"STABLE (Traffic Intensity ρ = {rho*100:.1f}% < 100%)",
            "unstable_warning": None,
            "p0": p0,
            "prob_waiting_erlang_c": erlang_c,
            "expected_queue_length_Lq": lq,
            "expected_wait_time_Wq": wq,
            "expected_system_time_W": w,
            "expected_system_size_L": l_sys,
        }


# Standard Presets for the 3 Course Scenarios
SIMULATION_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Normal": {
        "lambda": 0.60,
        "mu": 0.30,
        "counters": 3,
        "description": "Normal operation (λ=0.60, ρ=66.7%). System is stable with low delays.",
    },
    "Peak": {
        "lambda": 0.80,
        "mu": 0.30,
        "counters": 3,
        "description": "Peak rush hour (λ=0.80, ρ=88.9%). System is stable but highly congested.",
    },
    "Overloaded": {
        "lambda": 1.00,
        "mu": 0.30,
        "counters": 3,
        "description": "Overloaded demand (λ=1.00, ρ=111.1%). Unstable in theory; stabilized by balking/reneging.",
    },
}


