"""
simulation.py
Core simulation engine for Railway Ticket-Counter Simulation.
Course: Solving Problems Using Modelling and Simulation (NEC00091)

This module implements the discrete-event simulation orchestrator, coordinating
passenger arrivals, queuing dynamics, counter resource management, periodic system
monitoring, and conversion of simulation metrics into Pandas DataFrames.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import numpy as np
import pandas as pd
import simpy

from config import SimulationConfig
from counter import TicketCounterSystem
from passenger import Passenger, passenger_process


class RailwaySimulation:
    """
    Simulation coordinator for the Railway Ticket Counter M/M/c queuing system.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config if config is not None else SimulationConfig()
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(self.config.random_seed)

        self.ticket_system = TicketCounterSystem(self.env, self.config)
        self.passengers: List[Passenger] = []
        self.events_log: List[Dict[str, Any]] = []
        self.queue_time_series: List[Dict[str, Any]] = []

        # DataFrames populated after run()
        self.passengers_df: Optional[pd.DataFrame] = None
        self.queue_df: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        self.counter_df: Optional[pd.DataFrame] = None
        self.summary_stats: Optional[Dict[str, Any]] = None

    def _passenger_generator(self):
        """
        SimPy generator process that creates passengers according to a Poisson process
        (exponentially distributed inter-arrival times with rate lambda).
        """
        passenger_id = 0
        while self.env.now < self.config.sim_time:
            # Sample inter-arrival time: Exp(lambda) -> mean = 1 / lambda
            interarrival_time = self.rng.exponential(scale=self.config.mean_interarrival_time)
            yield self.env.timeout(interarrival_time)

            if self.env.now >= self.config.sim_time:
                break

            passenger_id += 1
            # Check balking immediately at arrival to guarantee deterministic status
            q_len = self.ticket_system.queue_length
            sys_sz = self.ticket_system.total_passengers_in_system

            from passenger import decide_balk
            will_balk = decide_balk(q_len, self.config, self.rng)

            if will_balk:
                p = Passenger(
                    passenger_id=passenger_id,
                    arrival_time=self.env.now,
                    status="BALKED",
                    queue_length_at_arrival=q_len,
                    system_size_at_arrival=sys_sz,
                    departure_time=self.env.now,
                    wait_time=0.0,
                    service_duration=0.0,
                    total_system_time=0.0,
                )
                self.passengers.append(p)
                self.events_log.append({
                    "time": self.env.now,
                    "event": "BALK",
                    "passenger_id": p.passenger_id,
                    "queue_length": q_len,
                    "busy_counters": self.ticket_system.busy_counters_count,
                })
            else:
                p = Passenger(
                    passenger_id=passenger_id,
                    arrival_time=self.env.now,
                    status="IN_QUEUE",
                    queue_enter_time=self.env.now,
                    queue_length_at_arrival=q_len,
                    system_size_at_arrival=sys_sz,
                )
                self.passengers.append(p)

                # Spawn individual passenger lifecycle process
                self.env.process(
                    passenger_process(
                        env=self.env,
                        passenger=p,
                        ticket_system=self.ticket_system,
                        config=self.config,
                        rng=self.rng,
                        event_logger=self.events_log,
                    )
                )

    def _system_monitor(self):
        """
        SimPy generator process that periodically samples system state variables:
        current queue length, busy counters, and total passengers in system.
        """
        while self.env.now <= self.config.sim_time:
            q_len = self.ticket_system.queue_length
            busy = self.ticket_system.busy_counters_count
            total = q_len + busy
            idle = self.config.num_counters - busy

            self.queue_time_series.append({
                "time": round(self.env.now, 4),
                "queue_length": q_len,
                "busy_counters": busy,
                "idle_counters": idle,
                "passengers_in_system": total,
            })
            yield self.env.timeout(self.config.queue_sample_interval)

    def prepare(self) -> None:
        """Initialize and register simulation generator and monitor processes."""
        if getattr(self, "is_prepared", False):
            return
        self.passenger_gen_process = self.env.process(self._passenger_generator())
        self.monitor_process = self.env.process(self._system_monitor())
        self.is_prepared = True
        self.is_finalized = False

    def step_until(self, target_time: float) -> bool:
        """
        Step simulation forward until target_time.
        Returns True if simulation is active (env.now < config.sim_time), False if finished.
        """
        if not getattr(self, "is_prepared", False):
            self.prepare()

        horizon = min(self.config.sim_time, max(self.env.now, target_time))
        if self.env.now < horizon:
            self.env.run(until=horizon)

        if self.env.now >= self.config.sim_time:
            if not getattr(self, "is_finalized", False):
                self.ticket_system.finalize(self.config.sim_time)
                self._build_dataframes()
                self.summary_stats = self._compute_summary_stats()
                self.is_finalized = True
            return False
        return True

    def inject_passenger(self) -> Passenger:
        """Manually inject a passenger at current simulation time (useful for live demonstrations)."""
        if not getattr(self, "is_prepared", False):
            self.prepare()
        new_id = len(self.passengers) + 1
        q_len = self.ticket_system.queue_length
        sys_sz = self.ticket_system.total_passengers_in_system

        from passenger import decide_balk
        will_balk = decide_balk(q_len, self.config, self.rng)

        if will_balk:
            p = Passenger(
                passenger_id=new_id,
                arrival_time=self.env.now,
                status="BALKED",
                queue_length_at_arrival=q_len,
                system_size_at_arrival=sys_sz,
                departure_time=self.env.now,
                wait_time=0.0,
                service_duration=0.0,
                total_system_time=0.0,
            )
            self.passengers.append(p)
            self.events_log.append({
                "time": self.env.now,
                "event": "BALK",
                "passenger_id": p.passenger_id,
                "queue_length": q_len,
                "busy_counters": self.ticket_system.busy_counters_count,
            })
        else:
            p = Passenger(
                passenger_id=new_id,
                arrival_time=self.env.now,
                status="IN_QUEUE",
                queue_enter_time=self.env.now,
                queue_length_at_arrival=q_len,
                system_size_at_arrival=sys_sz,
            )
            self.passengers.append(p)
            self.env.process(
                passenger_process(
                    env=self.env,
                    passenger=p,
                    ticket_system=self.ticket_system,
                    config=self.config,
                    rng=self.rng,
                    event_logger=self.events_log,
                )
            )
        return p

    def run(self) -> Dict[str, Any]:
        """
        Execute the simulation up to config.sim_time, post-process metrics,
        and generate structured Pandas DataFrames.

        Returns:
            summary_stats dictionary containing comprehensive empirical and theoretical metrics.
        """
        if not getattr(self, "is_prepared", False):
            self.prepare()

        # Execute discrete-event simulation
        self.env.run(until=self.config.sim_time)

        # Finalize counter busy time accounting
        if not getattr(self, "is_finalized", False):
            self.ticket_system.finalize(self.config.sim_time)
            self._build_dataframes()
            self.summary_stats = self._compute_summary_stats()
            self.is_finalized = True

        return self.summary_stats

    def _build_dataframes(self) -> None:
        """Convert raw event logs and entity lists into structured Pandas DataFrames."""
        # 1. Passengers DataFrame
        passenger_dicts = [p.to_dict() for p in self.passengers]
        self.passengers_df = pd.DataFrame(passenger_dicts)
        if self.passengers_df.empty:
            # Handle edge case where no passengers were generated
            self.passengers_df = pd.DataFrame(columns=[
                "passenger_id", "arrival_time", "status", "queue_length_at_arrival",
                "system_size_at_arrival", "patience_time", "queue_enter_time",
                "service_start_time", "service_end_time", "departure_time",
                "wait_time", "service_duration", "total_system_time", "assigned_counter_id"
            ])

        # 2. Queue Time Series DataFrame
        self.queue_df = pd.DataFrame(self.queue_time_series)
        if not self.queue_df.empty:
            self.queue_df["rolling_avg_queue"] = (
                self.queue_df["queue_length"].rolling(window=10, min_periods=1).mean()
            )

        # 3. Events DataFrame
        self.events_df = pd.DataFrame(self.events_log)

        # 4. Counter Utilization DataFrame
        effective_time = (
            self.config.sim_time if (getattr(self, "is_finalized", False) or self.env.now >= self.config.sim_time)
            else max(self.env.now, 0.001)
        )
        util_summary = self.ticket_system.get_utilization_summary(effective_time)
        self.counter_df = pd.DataFrame(util_summary["counters"])
        self.counter_df["utilization_percent"] = self.counter_df["utilization"] * 100.0

    def _compute_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregate simulation performance metrics and compare against theoretical values."""
        if self.passengers_df is None:
            self._build_dataframes()
        df = self.passengers_df
        total_arrivals = len(df)

        served_df = df[df["status"] == "SERVED"]
        balked_df = df[df["status"] == "BALKED"]
        reneged_df = df[df["status"] == "RENEGED"]
        waiting_df = df[df["status"] == "IN_QUEUE"]
        in_service_df = df[df["status"] == "IN_SERVICE"]

        num_served = len(served_df)
        num_balked = len(balked_df)
        num_reneged = len(reneged_df)
        num_waiting = len(waiting_df)
        num_in_service = len(in_service_df)
        num_abandoned = num_balked + num_reneged
        num_active = num_waiting + num_in_service

        pct_served = (num_served / total_arrivals * 100.0) if total_arrivals > 0 else 0.0
        pct_balked = (num_balked / total_arrivals * 100.0) if total_arrivals > 0 else 0.0
        pct_reneged = (num_reneged / total_arrivals * 100.0) if total_arrivals > 0 else 0.0
        pct_abandoned = (num_abandoned / total_arrivals * 100.0) if total_arrivals > 0 else 0.0
        pct_waiting = (num_waiting / total_arrivals * 100.0) if total_arrivals > 0 else 0.0
        pct_in_service = (num_in_service / total_arrivals * 100.0) if total_arrivals > 0 else 0.0
        pct_active = (num_active / total_arrivals * 100.0) if total_arrivals > 0 else 0.0

        # Verification of Passenger Accounting Identity:
        # Total Arrivals = Served + Balked + Reneged + Currently Waiting + Currently In Service
        accounted_total = num_served + num_balked + num_reneged + num_waiting + num_in_service
        discrepancy = total_arrivals - accounted_total
        is_accounting_valid = (discrepancy == 0)

        # Wait-time metrics
        # For passengers who joined queue (served + reneged)
        queued_passengers = df[df["status"].isin(["SERVED", "RENEGED"])]
        avg_wait_queued = queued_passengers["wait_time"].mean() if not queued_passengers.empty else 0.0
        avg_wait_served = served_df["wait_time"].mean() if not served_df.empty else 0.0
        max_wait_served = served_df["wait_time"].max() if not served_df.empty else 0.0
        median_wait_served = served_df["wait_time"].median() if not served_df.empty else 0.0

        avg_service_time = served_df["service_duration"].mean() if not served_df.empty else 0.0
        avg_time_in_system = served_df["total_system_time"].mean() if not served_df.empty else 0.0

        # Queue length metrics from time-series log
        if not self.queue_df.empty:
            avg_queue_len = float(self.queue_df["queue_length"].mean())
            max_queue_len = int(self.queue_df["queue_length"].max())
            avg_system_size = float(self.queue_df["passengers_in_system"].mean())
        else:
            avg_queue_len = 0.0
            max_queue_len = 0
            avg_system_size = 0.0

        # Server utilization
        effective_time = (
            self.config.sim_time if (getattr(self, "is_finalized", False) or self.env.now >= self.config.sim_time)
            else max(self.env.now, 0.001)
        )
        counter_summary = self.ticket_system.get_utilization_summary(effective_time)
        overall_utilization = counter_summary["overall_utilization"]
        individual_counter_utilization = {
            c["counter_id"]: round(c["utilization"] * 100.0, 2)
            for c in counter_summary["counters"]
        }

        # Theoretical Erlang-C metrics
        theoretical = self.config.calculate_theoretical_mmc()

        return {
            "simulation_parameters": {
                "sim_time": self.config.sim_time,
                "num_counters": self.config.num_counters,
                "arrival_rate_lambda": self.config.arrival_rate,
                "service_rate_mu": self.config.service_rate,
                "scenario_name": getattr(self.config, "scenario_name", "Normal"),
                "balking_enabled": self.config.enable_balking,
                "balking_threshold": self.config.balking_threshold,
                "balking_max_queue": self.config.balking_max_queue,
                "reneging_enabled": self.config.enable_reneging,
                "renege_mean_patience": self.config.renege_time_mean,
            },
            "passenger_counts": {
                "total_arrivals": total_arrivals,
                "num_served": num_served,
                "num_balked": num_balked,
                "num_reneged": num_reneged,
                "num_abandoned": num_abandoned,
                "num_waiting": num_waiting,
                "num_in_service": num_in_service,
                "num_in_system_at_end": num_active,
                "percent_served": round(pct_served, 2),
                "percent_balked": round(pct_balked, 2),
                "percent_reneged": round(pct_reneged, 2),
                "percent_abandoned": round(pct_abandoned, 2),
                "percent_waiting": round(pct_waiting, 2),
                "percent_in_service": round(pct_in_service, 2),
                "percent_in_system_at_end": round(pct_active, 2),
            },
            "timing_metrics": {
                "avg_wait_time_served": round(avg_wait_served, 4),
                "median_wait_time_served": round(median_wait_served, 4),
                "max_wait_time_served": round(max_wait_served, 4),
                "avg_wait_time_queued": round(avg_wait_queued, 4),
                "avg_service_time_served": round(avg_service_time, 4),
                "avg_system_time_served": round(avg_time_in_system, 4),
            },
            "queue_metrics": {
                "time_averaged_queue_length": round(avg_queue_len, 4),
                "max_queue_length": max_queue_len,
                "time_averaged_system_size": round(avg_system_size, 4),
            },
            "utilization": {
                "empirical_overall_utilization": round(overall_utilization, 4),
                "theoretical_traffic_intensity_rho": round(self.config.traffic_intensity, 4),
                "individual_counter_utilization": individual_counter_utilization,
            },
            "accounting": {
                "total_arrivals": total_arrivals,
                "accounted_total": accounted_total,
                "discrepancy": discrepancy,
                "is_valid": is_accounting_valid,
                "formula": (
                    f"Total Arrivals ({total_arrivals}) = Served ({num_served}) + "
                    f"Balked ({num_balked}) + Reneged ({num_reneged}) + "
                    f"Waiting ({num_waiting}) + In Service ({num_in_service})"
                ),
            },
            "theoretical_mmc": theoretical,
        }

    def save_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save simulation logs and DataFrames to CSV files.

        Returns:
            Dict of exported file paths.
        """
        target_dir = output_dir or self.config.output_dir
        os.makedirs(target_dir, exist_ok=True)

        paths = {
            "passengers": os.path.join(target_dir, "passenger_logs.csv"),
            "queue": os.path.join(target_dir, "queue_time_series.csv"),
            "events": os.path.join(target_dir, "events_log.csv"),
            "counters": os.path.join(target_dir, "counter_utilization.csv"),
        }

        if self.passengers_df is not None:
            self.passengers_df.to_csv(paths["passengers"], index=False)
        if self.queue_df is not None:
            self.queue_df.to_csv(paths["queue"], index=False)
        if self.events_df is not None:
            self.events_df.to_csv(paths["events"], index=False)
        if self.counter_df is not None:
            self.counter_df.to_csv(paths["counters"], index=False)

        return paths


def run_scenario_comparison(
    sim_time: float = 480.0,
    random_seed: int = 42,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Execute simulation across the 3 standard course scenarios:
    1. Normal: lambda = 0.60, mu = 0.30, c = 3
    2. Peak: lambda = 0.80, mu = 0.30, c = 3
    3. Overloaded: lambda = 1.00, mu = 0.30, c = 3

    Returns:
        Dictionary mapping scenario name to its simulation instance, config, and statistics.
    """
    from config import SIMULATION_SCENARIOS

    results = {}
    for scenario_name, spec in SIMULATION_SCENARIOS.items():
        cfg = SimulationConfig(
            random_seed=random_seed,
            sim_time=sim_time,
            num_counters=spec["counters"],
            arrival_rate=spec["lambda"],
            service_rate=spec["mu"],
            scenario_name=scenario_name,
            output_dir=output_dir,
        )
        sim = RailwaySimulation(cfg)
        stats = sim.run()
        results[scenario_name] = {
            "config": cfg,
            "sim": sim,
            "stats": stats,
            "passengers_df": sim.passengers_df,
            "queue_df": sim.queue_df,
            "counter_df": sim.counter_df,
        }
    return results

