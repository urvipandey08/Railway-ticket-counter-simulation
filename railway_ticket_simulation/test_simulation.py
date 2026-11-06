"""
test_simulation.py
Unit and validation test suite for Railway Ticket-Counter Simulation.
Course: Solving Problems Using Modelling and Simulation (NEC00091)
"""

import unittest
import numpy as np
import pandas as pd

from config import SimulationConfig
from counter import TicketCounter, TicketCounterSystem
from passenger import Passenger, decide_balk, generate_patience_time
from simulation import RailwaySimulation


class TestQueueTheoryAndSimulation(unittest.TestCase):
    """Test suite covering theoretical calculations, entity logic, and simulation execution."""

    def test_theoretical_mmc_values(self):
        """Verify Erlang-C theoretical calculation against known analytical properties."""
        # For c=1 (M/M/1 queue): rho = lambda / mu = 0.5 / 1.0 = 0.5
        # Lq = rho^2 / (1 - rho) = 0.25 / 0.5 = 0.5
        # Wq = Lq / lambda = 0.5 / 0.5 = 1.0
        config = SimulationConfig(
            num_counters=1,
            arrival_rate=0.5,
            service_rate=1.0,
        )
        theo = config.calculate_theoretical_mmc()
        self.assertTrue(theo["is_stable"])
        self.assertAlmostEqual(theo["traffic_intensity_rho"], 0.5, places=4)
        self.assertAlmostEqual(theo["expected_queue_length_Lq"], 0.5, places=4)
        self.assertAlmostEqual(theo["expected_wait_time_Wq"], 1.0, places=4)
        self.assertAlmostEqual(theo["expected_system_size_L"], 1.0, places=4)

    def test_unstable_system_detection(self):
        """Ensure system detects unstable conditions (rho >= 1.0)."""
        config = SimulationConfig(
            num_counters=2,
            arrival_rate=2.0,
            service_rate=0.5,  # rho = 2.0 / (2 * 0.5) = 2.0 >= 1
        )
        theo = config.calculate_theoretical_mmc()
        self.assertFalse(theo["is_stable"])
        self.assertIsNone(theo["expected_queue_length_Lq"])

    def test_balking_decision_logic(self):
        """Test balking behavior at boundary and threshold conditions."""
        config = SimulationConfig(
            enable_balking=True,
            balking_threshold=5,
            balking_max_queue=10,
            balking_strategy="linear",
        )
        rng = np.random.default_rng(42)

        # Queue <= 5: Should NEVER balk
        for q in range(6):
            self.assertFalse(decide_balk(q, config, rng))

        # Queue >= 10: Should ALWAYS balk
        for q in [10, 12, 20]:
            self.assertTrue(decide_balk(q, config, rng))

    def test_reproducibility_with_seed(self):
        """Ensure identical seeds yield identical passenger event counts and stats."""
        config1 = SimulationConfig(random_seed=123, sim_time=120.0, num_counters=3)
        config2 = SimulationConfig(random_seed=123, sim_time=120.0, num_counters=3)

        sim1 = RailwaySimulation(config1)
        sim2 = RailwaySimulation(config2)

        res1 = sim1.run()
        res2 = sim2.run()

        self.assertEqual(res1["passenger_counts"]["total_arrivals"], res2["passenger_counts"]["total_arrivals"])
        self.assertEqual(res1["passenger_counts"]["num_served"], res2["passenger_counts"]["num_served"])
        self.assertEqual(res1["passenger_counts"]["num_balked"], res2["passenger_counts"]["num_balked"])
        self.assertEqual(res1["passenger_counts"]["num_reneged"], res2["passenger_counts"]["num_reneged"])

    def test_dataframe_structure(self):
        """Ensure output DataFrames have all required columns and valid records."""
        config = SimulationConfig(random_seed=42, sim_time=60.0, num_counters=2)
        sim = RailwaySimulation(config)
        sim.run()

        # Passenger DataFrame check
        self.assertIsInstance(sim.passengers_df, pd.DataFrame)
        required_cols = [
            "passenger_id", "arrival_time", "status", "wait_time",
            "service_duration", "total_system_time"
        ]
        for col in required_cols:
            self.assertIn(col, sim.passengers_df.columns)

        # Counter DataFrame check
        self.assertIsInstance(sim.counter_df, pd.DataFrame)
        self.assertEqual(len(sim.counter_df), config.num_counters)
        self.assertTrue((sim.counter_df["utilization"] >= 0).all())
        self.assertTrue((sim.counter_df["utilization"] <= 1.05).all())

        # Queue time series check
        self.assertIsInstance(sim.queue_df, pd.DataFrame)
        self.assertTrue((sim.queue_df["queue_length"] >= 0).all())

    def test_default_simulation_parameters(self):
        """Verify default simulation parameters match course specifications."""
        config = SimulationConfig()
        self.assertEqual(config.num_counters, 3)
        self.assertAlmostEqual(config.arrival_rate, 0.60, places=4)
        self.assertAlmostEqual(config.service_rate, 0.30, places=4)
        self.assertAlmostEqual(config.mean_service_time, 1.0 / 0.30, places=2)
        self.assertAlmostEqual(config.traffic_intensity, 0.60 / (3 * 0.30), places=4)

    def test_three_simulation_scenarios_stability(self):
        """Verify Erlang-C stability and metrics for Normal, Peak, and Overloaded scenarios."""
        # 1. Normal: lambda = 0.60, rho = 0.667 < 1.0 (Stable)
        cfg_normal = SimulationConfig(num_counters=3, arrival_rate=0.60, service_rate=0.30)
        theo_normal = cfg_normal.calculate_theoretical_mmc()
        self.assertTrue(theo_normal["is_stable"])
        self.assertIsNotNone(theo_normal["expected_wait_time_Wq"])
        self.assertAlmostEqual(theo_normal["traffic_intensity_rho"], 2.0 / 3.0, places=4)

        # 2. Peak: lambda = 0.80, rho = 0.889 < 1.0 (Stable)
        cfg_peak = SimulationConfig(num_counters=3, arrival_rate=0.80, service_rate=0.30)
        theo_peak = cfg_peak.calculate_theoretical_mmc()
        self.assertTrue(theo_peak["is_stable"])
        self.assertIsNotNone(theo_peak["expected_wait_time_Wq"])
        self.assertAlmostEqual(theo_peak["traffic_intensity_rho"], 0.80 / 0.90, places=4)

        # 3. Overloaded: lambda = 1.00, rho = 1.111 >= 1.0 (Unstable)
        cfg_overloaded = SimulationConfig(num_counters=3, arrival_rate=1.00, service_rate=0.30)
        theo_overloaded = cfg_overloaded.calculate_theoretical_mmc()
        self.assertFalse(theo_overloaded["is_stable"])
        self.assertIsNone(theo_overloaded["expected_wait_time_Wq"])
        self.assertIsNone(theo_overloaded["expected_queue_length_Lq"])
        self.assertIn("UNSTABLE", theo_overloaded["status_message"])

    def test_passenger_accounting_identity(self):
        """Verify strict passenger accounting: Total Arrivals = Served + Balked + Reneged + Waiting + In_Service."""
        for scenario_lam in [0.60, 0.80, 1.00]:
            config = SimulationConfig(random_seed=42, sim_time=240.0, num_counters=3, arrival_rate=scenario_lam)
            sim = RailwaySimulation(config)
            stats = sim.run()

            counts = stats["passenger_counts"]
            total = counts["total_arrivals"]
            served = counts["num_served"]
            balked = counts["num_balked"]
            reneged = counts["num_reneged"]
            waiting = counts["num_waiting"]
            in_srv = counts["num_in_service"]

            accounted_sum = served + balked + reneged + waiting + in_srv
            self.assertEqual(total, accounted_sum, f"Accounting discrepancy in scenario lambda={scenario_lam}: {total} != {accounted_sum}")
            self.assertEqual(stats["accounting"]["discrepancy"], 0)
            self.assertTrue(stats["accounting"]["is_valid"])


if __name__ == "__main__":
    unittest.main()

