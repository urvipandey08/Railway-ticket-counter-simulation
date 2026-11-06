"""
main.py
Execution script for Railway Ticket-Counter Simulation.
Course: Solving Problems Using Modelling and Simulation (NEC00091)

Usage:
    python main.py
    python main.py --counters 4 --lambda-arr 1.2 --mu-srv 0.35 --sim-time 480
"""

import argparse
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from config import SimulationConfig, SIMULATION_SCENARIOS
from simulation import RailwaySimulation, run_scenario_comparison
from visualize import (
    plot_queue_length_over_time,
    plot_wait_time_distribution,
    plot_counter_utilization,
    plot_passenger_outcomes,
    plot_simulation_dashboard,
    plot_scenario_comparison,
)


def print_banner():
    banner = """
================================================================================
       RAILWAY TICKET-COUNTER SIMULATION SYSTEM (M/M/c Queuing Model)
        Course: Solving Problems Using Modelling and Simulation (NEC00091)
================================================================================
"""
    print(banner)


def print_summary_report(stats: dict):
    params = stats["simulation_parameters"]
    counts = stats["passenger_counts"]
    timing = stats["timing_metrics"]
    queue = stats["queue_metrics"]
    util = stats["utilization"]
    theo = stats.get("theoretical_mmc", {})
    acct = stats.get("accounting", {})

    print("\n" + "=" * 70)
    print("                     SIMULATION EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\n[1] CONFIGURATION & SYSTEM PARAMETERS:")
    print(f"  - Scenario                        : {params.get('scenario_name', 'Custom')}")
    print(f"  - Simulation Horizon (T)          : {params['sim_time']} minutes ({params['sim_time']/60:.1f} hours)")
    print(f"  - Number of Counters (c)          : {params['num_counters']}")
    print(f"  - Arrival Rate (lambda)           : {params['arrival_rate_lambda']} passengers/min")
    print(f"  - Service Rate (mu)               : {params['service_rate_mu']} passengers/min/counter")
    print(f"  - Mean Service Time (1/mu)        : {1.0/params['service_rate_mu']:.2f} minutes")
    print(f"  - Balking Enabled                 : {params['balking_enabled']} (Threshold: {params['balking_threshold']}, Max: {params['balking_max_queue']})")
    print(f"  - Reneging Enabled                : {params['reneging_enabled']} (Mean Patience: {params['renege_mean_patience']} min)")

    print("\n[2] PASSENGER LIFECYCLE & OUTCOMES:")
    print(f"  - Total Arrivals                  : {counts['total_arrivals']}")
    print(f"  - Served Successfully             : {counts['num_served']} ({counts['percent_served']}%)")
    print(f"  - Balked (Turned Away)            : {counts['num_balked']} ({counts['percent_balked']}%)")
    print(f"  - Reneged (Left Queue)            : {counts['num_reneged']} ({counts['percent_reneged']}%)")
    print(f"  - Total Abandoned (Balk + Renege) : {counts.get('num_abandoned', counts['num_balked'] + counts['num_reneged'])} ({counts.get('percent_abandoned', 0.0)}%)")
    print(f"  - Currently Waiting in Queue      : {counts.get('num_waiting', 0)}")
    print(f"  - Currently In Service            : {counts.get('num_in_service', 0)}")
    if counts.get("num_in_system_at_end", 0) > 0:
        print(f"  - Active in System at Cutoff      : {counts['num_in_system_at_end']} ({counts['percent_in_system_at_end']}%)")

    print("\n[3] PASSENGER ACCOUNTING IDENTITY CHECK:")
    if acct:
        status_tag = "VERIFIED [OK]" if acct.get("is_valid") else f"FAILED [Discrepancy: {acct.get('discrepancy')}]"
        print(f"  - Status                          : {status_tag}")
        print(f"  - Equation                        : {acct.get('formula')}")

    print("\n[4] QUEUE & WAITING PERFORMANCE METRICS:")
    print(f"  - Average Wait Time (Served)      : {timing['avg_wait_time_served']:.2f} minutes")
    print(f"  - Median Wait Time (Served)       : {timing['median_wait_time_served']:.2f} minutes")
    print(f"  - Maximum Wait Time (Served)      : {timing['max_wait_time_served']:.2f} minutes")
    print(f"  - Average Service Time            : {timing['avg_service_time_served']:.2f} minutes")
    print(f"  - Average Total System Time       : {timing['avg_system_time_served']:.2f} minutes")
    print(f"  - Time-Averaged Queue Length (Lq) : {queue['time_averaged_queue_length']:.2f} passengers")
    print(f"  - Peak Queue Length Recorded      : {queue['max_queue_length']} passengers")
    print(f"  - Time-Averaged System Size (L)   : {queue['time_averaged_system_size']:.2f} passengers")

    print("\n[5] SERVER UTILIZATION & CAPACITY:")
    print(f"  - Empirical System Utilization    : {util['empirical_overall_utilization']*100:.2f}%")
    print(f"  - Theoretical Traffic Intensity   : {util['theoretical_traffic_intensity_rho']*100:.2f}%")
    if "individual_counter_utilization" in util:
        c_utils = util["individual_counter_utilization"]
        for cid, u_val in c_utils.items():
            print(f"    • Counter {cid} Utilization       : {u_val:.1f}%")

    print("\n[6] THEORETICAL ERLANG-C BENCHMARK (Infinite Queue, No Abandonment):")
    if theo.get("is_stable"):
        print(f"  - System Stability                : STABLE (Traffic Intensity ρ < 100%)")
        print(f"  - Idle Probability (P0)           : {theo.get('p0', 0)*100:.2f}%")
        print(f"  - Queuing Probability (Erlang-C)  : {theo.get('prob_waiting_erlang_c', 0)*100:.2f}%")
        print(f"  - Expected Queue Length (Lq)      : {theo.get('expected_queue_length_Lq', 0):.2f} passengers")
        print(f"  - Expected Queue Wait Time (Wq)   : {theo.get('expected_wait_time_Wq', 0):.2f} minutes")
        print(f"  - Expected System Time (W)        : {theo.get('expected_system_time_W', 0):.2f} minutes")
        print(f"  - Insight                         : Actual simulation wait is bounded below theoretical Erlang-C by balking/reneging.")
    else:
        print(f"  - System Stability                : ⚠️ UNSTABLE (Traffic Intensity ρ = {theo.get('traffic_intensity_rho', 0)*100:.1f}% ≥ 100%)")
        print(f"  - Theoretical Lq and Wq           : N/A (Grows infinitely in an infinite queue without abandonment)")
        print(f"  - Simulation Stabilization        : Actual queue remains finite because balking and reneging limit backlog buildup.")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Railway Ticket-Counter Simulation (M/M/c Queuing System with Balking and Reneging)"
    )
    parser.add_argument("-c", "--counters", type=int, default=3, help="Number of ticket counters (default: 3)")
    parser.add_argument("-l", "--lambda-arr", type=float, default=0.60, help="Arrival rate lambda (passengers/min, default: 0.60)")
    parser.add_argument("-m", "--mu-srv", type=float, default=0.30, help="Service rate mu per counter (passengers/min, default: 0.30)")
    parser.add_argument("-t", "--sim-time", type=float, default=480.0, help="Simulation time horizon in minutes (default: 480.0)")
    parser.add_argument("--scenario", type=str, choices=["normal", "peak", "overloaded"], default=None, help="Select scenario preset: normal (0.60), peak (0.80), overloaded (1.00)")
    parser.add_argument("--compare-scenarios", action="store_true", help="Execute comparison across Normal, Peak, and Overloaded scenarios")
    parser.add_argument("--balk-thresh", type=int, default=5, help="Balking queue onset threshold (default: 5)")
    parser.add_argument("--balk-max", type=int, default=15, help="Balking maximum queue threshold (default: 15)")
    parser.add_argument("--no-balk", action="store_true", help="Disable passenger balking")
    parser.add_argument("--no-renege", action="store_true", help="Disable passenger reneging")
    parser.add_argument("--patience", type=float, default=10.0, help="Mean reneging patience in minutes (default: 10.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random number generator seed (default: 42)")
    parser.add_argument("-o", "--output-dir", type=str, default="outputs", help="Output directory for plots and CSV logs")

    args = parser.parse_args()

    print_banner()

    # Handle Scenario Comparison Mode
    if args.compare_scenarios:
        print("[*] Running comparative simulation across Normal, Peak, and Overloaded scenarios...")
        results = run_scenario_comparison(sim_time=args.sim_time, random_seed=args.seed, output_dir=args.output_dir)

        print("\n" + "=" * 80)
        print("                  SCENARIO COMPARISON MATRIX (3 COUNTERS, μ=0.30)")
        print("=" * 80)
        header = f"{'Scenario':<12} | {'λ':<5} | {'Theo ρ':<8} | {'Sim Util':<9} | {'Avg Wait':<9} | {'Theo Wq':<9} | {'Served':<8} | {'Abandoned':<10}"
        print(header)
        print("-" * 80)

        for name, data in results.items():
            s = data["stats"]
            cfg = data["config"]
            t_wq = s["theoretical_mmc"].get("expected_wait_time_Wq")
            t_wq_str = f"{t_wq:.2f}m" if t_wq is not None else "UNSTABLE"
            abn_pct = s["passenger_counts"].get("percent_abandoned", 0.0)
            abn_str = f"{s['passenger_counts']['num_abandoned']} ({abn_pct:.1f}%)"
            row = (
                f"{name:<12} | {cfg.arrival_rate:<5.2f} | "
                f"{s['utilization']['theoretical_traffic_intensity_rho']*100:<7.1f}% | "
                f"{s['utilization']['empirical_overall_utilization']*100:<8.1f}% | "
                f"{s['timing_metrics']['avg_wait_time_served']:<8.2f}m | "
                f"{t_wq_str:<9} | "
                f"{s['passenger_counts']['num_served']:<8} | "
                f"{abn_str:<10}"
            )
            print(row)
        print("=" * 80 + "\n")

        chart_path = os.path.join(args.output_dir, "scenario_comparison.png")
        plot_scenario_comparison(results, save_path=chart_path)
        print(f"[*] Comparative visualization saved to: {chart_path}")
        return

    # Scenario Selection Override
    arrival_rate = args.lambda_arr
    scenario_name = "Custom"
    if args.scenario:
        scenario_map = {"normal": ("Normal", 0.60), "peak": ("Peak", 0.80), "overloaded": ("Overloaded", 1.00)}
        scenario_name, arrival_rate = scenario_map[args.scenario.lower()]

    config = SimulationConfig(
        random_seed=args.seed,
        sim_time=args.sim_time,
        num_counters=args.counters,
        arrival_rate=arrival_rate,
        service_rate=args.mu_srv,
        enable_balking=not args.no_balk,
        balking_threshold=args.balk_thresh,
        balking_max_queue=args.balk_max,
        enable_reneging=not args.no_renege,
        renege_time_mean=args.patience,
        scenario_name=scenario_name,
        output_dir=args.output_dir,
    )

    print(f"[*] Initializing simulation: Scenario '{config.scenario_name}' with {config.num_counters} counters over {config.sim_time} minutes...")
    sim = RailwaySimulation(config)

    print("[*] Running discrete-event queuing simulation engine...")
    stats = sim.run()

    print("[*] Simulation complete. Logging event metrics and dataframes...")
    saved_csvs = sim.save_results()
    for name, path in saved_csvs.items():
        print(f"    -> Exported {name.capitalize()} Data: {path}")

    print("\n[*] Generating high-resolution analytical plots...")
    os.makedirs(config.output_dir, exist_ok=True)

    plot_paths = {
        "queue_length": os.path.join(config.output_dir, "queue_length_over_time.png"),
        "wait_time": os.path.join(config.output_dir, "wait_time_distribution.png"),
        "utilization": os.path.join(config.output_dir, "counter_utilization.png"),
        "outcomes": os.path.join(config.output_dir, "passenger_outcomes.png"),
        "dashboard": os.path.join(config.output_dir, "railway_simulation_dashboard.png"),
    }

    plot_queue_length_over_time(sim, save_path=plot_paths["queue_length"])
    print(f"    -> Generated: {plot_paths['queue_length']}")

    plot_wait_time_distribution(sim, save_path=plot_paths["wait_time"])
    print(f"    -> Generated: {plot_paths['wait_time']}")

    plot_counter_utilization(sim, save_path=plot_paths["utilization"])
    print(f"    -> Generated: {plot_paths['utilization']}")

    plot_passenger_outcomes(sim, save_path=plot_paths["outcomes"])
    print(f"    -> Generated: {plot_paths['outcomes']}")

    plot_simulation_dashboard(sim, save_path=plot_paths["dashboard"])
    print(f"    -> Generated: {plot_paths['dashboard']}")

    print_summary_report(stats)
    print(f"All artifacts successfully saved to '{os.path.abspath(config.output_dir)}'.")


if __name__ == "__main__":
    main()
