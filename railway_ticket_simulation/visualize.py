"""
visualize.py
Visualization and plotting module for Railway Ticket-Counter Simulation.
Course: Solving Problems Using Modelling and Simulation (NEC00091)

This module generates publication-quality plots:
1. Queue length over time (with moving average, thresholds, and event markers)
2. Wait-time distribution histogram and KDE (with theoretical vs empirical comparison)
3. Counter utilization bar charts and breakdown
4. Passenger outcome breakdown (Served, Balked, Reneged)
5. Comprehensive multi-panel simulation dashboard
6. Scenario comparison across Normal (lambda=0.60), Peak (lambda=0.80), and Overloaded (lambda=1.00)
"""

import os
import tempfile

# Prevent matplotlib sandbox write permission warnings on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from typing import Optional, Dict, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless and scripted execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from simulation import RailwaySimulation

# Configure global aesthetic styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})


def safe_savefig(fig: plt.Figure, save_path: str, **kwargs) -> None:
    """
    Safely save a matplotlib figure to disk, handling Windows file locking and
    pre-existing file truncation issues gracefully via pre-removal or atomic replace.
    """
    abs_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    if os.path.exists(abs_path):
        try:
            os.remove(abs_path)
        except OSError:
            pass
    try:
        fig.savefig(abs_path, bbox_inches="tight", **kwargs)
    except OSError:
        tmp_path = abs_path + f".tmp_{os.getpid()}.png"
        fig.savefig(tmp_path, bbox_inches="tight", **kwargs)
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except OSError:
                pass
        os.replace(tmp_path, abs_path)


def plot_queue_length_over_time(
    sim: RailwaySimulation,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot queue length over simulation time with rolling average,
    balking threshold indicators, and balking/reneging incident markers.
    """
    queue_df = sim.queue_df
    events_df = sim.events_df
    config = sim.config

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    if queue_df is not None and not queue_df.empty:
        # Instantaneous Queue Length
        ax.plot(
            queue_df["time"],
            queue_df["queue_length"],
            label="Instantaneous Queue Length",
            color="#1f77b4",
            alpha=0.45,
            linewidth=1.2,
        )

        # Rolling Average Queue Length
        if "rolling_avg_queue" in queue_df.columns:
            ax.plot(
                queue_df["time"],
                queue_df["rolling_avg_queue"],
                label="Moving Average (Smoothed)",
                color="#0d47a1",
                linewidth=2.2,
            )

    # Threshold horizontal lines
    if config.enable_balking:
        ax.axhline(
            y=config.balking_threshold,
            color="#ff9800",
            linestyle="--",
            linewidth=1.5,
            label=f"Balking Onset Threshold ({config.balking_threshold})",
        )
        ax.axhline(
            y=config.balking_max_queue,
            color="#d32f2f",
            linestyle=":",
            linewidth=1.8,
            label=f"Balking Max Cap ({config.balking_max_queue})",
        )

    # Mark Balking and Reneging occurrences along timeline
    if events_df is not None and not events_df.empty:
        balk_events = events_df[events_df["event"] == "BALK"]
        renege_events = events_df[events_df["event"] == "RENEGE"]

        if not balk_events.empty:
            ax.scatter(
                balk_events["time"],
                balk_events["queue_length"],
                color="#e65100",
                marker="x",
                s=40,
                alpha=0.8,
                label=f"Balk Incident (n={len(balk_events)})",
                zorder=5,
            )
        if not renege_events.empty:
            ax.scatter(
                renege_events["time"],
                renege_events["queue_length"],
                color="#c2185b",
                marker="v",
                s=40,
                alpha=0.8,
                label=f"Renege Incident (n={len(renege_events)})",
                zorder=5,
            )

    ax.set_title(
        f"Railway Ticket Counter Queue Length Over Time (c={config.num_counters}, "
        f"$\\lambda$={config.arrival_rate}/min, $\\mu$={config.service_rate}/min)",
        pad=12,
        fontweight="bold",
    )
    ax.set_xlabel("Simulation Time (minutes)")
    ax.set_ylabel("Queue Length (Passengers Waiting)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=config.sim_time)
    handles, lbls = ax.get_legend_handles_labels()
    if lbls:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        safe_savefig(fig, save_path)

    return fig


def plot_wait_time_distribution(
    sim: RailwaySimulation,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of passenger waiting times (histogram + KDE),
    comparing empirical mean/median and theoretical Erlang-C wait time Wq.
    """
    df = sim.passengers_df
    config = sim.config
    stats = sim.summary_stats or sim._compute_summary_stats()

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    served = df[df["status"] == "SERVED"] if df is not None else pd.DataFrame()
    reneged = df[df["status"] == "RENEGED"] if df is not None else pd.DataFrame()

    if not served.empty:
        # Histogram with KDE for Served Passengers
        sns.histplot(
            data=served,
            x="wait_time",
            kde=True,
            stat="density",
            color="#2e7d32",
            bins=25,
            label=f"Served Passengers (n={len(served)})",
            ax=ax,
            alpha=0.55,
        )

        mean_wait = served["wait_time"].mean()
        median_wait = served["wait_time"].median()

        ax.axvline(
            mean_wait,
            color="#1b5e20",
            linestyle="-",
            linewidth=2.2,
            label=f"Empirical Mean Wait ({mean_wait:.2f} min)",
        )
        ax.axvline(
            median_wait,
            color="#00796b",
            linestyle="--",
            linewidth=1.8,
            label=f"Empirical Median Wait ({median_wait:.2f} min)",
        )
    else:
        ax.text(
            0.5, 0.5,
            "No passengers have completed service yet\n(Waiting time distribution unavailable)",
            ha="center", va="center", fontsize=11, color="#64748b", transform=ax.transAxes
        )

    # Distribution of wait time experienced by reneged passengers before quitting
    if not reneged.empty and len(reneged) >= 3:
        sns.kdeplot(
            data=reneged["wait_time"],
            color="#d32f2f",
            linestyle=":",
            linewidth=2.0,
            label=f"Reneged Abandonment Time KDE (n={len(reneged)})",
            ax=ax,
        )

    # Theoretical Wq indicator
    theo_wq = stats.get("theoretical_mmc", {}).get("expected_wait_time_Wq")
    is_stable = stats.get("theoretical_mmc", {}).get("is_stable", True)
    if is_stable and theo_wq is not None and not np.isnan(theo_wq):
        ax.axvline(
            theo_wq,
            color="#d84315",
            linestyle="-.",
            linewidth=2.0,
            label=f"Theoretical $W_q$ Erlang-C ({theo_wq:.2f} min)",
        )
    elif not is_stable:
        ax.text(
            0.04,
            0.88,
            "Theoretical Erlang-C: UNSTABLE (ρ ≥ 100%)\nActual queue bounded by balking & reneging",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#b71c1c",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9),
        )

    # Info summary box
    avg_service = stats["timing_metrics"]["avg_service_time_served"]
    theo_service = 1.0 / config.service_rate
    info_text = (
        f"Sim Time: {config.sim_time} min\n"
        f"Arrival Rate $\\lambda$: {config.arrival_rate:.2f}/min\n"
        f"Mean Service $1/\\mu$: {theo_service:.2f} min\n"
        f"Observed Avg Service: {avg_service:.2f} min\n"
        f"Counters ($c$): {config.num_counters}\n"
        f"Traffic Intensity $\\rho$: {config.traffic_intensity*100:.1f}%"
    )
    ax.text(
        0.97,
        0.55,
        info_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="center",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_title(
        "Passenger Waiting Time Distribution (Queue Delay Before Service)",
        pad=12,
        fontweight="bold",
    )
    ax.set_xlabel("Wait Time in Queue (minutes)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(left=0)
    handles, lbls = ax.get_legend_handles_labels()
    if lbls:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        safe_savefig(fig, save_path)

    return fig


def plot_counter_utilization(
    sim: RailwaySimulation,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot individual counter utilization rates and passenger throughput,
    alongside theoretical system utilization (traffic intensity rho).
    """
    counter_df = sim.counter_df
    config = sim.config

    if counter_df is None or counter_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Counter Data Available", ha="center", va="center")
        if save_path:
            safe_savefig(fig, save_path)
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    counters_labels = [f"Counter {cid}" for cid in counter_df["counter_id"]]
    util_percentages = counter_df["utilization_percent"]
    served_counts = counter_df["customers_served"]

    # 1. Utilization Bar Chart
    bars = ax1.bar(
        counters_labels,
        util_percentages,
        color="#3f51b5",
        edgecolor="#1a237e",
        width=0.55,
        alpha=0.85,
    )
    # Add values on top of bars
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )

    # Theoretical rho reference line
    theo_rho_pct = config.traffic_intensity * 100.0
    theo_label = f"Theoretical $\\rho$ ({theo_rho_pct:.1f}%)" if theo_rho_pct < 100 else f"Theoretical $\\rho$ ({theo_rho_pct:.1f}% - Unstable)"
    ax1.axhline(
        theo_rho_pct,
        color="#e91e63",
        linestyle="--",
        linewidth=2,
        label=theo_label,
    )

    # System empirical mean utilization line
    empirical_mean_util = counter_df["utilization_percent"].mean()
    ax1.axhline(
        empirical_mean_util,
        color="#009688",
        linestyle=":",
        linewidth=2,
        label=f"Empirical Average ({empirical_mean_util:.1f}%)",
    )

    ax1.set_title("Ticket Counter Utilization Rate (%)", fontweight="bold", pad=10)
    ax1.set_ylabel("Utilization Percentage (%)")
    ax1.set_ylim(0, max(100.0, max(util_percentages) + 15, theo_rho_pct + 10))
    ax1.legend(loc="upper right", frameon=True)

    # 2. Passenger Throughput per Counter
    bars2 = ax2.bar(
        counters_labels,
        served_counts,
        color="#00897b",
        edgecolor="#004d40",
        width=0.55,
        alpha=0.85,
    )
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"{int(h)}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )

    ax2.set_title("Passenger Throughput Served per Counter", fontweight="bold", pad=10)
    ax2.set_ylabel("Total Passengers Served")
    ax2.set_ylim(0, max(served_counts) * 1.15 if max(served_counts) > 0 else 10)

    fig.suptitle(
        f"Multi-Server Ticket Counter Performance Analysis (c={config.num_counters})",
        fontweight="bold",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        safe_savefig(fig, save_path)

    return fig


def plot_passenger_outcomes(
    sim: RailwaySimulation,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot breakdown of passenger service outcomes: Served, Balked, Reneged.
    """
    stats = sim.summary_stats or sim._compute_summary_stats()
    counts = stats["passenger_counts"]

    labels = ["Served", "Balked", "Reneged"]
    values = [counts["num_served"], counts["num_balked"], counts["num_reneged"]]
    colors = ["#2e7d32", "#f57c00", "#d32f2f"]

    if counts.get("num_in_system_at_end", 0) > 0:
        labels.append("Active at Cutoff")
        values.append(counts["num_in_system_at_end"])
        colors.append("#1976d2")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Donut Chart - handle zero-outcome state safely
    if sum(values) == 0:
        ax1.text(0.5, 0.5, "No completed passenger\nevents logged yet", ha="center", va="center", fontsize=11, color="#64748b")
        ax1.set_title("Passenger Outcome Proportion", fontweight="bold", pad=10)
        ax1.axis("off")
    else:
        explode = tuple(0.04 for _ in labels)
        wedges, texts, autotexts = ax1.pie(
            values,
            labels=labels,
            autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(values) / 100.0))})" if p > 0 else "",
            colors=colors,
            explode=explode,
            startangle=140,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        )
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_fontsize(9.5)
        ax1.set_title("Passenger Outcome Proportion", fontweight="bold", pad=10)

    # Bar breakdown
    ax2.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    for i, v in enumerate(values):
        ax2.annotate(
            f"{v} ({v / sum(values) * 100:.1f}%)" if sum(values) > 0 else "0",
            xy=(i, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax2.set_ylabel("Passenger Count")
    ax2.set_title("Passenger Outcome Headcounts", fontweight="bold", pad=10)
    ax2.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 10)

    fig.suptitle(
        f"Passenger Service Lifecycle Outcomes (Total Arrivals = {counts['total_arrivals']})",
        fontweight="bold",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        safe_savefig(fig, save_path)

    return fig


def plot_simulation_dashboard(
    sim: RailwaySimulation,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate an integrated 4-panel comprehensive dashboard summarizing:
    1. Queue Length Over Time with Events & Thresholds
    2. Wait-Time Distribution with Theoretical Wq
    3. Counter Utilization & Server Throughput
    4. Passenger Outcome Breakdown (Served / Balked / Reneged)
    """
    queue_df = sim.queue_df
    events_df = sim.events_df
    counter_df = sim.counter_df
    df = sim.passengers_df
    config = sim.config
    stats = sim.summary_stats or sim._compute_summary_stats()

    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 1. Panel: Queue Length Over Time
    if queue_df is not None and not queue_df.empty:
        ax1.plot(
            queue_df["time"],
            queue_df["queue_length"],
            label="Instantaneous Queue",
            color="#64b5f6",
            alpha=0.6,
            linewidth=1.2,
        )
        if "rolling_avg_queue" in queue_df.columns:
            ax1.plot(
                queue_df["time"],
                queue_df["rolling_avg_queue"],
                label="Smoothed Moving Avg",
                color="#0d47a1",
                linewidth=2.2,
            )

    if config.enable_balking:
        ax1.axhline(
            y=config.balking_threshold,
            color="#ff9800",
            linestyle="--",
            linewidth=1.3,
            label=f"Balk Threshold ({config.balking_threshold})",
        )
        ax1.axhline(
            y=config.balking_max_queue,
            color="#d32f2f",
            linestyle=":",
            linewidth=1.5,
            label=f"Max Queue Cap ({config.balking_max_queue})",
        )

    if events_df is not None and not events_df.empty:
        balks = events_df[events_df["event"] == "BALK"]
        reneges = events_df[events_df["event"] == "RENEGE"]
        if not balks.empty:
            ax1.scatter(balks["time"], balks["queue_length"], color="#e65100", marker="x", s=30, label="Balked")
        if not reneges.empty:
            ax1.scatter(reneges["time"], reneges["queue_length"], color="#ad1457", marker="v", s=30, label="Reneged")

    ax1.set_title("A: Queue Length Dynamics Over Time", fontweight="bold")
    ax1.set_xlabel("Simulation Time (min)")
    ax1.set_ylabel("Queue Length (Passengers)")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(0, config.sim_time)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

    # 2. Panel: Wait-Time Distribution
    served = df[df["status"] == "SERVED"] if df is not None else pd.DataFrame()
    if not served.empty:
        sns.histplot(
            data=served,
            x="wait_time",
            kde=True,
            stat="density",
            color="#388e3c",
            bins=22,
            label="Served Passengers",
            ax=ax2,
            alpha=0.55,
        )
        mean_wait = served["wait_time"].mean()
        median_wait = served["wait_time"].median()
        ax2.axvline(mean_wait, color="#1b5e20", linestyle="-", linewidth=2, label=f"Mean: {mean_wait:.2f}m")
        ax2.axvline(median_wait, color="#00796b", linestyle="--", linewidth=1.8, label=f"Median: {median_wait:.2f}m")

    theo_wq = stats.get("theoretical_mmc", {}).get("expected_wait_time_Wq")
    is_stable = stats.get("theoretical_mmc", {}).get("is_stable", True)
    if is_stable and theo_wq is not None and not np.isnan(theo_wq):
        ax2.axvline(theo_wq, color="#d84315", linestyle="-.", linewidth=2, label=f"Theoretical $W_q$: {theo_wq:.2f}m")
    elif not is_stable:
        ax2.text(
            0.04, 0.88, "Theoretical: UNSTABLE (ρ ≥ 100%)",
            transform=ax2.transAxes, fontsize=8.5, color="#b71c1c", fontweight="bold"
        )

    ax2.set_title("B: Wait Time Distribution (Served Passengers)", fontweight="bold")
    ax2.set_xlabel("Queue Wait Time (min)")
    ax2.set_ylabel("Density")
    ax2.set_xlim(left=0)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

    # 3. Panel: Counter Utilization
    if counter_df is not None and not counter_df.empty:
        c_names = [f"Counter {i}" for i in counter_df["counter_id"]]
        c_utils = counter_df["utilization_percent"]
        bars = ax3.bar(c_names, c_utils, color="#3949ab", edgecolor="#1a237e", width=0.5, alpha=0.85)
        for b in bars:
            h = b.get_height()
            ax3.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        theo_rho = config.traffic_intensity * 100.0
        ax3.axhline(theo_rho, color="#e91e63", linestyle="--", linewidth=1.8, label=f"Theoretical $\\rho$ ({theo_rho:.1f}%)")
        mean_u = c_utils.mean()
        ax3.axhline(mean_u, color="#00897b", linestyle=":", linewidth=1.8, label=f"Observed Avg ({mean_u:.1f}%)")

    ax3.set_title("C: Ticket Counter Utilization Comparison", fontweight="bold")
    ax3.set_ylabel("Utilization (%)")
    ax3.set_ylim(0, max(100.0, max(counter_df["utilization_percent"]) + 15, config.traffic_intensity * 100.0 + 10))
    ax3.legend(loc="upper right", fontsize=8.5)

    # 4. Panel: Passenger Outcomes Breakdown
    p_counts = stats["passenger_counts"]
    outcomes = ["Served", "Balked", "Reneged"]
    vals = [p_counts["num_served"], p_counts["num_balked"], p_counts["num_reneged"]]
    cols = ["#43a047", "#fb8c00", "#e53935"]

    if p_counts.get("num_in_system_at_end", 0) > 0:
        outcomes.append("Active at Cutoff")
        vals.append(p_counts["num_in_system_at_end"])
        cols.append("#1976d2")

    if sum(vals) == 0:
        ax4.text(0.5, 0.5, "No completed passenger\nevents logged yet", ha="center", va="center", fontsize=10, color="#64748b")
        ax4.axis("off")
    else:
        wedge_explodes = tuple(0.03 for _ in outcomes)
        wedges, texts, autotexts = ax4.pie(
            vals,
            labels=outcomes,
            autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(vals) / 100.0))})" if p > 0 else "",
            colors=cols,
            startangle=135,
            explode=wedge_explodes,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")
    ax4.set_title("D: Passenger Service Lifecycle Outcomes", fontweight="bold")

    # Overall Header & Subtitle
    fig.suptitle(
        f"Railway Ticket-Counter Simulation Dashboard [Course: NEC00091]\n"
        f"M/M/{config.num_counters} Model with Balking & Reneging | "
        f"$\\lambda$={config.arrival_rate:.2f}/min, $\\mu$={config.service_rate:.2f}/min, Sim Time={config.sim_time:.0f} min",
        fontweight="bold",
        fontsize=14,
        y=0.98,
    )

    if save_path:
        safe_savefig(fig, save_path)

    return fig


def plot_scenario_comparison(
    scenario_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate comparative graphs across Normal (lambda=0.60), Peak (lambda=0.80),
    and Overloaded (lambda=1.00) simulation scenarios:
    1. Average Passenger Waiting Time (Empirical vs Theoretical Wq)
    2. Counter Utilization (Per-Counter and Overall System)
    3. Passengers Served vs. Passengers Abandoned (Headcount & Breakdown)
    4. Abandonment Rate (Balking vs Reneging %)
    """
    scenarios = list(scenario_results.keys())
    x = np.arange(len(scenarios))
    bar_width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Data extraction
    empirical_waits = []
    theoretical_waits = []
    overall_utils = []
    theo_rhos = []
    served_counts = []
    abandoned_counts = []
    balked_pcts = []
    reneged_pcts = []
    c1_utils = []
    c2_utils = []
    c3_utils = []

    for name in scenarios:
        s = scenario_results[name]["stats"]
        empirical_waits.append(s["timing_metrics"]["avg_wait_time_served"])

        theo_wq = s["theoretical_mmc"].get("expected_wait_time_Wq")
        theoretical_waits.append(theo_wq if theo_wq is not None else 0.0)

        overall_utils.append(s["utilization"]["empirical_overall_utilization"] * 100.0)
        theo_rhos.append(s["utilization"]["theoretical_traffic_intensity_rho"] * 100.0)

        served_counts.append(s["passenger_counts"]["num_served"])
        abandoned_counts.append(s["passenger_counts"].get("num_abandoned", s["passenger_counts"]["num_balked"] + s["passenger_counts"]["num_reneged"]))
        balked_pcts.append(s["passenger_counts"]["percent_balked"])
        reneged_pcts.append(s["passenger_counts"]["percent_reneged"])

        # Per counter
        c_utils = s["utilization"].get("individual_counter_utilization", {})
        c1_utils.append(c_utils.get(1, 0.0))
        c2_utils.append(c_utils.get(2, 0.0))
        c3_utils.append(c_utils.get(3, 0.0))

    # =========================================================================
    # Panel 1: Average Waiting Time (Empirical vs Theoretical)
    # =========================================================================
    bars1 = ax1.bar(x - bar_width / 2, empirical_waits, width=bar_width, color="#0284c7", label="Actual Simulation Wait (Served)")
    bars2 = ax1.bar(x + bar_width / 2, theoretical_waits, width=bar_width, color="#d97706", alpha=0.85, label="Theoretical Wq (Erlang-C)")

    for i, b in enumerate(bars1):
        h = b.get_height()
        ax1.annotate(f"{h:.2f}m", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, b in enumerate(bars2):
        if scenarios[i] == "Overloaded":
            ax1.annotate("UNSTABLE\n(Theory: ∞)", xy=(b.get_x() + b.get_width() / 2, 0.3), xytext=(0, 15),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#b91c1c")
        else:
            h = b.get_height()
            ax1.annotate(f"{h:.2f}m", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_title("A: Average Passenger Waiting Time Across Scenarios", fontweight="bold", pad=10)
    ax1.set_ylabel("Wait Time (minutes)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{name}\n(λ={scenario_results[name]['config'].arrival_rate:.2f})" for name in scenarios], fontweight="bold")
    ax1.set_ylim(0, max(max(empirical_waits), max(theoretical_waits)) * 1.25)
    ax1.legend(loc="upper left")

    # =========================================================================
    # Panel 2: Counter Utilization Across Scenarios
    # =========================================================================
    w_sub = 0.18
    ax2.bar(x - w_sub * 1.5, c1_utils, width=w_sub, color="#6366f1", label="Counter 1")
    ax2.bar(x - w_sub * 0.5, c2_utils, width=w_sub, color="#8b5cf6", label="Counter 2")
    ax2.bar(x + w_sub * 0.5, c3_utils, width=w_sub, color="#a855f7", label="Counter 3")
    ax2.bar(x + w_sub * 1.5, overall_utils, width=w_sub, color="#059669", label="Overall Average")

    for i, name in enumerate(scenarios):
        t_rho = theo_rhos[i]
        ax2.scatter(x[i], t_rho, color="#ef4444", marker="D", s=50, zorder=5)

    ax2.scatter([], [], color="#ef4444", marker="D", s=50, label="Theoretical Traffic Intensity (ρ)")

    ax2.set_title("B: Counter Utilization Rates Across Scenarios (%)", fontweight="bold", pad=10)
    ax2.set_ylabel("Utilization Percentage (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{name}\n(λ={scenario_results[name]['config'].arrival_rate:.2f})" for name in scenarios], fontweight="bold")
    ax2.set_ylim(0, 125)
    ax2.axhline(100, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.legend(loc="upper left", fontsize=8)

    # =========================================================================
    # Panel 3: Passengers Served vs. Abandoned
    # =========================================================================
    b_srv = ax3.bar(x - bar_width / 2, served_counts, width=bar_width, color="#10b981", label="Served Successfully")
    b_abn = ax3.bar(x + bar_width / 2, abandoned_counts, width=bar_width, color="#ef4444", label="Abandoned (Balked + Reneged)")

    for i, b in enumerate(b_srv):
        h = b.get_height()
        ax3.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, b in enumerate(b_abn):
        h = b.get_height()
        total_arr = scenario_results[scenarios[i]]["stats"]["passenger_counts"]["total_arrivals"]
        pct = (h / total_arr * 100.0) if total_arr > 0 else 0.0
        ax3.annotate(f"{int(h)}\n({pct:.1f}%)", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#b91c1c")

    ax3.set_title("C: Passengers Served vs. Abandoned Headcount", fontweight="bold", pad=10)
    ax3.set_ylabel("Number of Passengers")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{name}\n(λ={scenario_results[name]['config'].arrival_rate:.2f})" for name in scenarios], fontweight="bold")
    ax3.set_ylim(0, max(served_counts) * 1.25)
    ax3.legend(loc="upper left")

    # =========================================================================
    # Panel 4: Abandonment Breakdown (Balking vs Reneging %)
    # =========================================================================
    ax4.bar(x, balked_pcts, width=0.45, color="#f59e0b", label="Balked % (Line Too Long)")
    ax4.bar(x, reneged_pcts, bottom=balked_pcts, width=0.45, color="#dc2626", label="Reneged % (Patience Expired)")

    for i in range(len(scenarios)):
        tot = balked_pcts[i] + reneged_pcts[i]
        ax4.annotate(f"Total: {tot:.1f}%\n(Balk: {balked_pcts[i]:.1f}%, Ren: {reneged_pcts[i]:.1f}%)",
                     xy=(x[i], tot), xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax4.set_title("D: Passenger Abandonment Rate Breakdown (%)", fontweight="bold", pad=10)
    ax4.set_ylabel("Abandonment Rate (% of Total Arrivals)")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{name}\n(λ={scenario_results[name]['config'].arrival_rate:.2f})" for name in scenarios], fontweight="bold")
    ax4.set_ylim(0, max([b + r for b, r in zip(balked_pcts, reneged_pcts)]) * 1.35 if any(balked_pcts) else 25)
    ax4.legend(loc="upper left")

    fig.suptitle(
        "Railway Ticket-Counter Simulation: Scenario Comparison Analysis [Course: NEC00091]\n"
        "Normal (λ=0.60) vs. Peak (λ=0.80) vs. Overloaded (λ=1.00) | 3 Counters (μ=0.30)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    if save_path:
        safe_savefig(fig, save_path)

    return fig
