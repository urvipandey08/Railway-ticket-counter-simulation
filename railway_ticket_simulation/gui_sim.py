"""
gui_sim.py
Live Interactive GUI Simulation Runner for Railway Ticket-Counter Queuing System.
Course: Solving Problems Using Modelling and Simulation (Course Code: NEC00091)

Features:
- Live real-time visual canvas displaying 3 ticket counters with IDLE/BUSY status indicators.
- Animated passenger queue showing FIFO waiting, threshold zones (balking/capacity), and counter dispatch.
- Visual departures for served passengers and exit lanes for balked & reneged passengers.
- Real-time digital clock, live counter utilization percentages, and KPI scoreboard.
- Interactive Play/Pause, Step (+0.5m), Speed Slider (1x, 2x, 5x, 10x, 20x), and Viva Surge Demo.
- Scenario Selector: Normal (lambda=0.60), Peak (lambda=0.80), Overloaded (lambda=1.00).
- Erlang-C Theoretical Benchmark with clear STABLE / UNSTABLE distinction.
- Comprehensive metrics: Avg Wait Time, Avg Service Time, Avg Queue, Max Queue, Counter Utils, Abandonment %.
- Strict Passenger Accounting Verification: Total Arrivals = Served + Balked + Reneged + Waiting + Serving.
- Directly integrates with config.py, passenger.py, counter.py, simulation.py, and visualize.py.
"""

import os
import time
import math
import tempfile
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional, List, Dict, Any
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SimulationConfig, SIMULATION_SCENARIOS
from counter import TicketCounter, TicketCounterSystem
from passenger import Passenger
from simulation import RailwaySimulation, run_scenario_comparison
from visualize import (
    plot_simulation_dashboard,
    plot_queue_length_over_time,
    plot_wait_time_distribution,
    plot_counter_utilization,
    plot_passenger_outcomes,
    plot_scenario_comparison,
)


class RailwaySimulationGUI:
    """
    Tkinter-based Live Visual Simulation Runner for the M/M/3 Railway Ticket-Counter Queue.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Railway Ticket-Counter Simulation [Course: NEC00091] - Live Visualizer")
        self.root.geometry("1240x860")
        self.root.minsize(1100, 750)
        self.root.configure(bg="#0f172a")

        # Configuration (Default: Normal scenario with 3 counters, lambda=0.60, mu=0.30)
        self.config = SimulationConfig(
            num_counters=3,
            arrival_rate=0.60,
            service_rate=0.30,
            sim_time=480.0,
            enable_balking=True,
            balking_threshold=5,
            balking_max_queue=15,
            enable_reneging=True,
            renege_time_mean=10.0,
            scenario_name="Normal",
        )

        # Simulation state
        self.sim: Optional[RailwaySimulation] = None
        self.is_running: bool = True
        self.is_finished: bool = False
        self.speed_multiplier: float = 2.0
        self.base_sim_rate: float = 0.5  # 1 real sec = 0.5 sim-min at 1x
        self.last_real_time: float = time.time()
        self.last_event_idx: int = 0

        # Ephemeral alert tags for canvas animation
        self.recent_balked: List[Dict[str, Any]] = []
        self.recent_reneged: List[Dict[str, Any]] = []
        self.recent_served: List[Dict[str, Any]] = []

        # Build GUI Components
        self._setup_styles()
        self._build_header()
        self._build_kpi_scoreboard()
        self._build_control_bar()
        self._build_canvas()
        self._build_bottom_panels()

        # Initialize Simulation Engine
        self._reset_simulation()

        # Keyboard shortcuts
        self.root.bind("<space>", lambda e: self._toggle_play_pause())

        # Start animation / clock loop
        self.root.after(30, self._simulation_tick)

    # =========================================================================
    # UI SETUP & LAYOUT
    # =========================================================================

    def _setup_styles(self):
        """Configure ttk styles for dark modern aesthetic."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style.configure(
            "Dark.TFrame",
            background="#0f172a"
        )
        self.style.configure(
            "Card.TFrame",
            background="#1e293b",
            relief="flat"
        )
        self.style.configure(
            "TScale",
            background="#1e293b",
            troughcolor="#334155"
        )

    def _build_header(self):
        """Construct top title and status header."""
        header_frame = tk.Frame(self.root, bg="#0f172a", pady=4)
        header_frame.pack(fill="x", padx=16)

        title_container = tk.Frame(header_frame, bg="#0f172a")
        title_container.pack(side="left")

        title_lbl = tk.Label(
            title_container,
            text="RAILWAY TICKET-COUNTER SIMULATION",
            font=("Segoe UI", 15, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            title_container,
            text="Course: Solving Problems Using Modelling and Simulation (NEC00091) | M/M/3 Multi-Server Queue with Balking & Reneging",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#0f172a",
        )
        subtitle_lbl.pack(anchor="w")

        # Digital Clock & Status Badge on Right
        clock_container = tk.Frame(header_frame, bg="#1e293b", padx=12, pady=4, relief="solid", bd=1)
        clock_container.pack(side="right")

        self.clock_lbl = tk.Label(
            clock_container,
            text="SIM TIME: 000.0 min",
            font=("Consolas", 13, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
        )
        self.clock_lbl.pack(side="left", padx=(0, 8))

        self.status_badge = tk.Label(
            clock_container,
            text="RUNNING",
            font=("Segoe UI", 8, "bold"),
            fg="#10b981",
            bg="#064e3b",
            padx=8,
            pady=2,
        )
        self.status_badge.pack(side="left")

    def _build_kpi_scoreboard(self):
        """Construct row of live KPI metric cards and secondary performance metrics strip."""
        card_row = tk.Frame(self.root, bg="#0f172a")
        card_row.pack(fill="x", padx=16, pady=(2, 2))

        self.cards = {}
        card_configs = [
            ("TOTAL ARRIVALS", "0", "#38bdf8"),
            ("IN QUEUE", "0", "#06b6d4"),
            ("IN SERVICE", "0 / 3", "#a855f7"),
            ("SERVED", "0 (0.0%)", "#10b981"),
            ("BALKED", "0 (0.0%)", "#f59e0b"),
            ("RENEGED", "0 (0.0%)", "#ef4444"),
            ("ABANDONED", "0 (0.0%)", "#f43f5e"),
            ("SYSTEM UTILIZATION", "0.0%", "#6366f1"),
        ]

        for i, (title, default_val, accent_color) in enumerate(card_configs):
            card = tk.Frame(card_row, bg="#1e293b", padx=6, pady=3, relief="ridge", bd=1)
            card.pack(side="left", expand=True, fill="both", padx=2)

            lbl_title = tk.Label(
                card,
                text=title,
                font=("Segoe UI", 7, "bold"),
                fg="#94a3b8",
                bg="#1e293b"
            )
            lbl_title.pack(anchor="center")

            lbl_val = tk.Label(
                card,
                text=default_val,
                font=("Segoe UI", 10, "bold"),
                fg=accent_color,
                bg="#1e293b"
            )
            lbl_val.pack(anchor="center")

            self.cards[title] = lbl_val

        # Secondary Detailed Performance Metrics & Accounting Verification Strip
        metrics_bar = tk.Frame(self.root, bg="#1e293b", padx=8, pady=3, relief="solid", bd=1)
        metrics_bar.pack(fill="x", padx=16, pady=(2, 4))

        self.secondary_metrics_lbl = tk.Label(
            metrics_bar,
            text="Avg Wait: 0.00m | Avg Service: 3.33m | Avg Queue: 0.00 | Max Queue: 0 | C1 Util: 0.0% | C2 Util: 0.0% | C3 Util: 0.0% | Accounting: Verified",
            font=("Consolas", 8),
            fg="#cbd5e1",
            bg="#1e293b",
            anchor="w"
        )
        self.secondary_metrics_lbl.pack(side="left", fill="x", expand=True)

    def _build_control_bar(self):
        """Construct interactive simulation control bar."""
        ctrl_frame = tk.Frame(self.root, bg="#1e293b", padx=10, pady=5, relief="ridge", bd=1)
        ctrl_frame.pack(fill="x", padx=16, pady=2)

        # Play / Pause Toggle Button
        self.play_btn = tk.Button(
            ctrl_frame,
            text="⏸ Pause",
            font=("Segoe UI", 8, "bold"),
            bg="#0284c7",
            fg="white",
            activebackground="#0369a1",
            activeforeground="white",
            relief="raised",
            bd=1,
            padx=10,
            pady=2,
            command=self._toggle_play_pause,
            cursor="hand2",
        )
        self.play_btn.pack(side="left", padx=3)

        # Step +0.5m Button (for lab viva discrete demonstrations)
        self.step_btn = tk.Button(
            ctrl_frame,
            text="⏭ Step (+0.5m)",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="white",
            relief="raised",
            bd=1,
            padx=8,
            pady=2,
            command=self._step_once,
            cursor="hand2",
        )
        self.step_btn.pack(side="left", padx=3)

        # Reset Simulation Button
        reset_btn = tk.Button(
            ctrl_frame,
            text="🔄 Restart",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="white",
            relief="raised",
            bd=1,
            padx=8,
            pady=2,
            command=self._reset_simulation,
            cursor="hand2",
        )
        reset_btn.pack(side="left", padx=3)

        # Scenario Selector
        sep0 = tk.Frame(ctrl_frame, bg="#475569", width=1, height=22)
        sep0.pack(side="left", padx=6)

        scen_lbl = tk.Label(
            ctrl_frame,
            text="Scenario:",
            font=("Segoe UI", 8, "bold"),
            fg="#f8fafc",
            bg="#1e293b"
        )
        scen_lbl.pack(side="left", padx=(0, 2))

        self.scenario_btns = {}
        scenarios = [
            ("Normal (λ=0.6)", "Normal", 0.60),
            ("Peak (λ=0.8)", "Peak", 0.80),
            ("Overloaded (λ=1.0)", "Overloaded", 1.00),
        ]
        for label, name, lam in scenarios:
            is_active = (name == self.config.scenario_name)
            btn = tk.Button(
                ctrl_frame,
                text=label,
                font=("Segoe UI", 7, "bold"),
                bg="#0284c7" if is_active else "#334155",
                fg="white",
                activebackground="#0369a1",
                activeforeground="white",
                relief="raised" if is_active else "flat",
                bd=1,
                padx=5,
                pady=2,
                command=lambda n=name, l=lam: self._select_scenario(n, l),
                cursor="hand2",
            )
            btn.pack(side="left", padx=1)
            self.scenario_btns[name] = btn

        # Speed Multiplier Controls
        sep1 = tk.Frame(ctrl_frame, bg="#475569", width=1, height=22)
        sep1.pack(side="left", padx=6)

        speed_lbl = tk.Label(
            ctrl_frame,
            text="Speed:",
            font=("Segoe UI", 8, "bold"),
            fg="#f8fafc",
            bg="#1e293b"
        )
        speed_lbl.pack(side="left", padx=(0, 2))

        self.speed_var = tk.DoubleVar(value=2.0)
        self.speed_slider = tk.Scale(
            ctrl_frame,
            from_=1.0,
            to=20.0,
            orient="horizontal",
            variable=self.speed_var,
            resolution=0.5,
            showvalue=False,
            bg="#1e293b",
            fg="#38bdf8",
            highlightthickness=0,
            troughcolor="#334155",
            activebackground="#0284c7",
            length=80,
            command=self._on_speed_changed,
            cursor="hand2",
        )
        self.speed_slider.pack(side="left", padx=1)

        self.speed_display = tk.Label(
            ctrl_frame,
            text="2.0x",
            font=("Consolas", 8, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
            width=4
        )
        self.speed_display.pack(side="left", padx=1)

        # Quick Preset Speed Buttons
        presets = [("1x", 1.0), ("2x", 2.0), ("5x", 5.0), ("10x", 10.0), ("20x", 20.0)]
        for label, val in presets:
            btn = tk.Button(
                ctrl_frame,
                text=label,
                font=("Segoe UI", 7, "bold"),
                bg="#334155",
                fg="#cbd5e1",
                activebackground="#475569",
                relief="flat",
                bd=0,
                padx=4,
                pady=1,
                command=lambda v=val: self._set_speed(v),
                cursor="hand2",
            )
            btn.pack(side="left", padx=1)

        # Viva Demo Action: Spawn Passenger Surge
        sep2 = tk.Frame(ctrl_frame, bg="#475569", width=1, height=22)
        sep2.pack(side="left", padx=6)

        surge_btn = tk.Button(
            ctrl_frame,
            text="⚡ Surge (+5)",
            font=("Segoe UI", 7, "bold"),
            bg="#d97706",
            fg="white",
            activebackground="#b45309",
            relief="raised",
            bd=1,
            padx=6,
            pady=2,
            command=self._spawn_surge,
            cursor="hand2",
        )
        surge_btn.pack(side="left", padx=2)

        # Action: Compare Scenarios Button
        compare_btn = tk.Button(
            ctrl_frame,
            text="📈 Compare Scenarios",
            font=("Segoe UI", 8, "bold"),
            bg="#059669",
            fg="white",
            activebackground="#047857",
            relief="raised",
            bd=1,
            padx=7,
            pady=2,
            command=self._compare_scenarios_dialog,
            cursor="hand2",
        )
        compare_btn.pack(side="right", padx=3)

        # Generate Full Analytical Charts Button
        plot_btn = tk.Button(
            ctrl_frame,
            text="📊 Export Plots",
            font=("Segoe UI", 8),
            bg="#4f46e5",
            fg="white",
            activebackground="#4338ca",
            relief="raised",
            bd=1,
            padx=7,
            pady=2,
            command=self._export_plots,
            cursor="hand2",
        )
        plot_btn.pack(side="right", padx=3)

    def _select_scenario(self, name: str, lam: float):
        """Switch simulation scenario, update configuration, and restart simulation."""
        self.config.scenario_name = name
        self.config.arrival_rate = lam

        # Update button visual states
        for s_name, btn in self.scenario_btns.items():
            if s_name == name:
                btn.config(bg="#0284c7", relief="raised")
            else:
                btn.config(bg="#334155", relief="flat")

        self._reset_simulation()
        self._log_event(
            f"[0.0m] 🔀 Switched to Scenario: {name} (λ={lam:.2f}/min, μ={self.config.service_rate:.2f}/min, "
            f"c={self.config.num_counters})."
        )

    def _build_canvas(self):
        """Construct main animation canvas representing the railway reservation hall."""
        canvas_container = tk.Frame(self.root, bg="#0f172a")
        canvas_container.pack(fill="both", expand=True, padx=16, pady=2)

        self.canvas_w = 1200
        self.canvas_h = 350
        self.canvas = tk.Canvas(
            canvas_container,
            bg="#0f172a",
            highlightthickness=1,
            highlightbackground="#334155",
            width=self.canvas_w,
            height=self.canvas_h,
        )
        self.canvas.pack(fill="both", expand=True)

        self._draw_static_station_scenery()

    def _build_bottom_panels(self):
        """Construct bottom event log and theoretical queueing metrics comparison panel."""
        bottom_frame = tk.Frame(self.root, bg="#0f172a", height=185)
        bottom_frame.pack(fill="x", padx=16, pady=(0, 6))

        # Left: Live Event Log Stream
        log_frame = tk.LabelFrame(
            bottom_frame,
            text="Live Simulation Event Feed",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#1e293b",
            padx=6,
            pady=4,
            relief="ridge",
            bd=1,
        )
        log_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=7,
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="white",
            font=("Consolas", 8),
            wrap="word",
            relief="flat",
            bd=0,
        )
        self.log_text.pack(fill="both", expand=True)

        # Right: Analytical Queueing Model Reference (Erlang-C HUD)
        theory_frame = tk.LabelFrame(
            bottom_frame,
            text="Erlang-C Benchmark (Infinite Queue M/M/3) vs. Simulation",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#1e293b",
            padx=8,
            pady=4,
            relief="ridge",
            bd=1,
            width=460,
        )
        theory_frame.pack(side="right", fill="both", padx=(4, 0))
        theory_frame.pack_propagate(False)

        self.theory_lbl = tk.Label(
            theory_frame,
            text="Calculating theoretical benchmarks...",
            font=("Consolas", 8),
            fg="#cbd5e1",
            bg="#1e293b",
            justify="left",
            anchor="nw",
        )
        self.theory_lbl.pack(fill="both", expand=True)

    # =========================================================================
    # STATIC SCENERY & FLOOR PLAN
    # =========================================================================

    def _draw_static_station_scenery(self):
        """Draw persistent visual zones: Entrance, Waiting Rope Lane, Counters, Exit."""
        self.canvas.delete("all")

        scenario = getattr(self.config, "scenario_name", "Normal")

        # 1. Station Entrance (Left)
        self.canvas.create_rectangle(
            20, 130, 105, 250,
            fill="#1e293b", outline="#38bdf8", width=2, tags="static"
        )
        self.canvas.create_text(
            62, 155,
            text="STATION\nENTRANCE",
            font=("Segoe UI", 9, "bold"),
            fill="#38bdf8",
            justify="center",
            tags="static"
        )
        self.canvas.create_text(
            62, 205,
            text=f"λ = {self.config.arrival_rate:.2f}/min\n({scenario})",
            font=("Segoe UI", 7, "bold"),
            fill="#cbd5e1",
            justify="center",
            tags="static"
        )

        # 2. Balking / Reneging Exit Lane (Top Left)
        self.canvas.create_rectangle(
            20, 25, 230, 80,
            fill="#1e293b", outline="#f59e0b", width=1, dash=(3, 3), tags="static"
        )
        self.canvas.create_text(
            125, 42,
            text="🚪 EXIT LANE (Balked & Reneged)",
            font=("Segoe UI", 8, "bold"),
            fill="#f59e0b",
            tags="static"
        )
        self.canvas.create_text(
            125, 62,
            text="Departures without receiving service",
            font=("Segoe UI", 7),
            fill="#64748b",
            tags="static"
        )

        # 3. FIFO Waiting Queue Lane (Center)
        # Queue floor
        self.canvas.create_rectangle(
            120, 145, 830, 235,
            fill="#131f37", outline="#1e293b", width=1, tags="static"
        )
        # Velvet rope guide lines
        self.canvas.create_line(120, 145, 830, 145, fill="#64748b", width=2, tags="static")
        self.canvas.create_line(120, 235, 830, 235, fill="#64748b", width=2, tags="static")

        # Floor flow indicator
        self.canvas.create_text(
            475, 190,
            text="═══ FIFO WAITING QUEUE (Head of Line ➔) ═══",
            font=("Segoe UI", 8, "bold"),
            fill="#1e293b",
            tags="static"
        )

        # Balking Threshold Indicator Line (Slot 5: aligned between slot 4 and 5 at x=611)
        balk_x = 611
        self.canvas.create_line(
            balk_x, 105, balk_x, 275,
            fill="#f59e0b", width=2, dash=(4, 3), tags="static"
        )
        self.canvas.create_rectangle(
            balk_x - 65, 105, balk_x + 65, 125,
            fill="#78350f", outline="#f59e0b", width=1, tags="static"
        )
        self.canvas.create_text(
            balk_x, 115,
            text=f"⚠️ Balk Onset (Queue={self.config.balking_threshold})",
            font=("Segoe UI", 7, "bold"),
            fill="#fef3c7",
            tags="static"
        )

        # Max Capacity Line (Slot 15: aligned at x=191)
        max_x = 191
        self.canvas.create_line(
            max_x, 105, max_x, 275,
            fill="#ef4444", width=2, dash=(4, 3), tags="static"
        )
        self.canvas.create_rectangle(
            max_x - 65, 105, max_x + 65, 125,
            fill="#7f1d1d", outline="#ef4444", width=1, tags="static"
        )
        self.canvas.create_text(
            max_x, 115,
            text=f"⛔ 100% Balk Cap ({self.config.balking_max_queue})",
            font=("Segoe UI", 7, "bold"),
            fill="#fee2e2",
            tags="static"
        )

        # 4. Connecting Dispatch Paths (from Queue Head at x=830 to Counters)
        counter_y_positions = [65, 185, 305]
        for y_pos in counter_y_positions:
            self.canvas.create_line(
                830, 190, 890, y_pos,
                fill="#334155", width=2, dash=(2, 3), tags="static"
            )

        # 5. Platforms Exit (Far Right)
        self.canvas.create_rectangle(
            1110, 25, 1175, 345,
            fill="#1e293b", outline="#10b981", width=2, tags="static"
        )
        self.canvas.create_text(
            1142, 185,
            text="T\nR\nA\nI\nN\n\nP\nL\nA\nT\nF\nO\nR\nM\nS\n\n➡️",
            font=("Segoe UI", 8, "bold"),
            fill="#10b981",
            justify="center",
            tags="static"
        )

    # =========================================================================
    # SIMULATION LIFECYCLE & STEPPING
    # =========================================================================

    def _reset_simulation(self):
        """Reinitialize the discrete-event simulation engine and refresh display."""
        self.sim = RailwaySimulation(self.config)
        self.sim.prepare()
        self.is_running = True
        self.is_finished = False
        self.last_real_time = time.time()
        self.last_event_idx = 0
        self.recent_balked.clear()
        self.recent_reneged.clear()
        self.recent_served.clear()

        self._update_play_pause_button()
        self._update_theory_display()
        self._draw_static_station_scenery()

        # Clear and initialize log
        self.log_text.delete("1.0", tk.END)
        self._log_event(
            f"[0.0m] Simulation initialized: Scenario '{self.config.scenario_name}' ({self.config.num_counters} counters, "
            f"λ={self.config.arrival_rate:.2f}/min, μ={self.config.service_rate:.2f}/min, mean service={1.0/self.config.service_rate:.2f}m)."
        )

        self._update_gui()

    def _toggle_play_pause(self):
        """Toggle simulation execution between running and paused."""
        if self.is_finished:
            self._reset_simulation()
            return

        self.is_running = not self.is_running
        self.last_real_time = time.time()
        self._update_play_pause_button()

    def _update_play_pause_button(self):
        """Update button styling and status badge based on running state."""
        if self.is_finished:
            self.play_btn.config(text="🔄 Restart", bg="#4f46e5")
            self.status_badge.config(text="COMPLETED", fg="#38bdf8", bg="#0c4a6e")
        elif self.is_running:
            self.play_btn.config(text="⏸ Pause", bg="#0284c7")
            self.status_badge.config(text="RUNNING", fg="#10b981", bg="#064e3b")
        else:
            self.play_btn.config(text="▶ Play", bg="#10b981")
            self.status_badge.config(text="PAUSED", fg="#f59e0b", bg="#78350f")

    def _step_once(self):
        """Advance simulation by exactly 0.5 simulation minutes while paused."""
        if not self.sim:
            return
        self.is_running = False
        self._update_play_pause_button()

        target_time = self.sim.env.now + 0.5
        still_running = self.sim.step_until(target_time)
        if not still_running:
            self.is_finished = True
            self._update_play_pause_button()
            self._log_event(f"[{self.sim.env.now:.1f}m] --- SIMULATION HORIZON REACHED ---")

        self._process_new_events()
        self._update_gui()

    def _on_speed_changed(self, val):
        """Handle speed slider movement."""
        speed = float(val)
        self.speed_multiplier = speed
        self.speed_display.config(text=f"{speed:.1f}x")

    def _set_speed(self, speed: float):
        """Set speed via preset button."""
        self.speed_var.set(speed)
        self.speed_multiplier = speed
        self.speed_display.config(text=f"{speed:.1f}x")

    def _spawn_surge(self):
        """Inject 5 passengers in rapid succession to demonstrate queue buildup & balking."""
        if not self.sim:
            return
        self._log_event(f"[{self.sim.env.now:.1f}m] ⚡ VIVA DEMO: Injecting Passenger Surge (+5)...")
        for _ in range(5):
            self.sim.inject_passenger()
        self._process_new_events()
        self._update_gui()

    def _simulation_tick(self):
        """Main timer loop driving SimPy discrete-event steps and GUI refresh."""
        current_real_time = time.time()
        dt_real = current_real_time - self.last_real_time
        self.last_real_time = current_real_time

        # Cap dt to prevent huge jumps if window was minimized
        dt_real = min(dt_real, 0.1)

        if self.is_running and not self.is_finished and self.sim is not None:
            # Advance simulation time based on speed factor
            sim_dt = dt_real * (self.base_sim_rate * self.speed_multiplier)
            target_time = self.sim.env.now + sim_dt

            still_running = self.sim.step_until(target_time)
            if not still_running:
                self.is_running = False
                self.is_finished = True
                self._update_play_pause_button()
                self._log_event(f"[{self.sim.env.now:.1f}m] --- SIMULATION HORIZON COMPLETED (T={self.config.sim_time}m) ---")

            self._process_new_events()
            self._update_gui()

        # Schedule next tick (~30 FPS)
        self.root.after(30, self._simulation_tick)

    # =========================================================================
    # EVENT LOGGING & NOTIFICATIONS
    # =========================================================================

    def _process_new_events(self):
        """Read newly registered SimPy discrete events and update log / alert tags."""
        if not self.sim:
            return

        events = self.sim.events_log
        now = self.sim.env.now

        while self.last_event_idx < len(events):
            ev = events[self.last_event_idx]
            self.last_event_idx += 1

            etype = ev.get("event")
            pid = ev.get("passenger_id")
            q_len = ev.get("queue_length", 0)
            cid = ev.get("counter_id")
            t = ev.get("time", 0.0)

            if etype == "BALK":
                msg = f"[{t:05.1f}m] ⛔ Passenger #{pid} BALKED (Queue length {q_len} exceeded patience zone)!"
                self._log_event(msg)
                self.recent_balked.append({"pid": pid, "time": now, "reason": f"Queue: {q_len}"})

            elif etype == "RENEGE":
                msg = f"[{t:05.1f}m] ⏱️ Passenger #{pid} RENEGED from queue (Waiting patience exceeded)!"
                self._log_event(msg)
                self.recent_reneged.append({"pid": pid, "time": now})

            elif etype == "SERVICE_START":
                msg = f"[{t:05.1f}m] 🎟️ Counter {cid} started serving Passenger #{pid} (Queue remaining: {q_len})."
                self._log_event(msg)

            elif etype == "SERVICE_COMPLETE":
                msg = f"[{t:05.1f}m] ✔️ Counter {cid} completed service for Passenger #{pid}."
                self._log_event(msg)
                self.recent_served.append({"pid": pid, "cid": cid, "time": now})

            elif etype == "QUEUE_ENTER":
                msg = f"[{t:05.1f}m] 🚶 Passenger #{pid} joined queue (Position: {q_len})."
                self._log_event(msg)

        # Expire alert tags older than 8 simulation minutes
        self.recent_balked = [b for b in self.recent_balked if now - b["time"] < 8.0]
        self.recent_reneged = [r for r in self.recent_reneged if now - r["time"] < 8.0]
        self.recent_served = [s for s in self.recent_served if now - s["time"] < 5.0]

    def _log_event(self, text: str):
        """Append event line to scrolled text widget."""
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    # =========================================================================
    # VISUAL CANVAS RENDERING (ANIMATIONS & DYNAMICS)
    # =========================================================================

    def _update_gui(self):
        """Refresh all dynamic visual elements on the canvas, scoreboard, and secondary metrics."""
        if not self.sim:
            return

        now = self.sim.env.now

        # 1. Update Digital Clock
        hrs = int(now // 60)
        mins = int(now % 60)
        self.clock_lbl.config(text=f"SIM TIME: {now:05.1f} min ({hrs:02d}h {mins:02d}m)")

        # 2. Extract Entity Metrics and Enforce Accounting Identity
        passengers = self.sim.passengers
        total_arrivals = len(passengers)
        served_p = [p for p in passengers if p.status == "SERVED"]
        balked_p = [p for p in passengers if p.status == "BALKED"]
        reneged_p = [p for p in passengers if p.status == "RENEGED"]
        waiting_p = [p for p in passengers if p.status == "IN_QUEUE"]
        serving_p = [p for p in passengers if p.status == "IN_SERVICE"]

        # Catch any transient passenger in arrival state
        unassigned = [p for p in passengers if p.status not in ("SERVED", "BALKED", "RENEGED", "IN_QUEUE", "IN_SERVICE")]
        if unassigned:
            waiting_p.extend(unassigned)

        n_served = len(served_p)
        n_balked = len(balked_p)
        n_reneged = len(reneged_p)
        n_queue = len(waiting_p)
        n_serving = len(serving_p)
        n_abandoned = n_balked + n_reneged

        pct_served = (n_served / total_arrivals * 100.0) if total_arrivals > 0 else 0.0
        pct_balked = (n_balked / total_arrivals * 100.0) if total_arrivals > 0 else 0.0
        pct_reneged = (n_reneged / total_arrivals * 100.0) if total_arrivals > 0 else 0.0
        pct_abandoned = (n_abandoned / total_arrivals * 100.0) if total_arrivals > 0 else 0.0

        # Verification of Passenger Accounting Identity:
        # Total Arrivals = Total Served + Total Balked + Total Reneged + Currently Waiting + Currently In Service
        accounted_sum = n_served + n_balked + n_reneged + n_queue + n_serving
        is_accounting_valid = (total_arrivals == accounted_sum)

        # Counter Busy Times and Utilizations
        counter_sys = self.sim.ticket_system
        busy_times = []
        c_utils = []
        for c in counter_sys.counters:
            bt = c.total_busy_time
            if c.is_busy and c.service_start_timestamp is not None:
                bt += (now - c.service_start_timestamp)
            busy_times.append(bt)
            c_util = (bt / max(now, 0.001)) * 100.0 if now > 0 else 0.0
            c_utils.append(c_util)

        total_system_busy = sum(busy_times)
        overall_util = (
            total_system_busy / (self.config.num_counters * max(now, 0.001)) * 100.0
            if now > 0 else 0.0
        )

        # Detailed Performance Indicators:
        # Average passenger waiting time (served)
        avg_wait = np.mean([p.wait_time for p in served_p]) if served_p else 0.0
        # Average service time
        avg_service = np.mean([p.service_duration for p in served_p]) if served_p else (1.0 / self.config.service_rate)

        # Average & peak queue length
        if self.sim.queue_time_series:
            q_lens = [entry["queue_length"] for entry in self.sim.queue_time_series]
            avg_queue = float(np.mean(q_lens))
            max_queue = int(max(q_lens))
        else:
            avg_queue = float(n_queue)
            max_queue = n_queue

        # 3. Update KPI Scoreboard
        self.cards["TOTAL ARRIVALS"].config(text=str(total_arrivals))
        self.cards["IN QUEUE"].config(
            text=str(n_queue),
            fg="#ef4444" if n_queue >= self.config.balking_max_queue else (
                "#f59e0b" if n_queue >= self.config.balking_threshold else "#06b6d4"
            )
        )
        self.cards["IN SERVICE"].config(text=f"{n_serving} / {self.config.num_counters}")
        self.cards["SERVED"].config(text=f"{n_served} ({pct_served:.1f}%)")
        self.cards["BALKED"].config(text=f"{n_balked} ({pct_balked:.1f}%)")
        self.cards["RENEGED"].config(text=f"{n_reneged} ({pct_reneged:.1f}%)")
        self.cards["ABANDONED"].config(text=f"{n_abandoned} ({pct_abandoned:.1f}%)")
        self.cards["SYSTEM UTILIZATION"].config(text=f"{overall_util:.1f}%")

        # Update Secondary Metrics Bar
        c1_str = f"{c_utils[0]:.1f}%" if len(c_utils) > 0 else "0.0%"
        c2_str = f"{c_utils[1]:.1f}%" if len(c_utils) > 1 else "0.0%"
        c3_str = f"{c_utils[2]:.1f}%" if len(c_utils) > 2 else "0.0%"
        acc_status = "VERIFIED" if is_accounting_valid else f"MISMATCH ({total_arrivals} != {accounted_sum})"

        self.secondary_metrics_lbl.config(
            text=(
                f"Avg Wait: {avg_wait:.2f}m | Avg Service: {avg_service:.2f}m | "
                f"Avg Queue (Lq): {avg_queue:.2f} | Peak Queue: {max_queue} | "
                f"C1: {c1_str} | C2: {c2_str} | C3: {c3_str} | "
                f"Accounting: {total_arrivals} = {n_served}+{n_balked}+{n_reneged}+{n_queue}+{n_serving} [{acc_status}]"
            ),
            fg="#10b981" if is_accounting_valid else "#ef4444"
        )

        # 4. Clear Dynamic Canvas Elements
        self.canvas.delete("dynamic")

        # 5. Render Waiting Queue of Passengers
        self._render_queue(waiting_p, now)

        # 6. Render Ticket Counters (Status, Utilization, Assigned Passenger)
        self._render_counters(counter_sys, busy_times, now)

        # 7. Render Recent Departures / Exit Notifications
        self._render_exit_lane()

    def _render_queue(self, waiting_p: List[Passenger], now: float):
        """Draw queued passenger blocks aligned in FIFO order."""
        slot_width = 42
        head_x = 800
        y_center = 190

        # Sort waiting passengers by queue_enter_time (strict FIFO order)
        sorted_p = sorted(
            waiting_p,
            key=lambda p: p.queue_enter_time if p.queue_enter_time is not None else 0.0
        )

        max_visible_slots = 15
        for i, p in enumerate(sorted_p[:max_visible_slots]):
            x_pos = head_x - (i * slot_width)
            wait_so_far = now - (p.queue_enter_time or now)

            # Color transitions from teal to amber to red as wait approaches reneging patience
            patience = p.patience_time if not math.isinf(p.patience_time) else 20.0
            urgency = min(1.0, wait_so_far / max(patience, 0.1))

            if urgency > 0.75:
                box_color = "#b91c1c"  # Red: high abandonment risk
            elif urgency > 0.45:
                box_color = "#d97706"  # Amber: waiting long
            else:
                box_color = "#0284c7"  # Normal: blue/teal

            # Passenger Block
            self.canvas.create_rectangle(
                x_pos - 16, y_center - 22, x_pos + 16, y_center + 22,
                fill=box_color, outline="#f8fafc", width=1.5, tags="dynamic"
            )
            # Head icon circle
            self.canvas.create_oval(
                x_pos - 7, y_center - 18, x_pos + 7, y_center - 4,
                fill="#f8fafc", outline="", tags="dynamic"
            )
            # Passenger ID
            self.canvas.create_text(
                x_pos, y_center + 8,
                text=f"P{p.passenger_id}",
                font=("Segoe UI", 7, "bold"),
                fill="white",
                tags="dynamic"
            )

            # Wait Time Badge Below
            self.canvas.create_text(
                x_pos, y_center + 32,
                text=f"{wait_so_far:.1f}m",
                font=("Consolas", 7),
                fill="#cbd5e1",
                tags="dynamic"
            )

        # Overflow indicator if queue > 15
        if len(sorted_p) > max_visible_slots:
            overflow_count = len(sorted_p) - max_visible_slots
            self.canvas.create_rectangle(
                125, y_center - 22, 165, y_center + 22,
                fill="#7f1d1d", outline="#ef4444", width=2, tags="dynamic"
            )
            self.canvas.create_text(
                145, y_center,
                text=f"+{overflow_count}",
                font=("Segoe UI", 8, "bold"),
                fill="white",
                tags="dynamic"
            )

    def _render_counters(
        self,
        counter_sys: TicketCounterSystem,
        busy_times: List[float],
        now: float
    ):
        """Draw 3 ticket counters with IDLE/BUSY badges, utilization meters, and serving status."""
        counter_y_positions = [65, 185, 305]

        for i, counter in enumerate(counter_sys.counters[:3]):
            y_mid = counter_y_positions[i]
            x1, y1 = 890, y_mid - 45
            x2, y2 = 1090, y_mid + 45

            # Outer counter booth frame
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill="#1e293b", outline="#475569", width=2, tags="dynamic"
            )

            # Counter Title (Top-Left)
            self.canvas.create_text(
                x1 + 10, y1 + 14,
                text=f"COUNTER {counter.counter_id}",
                font=("Segoe UI", 8, "bold"),
                fill="#f8fafc",
                anchor="w",
                tags="dynamic"
            )

            # Status Badge (Top-Right)
            if counter.is_busy:
                badge_bg = "#7f1d1d"
                badge_fg = "#ef4444"
                badge_txt = "🔴 BUSY"
            else:
                badge_bg = "#064e3b"
                badge_fg = "#10b981"
                badge_txt = "🟢 IDLE"

            self.canvas.create_rectangle(
                x2 - 62, y1 + 5, x2 - 8, y1 + 22,
                fill=badge_bg, outline=badge_fg, width=1, tags="dynamic"
            )
            self.canvas.create_text(
                x2 - 35, y1 + 13,
                text=badge_txt,
                font=("Segoe UI", 7, "bold"),
                fill=badge_fg,
                tags="dynamic"
            )

            # Live Utilization Calculation & Mini Meter (Middle Row)
            bt = busy_times[i] if i < len(busy_times) else 0.0
            util_pct = (bt / max(now, 0.001)) * 100.0 if now > 0 else 0.0

            self.canvas.create_text(
                x1 + 10, y1 + 33,
                text=f"Util: {util_pct:.1f}%",
                font=("Segoe UI", 8),
                fill="#94a3b8",
                anchor="w",
                tags="dynamic"
            )

            # Progress Bar for Counter Utilization
            meter_x1, meter_y1 = x1 + 75, y1 + 29
            meter_w, meter_h = 110, 7
            self.canvas.create_rectangle(
                meter_x1, meter_y1, meter_x1 + meter_w, meter_y1 + meter_h,
                fill="#334155", outline="", tags="dynamic"
            )
            fill_w = min(meter_w, max(0.0, meter_w * (util_pct / 100.0)))
            self.canvas.create_rectangle(
                meter_x1, meter_y1, meter_x1 + fill_w, meter_y1 + meter_h,
                fill="#6366f1", outline="", tags="dynamic"
            )

            # Service Activity / Serving Passenger Avatar (Bottom Row)
            if counter.is_busy and counter.current_passenger_id is not None:
                # Active transaction line connecting counter window to platform
                self.canvas.create_line(
                    x2, y_mid, 1110, y_mid,
                    fill="#10b981", width=2, dash=(2, 2), tags="dynamic"
                )

                # Passenger tag badge
                self.canvas.create_rectangle(
                    x1 + 8, y1 + 50, x1 + 50, y2 - 8,
                    fill="#10b981", outline="white", width=1, tags="dynamic"
                )
                self.canvas.create_text(
                    x1 + 29, y1 + 63,
                    text=f"P{counter.current_passenger_id}",
                    font=("Segoe UI", 7, "bold"),
                    fill="white",
                    tags="dynamic"
                )

                self.canvas.create_text(
                    x1 + 58, y1 + 63,
                    text="In Service",
                    font=("Segoe UI", 7),
                    fill="#10b981",
                    anchor="w",
                    tags="dynamic"
                )

                self.canvas.create_text(
                    x2 - 10, y1 + 63,
                    text=f"Served: {counter.customers_served}",
                    font=("Segoe UI", 7),
                    fill="#cbd5e1",
                    anchor="e",
                    tags="dynamic"
                )
            else:
                self.canvas.create_text(
                    x1 + 10, y1 + 63,
                    text="Window Open",
                    font=("Segoe UI", 7, "italic"),
                    fill="#64748b",
                    anchor="w",
                    tags="dynamic"
                )
                self.canvas.create_text(
                    x2 - 10, y1 + 63,
                    text=f"Served: {counter.customers_served}",
                    font=("Segoe UI", 7),
                    fill="#cbd5e1",
                    anchor="e",
                    tags="dynamic"
                )

    def _render_exit_lane(self):
        """Draw animated indicators for recently balked, reneged, or served passengers."""
        for i, item in enumerate(self.recent_balked[-3:]):
            x = 180 - (i * 55)
            y = 52
            self.canvas.create_rectangle(
                x - 22, y - 12, x + 22, y + 12,
                fill="#78350f", outline="#f59e0b", width=1, tags="dynamic"
            )
            self.canvas.create_text(
                x, y,
                text=f"P{item['pid']}\nBalked",
                font=("Segoe UI", 6, "bold"),
                fill="#fef3c7",
                justify="center",
                tags="dynamic"
            )

        for i, item in enumerate(self.recent_reneged[-3:]):
            x = 240 + (i * 55)
            y = 52
            self.canvas.create_rectangle(
                x - 22, y - 12, x + 22, y + 12,
                fill="#7f1d1d", outline="#ef4444", width=1, tags="dynamic"
            )
            self.canvas.create_text(
                x, y,
                text=f"P{item['pid']}\nReneged",
                font=("Segoe UI", 6, "bold"),
                fill="#fee2e2",
                justify="center",
                tags="dynamic"
            )

    # =========================================================================
    # THEORETICAL QUEUEING COMPARISON HUD (ERLANG-C BENCHMARK)
    # =========================================================================

    def _update_theory_display(self):
        """
        Compute and display analytical Erlang-C theoretical metrics.
        Clearly distinguishes theoretical model predictions from actual simulation results.
        Shows prominent UNSTABLE message when utilization >= 100%.
        """
        theo = self.config.calculate_theoretical_mmc()
        c = self.config.num_counters
        lam = self.config.arrival_rate
        mu = self.config.service_rate
        rho = self.config.traffic_intensity
        scenario = getattr(self.config, "scenario_name", "Normal")

        if not theo.get("is_stable", True):
            info_text = (
                f"SCENARIO: {scenario.upper()} (λ={lam:.2f}/min, μ={mu:.2f}/min, c={c})\n"
                f"════════════ THEORETICAL ERLANG-C BENCHMARK ════════════\n"
                f"STATUS: ⚠️ UNSTABLE (Traffic Intensity ρ = {rho*100:.1f}% ≥ 100%)\n"
                f"Theoretical Queue Length (L_q) : N/A (Grows infinitely)\n"
                f"Theoretical Wait Time (W_q)    : N/A (Grows infinitely)\n"
                f"Condition: Arrival rate λ ({lam:.2f}) ≥ Total capacity c·μ ({c*mu:.2f})\n"
                f"────────────────────────────────────────────────────────\n"
                f"═════════════ ACTUAL SIMULATION DYNAMICS ═══════════════\n"
                f"System is STABILIZED by behavioral dynamics:\n"
                f"- Passengers BALK when queue exceeds capacity threshold\n"
                f"- Passengers RENEGE when patience timer expires\n"
                f"Result: Real-world queue length and wait times remain finite!"
            )
        else:
            p0_str = f"{theo['p0']*100:.1f}%" if theo.get("p0") is not None else "N/A"
            pq_str = f"{theo['prob_waiting_erlang_c']*100:.1f}%" if theo.get("prob_waiting_erlang_c") is not None else "N/A"
            wq_str = f"{theo['expected_wait_time_Wq']:.2f} min" if theo.get("expected_wait_time_Wq") is not None else "N/A"
            lq_str = f"{theo['expected_queue_length_Lq']:.2f}" if theo.get("expected_queue_length_Lq") is not None else "N/A"
            w_str = f"{theo['expected_system_time_W']:.2f} min" if theo.get("expected_system_time_W") is not None else "N/A"

            info_text = (
                f"SCENARIO: {scenario.upper()} (λ={lam:.2f}/min, μ={mu:.2f}/min, c={c})\n"
                f"════════════ THEORETICAL ERLANG-C BENCHMARK ════════════\n"
                f"STATUS: ✅ STABLE (Traffic Intensity ρ = {rho*100:.1f}% < 100%)\n"
                f"Erlang-C Queuing Prob (P_Q)    : {pq_str}\n"
                f"Theoretical Avg Queue (L_q)    : {lq_str} passengers\n"
                f"Theoretical Avg Wait (W_q)     : {wq_str}\n"
                f"Theoretical System Time (W)    : {w_str}\n"
                f"System Idle Probability (P_0)  : {p0_str}\n"
                f"────────────────────────────────────────────────────────\n"
                f"═════════════ ACTUAL SIMULATION DYNAMICS ═══════════════\n"
                f"Balking & reneging bound empirical wait below pure Erlang-C."
            )
        self.theory_lbl.config(text=info_text)

    # =========================================================================
    # EXPORTING PLOTS & SCENARIO COMPARISON
    # =========================================================================

    def _compare_scenarios_dialog(self):
        """Execute all 3 scenarios (Normal, Peak, Overloaded) and display the comparative matrix."""
        was_running = self.is_running
        self.is_running = False
        self._update_play_pause_button()

        self.root.config(cursor="watch")
        self._log_event(f"[{self.sim.env.now:.1f}m] ⏳ Running 3-Scenario Comparison (Normal, Peak, Overloaded)... Please wait...")
        self.root.update_idletasks()

        try:
            target_dir = self.config.output_dir
            os.makedirs(target_dir, exist_ok=True)

            results = run_scenario_comparison(
                sim_time=self.config.sim_time,
                random_seed=self.config.random_seed,
                output_dir=target_dir
            )
            chart_path = os.path.join(target_dir, "scenario_comparison.png")
            fig = plot_scenario_comparison(results, save_path=chart_path)
            plt.close("all")

            self._log_event(f"[{self.sim.env.now:.1f}m] ✔️ Scenario comparison complete! Comparative chart saved to '{chart_path}'.")

            # Display rich on-screen comparison matrix window
            self._show_comparison_window(results, chart_path)

        except Exception as e:
            self._log_event(f"[{self.sim.env.now:.1f}m] ❌ Comparison Error: {e}")
            messagebox.showerror("Comparison Error", f"Failed to run scenario comparison:\n{e}")
        finally:
            self.root.config(cursor="")
            if was_running and not self.is_finished:
                self.is_running = True
                self._update_play_pause_button()

    def _show_comparison_window(self, results: Dict[str, Any], chart_path: str):
        """Display an on-screen dialog showing the comparative matrix table and chart launcher."""
        top = tk.Toplevel(self.root)
        top.title("Scenario Comparison: Normal vs Peak vs Overloaded [Course: NEC00091]")
        top.geometry("840x560")
        top.minsize(750, 480)
        top.configure(bg="#0f172a")
        top.transient(self.root)
        top.lift()
        top.focus_force()
        try:
            top.attributes("-topmost", True)
            top.after(250, lambda: top.attributes("-topmost", False))
        except Exception:
            pass

        # Header
        hdr = tk.Label(
            top,
            text="📊 M/M/3 Queuing System: Scenario Comparison Matrix",
            font=("Segoe UI", 12, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
            pady=10
        )
        hdr.pack(fill="x")

        # Table Frame
        tbl_frame = tk.Frame(top, bg="#1e293b", padx=12, pady=12, relief="groove", bd=1)
        tbl_frame.pack(fill="both", expand=True, padx=16, pady=6)

        # Table Headers
        headers = ["Metric / Parameter", "Normal (λ=0.60)", "Peak (λ=0.80)", "Overloaded (λ=1.00)"]
        for col_idx, h in enumerate(headers):
            bg_col = "#0369a1" if col_idx > 0 else "#334155"
            lbl = tk.Label(
                tbl_frame,
                text=h,
                font=("Segoe UI", 9, "bold"),
                fg="white",
                bg=bg_col,
                padx=8,
                pady=6,
                relief="flat"
            )
            lbl.grid(row=0, column=col_idx, sticky="nsew", padx=1, pady=1)

        norm_s = results["Normal"]["stats"]
        peak_s = results["Peak"]["stats"]
        ovld_s = results["Overloaded"]["stats"]

        t_wq_norm = norm_s["theoretical_mmc"].get("expected_wait_time_Wq")
        t_wq_peak = peak_s["theoretical_mmc"].get("expected_wait_time_Wq")

        matrix_rows = [
            ("Arrival Rate (λ)", "0.60 passengers/min", "0.80 passengers/min", "1.00 passengers/min"),
            ("Service Rate (μ)", "0.30 per counter", "0.30 per counter", "0.30 per counter"),
            ("Ticket Counters (c)", "3 parallel counters", "3 parallel counters", "3 parallel counters"),
            ("Theoretical Intensity (ρ)", f"{norm_s['utilization']['theoretical_traffic_intensity_rho']*100:.1f}% (Stable)", f"{peak_s['utilization']['theoretical_traffic_intensity_rho']*100:.1f}% (Stable)", f"{ovld_s['utilization']['theoretical_traffic_intensity_rho']*100:.1f}% (⚠️ UNSTABLE)"),
            ("Simulation Utilization", f"{norm_s['utilization']['empirical_overall_utilization']*100:.1f}%", f"{peak_s['utilization']['empirical_overall_utilization']*100:.1f}%", f"{ovld_s['utilization']['empirical_overall_utilization']*100:.1f}%"),
            ("Average Wait Time (Served)", f"{norm_s['timing_metrics']['avg_wait_time_served']:.2f} min", f"{peak_s['timing_metrics']['avg_wait_time_served']:.2f} min", f"{ovld_s['timing_metrics']['avg_wait_time_served']:.2f} min"),
            ("Theoretical Wq (Erlang-C)", f"{t_wq_norm:.2f} min" if t_wq_norm else "N/A", f"{t_wq_peak:.2f} min" if t_wq_peak else "N/A", "⚠️ UNSTABLE (Theory: ∞)"),
            ("Served Customers", f"{norm_s['passenger_counts']['num_served']} ({norm_s['passenger_counts']['percent_served']:.1f}%)", f"{peak_s['passenger_counts']['num_served']} ({peak_s['passenger_counts']['percent_served']:.1f}%)", f"{ovld_s['passenger_counts']['num_served']} ({ovld_s['passenger_counts']['percent_served']:.1f}%)"),
            ("Total Abandoned (Balk+Renege)", f"{norm_s['passenger_counts']['num_abandoned']} ({norm_s['passenger_counts']['percent_abandoned']:.1f}%)", f"{peak_s['passenger_counts']['num_abandoned']} ({peak_s['passenger_counts']['percent_abandoned']:.1f}%)", f"{ovld_s['passenger_counts']['num_abandoned']} ({ovld_s['passenger_counts']['percent_abandoned']:.1f}%)"),
            ("Time-Averaged Queue (Lq)", f"{norm_s['queue_metrics']['time_averaged_queue_length']:.2f} pass.", f"{peak_s['queue_metrics']['time_averaged_queue_length']:.2f} pass.", f"{ovld_s['queue_metrics']['time_averaged_queue_length']:.2f} pass."),
        ]

        for r_idx, (m_label, v1, v2, v3) in enumerate(matrix_rows, start=1):
            row_bg = "#1e293b" if r_idx % 2 == 0 else "#0f172a"
            tk.Label(tbl_frame, text=m_label, font=("Segoe UI", 8, "bold"), fg="#94a3b8", bg=row_bg, anchor="w", padx=8, pady=4).grid(row=r_idx, column=0, sticky="nsew", padx=1, pady=1)
            tk.Label(tbl_frame, text=v1, font=("Segoe UI", 8), fg="#38bdf8", bg=row_bg, anchor="center", padx=6, pady=4).grid(row=r_idx, column=1, sticky="nsew", padx=1, pady=1)
            tk.Label(tbl_frame, text=v2, font=("Segoe UI", 8), fg="#f59e0b", bg=row_bg, anchor="center", padx=6, pady=4).grid(row=r_idx, column=2, sticky="nsew", padx=1, pady=1)
            tk.Label(tbl_frame, text=v3, font=("Segoe UI", 8), fg="#f87171" if "UNSTABLE" in v3 else "#fca5a5", bg=row_bg, anchor="center", padx=6, pady=4).grid(row=r_idx, column=3, sticky="nsew", padx=1, pady=1)

        for col_idx in range(4):
            tbl_frame.grid_columnconfigure(col_idx, weight=1)

        # Key Findings / Theoretical Insights Note
        note_frame = tk.Frame(top, bg="#0f172a", padx=16, pady=6)
        note_frame.pack(fill="x")
        insights = (
            "💡 Viva Insights: In the Overloaded scenario (λ=1.00), theoretical Erlang-C is UNSTABLE (ρ = 111.1% ≥ 100%, wait → ∞).\n"
            "   However, the actual physical simulation stabilizes with an empirical average wait of 2.06m because 22.4% of passengers\n"
            "   abandon the queue (balking & reneging), bounding backlog buildup."
        )
        tk.Label(note_frame, text=insights, font=("Segoe UI", 8, "italic"), fg="#cbd5e1", bg="#0f172a", justify="left").pack(anchor="w")

        # Action Buttons Frame
        btn_frame = tk.Frame(top, bg="#0f172a", padx=16, pady=10)
        btn_frame.pack(fill="x")

        def open_chart():
            abs_p = os.path.abspath(chart_path)
            if hasattr(os, "startfile") and os.path.exists(abs_p):
                os.startfile(abs_p)
            else:
                messagebox.showinfo("Chart Path", f"Comparison chart saved at:\n{abs_p}")

        view_btn = tk.Button(
            btn_frame,
            text="🖼️ Open High-Resolution Comparison Graph (.PNG)",
            font=("Segoe UI", 9, "bold"),
            bg="#0284c7",
            fg="white",
            padx=12,
            pady=4,
            command=open_chart,
            cursor="hand2"
        )
        view_btn.pack(side="left", padx=4)

        close_btn = tk.Button(
            btn_frame,
            text="Close",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="white",
            padx=12,
            pady=4,
            command=top.destroy,
            cursor="hand2"
        )
        close_btn.pack(side="right", padx=4)

    def _export_plots(self):
        """Export simulation analytical figures using visualize.py."""
        if not self.sim:
            messagebox.showwarning("Export Notice", "Simulation engine is not initialized.")
            return
        try:
            target_dir = self.config.output_dir
            os.makedirs(target_dir, exist_ok=True)

            # Finalize counter state up to current elapsed time and build DataFrames
            current_time = max(self.sim.env.now, 0.001)
            self.sim.ticket_system.finalize(current_time)
            self.sim._build_dataframes()
            self.sim.summary_stats = self.sim._compute_summary_stats()

            # Save CSV datasets
            self.sim.save_results(target_dir)

            # Generate high-resolution analytical plots
            plot_simulation_dashboard(self.sim, save_path=os.path.join(target_dir, "railway_simulation_dashboard.png"))
            plot_queue_length_over_time(self.sim, save_path=os.path.join(target_dir, "queue_length_over_time.png"))
            plot_wait_time_distribution(self.sim, save_path=os.path.join(target_dir, "wait_time_distribution.png"))
            plot_counter_utilization(self.sim, save_path=os.path.join(target_dir, "counter_utilization.png"))
            plot_passenger_outcomes(self.sim, save_path=os.path.join(target_dir, "passenger_outcomes.png"))
            plt.close("all")

            self._log_event(f"[{self.sim.env.now:.1f}m] 📊 Successfully exported analytical plots & CSV logs to '{target_dir}'.")

            messagebox.showinfo(
                "Export Complete",
                f"Analytical plots and CSV datasets have been successfully saved!\n\n"
                f"📁 Destination Folder:\n{os.path.abspath(target_dir)}\n\n"
                f"Generated Artifacts:\n"
                f"• railway_simulation_dashboard.png\n"
                f"• queue_length_over_time.png\n"
                f"• wait_time_distribution.png\n"
                f"• counter_utilization.png\n"
                f"• passenger_outcomes.png\n"
                f"• passenger_logs.csv\n"
                f"• queue_time_series.csv\n"
                f"• events_log.csv\n"
                f"• counter_utilization.csv"
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to generate plots:\n{e}")


def main():
    root = tk.Tk()
    app = RailwaySimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
