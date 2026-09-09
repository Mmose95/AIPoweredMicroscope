"""Presentation edition of the microscopy QA desktop workflow.

Run in AI_POWMIC: python qa_workflow_ui_v2.py
All inference, queue, filtering and inspection behaviour is inherited from V1.
Keep this file alongside qa_workflow_ui.py.
"""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import qa_workflow_ui as core
from qa_workflow_ui import (
    APP_BG, CARD_BG, TEXT_PRIMARY, TEXT_SECONDARY, SLIDE_BG,
    QA_COLORS, OVERALL_RULES, HEATMAP_FILTERS, SELECTED_PREVIEW_SIZE,
)


class QAWorkflowUIV2(core.QAWorkflowUI):
    """A presentation layer over the same QA workflow and inference engine."""

    heatmap_footer_height = 26

    def __init__(self, root: tk.Tk, **kwargs) -> None:
        super().__init__(root, **kwargs)
        root.title("Microscopy | Quality Assessment · V2")
        root.geometry("1640x1000")
        root.minsize(1360, 860)

    def _configure_styles(self) -> None:
        super()._configure_styles()
        style = ttk.Style(self.root)
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Card.TLabelframe", borderwidth=0, relief="flat")
        style.configure("Card.TLabelframe.Label", foreground="#334155", font=("Segoe UI", 11, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 23, "bold"), foreground="#18354a")
        style.configure("TButton", background="#e8eef2", foreground="#243e50", borderwidth=0,
                        padding=(10, 7), font=("Segoe UI", 10))
        style.map("TButton", background=[("active", "#dce8ed"), ("pressed", "#cbdce4")],
                  foreground=[("disabled", "#94a3b8")])
        style.configure("Primary.TButton", background="#126c78", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#dce8ed"), ("active", "#0e5963")],
                  foreground=[("disabled", "#94a3b8"), ("!disabled", "white")])
        style.configure("TEntry", padding=6, fieldbackground="white", bordercolor="#d5dfe5")
        style.configure("TCombobox", padding=5, arrowsize=14, bordercolor="#d5dfe5")
        style.map("TCombobox", fieldbackground=[("readonly", "#f8fafc")],
                  foreground=[("readonly", "#243e50")])
        style.configure("TCheckbutton", background=CARD_BG, foreground=TEXT_SECONDARY, padding=(0, 4))
        style.map("TCheckbutton", background=[("active", CARD_BG)])
        style.configure("Horizontal.TProgressbar", background="#126c78", troughcolor="#e8eef2",
                        borderwidth=0, thickness=5)
        style.configure("TScrollbar", background="#d8e2e8", troughcolor="#f3f4f6", borderwidth=0, arrowsize=12)
        style.configure("TNotebook", background=CARD_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 9), background="#e8eef2", foreground="#496071")
        style.map("TNotebook.Tab", background=[("selected", "#126c78")], foreground=[("selected", "white")],
                  padding=[("selected", (16, 9)), ("!selected", (16, 9))],
                  expand=[("selected", (0, 0, 0, 0))])
        style.layout("Horizontal.TProgressbar", [("Horizontal.Progressbar.trough", {"sticky": "nswe", "children": [
            ("Horizontal.Progressbar.pbar", {"side": "left", "sticky": "ns"})]})])

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18, 24, 16))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="Microscopy  /  Quality assessment", style="Header.TLabel").grid(row=0, column=0, sticky="w")

        self.status_label = tk.Label(
            header,
            text="",
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Segoe UI", 11),
            padx=8,
            pady=4,
        )
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.overall_badge = tk.Label(
            header,
            text="Awaiting QA",
            bg=QA_COLORS["Pending"],
            fg="white",
            font=("Segoe UI", 12, "bold"),
            padx=12,
            pady=6,
        )
        self.overall_badge.grid(row=0, column=2, sticky="e")

        body = ttk.Frame(self.root, style="App.TFrame", padding=(24, 0, 24, 20))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=6, minsize=440)
        body.columnconfigure(1, weight=4, minsize=360)
        body.columnconfigure(2, weight=0)
        body.rowconfigure(1, weight=1)

        slide_frame = ttk.Labelframe(body, text="01   Slide overview", style="Card.TLabelframe", padding=12)
        slide_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        slide_frame.rowconfigure(0, weight=0)
        slide_frame.rowconfigure(1, weight=1)
        slide_frame.columnconfigure(0, weight=1)

        slide_toolbar = ttk.Frame(slide_frame, style="Card.TFrame")
        slide_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        slide_toolbar.columnconfigure(4, weight=1)
        ttk.Button(slide_toolbar, text="Fit slide", command=self.fit_slide_to_view).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(slide_toolbar, text="-", width=3, command=lambda: self.zoom_slide(0.8)).grid(row=0, column=1, padx=(0, 4))
        ttk.Button(slide_toolbar, text="+", width=3, command=lambda: self.zoom_slide(1.25)).grid(row=0, column=2, padx=(0, 8))
        tk.Label(
            slide_toolbar,
            textvariable=self.slide_zoom_label_var,
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            width=6,
            anchor="w",
        ).grid(row=0, column=3, sticky="w")

        self.slide_views = ttk.Notebook(slide_frame)
        self.slide_views.grid(row=1, column=0, sticky="nsew")
        slide_view = ttk.Frame(self.slide_views, style="Card.TFrame")
        self.slide_views.add(slide_view, text="Slide images")
        slide_view.rowconfigure(0, weight=1)
        slide_view.columnconfigure(0, weight=1)

        self.slide_canvas = tk.Canvas(slide_view, bg=SLIDE_BG, width=100, height=100, highlightthickness=0, relief="flat")
        self.slide_canvas.grid(row=0, column=0, sticky="nsew")
        self.slide_y_scrollbar = ttk.Scrollbar(slide_view, orient="vertical", command=self.slide_canvas.yview)
        self.slide_y_scrollbar.grid(row=0, column=1, sticky="ns")
        self.slide_x_scrollbar = ttk.Scrollbar(slide_view, orient="horizontal", command=self.slide_canvas.xview)
        self.slide_x_scrollbar.grid(row=1, column=0, sticky="ew")
        self.slide_canvas.configure(
            xscrollcommand=self.slide_x_scrollbar.set,
            yscrollcommand=self.slide_y_scrollbar.set,
        )

        ranking = ttk.Frame(self.slide_views, style="Card.TFrame", padding=0)
        self.slide_views.add(ranking, text="FOV ranking")
        ranking.rowconfigure(0, weight=1)
        ranking.columnconfigure(0, weight=1)

        self.ranking_canvas = tk.Canvas(ranking, bg=SLIDE_BG, width=100, height=100, highlightthickness=0, relief="flat")
        self.ranking_canvas.grid(row=0, column=0, sticky="nsew")

        heatmap_panel = ttk.Labelframe(body, text="02   Cell distributions", style="Card.TLabelframe", padding=12)
        heatmap_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 16))
        heatmap_panel.columnconfigure(0, weight=1)
        heatmap_panel.rowconfigure(1, weight=1)
        heatmap_panel.rowconfigure(2, weight=1)

        heatmap_filter_bar = ttk.Frame(heatmap_panel, style="Card.TFrame")
        heatmap_filter_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        heatmap_filter_bar.columnconfigure(2, weight=1)
        tk.Label(
            heatmap_filter_bar,
            text="Show fields",
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.heatmap_filter_combo = ttk.Combobox(
            heatmap_filter_bar,
            textvariable=self.heatmap_filter,
            values=HEATMAP_FILTERS,
            state="readonly",
            width=22,
        )
        self.heatmap_filter_combo.grid(row=0, column=1, sticky="w")

        leucocyte_panel = ttk.Frame(heatmap_panel, style="Card.TFrame")
        leucocyte_panel.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        leucocyte_panel.columnconfigure(0, weight=1)
        leucocyte_panel.rowconfigure(1, weight=1)
        tk.Label(
            leucocyte_panel,
            text="Leucocyte distribution",
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.leucocyte_heatmap_canvas = tk.Canvas(leucocyte_panel, bg="#111827", height=220, highlightthickness=0)
        self.leucocyte_heatmap_canvas.grid(row=1, column=0, sticky="nsew")

        epithelial_panel = ttk.Frame(heatmap_panel, style="Card.TFrame")
        epithelial_panel.grid(row=2, column=0, sticky="nsew")
        epithelial_panel.columnconfigure(0, weight=1)
        epithelial_panel.rowconfigure(1, weight=1)
        tk.Label(
            epithelial_panel,
            text="Squamous epithelial-cell distribution",
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.epithelial_heatmap_canvas = tk.Canvas(epithelial_panel, bg="#111827", height=220, highlightthickness=0)
        self.epithelial_heatmap_canvas.grid(row=1, column=0, sticky="nsew")

        side_panel = ttk.Frame(body, style="App.TFrame")
        side_panel.grid(row=1, column=2, sticky="nsew")
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(0, weight=1)
        self.review_tabs = ttk.Notebook(side_panel)
        self.review_tabs.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Labelframe(self.review_tabs, text="Run configuration", style="Card.TLabelframe", padding=12)
        self.review_tabs.add(controls, text="Run setup")
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Model checkpoint", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        checkpoint_row = ttk.Frame(controls, style="Card.TFrame")
        checkpoint_row.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        checkpoint_row.columnconfigure(0, weight=1)
        self.checkpoint_entry = ttk.Entry(checkpoint_row, textvariable=self.checkpoint_path)
        self.checkpoint_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(checkpoint_row, text="Browse", command=self.select_checkpoint).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(controls, text="Sample folders", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        sample_row = ttk.Frame(controls, style="Card.TFrame")
        sample_row.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        sample_row.columnconfigure(0, weight=1)
        self.sample_entry = ttk.Entry(sample_row, textvariable=self.sample_folder)
        self.sample_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(sample_row, text="Add", command=self.select_sample_folder).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(sample_row, text="Clear", command=self.clear_sample_folders).grid(row=0, column=2, padx=(6, 0))

        sample_nav = ttk.Frame(controls, style="Card.TFrame")
        sample_nav.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        sample_nav.columnconfigure(1, weight=1)
        self.previous_sample_button = ttk.Button(sample_nav, text="<", width=3, command=self.show_previous_sample)
        self.previous_sample_button.grid(row=0, column=0, sticky="w")
        tk.Label(
            sample_nav,
            textvariable=self.sample_nav_var,
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            anchor="center",
        ).grid(row=0, column=1, sticky="ew", padx=6)
        self.next_sample_button = ttk.Button(sample_nav, text=">", width=3, command=self.show_next_sample)
        self.next_sample_button.grid(row=0, column=2, sticky="e")

        ttk.Label(controls, text="Sample assessment rule", style="Muted.TLabel").grid(row=5, column=0, sticky="w")
        self.rule_combo = ttk.Combobox(controls, textvariable=self.overall_rule, values=OVERALL_RULES, state="readonly")
        self.rule_combo.grid(row=6, column=0, sticky="ew", pady=(2, 12))

        button_row = ttk.Frame(controls, style="Card.TFrame")
        button_row.grid(row=7, column=0, sticky="ew")
        for column in range(4):
            button_row.columnconfigure(column, weight=1)

        self.load_button = ttk.Button(button_row, text="Load", style="Primary.TButton", command=lambda: self.load_selected_sample(auto_start=True))
        self.load_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.start_button = ttk.Button(button_row, text="Start", command=self.start_processing)
        self.start_button.grid(row=0, column=1, sticky="ew", padx=3)

        self.pause_button = ttk.Button(button_row, text="Pause", command=self.pause_processing)
        self.pause_button.grid(row=0, column=2, sticky="ew", padx=3)

        self.reset_button = ttk.Button(button_row, text="Reset", command=self.reset_processing)
        self.reset_button.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        live_row = ttk.Frame(controls, style="Card.TFrame")
        live_row.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        live_row.columnconfigure(0, weight=1)
        live_row.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            live_row,
            text="Live slide",
            variable=self.live_slide_updates,
            command=self.refresh_ui,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            live_row,
            text="Live heatmaps",
            variable=self.live_heatmap_updates,
            command=self.draw_heatmaps,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        summary = ttk.Frame(body, style="Card.TFrame", padding=(16, 12))
        summary.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 18))
        metrics = (
            ("Fields present", self.present_var, TEXT_PRIMARY),
            ("QA complete", self.completed_var, TEXT_PRIMARY),
            ("Pending", self.pending_var, TEXT_SECONDARY),
            ("Qualified", self.qualified_var, QA_COLORS["Qualified"]),
            ("Partially qualified", self.partial_var, QA_COLORS["Partially Qualified"]),
            ("Not qualified", self.not_qualified_var, QA_COLORS["Not Qualified"]),
            ("Leucocytes", self.leucocyte_total_var, TEXT_PRIMARY),
            ("Squamous cells", self.squamous_total_var, TEXT_PRIMARY),
            ("Time / field", self.inference_time_var, TEXT_PRIMARY),
        )
        for column, (label, variable, color) in enumerate(metrics):
            summary.columnconfigure(column, weight=1)
            metric = ttk.Frame(summary, style="Card.TFrame", padding=(8, 2))
            metric.grid(row=0, column=column, sticky="nsew")
            tk.Label(metric, text=label, bg=CARD_BG, fg=TEXT_SECONDARY,
                     font=("Segoe UI", 9), anchor="w").pack(anchor="w")
            tk.Label(metric, textvariable=variable, bg=CARD_BG, fg=color,
                     font=("Segoe UI", 20, "bold"), anchor="w").pack(anchor="w", pady=(4, 0))
        self.inference_progress = ttk.Progressbar(summary, maximum=1, mode="determinate")
        self.inference_progress.grid(row=1, column=0, columnspan=9, sticky="ew", pady=(12, 0))

        detail = ttk.Labelframe(self.review_tabs, text="Selected field of view", style="Card.TLabelframe", padding=12)
        self.detail_panel = detail
        self.review_tabs.add(detail, text="Selected field")
        detail.columnconfigure(0, weight=1)

        preview_frame = tk.Frame(detail, bg="#d1d5db", width=SELECTED_PREVIEW_SIZE, height=SELECTED_PREVIEW_SIZE)
        preview_frame.grid(row=0, column=0, pady=(0, 12))
        preview_frame.grid_propagate(False)
        self.preview_label = tk.Label(preview_frame, bg="#d1d5db", bd=0, highlightthickness=0)
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        self._make_detail_row(detail, 1, "FOV", self.fov_name_var)
        self._make_detail_row(detail, 2, "Position", self.fov_position_var)
        self._make_detail_row(detail, 3, "Stage", self.fov_stage_var)
        self._make_detail_row(detail, 4, "QA", self.fov_qa_var)
        self._make_detail_row(detail, 5, "Leucocytes", self.fov_leucocyte_var)
        self._make_detail_row(detail, 6, "Squamous cells", self.fov_squamous_var)

        legend = ttk.Frame(slide_frame, style="Card.TFrame")
        legend.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for label in ("Qualified", "Partially Qualified", "Not Qualified"):
            tk.Label(legend, text="● " + label, bg=CARD_BG, fg=QA_COLORS[label],
                     font=("Segoe UI", 9)).pack(side="left", padx=(0, 12))
        ttk.Label(detail, text="Double-click a slide or ranking tile to inspect\nthe original image and detections side by side.",
                  style="Muted.TLabel", wraplength=290).grid(row=7, column=0, sticky="w", pady=(14, 0))

    def _bind_events(self) -> None:
        super()._bind_events()
        for canvas in (self.slide_canvas, self.ranking_canvas):
            canvas.bind("<Button-1>", self._show_selected_field, add="+")

    def _show_selected_field(self, _event=None) -> None:
        if self.selected_fov is not None:
            self.review_tabs.select(self.detail_panel)

    def draw_heatmaps(self) -> None:
        # Sequential colour scales encode abundance, independently of QA status.
        source_fovs = self._filtered_heatmap_fovs()
        for canvas, class_type, color, title in (
            (self.leucocyte_heatmap_canvas, "leucocyte", "#2dd4bf", "Leucocyte map awaits QA"),
            (self.epithelial_heatmap_canvas, "epithelial", "#a78bfa", "Epithelial map awaits QA"),
        ):
            self._draw_count_heatmap(canvas, self._class_index_for_heatmap(class_type),
                                     source_fovs, self.heatmap_filter.get(), color, title)

    def _draw_count_heatmap(self, canvas, class_index, source_fovs, filter_label, high_color, empty_text) -> None:
        super()._draw_count_heatmap(canvas, class_index, source_fovs, filter_label, high_color, empty_text)
        if self.sample is None or class_index is None:
            return
        # Keep the legend outside the map's fitted area by reserving a bottom band.
        width, height = canvas.winfo_width(), canvas.winfo_height()
        canvas.create_rectangle(0, height - 26, width, height, fill=SLIDE_BG, outline="")
        for step in range(60):
            x = 12 + step * 1.4
            canvas.create_rectangle(x, height - 16, x + 1.5, height - 8,
                                    fill=core._blend_hex_color("#172033", high_color, step / 59), outline="")
        canvas.create_text(104, height - 12, anchor="w", text="0 → max cells / field",
                           fill="#cbd5e1", font=("Segoe UI", 8))


def main() -> None:
    args = core.build_parser().parse_args()
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else None
    sample_folder = Path(args.sample_folder).expanduser() if args.sample_folder else None
    if args.check:
        # Reuse V1's command-line validation and diagnostic output.
        core.main()
        return
    root = tk.Tk()
    QAWorkflowUIV2(root, initial_checkpoint=checkpoint, initial_sample_folder=sample_folder,
                   auto_load=bool(args.auto_load))
    root.mainloop()


if __name__ == "__main__":
    main()
