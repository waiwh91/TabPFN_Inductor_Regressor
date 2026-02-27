import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# -----------------------------
# Parameter grouping
# -----------------------------
CORE_KEYS = {
    "t_corelam_nm", "N_lam", "t_lam_nm",
    "t_Ti_bottom_nm", "t_Ti_top_nm",
    "w_fc_um", "w_sc_um", "alpha_deg"
}
WINDING_KEYS = {"w_Cu_um", "t_Cu_um", "w_gap_um", "w_centre_gap_um", "N_turns"}
DIELECTRIC_KEYS = {"t_Su8_bottom_um", "t_Su8_top_um"}
# anything else goes to GENERAL

PARAM_DEFS = [
    # Core
    ("t_corelam_nm", "Core lam thickness (nm)", "float"),
    ("N_lam", "Number of laminations (N_lam)", "int"),
    ("t_lam_nm", "Lamination thickness t_lam (nm)", "float"),
    ("t_Ti_bottom_nm", "Ti bottom thickness (nm)", "float"),
    ("t_Ti_top_nm", "Ti top thickness (nm)", "float"),
    ("w_fc_um", "Core first layer width w_fc (µm)", "float"),
    ("w_sc_um", "Core side extension w_sc (µm)", "float"),
    ("alpha_deg", "Deposition angle alpha (deg)", "float"),

    # Winding
    ("w_Cu_um", "Cu width (µm)", "float"),
    ("t_Cu_um", "Cu thickness (µm)", "float"),
    ("w_gap_um", "Turn gap w_gap (µm)", "float"),
    ("w_centre_gap_um", "Centre gap (µm)", "float"),
    ("N_turns", "Number of turns (N)", "int"),

    # Dielectric
    ("t_Su8_bottom_um", "SU8 bottom thickness (µm)", "float"),
    ("t_Su8_top_um", "SU8 top thickness (µm)", "float"),

    # General
    ("lam_bool", "Use laminations (lam_bool)", "bool"),
    ("padding_percentage", "Solution padding (%)", "float"),
    ("racetrack_tunel_depth_um", "Tunnel depth (µm)", "float"),
    ("core_material", "Core material", "str"),
]

DEFAULTS = {
    "t_corelam_nm": 150,
    "w_Cu_um": 70.5,
    "t_Cu_um": 15,
    "N_lam": 12,
    "t_lam_nm": 15,
    "t_Su8_bottom_um": 4,
    "t_Su8_top_um": 4,

    "t_Ti_bottom_nm": 20,
    "t_Ti_top_nm": 20,

    "w_fc_um": 231,
    "w_sc_um": 20,
    "w_gap_um": 30,

    "alpha_deg": 70,
    "lam_bool": True,
    "padding_percentage": 300,

    "w_centre_gap_um": 100,
    "racetrack_tunel_depth_um": 1.1,

    "N_turns": 2,
    "core_material": "Z9713_H2",
}


def _is_int_string(s: str) -> bool:
    s = s.strip()
    if s.startswith("-"):
        s = s[1:]
    return s.isdigit()


def parse_list(text: str, kind: str):
    raw = [t.strip() for t in text.split(",") if t.strip() != ""]
    if not raw:
        raise ValueError("List is empty.")

    out = []
    for token in raw:
        if kind == "int":
            if not _is_int_string(token):
                raise ValueError(f"Expected int, got '{token}'.")
            out.append(int(token))
        elif kind == "float":
            out.append(float(token))
        elif kind == "bool":
            tl = token.lower()
            if tl in ("true", "1", "yes", "y"):
                out.append(True)
            elif tl in ("false", "0", "no", "n"):
                out.append(False)
            else:
                raise ValueError(f"Expected bool, got '{token}'. Use true/false.")
        elif kind == "str":
            out.append(token)
        else:
            raise ValueError(f"Unknown kind: {kind}")
    return out


def expand_range(start, stop, step, kind: str):
    if step == 0:
        raise ValueError("Step cannot be 0.")
    vals = []
    k = 0
    eps = 1e-12
    while True:
        v = start + k * step
        if (step > 0 and v > stop + eps) or (step < 0 and v < stop - eps):
            break
        vals.append(v)
        k += 1
        if k > 2_000_000:
            raise RuntimeError("Too many sweep points; check step.")
    if kind == "int":
        casted = []
        for v in vals:
            if abs(v - round(v)) > 1e-9:
                raise ValueError("Integer sweep produced non-integer values. Check start/stop/step.")
            casted.append(int(round(v)))
        return casted
    if kind == "float":
        return [float(v) for v in vals]
    raise ValueError("Range sweep is only supported for int/float parameters.")


def estimate_total_runs(param_specs: dict) -> int:
    total = 1
    for _, spec in param_specs.items():
        if spec.get("mode") == "sweep":
            if "values" in spec:
                n = len(spec["values"])
            else:
                vals = expand_range(float(spec["start"]), float(spec["stop"]), float(spec["step"]), "float")
                n = len(vals)
            total *= max(1, n)
    return total


def apply_dark_theme(root: tk.Tk):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    bg = "#121417"
    fg = "#E6E6E6"
    panel = "#1A1F24"
    panel2 = "#141A1F"

    root.configure(bg=bg)
    style.configure(".", background=bg, foreground=fg, fieldbackground=panel, bordercolor=panel2)
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=fg)
    style.configure("TButton", background=panel, foreground=fg, padding=7)
    style.map("TButton",
              background=[("active", panel2), ("pressed", panel2)],
              foreground=[("disabled", "#777777")])
    style.configure("TEntry", fieldbackground=panel, foreground=fg, insertcolor=fg)
    style.configure("TCombobox", fieldbackground=panel, foreground=fg, arrowcolor=fg)
    style.map("TCombobox", fieldbackground=[("readonly", panel)], foreground=[("readonly", fg)])

    style.configure("TLabelframe", background=bg, foreground=fg, bordercolor=panel2)
    style.configure("TLabelframe.Label", background=bg, foreground=fg)
    style.configure("TSeparator", background=panel2)
    style.configure("Section.TButton", background=panel2, foreground=fg, padding=6)
    style.map("Section.TButton", background=[("active", panel), ("pressed", panel)])


class ParamRow:
    def __init__(self, parent, key, label, kind, default_value):
        self.key = key
        self.kind = kind

        self.mode = tk.StringVar(value="const")         # const | sweep
        self.sweep_type = tk.StringVar(value="list")    # list | range

        self.const_value = tk.StringVar(value=str(default_value))

        self.list_values = tk.StringVar(value="")
        self.range_start = tk.StringVar(value=str(default_value))
        self.range_stop = tk.StringVar(value=str(default_value))
        self.range_step = tk.StringVar(value="1" if kind == "int" else "0.1")

        self.frame = ttk.Frame(parent)

        ttk.Label(self.frame, text=label).grid(row=0, column=0, sticky="w", padx=8)

        self.mode_cb = ttk.Combobox(
            self.frame, textvariable=self.mode, values=["const", "sweep"], width=7, state="readonly"
        )
        self.mode_cb.grid(row=0, column=1, padx=6)

        self.const_entry = ttk.Entry(self.frame, textvariable=self.const_value, width=18)
        self.const_entry.grid(row=0, column=2, padx=6)

        self.sweep_cb = ttk.Combobox(
            self.frame, textvariable=self.sweep_type, values=["list", "range"], width=7, state="readonly"
        )
        self.sweep_cb.grid(row=0, column=3, padx=6)

        self.list_entry = ttk.Entry(self.frame, textvariable=self.list_values, width=30)
        self.list_entry.grid(row=0, column=4, padx=6)

        self.range_start_entry = ttk.Entry(self.frame, textvariable=self.range_start, width=10)
        self.range_stop_entry = ttk.Entry(self.frame, textvariable=self.range_stop, width=10)
        self.range_step_entry = ttk.Entry(self.frame, textvariable=self.range_step, width=10)
        self.range_start_entry.grid(row=0, column=5, padx=(6, 2))
        self.range_stop_entry.grid(row=0, column=6, padx=2)
        self.range_step_entry.grid(row=0, column=7, padx=(2, 6))

        ttk.Label(self.frame, text="start").grid(row=1, column=5, sticky="n", pady=(0, 4))
        ttk.Label(self.frame, text="stop").grid(row=1, column=6, sticky="n", pady=(0, 4))
        ttk.Label(self.frame, text="step").grid(row=1, column=7, sticky="n", pady=(0, 4))

        self.mode.trace_add("write", lambda *_: self._refresh_visibility())
        self.sweep_type.trace_add("write", lambda *_: self._refresh_visibility())
        self._refresh_visibility()

    def _refresh_visibility(self):
        mode = self.mode.get()
        if mode == "const":
            self.const_entry.state(["!disabled"])
            self.sweep_cb.state(["disabled"])
            self.list_entry.state(["disabled"])
            self.range_start_entry.state(["disabled"])
            self.range_stop_entry.state(["disabled"])
            self.range_step_entry.state(["disabled"])
        else:
            self.const_entry.state(["disabled"])
            self.sweep_cb.state(["!disabled"])
            if self.sweep_type.get() == "list":
                self.list_entry.state(["!disabled"])
                self.range_start_entry.state(["disabled"])
                self.range_stop_entry.state(["disabled"])
                self.range_step_entry.state(["disabled"])
            else:
                self.list_entry.state(["disabled"])
                self.range_start_entry.state(["!disabled"])
                self.range_stop_entry.state(["!disabled"])
                self.range_step_entry.state(["!disabled"])

    def _parse_scalar(self, text: str):
        if self.kind == "int":
            if not _is_int_string(text):
                raise ValueError(f"{self.key}: expected int, got '{text}'")
            return int(text)
        if self.kind == "float":
            return float(text)
        if self.kind == "bool":
            tl = text.strip().lower()
            if tl in ("true", "1", "yes", "y"):
                return True
            if tl in ("false", "0", "no", "n"):
                return False
            raise ValueError(f"{self.key}: expected bool (true/false), got '{text}'")
        if self.kind == "str":
            return text.strip()
        raise ValueError(f"Unknown kind: {self.kind}")

    def build_spec(self):
        mode = self.mode.get()
        if mode == "const":
            val = self._parse_scalar(self.const_value.get())
            return {"mode": "const", "value": val}

        if self.sweep_type.get() == "list":
            vals = parse_list(self.list_values.get(), self.kind)
            return {"mode": "sweep", "values": vals}

        if self.kind in ("bool", "str"):
            raise ValueError(f"{self.key}: range sweep not supported for {self.kind}. Use list.")

        start = self._parse_scalar(self.range_start.get())
        stop = self._parse_scalar(self.range_stop.get())
        step = self._parse_scalar(self.range_step.get())
        _ = expand_range(start, stop, step, "int" if self.kind == "int" else "float")
        return {"mode": "sweep", "start": start, "stop": stop, "step": step}

    def set_mode(self, mode: str):
        self.mode.set(mode)

    def set_sweep_type(self, sweep_type: str):
        self.sweep_type.set(sweep_type)

    def load_from_spec(self, spec: dict):
        # spec: {"mode":..., ...}
        mode = spec.get("mode", "const")
        self.mode.set(mode)

        if mode == "const":
            self.const_value.set(str(spec.get("value", "")))
            return

        # sweep
        if "values" in spec:
            self.sweep_type.set("list")
            self.list_values.set(", ".join(str(v) for v in spec["values"]))
        else:
            self.sweep_type.set("range")
            self.range_start.set(str(spec.get("start", "")))
            self.range_stop.set(str(spec.get("stop", "")))
            self.range_step.set(str(spec.get("step", "")))


class CollapsibleSection:
    def __init__(self, parent, title: str):
        self.title = title
        self.is_open = tk.BooleanVar(value=True)

        self.outer = ttk.Frame(parent)
        self.outer.pack(fill="x", padx=6, pady=6)

        self.header = ttk.Frame(self.outer)
        self.header.pack(fill="x")

        self.btn = ttk.Button(
            self.header,
            text=f"▼  {title}",
            style="Section.TButton",
            command=self.toggle
        )
        self.btn.pack(side="left", fill="x", expand=True)

        self.body = ttk.Labelframe(self.outer, text=title)
        self.body.pack(fill="x", pady=(6, 0))

        hdr = ttk.Frame(self.body)
        hdr.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(hdr, text="Parameter").grid(row=0, column=0, padx=8, sticky="w")
        ttk.Label(hdr, text="Mode").grid(row=0, column=1, padx=6, sticky="w")
        ttk.Label(hdr, text="Const value").grid(row=0, column=2, padx=6, sticky="w")
        ttk.Label(hdr, text="Sweep type").grid(row=0, column=3, padx=6, sticky="w")
        ttk.Label(hdr, text="List (comma-separated)").grid(row=0, column=4, padx=6, sticky="w")
        ttk.Label(hdr, text="Range").grid(row=0, column=5, columnspan=3, padx=6, sticky="w")

    def toggle(self):
        open_now = not self.is_open.get()
        self.is_open.set(open_now)
        if open_now:
            self.btn.configure(text=f"▼  {self.title}")
            self.body.pack(fill="x", pady=(6, 0))
        else:
            self.btn.configure(text=f"►  {self.title}")
            self.body.pack_forget()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        apply_dark_theme(self)

        self.title("Sweep Config Builder (no simulation)")
        self.geometry("1260x820")

        self.total_runs_var = tk.StringVar(value="Total runs: (click Recompute)")

        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=12, pady=12)

        # Presets + actions
        ttk.Label(top, text="Preset:").grid(row=0, column=0, sticky="w")
        self.preset_var = tk.StringVar(value="—")
        preset_cb = ttk.Combobox(
            top, textvariable=self.preset_var, state="readonly", width=22,
            values=["—", "All Const", "Sweep Core only", "Sweep Copper only", "Sweep Dielectric only"]
        )
        preset_cb.grid(row=0, column=1, sticky="w", padx=(8, 12))
        ttk.Button(top, text="Apply", command=self.apply_preset).grid(row=0, column=2, sticky="w")

        ttk.Button(top, text="Recompute total runs", command=self.recompute_total_runs).grid(row=0, column=3, sticky="w", padx=(16, 0))
        ttk.Label(top, textvariable=self.total_runs_var).grid(row=0, column=4, sticky="w", padx=(10, 0))

        ttk.Separator(top, orient="horizontal").grid(row=1, column=0, columnspan=5, sticky="we", pady=12)

        ttk.Button(top, text="Import JSON", command=self.import_json).grid(row=2, column=0, sticky="w")
        ttk.Button(top, text="Export JSON", command=self.export_json).grid(row=2, column=1, sticky="w", padx=(8, 0))
        ttk.Button(top, text="Collapse all", command=self.collapse_all).grid(row=2, column=2, sticky="w", padx=(16, 0))
        ttk.Button(top, text="Expand all", command=self.expand_all).grid(row=2, column=3, sticky="w", padx=(8, 0))

        top.columnconfigure(4, weight=1)

        # Scrollable param area
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        canvas = tk.Canvas(mid, highlightthickness=0)
        scrollbar = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        container = ttk.Frame(canvas)

        container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Sections
        self.sec_core = CollapsibleSection(container, "Core")
        self.sec_winding = CollapsibleSection(container, "Winding")
        self.sec_diel = CollapsibleSection(container, "Dielectric")
        self.sec_general = CollapsibleSection(container, "General")
        self.sections = [self.sec_core, self.sec_winding, self.sec_diel, self.sec_general]

        # Param rows map
        self.param_rows = {}
        for (key, label, kind) in PARAM_DEFS:
            parent = self._choose_section_body(key)
            row = ParamRow(parent, key, label, kind, DEFAULTS.get(key, ""))
            row.frame.pack(fill="x", padx=6, pady=2)
            self.param_rows[key] = row

    def _choose_section_body(self, key: str):
        if key in CORE_KEYS:
            return self.sec_core.body
        if key in WINDING_KEYS:
            return self.sec_winding.body
        if key in DIELECTRIC_KEYS:
            return self.sec_diel.body
        return self.sec_general.body

    def collapse_all(self):
        for s in self.sections:
            if s.is_open.get():
                s.toggle()

    def expand_all(self):
        for s in self.sections:
            if not s.is_open.get():
                s.toggle()

    def apply_preset(self):
        name = self.preset_var.get()
        if name == "—":
            return
        if name == "All Const":
            for row in self.param_rows.values():
                row.set_mode("const")
            return
        if name == "Sweep Core only":
            self._set_group_modes(CORE_KEYS); return
        if name == "Sweep Copper only":
            self._set_group_modes(WINDING_KEYS); return
        if name == "Sweep Dielectric only":
            self._set_group_modes(DIELECTRIC_KEYS); return

    def _set_group_modes(self, sweep_keys: set):
        for key, row in self.param_rows.items():
            if key in sweep_keys:
                row.set_mode("sweep")
                if row.sweep_type.get() not in ("list", "range"):
                    row.set_sweep_type("list")
            else:
                row.set_mode("const")

    def build_config(self):
        params = {}
        for key, row in self.param_rows.items():
            params[key] = row.build_spec()
        return {"params": params}

    def recompute_total_runs(self):
        try:
            cfg = self.build_config()
            total = estimate_total_runs(cfg["params"])
            self.total_runs_var.set(f"Total runs: {total}")
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e))

    def export_json(self):
        try:
            cfg = self.build_config()
            # also compute for user feedback
            total = estimate_total_runs(cfg["params"])
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e))
            return

        default_name = "sweep_config.json"
        path = filedialog.asksaveasfilename(
            title="Save sweep config",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        messagebox.showinfo("Saved", f"Saved JSON:\n{path}\n\nTotal runs: {total}")

    def import_json(self):
        path = filedialog.askopenfilename(
            title="Open sweep config JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            params = cfg.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("Invalid config: 'params' must be an object/dict.")

            # Load known keys
            for key, row in self.param_rows.items():
                if key in params and isinstance(params[key], dict):
                    row.load_from_spec(params[key])

            total = estimate_total_runs(params)
            self.total_runs_var.set(f"Total runs: {total}")

            messagebox.showinfo("Loaded", f"Loaded JSON:\n{path}\n\nTotal runs: {total}")

        except Exception as e:
            messagebox.showerror("Import failed", str(e))


if __name__ == "__main__":
    App().mainloop()