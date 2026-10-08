"""Configuration validation and versioned, non-overwriting run artifacts."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import scipy
import yaml

from model import make_absorber, make_grid, make_pump, parameters, DOI

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"

ALLOWED = {
    "run": {"validation_mode"},
    "model": {"source", "mass_over_me", "gC_mev_um2", "gR_mev_um2",
              "R_um2_per_ps", "gammaC_per_ps", "gammaR_per_ps",
              "eta_times_gammaR", "background_blueshift"},
    "pump": {"kind", "P0", "center_um", "d_um", "r0_um"},
    "initial": {"mode", "psi_seed_density_um2_inv", "seed_correlation_um",
                "random_seed", "reservoir_initial_um2_inv"},
    "space": {"Lx_um", "Ly_um", "Nx", "Ny", "boundary", "absorbing_layer",
              "absorber_width_um", "absorber_gamma_max_per_ps"},
    "time": {"dt_ps", "t_final_ps", "integrator"},
    "save": {"diagnostic_every_ps", "observable_every_ps", "snapshot_every_ps",
             "stationarity_probe_every_ps", "stationarity_reference_every_ps",
             "checkpoint_every_ps"},
    "checks": {"edge_width_um", "edge_mass_warn", "edge_peak_warn",
               "high_k_cutoff_fraction", "high_k_warn", "signal_min_rel",
               "growth_signal_min_rel", "stationarity_window_ps",
               "stationarity_lags_ps", "stationary_count_span_rel",
               "stationary_density_change_rel", "stationary_reservoir_change_rel",
               "stationary_field_change_rel", "stationary_current_change_rel",
               "balance_warn_rel", "max_walltime_per_run_minutes"},
    "threshold": {"initial_P0_scan", "linear_fit_window_ps",
                  "linear_fit_R2_min", "sigma_agreement_rel",
                  "sigma_agreement_abs_per_ps", "target_bracket_rel"},
}


def _merge(base, override):
    out = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path, _visited=None):
    path = Path(path).resolve()
    visited = set() if _visited is None else _visited
    if path in visited:
        raise ValueError(f"Cyclic configuration inheritance: {path}")
    visited.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    parent = raw.pop("extends", None)
    config = _merge(load_config(path.parent/parent, visited), raw) if parent else raw
    visited.remove(path)
    check_config(config)
    return config


def check_config(config):
    if not isinstance(config, dict) or set(config) != set(ALLOWED):
        raise ValueError(f"Configuration sections must be {sorted(ALLOWED)}")
    for section, allowed in ALLOWED.items():
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
        unknown = set(config[section])-allowed
        if unknown:
            raise ValueError(f"Unknown {section} keys: {sorted(unknown)}")
    required = {k: v for k, v in ALLOWED.items()}
    required["initial"] = required["initial"]-{"reservoir_initial_um2_inv"}
    required["space"] = required["space"]-{
        "absorber_width_um", "absorber_gamma_max_per_ps"}
    for section, keys in required.items():
        missing = keys-set(config[section])
        if missing:
            raise ValueError(f"Missing {section} keys: {sorted(missing)}")
    m, p, s, t, save, checks = (config[k] for k in
                              ("model", "pump", "space", "time", "save", "checks"))
    mode = config["run"]["validation_mode"]
    if mode not in ("physical", "control", "analytic_test"):
        raise ValueError("Unknown validation mode")
    numbers = [v for section in config.values() for v in section.values()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    sequences = [p["center_um"], checks["stationarity_lags_ps"],
                 config["threshold"]["initial_P0_scan"]]
    try:
        finite = (np.all(np.isfinite(numbers)) and
                  all(np.all(np.isfinite(np.asarray(item, dtype=float)))
                      for item in sequences))
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("Non-finite numeric parameter")
    if m["mass_over_me"] <= 0:
        raise ValueError("Mass must be positive")
    coeffs = [m[k] for k in ("gC_mev_um2", "gR_mev_um2", "R_um2_per_ps",
                            "gammaC_per_ps", "gammaR_per_ps")]
    if mode == "physical" and any(v <= 0 for v in coeffs):
        raise ValueError("Physical model requires positive coefficients")
    if mode == "control" and (any(v < 0 for v in coeffs[:2])
                              or any(v <= 0 for v in coeffs[2:])):
        raise ValueError("Invalid control coefficients")
    if mode == "analytic_test" and any(v < 0 for v in coeffs):
        raise ValueError("Analytic-test coefficients cannot be negative")
    if m["eta_times_gammaR"] < 0:
        raise ValueError("eta_times_gammaR must be nonnegative")
    if m["gammaR_per_ps"] == 0 and m["background_blueshift"]:
        raise ValueError("Set background_blueshift=false when gammaR=0")
    if p["kind"] not in ("gaussian", "ring", "uniform", "off"):
        raise ValueError("Unknown pump kind")
    if p["kind"] != "off" and m["R_um2_per_ps"] == 0:
        raise ValueError("Relative P0 pump requires R>0")
    if p["P0"] < 0 or p["d_um"] <= 0 or p["r0_um"] < 0 or len(p["center_um"]) != 2:
        raise ValueError("Invalid pump geometry")
    if config["initial"]["mode"] not in ("physical_startup", "preloaded_reservoir", "deterministic_test"):
        raise ValueError("Unknown initial mode")
    if config["initial"]["psi_seed_density_um2_inv"] < 0 or config["initial"]["seed_correlation_um"] <= 0:
        raise ValueError("Invalid initial seed")
    if config["initial"]["mode"] == "preloaded_reservoir" and m["gammaR_per_ps"] <= 0:
        raise ValueError("Preloaded reservoir requires gammaR>0")
    if s["boundary"] != "periodic":
        raise ValueError("Only periodic FFT boundary is implemented")
    if any(s[k] <= 0 for k in ("Lx_um", "Ly_um", "Nx", "Ny")):
        raise ValueError("Invalid domain")
    if any(int(s[k]) != s[k] or s[k] < 4 for k in ("Nx", "Ny")):
        raise ValueError("Grid sizes must be integers >=4")
    if s["absorbing_layer"] and (
        s.get("absorber_width_um", 0) <= 0
        or s["absorber_width_um"] >= min(s["Lx_um"], s["Ly_um"])/2
        or s.get("absorber_gamma_max_per_ps", 0) <= 0):
        raise ValueError("Absorbing layer needs a finite width and positive loss")
    if t["integrator"] != "split_step_local_rk4" or t["dt_ps"] <= 0 or t["t_final_ps"] <= 0:
        raise ValueError("Invalid time integration")
    dt = t["dt_ps"]
    intervals = list(save.values()) + [checks["stationarity_window_ps"]] + checks["stationarity_lags_ps"]
    if any(v <= 0 or not np.isclose(v/dt, round(v/dt), rtol=0, atol=1e-9) for v in intervals):
        raise ValueError("Save, stationarity and final intervals must be positive integer steps")
    if not np.isclose(t["t_final_ps"]/dt, round(t["t_final_ps"]/dt), rtol=0, atol=1e-9):
        raise ValueError("Final time must be an integer number of steps")
    if (len(checks["stationarity_lags_ps"]) != 2
            or save["stationarity_reference_every_ps"] != save["observable_every_ps"]
            or not np.isclose(checks["stationarity_window_ps"]/
                              save["stationarity_probe_every_ps"],
                              round(checks["stationarity_window_ps"]/
                                    save["stationarity_probe_every_ps"]))
            or max(checks["stationarity_lags_ps"]) >= checks["stationarity_window_ps"]):
        raise ValueError("Stationarity samples and lags do not cover a full window")
    if checks["edge_width_um"] <= 0 or checks["edge_width_um"] >= min(s["Lx_um"], s["Ly_um"])/2:
        raise ValueError("Invalid edge width")
    if len(checks["stationarity_lags_ps"]) != 2 or any(
        not np.isclose(v/save["stationarity_probe_every_ps"], round(v/save["stationarity_probe_every_ps"]))
        for v in checks["stationarity_lags_ps"]
    ):
        raise ValueError("Stationarity lags must align with probe interval")
    if not (0 < checks["high_k_cutoff_fraction"] < 1):
        raise ValueError("Invalid high-k cutoff")
    threshold = config["threshold"]
    if (not threshold["initial_P0_scan"]
            or any(power <= 0 for power in threshold["initial_P0_scan"])
            or threshold["linear_fit_window_ps"] <= 0
            or not 0 < threshold["linear_fit_R2_min"] <= 1
            or threshold["sigma_agreement_rel"] < 0
            or threshold["sigma_agreement_abs_per_ps"] < 0
            or not 0 < threshold["target_bracket_rel"] <= 0.1):
        raise ValueError("Invalid threshold protocol")
    if checks["max_walltime_per_run_minutes"] <= 0 or any(
        checks[key] < 0 for key in ("edge_mass_warn", "edge_peak_warn",
                                  "high_k_warn", "balance_warn_rel",
                                  "signal_min_rel", "growth_signal_min_rel")):
        raise ValueError("Invalid diagnostic tolerance or walltime")
    grid = make_grid(config)
    pump = make_pump(config, grid)
    absorber = make_absorber(config, grid)
    if (not np.all(np.isfinite(pump)) or np.min(pump) < 0
            or not np.all(np.isfinite(absorber)) or np.min(absorber) < 0):
        raise ValueError("Invalid pump field")
    return True


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(config):
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False)+"\n", encoding="utf-8")


def _git_version():
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def create_run(config, root=RESULTS_DIR):
    check_config(config)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}_{config['pump']['kind']}_{config_hash(config)[:8]}"
    path = root/run_id
    path.mkdir(exist_ok=False)
    for child in ("snapshots", "checkpoints", "figures"):
        (path/child).mkdir()
    (path/"resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    grid = make_grid(config)
    pump = make_pump(config, grid)
    absorber = make_absorber(config, grid)
    physical = parameters(config)
    cells = grid.x.size*grid.y.size
    samples = round(config["checks"]["stationarity_window_ps"] /
                    config["save"]["stationarity_probe_every_ps"])+1
    snapshots = math.ceil(config["time"]["t_final_ps"] /
                          config["save"]["snapshot_every_ps"])+1
    checkpoints = math.ceil(config["time"]["t_final_ps"] /
                            config["save"]["checkpoint_every_ps"])+1
    # Conservative planning estimate: RK4/FFT temporaries and one full
    # stationarity window in RAM; uncompressed NPZ arrays on disk.
    memory_estimate = cells*(16*20+8*8+24*(samples+1))
    data_estimate = cells*(24*snapshots+24*(samples+3)*checkpoints)
    _write_json(path/"run_metadata.json", {
        "run_id": run_id, "utc_created": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash(config), "git_commit": _git_version(),
        "launch_argv": sys.argv,
        "source_doi": DOI, "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "device": "CPU", "fft_backend": "numpy.fft", "fft_threads": 1,
        "field_dtype": "complex128",
        "reservoir_dtype": "float64", "derived_parameters": physical,
        "grid": {"Nx": grid.x.size, "Ny": grid.y.size, "dx_um": grid.dx,
                 "dy_um": grid.dy, "Lx_um": grid.lx, "Ly_um": grid.ly,
                 "dkx_um_inv": 2*np.pi/grid.lx, "dky_um_inv": 2*np.pi/grid.ly,
                 "kx_nyquist_um_inv": np.pi/grid.dx,
                 "ky_nyquist_um_inv": np.pi/grid.dy},
        "resource_estimate": {
            "working_memory_bytes": memory_estimate,
            "output_bytes": data_estimate,
            "disk_free_bytes_at_start": shutil.disk_usage(root).free,
            "note": "Conservative uncompressed-array estimate"},
        "numerical_boundary": {
            "fft_boundary": "periodic",
            "absorbing_layer": config["space"]["absorbing_layer"],
            "absorber_width_um": config["space"].get("absorber_width_um")
            if config["space"]["absorbing_layer"] else None,
            "absorber_gamma_max_per_ps": float(np.max(absorber))},
        "pump": {"kind": config["pump"]["kind"], "P0": config["pump"]["P0"],
                 "Pmax_um2_inv_ps_inv": float(np.max(pump)),
                 "Ptotal_per_ps": float(np.sum(pump)*grid.da)},
    })
    (path/"run.log").touch()
    (path/"run_events.jsonl").touch()
    return path


def event(path, level, code, details=None):
    record = {"time": datetime.now(timezone.utc).isoformat(), "level": level,
              "event_code": code, "details": details or {}}
    with (Path(path)/"run_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(canonical_json(record)+"\n")


def log(path, message):
    print(message, flush=True)
    with (Path(path)/"run.log").open("a", encoding="utf-8") as f:
        f.write(message+"\n")


def append_csv(path, row):
    file = Path(path)/"diagnostics.csv"
    fields = list(row)
    if file.exists() and file.stat().st_size:
        with file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            if set(old_fields) != set(fields):
                rows = list(reader)
                fields = list(dict.fromkeys([*old_fields, *fields]))
                with file.open("w", newline="", encoding="utf-8") as out:
                    writer = csv.DictWriter(out, fieldnames=fields)
                    writer.writeheader()
                    for old in rows:
                        writer.writerow({key: old.get(key, "n/a") for key in fields})
            else:
                fields = old_fields
    with file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({k: "n/a" if row.get(k) is None else row[k]
                         for k in fields})


def save_snapshot(path, step, time_ps, psi, nr, grid):
    file = Path(path)/"snapshots"/f"step_{step:08d}.npz"
    np.savez(file, step=step, time_ps=time_ps, psi=psi, nR=nr,
             x_um=grid.x, y_um=grid.y)
    return file


def save_checkpoint(path, step, psi, nr, balance_integral, balance_scale,
                    initial_total, wall_seconds, tracker=None):
    file = Path(path)/"checkpoints"/f"step_{step:08d}.npz"
    np.savez(file, step=step, psi=psi, nR=nr,
             balance_integral=balance_integral, balance_scale=balance_scale,
             initial_total=initial_total, wall_seconds=wall_seconds,
             **(tracker.export_state() if tracker is not None else {}))
    return file


def latest_checkpoint(path):
    files = sorted((Path(path)/"checkpoints").glob("step_*.npz"))
    if not files:
        raise FileNotFoundError("No checkpoint exists")
    return files[-1]


def rewind_artifacts(path, checkpoint_step):
    """Discard samples after a checkpoint before replaying an interrupted run."""
    path = Path(path)
    removed = 0
    file = path/"diagnostics.csv"
    if file.exists():
        with file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        kept = [row for row in rows if int(row["step"]) <= checkpoint_step]
        removed += len(rows)-len(kept)
        if len(kept) != len(rows):
            with file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)
    for snapshot in (path/"snapshots").glob("step_*.npz"):
        if int(snapshot.stem.split("_")[-1]) > checkpoint_step:
            snapshot.unlink()
            removed += 1
    if removed:
        event(path, "INFO", "REWIND_TO_CHECKPOINT",
              {"step": checkpoint_step, "removed_records": removed})
    return removed


def load_run(path):
    path = Path(path)
    config = yaml.safe_load((path/"resolved_config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((path/"run_metadata.json").read_text(encoding="utf-8"))
    summary_file = path/"summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else None
    snapshots = sorted((path/"snapshots").glob("step_*.npz"))
    return {"path": path, "config": config, "metadata": metadata,
            "summary": summary, "snapshots": snapshots}


def artifact_statistics(path):
    path = Path(path)
    files = sorted(f for f in path.rglob("*") if f.is_file())
    return {"snapshot_count": len(list((path/"snapshots").glob("step_*.npz"))),
            "checkpoint_count": len(list((path/"checkpoints").glob("step_*.npz"))),
            "data_bytes": sum(f.stat().st_size for f in files),
            "files": [str(f.relative_to(path)) for f in files]}


def diagnostic_extrema(path):
    file = Path(path)/"diagnostics.csv"
    if not file.exists():
        return {"max_balance_absolute_residual": None,
                "max_balance_relative_residual": None}
    with file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        "max_balance_absolute_residual": max(
            (float(row["balance_absolute_residual"]) for row in rows), default=None),
        "max_balance_relative_residual": max(
            (float(row["balance_relative_residual"]) for row in rows), default=None),
    }


def warning_counts(path):
    counts = {}
    with (Path(path)/"run_events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["level"] == "WARN":
                code = item["event_code"]
                counts[code] = counts.get(code, 0)+1
    return counts


def finalize_metadata(path, summary):
    file = Path(path)/"run_metadata.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    data["performance"] = {
        "elapsed_wall_s": summary["elapsed_wall_s"],
        "mean_step_ms": summary["mean_step_ms"],
        "peak_rss_mb": summary["peak_rss_mb"],
        "completed_steps": summary["step"],
    }
    _write_json(file, data)


def write_summary(path, summary, checks):
    _write_json(Path(path)/"summary.json", summary)
    lines = ["# Диагностика расчёта", "", "| Проверка | Статус | Значение |",
             "|---|---|---|"]
    for name, (status, value) in checks.items():
        lines.append(f"| {name} | {status} | {value} |")
    (Path(path)/"diagnostic_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def compatible_for_resume(old, new):
    a, b = json.loads(canonical_json(old)), json.loads(canonical_json(new))
    old_final, new_final = a["time"].pop("t_final_ps"), b["time"].pop("t_final_ps")
    return a == b and new_final >= old_final
