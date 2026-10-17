"""Calculation, configuration, diagnostics and saved runs."""
from __future__ import annotations

# Configuration and saving

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
CONFIG_FILE = ROOT / "config.yaml"
RESULTS_DIR = ROOT / "results" / "runs"

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


def load_config(profile="gaussian"):
    """Named profile in config.yaml, or a saved resolved_config.yaml file."""
    raw = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if str(profile) == "base" or str(profile) in raw["profiles"]:
        config = _merge(raw["base"], raw["profiles"].get(str(profile), {}))
    else:
        config = yaml.safe_load(Path(profile).read_text(encoding="utf-8"))
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
    # Planning estimate: RK4/FFT temporaries, one stationarity window,
    # and process/library overhead; uncompressed NPZ arrays on disk.
    memory_estimate = 256*2**20+cells*(16*20+8*8+24*(samples+1))
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
            "note": "Approximate uncompressed-array estimate with process overhead"},
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


def compatible_for_resume(old, new):
    a, b = json.loads(canonical_json(old)), json.loads(canonical_json(new))
    old_final, new_final = a["time"].pop("t_final_ps"), b["time"].pop("t_final_ps")
    return a == b and new_final >= old_final


# Observables and stationarity

import json
import numpy as np

from model import make_absorber, parameters, pump_profile


def norm2(field, grid):
    return float(np.sqrt(np.sum(np.abs(field)**2)*grid.da))


def relative_change(a, b, grid, reference=1.0):
    return norm2(a-b, grid)/max(norm2(b, grid), 1e-12*reference)


def phase_aligned_change(a, b, grid, reference=1.0):
    na2 = norm2(a, grid)**2
    nb2 = norm2(b, grid)**2
    overlap = np.vdot(b, a)*grid.da
    return np.sqrt(max(0.0, na2+nb2-2*abs(overlap)))/max(
        np.sqrt(nb2), 1e-12*reference)


def field_gradients(psi, grid):
    """Spectral derivatives with the even-grid Nyquist derivative set to zero."""
    fft = np.fft.fft2(psi)
    kx, ky = grid.kx.copy(), grid.ky.copy()
    if kx.size % 2 == 0:
        kx[kx.size//2] = 0.0
    if ky.size % 2 == 0:
        ky[ky.size//2] = 0.0
    return (np.fft.ifft2(1j*kx[None, :]*fft),
            np.fft.ifft2(1j*ky[:, None]*fft))


def current(psi, grid, config):
    p = parameters(config)
    dxpsi, dypsi = field_gradients(psi, grid)
    factor = 2*p["K"]/p["hbar"]
    return factor*np.imag(np.conj(psi)*dxpsi), factor*np.imag(np.conj(psi)*dypsi)


def vortex_diagnostics(psi, grid, config, *, radii_um=(4, 6, 7.5, 9, 12),
                       n_angles=1024, density_floor_rel=0.05, center_um=None):
    """Charge on dense counterclockwise circles and angular momentum about centre.

    Interpolate the complex field, never the wrapped phase. Phase increments
    include the closing segment; reject weak contours or unresolved jumps.
    """
    from scipy.ndimage import map_coordinates
    if n_angles < 16 or not 0 < density_floor_rel < 1:
        raise ValueError('Use >=16 contour samples and a density floor in (0,1)')
    centre=config['pump']['center_um'] if center_um is None else center_um
    x0,y0=map(float,centre)
    n=abs(psi)**2;nc=float(n.sum()*grid.da);peak=float(n.max())
    if nc <= 0:
        raise ValueError('Charge and angular momentum require a nonzero field')
    dxpsi,dypsi=field_gradients(psi,grid)
    x=grid.x[None,:]-x0;y=grid.y[:,None]-y0
    p=parameters(config)
    expectation=np.sum(np.conj(psi)*(-1j)*(x*dypsi-y*dxpsi))*grid.da
    jx,jy=current(psi,grid,config);factor=2*p['K']/p['hbar']
    theta=np.linspace(0,2*np.pi,n_angles,endpoint=False)
    contours=[]
    for radius in radii_um:
        radius=float(radius)
        if radius<=0:raise ValueError('Contour radius must be positive')
        xx=x0+radius*np.cos(theta);yy=y0+radius*np.sin(theta)
        inside=bool(xx.min()>=grid.x[0] and xx.max()<=grid.x[-1] and
                    yy.min()>=grid.y[0] and yy.max()<=grid.y[-1])
        if not inside:
            contours.append({'radius_um':radius,'status':'OUTSIDE_GRID','q':None});continue
        coords=np.array([(yy-grid.y[0])/grid.dy,(xx-grid.x[0])/grid.dx])
        def sample(values):return map_coordinates(values,coords,order=1,mode='nearest')
        values=sample(psi.real)+1j*sample(psi.imag)
        increments=np.angle(np.roll(values,-1)*values.conj())
        raw=float(increments.sum()/(2*np.pi));density_min=float((abs(values)**2).min()/peak)
        jump=float(np.max(abs(increments)))
        valid=density_min>=density_floor_rel and jump<np.pi/2 and abs(raw-round(raw))<1e-6
        status='VALID' if valid else 'LOW_DENSITY' if density_min<density_floor_rel else 'UNRESOLVED_PHASE'
        sjx,sjy=sample(jx),sample(jy)
        density=sample(n)
        tangential=-np.sin(theta)*sjx+np.cos(theta)*sjy
        radial=np.cos(theta)*sjx+np.sin(theta)*sjy
        circulation=float(radius*2*np.pi*np.mean(tangential/np.maximum(density,1e-30)))
        contours.append({'radius_um':radius,'status':status,'q':int(round(raw)) if valid else None,
            'raw_winding':raw,'minimum_density_rel':density_min,'max_phase_increment_rad':jump,
            'velocity_circulation_um2_per_ps':circulation,'charge_from_current':circulation/(2*np.pi*factor),
            'mean_tangential_current_um_inv_ps_inv':float(tangential.mean()),
            'mean_radial_current_um_inv_ps_inv':float(radial.mean())})
    charges=[c['q'] for c in contours if c['status']=='VALID']
    consistent=len(charges)>=2 and len(set(charges))==1
    return {'NC':nc,'center_um':[x0,y0],'n_angles':int(n_angles),'density_floor_rel':density_floor_rel,
        'contour_orientation':'counterclockwise','contours':contours,
        'q':charges[0] if consistent else None,'contour_charge_consistent':consistent,
        'Lz_mev_ps':float(p['hbar']*expectation.real),
        'Lz_over_hbar':float(expectation.real),'Lz_over_hbar_NC':float(expectation.real/nc),
        'Lz_imag_over_hbar_NC':float(expectation.imag/nc),
        'classification':'VORTEX_CONDENSATE' if consistent and charges[0]!=0 else
                         'ZERO_NET_CHARGE' if consistent else 'UNRESOLVED'}


def spectrum(psi, grid):
    """Physical Fourier transform, with d²k/(2pi)² measure."""
    return grid.da*np.fft.fft2(psi)


def parseval_error(psi, grid):
    spatial = np.sum(np.abs(psi)**2)*grid.da
    spectral = (np.sum(np.abs(spectrum(psi, grid))**2)
                * (2*np.pi/grid.lx)*(2*np.pi/grid.ly)/(2*np.pi)**2)
    return float(abs(spatial-spectral)/max(spatial, 1e-30))


def number_reference(config, grid, initial_nc=1.0):
    p = parameters(config)
    if config["pump"]["kind"] == "off" or p["nRth"] is None:
        return max(initial_nc, 1.0)
    area = float(np.sum(pump_profile(config, grid))*grid.da)
    return p["nRth"]*area


def basic_observables(psi, nr, pump, grid, config, absorber=None):
    nc = np.abs(psi)**2
    p = parameters(config)
    if absorber is None and config["space"]["absorbing_layer"]:
        absorber = make_absorber(config, grid)
    return {
        "NC": float(np.sum(nc)*grid.da),
        "NR": float(np.sum(nr)*grid.da),
        "NC_max_um2_inv": float(np.max(nc)),
        "nR_min_um2_inv": float(np.min(nr)),
        "nR_max_um2_inv": float(np.max(nr)),
        "gain_max_ps_inv": float(np.max(p["R"]*nr-p["gammaC"])),
        "P_total_per_ps": float(np.sum(pump)*grid.da),
        "absorber_loss_per_ps": 0.0 if absorber is None else
        float(np.sum(absorber*nc)*grid.da),
    }


def spatial_diagnostics(psi, nr, pump, grid, config, nref):
    nc = np.abs(psi)**2
    total = np.sum(nc)
    checks = config["checks"]
    if total*grid.da < checks["signal_min_rel"]*nref:
        return {"edge_mass_fraction": None, "edge_peak_ratio": None,
                "high_k_fraction": None, "j_max_um_inv_ps_inv": None,
                "parseval_relative_error": None, "r_rms_um": None,
                "r_rms_pump_um": None, "l_z_over_hbar_NC": None}
    xdist = grid.lx/2-np.abs(grid.x[None, :])
    ydist = grid.ly/2-np.abs(grid.y[:, None])
    edge = (xdist < checks["edge_width_um"]) | (ydist < checks["edge_width_um"])
    energy = np.abs(np.fft.fft2(psi))**2
    cutoff = checks["high_k_cutoff_fraction"]
    high = ((np.abs(grid.kx[None, :]) >= cutoff*np.pi/grid.dx)
            | (np.abs(grid.ky[:, None]) >= cutoff*np.pi/grid.dy))
    jx, jy = current(psi, grid, config)
    nc_total = total*grid.da
    x = grid.x[None, :]
    y = grid.y[:, None]
    xcenter = float(np.sum(x*nc)/total)
    ycenter = float(np.sum(y*nc)/total)
    px, py = map(float, config["pump"]["center_um"])
    radius_com = float(np.sqrt(np.sum(((x-xcenter)**2+(y-ycenter)**2)*nc)/total))
    radius_pump = float(np.sqrt(np.sum(((x-px)**2+(y-py)**2)*nc)/total))
    factor = 2*parameters(config)["K"]/parameters(config)["hbar"]
    lz = float(np.sum(x*jy-y*jx)*grid.da/(factor*nc_total))
    return {
        "edge_mass_fraction": float(np.sum(nc[edge])/total),
        "edge_peak_ratio": float(np.max(nc[edge])/np.max(nc)),
        "high_k_fraction": float(np.sum(energy[high])/np.sum(energy)),
        "j_max_um_inv_ps_inv": float(np.max(np.hypot(jx, jy))),
        "parseval_relative_error": parseval_error(psi, grid),
        "r_rms_um": radius_com, "r_rms_pump_um": radius_pump,
        "l_z_over_hbar_NC": lz,
    }


def balance_residual(initial_total, current_total, integral, scale):
    absolute = abs(current_total-initial_total-integral)
    return absolute, absolute/max(scale, 1e-30)


def _span(values, nref):
    a = np.asarray(values)
    return float((np.max(a)-np.min(a))/max(float(np.mean(a)), 1e-12*nref))


class StationarityTracker:
    """A complete 100-ps window of dense observations and two-lag fields."""

    def __init__(self, config, grid, nref):
        self.config, self.grid, self.nref = config, grid, nref
        self.window = float(config["checks"]["stationarity_window_ps"])
        self.probe_every = float(config["save"]["stationarity_probe_every_ps"])
        self.reference_every = float(config["save"]["stationarity_reference_every_ps"])
        self.lags = tuple(float(x) for x in config["checks"]["stationarity_lags_ps"])
        self.start = None
        self.reference = None
        self.dense = []
        self.probes = []
        self.max_ref = {"density_reference_change_max": 0.0,
                        "reservoir_reference_change_max": 0.0,
                        "field_reference_change_max": 0.0,
                        "current_reference_change_max": 0.0}
        self.significant_current = False
        self.last = {"stationarity_status": "NOT_TESTED"}

    def export_state(self):
        return {
            "tracker_start_ps": np.array(self.start if self.start is not None else np.nan),
            "tracker_reference_psi": self.reference[0],
            "tracker_reference_nR": self.reference[1],
            "tracker_dense": np.asarray(self.dense, dtype=float).reshape(-1, 3),
            "tracker_probe_times": np.asarray([r[0] for r in self.probes], dtype=float),
            "tracker_probe_psi": np.stack([r[1] for r in self.probes]),
            "tracker_probe_nR": np.stack([r[2] for r in self.probes]),
            "tracker_max_ref_json": np.array(json.dumps(self.max_ref)),
            "tracker_last_json": np.array(json.dumps(self.last)),
            "tracker_significant_current": np.array(self.significant_current),
        }

    def restore_state(self, checkpoint):
        self.start = float(checkpoint["tracker_start_ps"])
        self.reference = (checkpoint["tracker_reference_psi"].copy(),
                          checkpoint["tracker_reference_nR"].copy())
        self.dense = [tuple(map(float, row)) for row in checkpoint["tracker_dense"]]
        times = checkpoint["tracker_probe_times"]
        fields = checkpoint["tracker_probe_psi"]
        reservoirs = checkpoint["tracker_probe_nR"]
        self.probes = [(float(t), psi.copy(), nr.copy()) for t, psi, nr
                       in zip(times, fields, reservoirs)]
        self.max_ref = json.loads(str(checkpoint["tracker_max_ref_json"]))
        self.last = json.loads(str(checkpoint["tracker_last_json"]))
        self.significant_current = bool(checkpoint["tracker_significant_current"])

    def _refs(self):
        p = parameters(self.config)
        area = self.nref/max(p["nRth"] or 1.0, 1e-30)
        density_ref = (p["nRth"] or 1.0)*np.sqrt(area)
        return density_ref, np.sqrt(self.nref)

    def _current_scale(self, psi):
        if self.config["pump"]["kind"] == "uniform":
            return None
        p = parameters(self.config)
        d = float(self.config["pump"]["d_um"])
        if d <= 0:
            return None
        return 1e-6*(2*p["K"]/p["hbar"])/d*norm2(np.abs(psi)**2, self.grid)

    def update(self, time_ps, psi, nr, obs, step):
        dt = float(self.config["time"]["dt_ps"])
        probe_steps = round(self.probe_every/dt)
        reference_steps = round(self.reference_every/dt)
        window_steps = round(self.window/dt)
        if self.start is None:
            self.start = time_ps
            self.reference = (psi.copy(), nr.copy())
        if step % reference_steps == 0:
            self.dense.append((time_ps, obs["NC"], obs["NR"]))
            pr, nr_reference = self.reference
            density_ref, _field_ref = self._refs()
            self.max_ref["density_reference_change_max"] = max(
                self.max_ref["density_reference_change_max"],
                relative_change(np.abs(psi)**2, np.abs(pr)**2, self.grid, density_ref))
            self.max_ref["reservoir_reference_change_max"] = max(
                self.max_ref["reservoir_reference_change_max"],
                relative_change(nr, nr_reference, self.grid, density_ref))
            self.max_ref["field_reference_change_max"] = max(
                self.max_ref["field_reference_change_max"],
                phase_aligned_change(psi, pr, self.grid, np.sqrt(self.nref)))
            jx, jy = current(psi, self.grid, self.config)
            rx, ry = current(pr, self.grid, self.config)
            jnow = norm2(np.stack((jx, jy)), self.grid)
            jref = norm2(np.stack((rx, ry)), self.grid)
            scale = self._current_scale(psi)
            if scale is not None and max(jnow, jref) >= scale:
                self.significant_current = True
                self.max_ref["current_reference_change_max"] = max(
                    self.max_ref["current_reference_change_max"],
                    norm2(np.stack((jx-rx, jy-ry)), self.grid)/max(jref, scale))
        if step % probe_steps == 0:
            self.probes.append((time_ps, psi.copy(), nr.copy()))
        if step > 0 and step % window_steps == 0:
            self.last = self.evaluate()
            self.start = time_ps
            self.reference = (psi.copy(), nr.copy())
            self.dense = [(time_ps, obs["NC"], obs["NR"])]
            self.probes = [(time_ps, psi.copy(), nr.copy())]
            self.max_ref = {key: 0.0 for key in self.max_ref}
            self.significant_current = False
        return self.last

    def evaluate(self):
        c = self.config["checks"]
        result = dict(self.max_ref)
        expected_dense = round(self.window/self.reference_every)+1
        expected_probes = round(self.window/self.probe_every)+1
        if (len(self.dense) < expected_dense or len(self.probes) < expected_probes
                or self.dense[-1][0]-self.dense[0][0] < self.window-1e-8
                or self.probes[-1][0]-self.probes[0][0] < self.window-1e-8):
            return {"stationarity_status": "NOT_TESTED"}
        result["NC_window_span"] = _span([r[1] for r in self.dense], self.nref)
        result["NR_window_span"] = _span([r[2] for r in self.dense], self.nref)
        result["dense_points"] = len(self.dense)
        result["probe_points"] = len(self.probes)
        result["time_window_ps"] = [self.dense[0][0], self.dense[-1][0]]
        if self.dense[-1][1] < c["signal_min_rel"]*self.nref:
            tail = self.dense[max(0, len(self.dense)*4//5):]
            times = np.array([r[0] for r in tail])
            values = np.array([r[1] for r in tail])
            if len(tail) >= 3 and np.all(values > 1e-200):
                slope = float(np.polyfit(times, np.log(values), 1)[0])
                result["final_log_NC_slope_per_ps"] = slope
                result["stationarity_status"] = "DECAYED" if slope < -1e-3 else "UNDETERMINED"
            else:
                result["stationarity_status"] = "UNDETERMINED"
            return result
        for name in ("density_change_max", "reservoir_change_max",
                     "field_aligned_change_max", "current_change_max"):
            result[name] = 0.0
        rd, _rf = self._refs()
        lag_counts = {lag: 0 for lag in self.lags}
        for t, psi, nr in self.probes:
            for lag in self.lags:
                previous = next(((p, r) for tp, p, r in self.probes
                                 if abs(tp-(t-lag)) < 1e-8), None)
                if previous is None:
                    continue
                lag_counts[lag] += 1
                pp, rr = previous
                result["density_change_max"] = max(result["density_change_max"],
                    relative_change(abs(psi)**2, abs(pp)**2, self.grid, rd))
                result["reservoir_change_max"] = max(result["reservoir_change_max"],
                    relative_change(nr, rr, self.grid, rd))
                result["field_aligned_change_max"] = max(result["field_aligned_change_max"],
                    phase_aligned_change(psi, pp, self.grid, np.sqrt(self.nref)))
                jx, jy = current(psi, self.grid, self.config)
                px, py = current(pp, self.grid, self.config)
                jnow = norm2(np.stack((jx, jy)), self.grid)
                jprev = norm2(np.stack((px, py)), self.grid)
                scale = self._current_scale(psi)
                if scale is not None and max(jnow, jprev) >= scale:
                    self.significant_current = True
                    result["current_change_max"] = max(result["current_change_max"],
                        norm2(np.stack((jx-px, jy-py)), self.grid)/max(jprev, scale))
        result["lag_pair_counts"] = {str(lag): count for lag, count in lag_counts.items()}
        result["dense_points"] = len(self.dense)
        result["probe_points"] = len(self.probes)
        result["time_window_ps"] = [self.dense[0][0], self.dense[-1][0]]
        if not all(lag_counts.values()):
            result["stationarity_status"] = "NOT_TESTED"
            return result
        if not self.significant_current:
            result["current_status"] = "NEGLIGIBLE_CURRENT"
            result["current_change_max"] = None
            result["current_reference_change_max"] = None
        else:
            result["current_status"] = "TESTED"
        passed = (
            result["NC_window_span"] <= c["stationary_count_span_rel"]
            and result["NR_window_span"] <= c["stationary_count_span_rel"]
            and all(result[k] <= c["stationary_density_change_rel"] for k in
                    ("density_change_max", "density_reference_change_max"))
            and all(result[k] <= c["stationary_reservoir_change_rel"] for k in
                    ("reservoir_change_max", "reservoir_reference_change_max"))
            and all(result[k] <= c["stationary_field_change_rel"] for k in
                    ("field_aligned_change_max", "field_reference_change_max"))
            and (not self.significant_current or all(
                result[k] <= c["stationary_current_change_rel"] for k in
                ("current_change_max", "current_reference_change_max")))
        )
        result["stationarity_status"] = "STATIONARY_MODE" if passed else (
            "NONSTATIONARY_FIELD" if result["NC_window_span"] <= c["stationary_count_span_rel"]
            and result["NR_window_span"] <= c["stationary_count_span_rel"]
            else "DYNAMIC_OR_UNSETTLED")
        return result


# Numerical methods

import numpy as np
from model import make_grid, make_absorber, parameters


def kinetic_half_factor(grid, config, dt):
    p = parameters(config)
    return np.exp(-0.5j*p["K"]*grid.k2*dt/p["hbar"])


def kinetic_step(psi, factor):
    return np.fft.ifft2(np.fft.fft2(psi)*factor)


def local_rhs(psi, nr, pump, config, absorber=None):
    p = parameters(config)
    density = np.abs(psi)**2
    background = p["eta"]*pump if p["background_blueshift"] else 0.0
    energy = p["gC"]*density+p["gR"]*(nr+background)
    loss = 0.0 if absorber is None else absorber
    growth = 0.5*(p["R"]*nr-p["gammaC"]-loss)
    return (growth-1j*energy/p["hbar"])*psi, pump-(p["gammaR"]+p["R"]*density)*nr


def local_rk4(psi, nr, pump, config, dt, absorber=None):
    a, b = local_rhs(psi, nr, pump, config, absorber)
    c, d = local_rhs(psi+dt*a/2, nr+dt*b/2, pump, config, absorber)
    e, f = local_rhs(psi+dt*c/2, nr+dt*d/2, pump, config, absorber)
    g, h = local_rhs(psi+dt*e, nr+dt*f, pump, config, absorber)
    return psi+dt*(a+2*c+2*e+g)/6, nr+dt*(b+2*d+2*f+h)/6


def split_step(psi, nr, pump, config, dt, half_factor=None, absorber=None):
    if half_factor is None:
        half_factor = kinetic_half_factor(make_grid(config), config, dt)
    if absorber is None and config["space"]["absorbing_layer"]:
        absorber = make_absorber(config, make_grid(config))
    psi = kinetic_step(psi, half_factor)
    psi, nr = local_rk4(psi, nr, pump, config, dt, absorber)
    return kinetic_step(psi, half_factor), nr


def linear_step(psi, pump, config, half_factor, absorber=None):
    """One step around psi=0 and the *frozen* nR=P/gammaR branch."""
    p = parameters(config)
    nr0 = pump/p["gammaR"]
    bg = p["eta"]*pump if p["background_blueshift"] else 0.0
    if absorber is None and config["space"]["absorbing_layer"]:
        absorber = make_absorber(config, make_grid(config))
    loss = 0.0 if absorber is None else absorber
    dt = float(config["time"]["dt_ps"])
    local = np.exp((0.5*(p["R"]*nr0-p["gammaC"]-loss)
                    -1j*p["gR"]*(nr0+bg)/p["hbar"])*dt)
    return kinetic_step(kinetic_step(psi, half_factor)*local, half_factor)


def simulate_2d(config, *, result_root=None, resume_dir=None, max_steps=None,
                on_initialized=None, on_observation=None):
    """Run and persist the full coupled model; max_steps explicitly stops a preview."""
    import json
    from pathlib import Path
    import resource
    import time

    from model import initial_fields, make_pump

    check_config(config)
    grid = make_grid(config)
    pump = make_pump(config, grid)
    absorber = make_absorber(config, grid)
    absorber_enabled = bool(config["space"]["absorbing_layer"])
    absorber_arg = absorber if absorber_enabled else None
    p = parameters(config)
    dt = float(config["time"]["dt_ps"])
    requested_steps = round(config["time"]["t_final_ps"]/dt)
    half = kinetic_half_factor(grid, config, dt)
    clock = time.perf_counter()
    tracker_state = None
    if resume_dir is None:
        path = create_run(config, root=result_root or RESULTS_DIR)
        psi, nr = initial_fields(config, grid, pump)
        first_step, integral, scale, old_wall = 0, 0.0, 0.0, 0.0
        nc0 = np.sum(abs(psi)**2)*grid.da
        nr0 = np.sum(nr)*grid.da
        initial_total = float(nc0+nr0)
        save_snapshot(path, 0, 0.0, psi, nr, grid)
        save_checkpoint(path, 0, psi, nr, 0.0, 0.0, initial_total, 0.0)
        event(path, "INFO", "RUN_STARTED", {"requested_steps": requested_steps})
        log(path, f"RUN_ID={path.name} config_sha256={config_hash(config)}")
        log(path, f"MODEL K_meV_um2={p['K']:.9g} hbar_meV_ps={p['hbar']:.9g} "
                    f"eta_ps={p['eta']} Pth0D_um2_inv_ps_inv={p['Pth0D']} "
                    f"nRth_um2_inv={p['nRth']} DOI=10.1126/sciadv.adj1589")
        log(path, f"GRID Nx={grid.x.size} Ny={grid.y.size} "
                    f"dx_um={grid.dx} dy_um={grid.dy} dt_ps={dt} "
                    f"steps={requested_steps} pump={config['pump']['kind']} "
                    f"P0={config['pump']['P0']} "
                    f"numerical_absorber={absorber_enabled} "
                    f"gamma_abs_max_per_ps={float(np.max(absorber))}")
    else:
        path = Path(resume_dir)
        old = load_run(path)
        if (old["summary"] and old["summary"]["run_status"] == "RUN_COMPLETED"
                and config["time"]["t_final_ps"] <= old["config"]["time"]["t_final_ps"]):
            raise ValueError("A completed run can only be extended to a later final time")
        if not compatible_for_resume(old["config"], config):
            raise ValueError("Incompatible resume configuration")
        with np.load(latest_checkpoint(path)) as checkpoint:
            psi, nr = checkpoint["psi"].copy(), checkpoint["nR"].copy()
            first_step = int(checkpoint["step"])
            integral = float(checkpoint["balance_integral"])
            scale = float(checkpoint["balance_scale"])
            initial_total = float(checkpoint["initial_total"])
            old_wall = float(checkpoint["wall_seconds"])
            if "tracker_start_ps" in checkpoint.files:
                tracker_state = {key: checkpoint[key].copy() for key in checkpoint.files
                                 if key.startswith("tracker_")}
        if psi.shape != grid.shape:
            raise ValueError("Checkpoint grid mismatch")
        rewind_artifacts(path, first_step)
        if config != old["config"]:
            import yaml
            (path/"resolved_config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
            meta_file = path/"run_metadata.json"
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta["config_sha256"] = config_hash(config)
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            event(path, "INFO", "FINAL_TIME_EXTENDED",
                          {"old_ps": old["config"]["time"]["t_final_ps"],
                           "new_ps": config["time"]["t_final_ps"]})
        event(path, "INFO", "RUN_RESUMED", {"from_step": first_step})

    if on_initialized is not None:
        on_initialized(path)

    nref = number_reference(config, grid, initial_total)
    tracker = StationarityTracker(config, grid, nref)
    obs = basic_observables(psi, nr, pump, grid, config, absorber_arg)
    if tracker_state is None:
        tracker.update(first_step*dt, psi, nr, obs, first_step)
    else:
        tracker.restore_state(tracker_state)
    if on_observation is not None:
        on_observation(first_step*dt, psi, nr, obs, tracker.last)
    ptotal = obs["P_total_per_ps"]
    previous_source = (ptotal-p["gammaC"]*obs["NC"]-p["gammaR"]*obs["NR"]
                       -obs["absorber_loss_per_ps"])
    previous_magnitude = (ptotal+p["gammaC"]*obs["NC"]+p["gammaR"]*obs["NR"]
                          +obs["absorber_loss_per_ps"])
    intervals = {key: round(config["save"][key]/dt) for key in config["save"]}
    # Legacy configuration key: this is a performance recommendation,
    # never a reason to stop a physical run.
    recommended_seconds = config["checks"]["max_walltime_per_run_minutes"]*60
    runtime_notice = False
    warnings = {}
    previous_diagnostic = None
    status, step = "RUN_INCOMPLETE", first_step
    try:
        for step in range(first_step+1, requested_steps+1):
            psi, nr = split_step(psi, nr, pump, config, dt, half, absorber_arg)
            if not np.all(np.isfinite(psi)) or not np.all(np.isfinite(nr)):
                raise FloatingPointError("Non-finite field or reservoir")
            minimum_nr = float(np.min(nr))
            if minimum_nr < -1e-10*max(1.0, p["nRth"] or 1.0):
                raise FloatingPointError(f"Negative reservoir: {minimum_nr}")
            if minimum_nr < 0 and step % round(config["save"]["diagnostic_every_ps"]/dt) == 0:
                warnings["WARN_NEGATIVE_RESERVOIR"] = warnings.get("WARN_NEGATIVE_RESERVOIR", 0)+1
                event(path, "WARN", "WARN_NEGATIVE_RESERVOIR",
                              {"minimum_um2_inv": minimum_nr, "step": step})
            nc = float(np.sum(abs(psi)**2)*grid.da)
            nr_total = float(np.sum(nr)*grid.da)
            loss = float(np.sum(absorber*abs(psi)**2)*grid.da) if absorber_enabled else 0.0
            source = ptotal-p["gammaC"]*nc-p["gammaR"]*nr_total-loss
            magnitude = ptotal+p["gammaC"]*nc+p["gammaR"]*nr_total+loss
            integral += dt*(previous_source+source)/2
            scale += dt*(previous_magnitude+magnitude)/2
            previous_source, previous_magnitude = source, magnitude
            time_ps = step*dt
            if step % intervals["observable_every_ps"] == 0:
                obs = basic_observables(psi, nr, pump, grid, config, absorber_arg)
                tracker.update(time_ps, psi, nr, obs, step)
                if on_observation is not None:
                    on_observation(time_ps, psi, nr, obs, tracker.last)
            if step % intervals["snapshot_every_ps"] == 0 or step == requested_steps:
                save_snapshot(path, step, time_ps, psi, nr, grid)
            if step % intervals["diagnostic_every_ps"] == 0 or step == requested_steps:
                obs = basic_observables(psi, nr, pump, grid, config, absorber_arg)
                extra = spatial_diagnostics(psi, nr, pump, grid, config, nref)
                absolute, relative = balance_residual(
                    initial_total, nc+nr_total, integral, scale)
                session_elapsed = time.perf_counter()-clock
                mean_step_ms = 1000*session_elapsed/max(step-first_step, 1)
                signal_floor = config["checks"]["signal_min_rel"]*nref
                growth_rate = None
                if (previous_diagnostic is not None
                        and min(nc, previous_diagnostic[1]) > signal_floor):
                    growth_rate = float(np.log(nc/previous_diagnostic[1]) /
                                        (time_ps-previous_diagnostic[0]))
                row = diagnostic_row(step, time_ps, obs, extra, absolute,
                                     relative, tracker.last, {
                                         "steps_total": requested_steps,
                                         "elapsed_s": old_wall+session_elapsed,
                                         "eta_remaining_s": (requested_steps-step)
                                         *mean_step_ms/1000,
                                         "step_ms": mean_step_ms,
                                         "NC_growth_rate_ps_inv": growth_rate,
                                         "phase_frequency_est_ps_inv": None,
                                         "nonfinite": 0})
                append_csv(path, row)
                log(path, "[STEP] "+", ".join(
                    f"{k}={row[k]}" for k in (
                        "t_ps", "step", "steps_total", "elapsed_s",
                        "eta_remaining_s", "step_ms", "NC", "NR",
                        "NC_growth_rate_ps_inv", "NC_max_um2_inv",
                        "nR_min_um2_inv", "nR_max_um2_inv", "gain_max_ps_inv",
                        "phase_frequency_est_ps_inv", "j_max_um_inv_ps_inv",
                        "edge_mass_fraction", "edge_peak_ratio",
                        "balance_relative_residual", "high_k_fraction",
                        "stationarity_status", "nonfinite")))
                previous_diagnostic = (time_ps, nc)
                if step % round(config["checks"]["stationarity_window_ps"]/dt) == 0:
                    log(path, "[STATIONARITY] "+", ".join(
                        f"{key}={tracker.last.get(key)}" for key in (
                            "NC_window_span", "NR_window_span",
                            "density_change_max", "reservoir_change_max",
                            "field_aligned_change_max", "field_reference_change_max",
                            "current_status", "dense_points", "probe_points",
                            "lag_pair_counts", "time_window_ps",
                            "stationarity_status")))
                for metric, limit, code in (
                    ("edge_mass_fraction", config["checks"]["edge_mass_warn"], "WARN_BOUNDARY"),
                    ("edge_peak_ratio", config["checks"]["edge_peak_warn"], "WARN_BOUNDARY"),
                    ("high_k_fraction", config["checks"]["high_k_warn"], "WARN_ALIASING"),
                    ("balance_relative_residual", config["checks"]["balance_warn_rel"], "WARN_BALANCE"),
                ):
                    value = row.get(metric)
                    if value is not None and value > limit and not (
                        code == "WARN_BOUNDARY" and config["pump"]["kind"] == "uniform"):
                        warnings[code] = warnings.get(code, 0)+1
                        event(path, "WARN", code, {"metric": metric, "value": value})
            session_wall = time.perf_counter()-clock
            wall = old_wall+session_wall
            checkpoint_due = step % intervals["checkpoint_every_ps"] == 0 or step == requested_steps
            explicit_stop = max_steps is not None and step-first_step >= max_steps
            if checkpoint_due or explicit_stop:
                save_checkpoint(path, step, psi, nr, integral, scale,
                                        initial_total, wall, tracker)
                if wall >= recommended_seconds and not runtime_notice:
                    event(path, "INFO", "RECOMMENDED_RUNTIME_EXCEEDED",
                          {"step": step, "elapsed_wall_s": wall,
                           "recommended_seconds": recommended_seconds})
                    runtime_notice = True
            if explicit_stop and step < requested_steps:
                event(path, "INFO", "EXPLICIT_STEP_LIMIT", {"step": step})
                break
        else:
            status = "RUN_COMPLETED"
    except KeyboardInterrupt:
        status = "RUN_INCOMPLETE"
        event(path, "INFO", "USER_INTERRUPT", {"attempted_step": step})
        with np.load(latest_checkpoint(path)) as checkpoint:
            psi, nr = checkpoint["psi"].copy(), checkpoint["nR"].copy()
            step = int(checkpoint["step"])
            integral = float(checkpoint["balance_integral"])
            scale = float(checkpoint["balance_scale"])
            if "tracker_start_ps" in checkpoint.files:
                tracker.restore_state(checkpoint)
            else:
                tracker.last = {"stationarity_status": "NOT_TESTED"}
        rewind_artifacts(path, step)
    except Exception as exc:
        status = "RUN_FAILED"
        failed_step = step
        event(path, "ERROR", "RUN_FAILED", {"step": failed_step, "message": str(exc)})
        log(path, f"RUN_FAILED step={failed_step} error={exc}")
        with np.load(latest_checkpoint(path)) as checkpoint:
            psi, nr = checkpoint["psi"].copy(), checkpoint["nR"].copy()
            step = int(checkpoint["step"])
            integral = float(checkpoint["balance_integral"])
            scale = float(checkpoint["balance_scale"])
    wall = old_wall+time.perf_counter()-clock
    obs = basic_observables(psi, nr, pump, grid, config, absorber_arg)
    extra = spatial_diagnostics(psi, nr, pump, grid, config, nref) if status != "RUN_FAILED" else {}
    absolute, relative = balance_residual(initial_total, obs["NC"]+obs["NR"],
                                          integral, scale)
    summary = {
        "run_status": status, "step": step, "time_ps": step*dt,
        "requested_time_ps": config["time"]["t_final_ps"], "elapsed_wall_s": wall,
        "runtime_recommendation_exceeded": wall > recommended_seconds,
        "mean_step_ms": 1000*wall/max(step, 1),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "config_sha256": config_hash(config),
        "observables": obs, "spatial_diagnostics": extra,
        "balance_absolute_residual": absolute, "balance_relative_residual": relative,
        "stationarity": tracker.last,
        "physics_validation_status": (
            "PHYSICS_PENDING_VALIDATION" if
            tracker.last["stationarity_status"] == "STATIONARY_MODE"
            else "NOT_STATIONARY_OR_UNDETERMINED"),
        "warnings": warning_counts(path),
        "latest_checkpoint": str(latest_checkpoint(path)),
        **diagnostic_extrema(path),
    }
    checks = {
        "completion": ("PASS" if status == "RUN_COMPLETED" else
                       "FAIL" if status == "RUN_FAILED" else "WARN", status),
        "balance": ("PASS" if relative <= config["checks"]["balance_warn_rel"] else
                    "FAIL" if relative > 10*config["checks"]["balance_warn_rel"]
                    else "WARN", relative),
        "edge": ("NOT_TESTED" if extra.get("edge_mass_fraction") is None else
                 "PASS" if extra["edge_mass_fraction"] <= config["checks"]["edge_mass_warn"] else "WARN",
                 extra.get("edge_mass_fraction")),
        "stationarity": ("PASS" if tracker.last["stationarity_status"] == "STATIONARY_MODE"
                         else "NOT_TESTED" if tracker.last["stationarity_status"] == "NOT_TESTED" else "WARN",
                         tracker.last["stationarity_status"]),
        "convergence": ("NOT_TESTED", "separate grid, time and domain study"),
        "physical validation": ("NOT_TESTED", "requires boundary and convergence checks"),
    }
    write_summary(path, summary, checks)
    summary["artifacts"] = artifact_statistics(path)
    write_summary(path, summary, checks)
    finalize_metadata(path, summary)
    log(path, f"{status} step={step} wall_s={wall:.3f}")
    return load_run(path)


def diagnostic_row(step, time_ps, obs, extra, absolute, relative, stationarity,
                   performance=None):
    keys = ("NC_window_span", "NR_window_span", "density_change_max",
            "reservoir_change_max", "field_aligned_change_max",
            "current_change_max", "density_reference_change_max",
            "reservoir_reference_change_max", "field_reference_change_max",
            "current_reference_change_max", "current_status",
            "dense_points", "probe_points", "final_log_NC_slope_per_ps")
    return {"step": step, "t_ps": time_ps, **(performance or {}), **obs, **extra,
            "balance_absolute_residual": absolute,
            "balance_relative_residual": relative,
            **{k: stationarity.get(k) for k in keys},
            "stationarity_status": stationarity.get("stationarity_status", "NOT_TESTED")}


def benchmark(config, steps=300, sizes=(256, 512)):
    """Measure actual CPU stepping and estimate a full run on this machine."""
    import copy
    import resource
    import time
    from model import initial_fields, make_pump

    reports = []
    for n in sizes:
        c = copy.deepcopy(config)
        c["space"]["Nx"] = c["space"]["Ny"] = int(n)
        check_config(c)
        grid = make_grid(c)
        pump = make_pump(c, grid)
        absorber = make_absorber(c, grid) if c["space"]["absorbing_layer"] else None
        psi, nr = initial_fields(c, grid, pump)
        factor = kinetic_half_factor(grid, c, c["time"]["dt_ps"])
        reference = number_reference(c, grid)
        tracker = StationarityTracker(c, grid, reference)
        obs = basic_observables(psi, nr, pump, grid, c, absorber)
        tracker.update(0.0, psi, nr, obs, 0)
        start = time.perf_counter()
        for step in range(1, steps+1):
            psi, nr = split_step(psi, nr, pump, c, c["time"]["dt_ps"], factor, absorber)
            if step % max(1, round(c["save"]["observable_every_ps"]/c["time"]["dt_ps"])) == 0:
                obs = basic_observables(psi, nr, pump, grid, c, absorber)
                tracker.update(step*c["time"]["dt_ps"], psi, nr, obs, step)
            if step % max(1, round(c["save"]["diagnostic_every_ps"]/c["time"]["dt_ps"])) == 0:
                spatial_diagnostics(psi, nr, pump, grid, c, reference)
        elapsed = time.perf_counter()-start
        total_steps = round(c["time"]["t_final_ps"]/c["time"]["dt_ps"])
        reports.append({
            "N": n, "steps_measured": steps, "elapsed_s": elapsed,
            "mean_step_ms": 1000*elapsed/steps,
            "projected_full_s": elapsed/steps*total_steps,
            "projection_excludes_io_and_full_window_comparisons": True,
            "within_20_minutes": elapsed/steps*total_steps <= 1200,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "estimated_state_mb": (n*n*(16+8))/2**20,
        })
    return reports


def run_validation_suite(config):
    """Fast measured residuals, independent of pytest's pass/fail summary."""
    import copy
    from model import (make_pump, solve_0d_reference, stationary_0d,
                       jacobian_0d, phase_frequency_0d, stuart_landau)

    c = copy.deepcopy(config)
    c["space"].update(Lx_um=16.0, Ly_um=16.0, Nx=32, Ny=32)
    c["checks"]["edge_width_um"] = 4.0
    c["run"]["validation_mode"] = "analytic_test"
    check_config(c)
    grid = make_grid(c)
    rng = np.random.default_rng(15)
    field = rng.normal(size=grid.shape)+1j*rng.normal(size=grid.shape)
    parseval = parseval_error(field, grid)

    free = copy.deepcopy(c)
    free["pump"]["kind"] = "off"
    free["model"].update(gC_mev_um2=0, gR_mev_um2=0, R_um2_per_ps=0,
                         gammaC_per_ps=0, gammaR_per_ps=0,
                         background_blueshift=False)
    x, y = np.meshgrid(grid.x, grid.y)
    kx, ky = grid.kx[2], grid.ky[3]
    wave = np.exp(1j*(kx*x+ky*y))
    dt = free["time"]["dt_ps"]
    moved, _ = split_step(wave, np.zeros(grid.shape), np.zeros(grid.shape), free, dt)
    expected = wave*np.exp(-1j*parameters(free)["K"]*(kx*kx+ky*ky)*dt/parameters(free)["hbar"])
    plane_wave = float(np.max(abs(moved-expected)))

    uniform = copy.deepcopy(c)
    uniform["pump"]["kind"] = "uniform"
    pump = make_pump(uniform, grid)
    psi0, nr0 = 0.1+0.03j, 3.0
    psi, nr = np.full(grid.shape, psi0), np.full(grid.shape, nr0)
    half = kinetic_half_factor(grid, uniform, dt)
    for _ in range(20):
        psi, nr = split_step(psi, nr, pump, uniform, dt, half)
    reference_psi, reference_nr = solve_0d_reference(
        uniform, float(pump[0, 0]), psi0, nr0, np.array([0.0, 20*dt]))
    uniform_error = float(max(np.max(abs(psi-reference_psi[-1])),
                              np.max(abs(nr-reference_nr[-1]))))
    nr0_field = np.full(grid.shape, 2.0)
    zero_field = np.zeros(grid.shape, complex)
    nr_field = nr0_field.copy()
    for _ in range(20):
        zero_field, nr_field = split_step(
            zero_field, nr_field, pump, uniform, dt, half)
    gamma_r = parameters(uniform)["gammaR"]
    exact_nr = pump/gamma_r+(nr0_field-pump/gamma_r)*np.exp(-gamma_r*20*dt)
    reservoir_error = float(np.max(abs(nr_field-exact_nr)))
    physical = parameters(c)
    p_stationary = 1.2*physical["Pth0D"]
    nc_star, nr_star = stationary_0d(p_stationary, c)["condensed"]
    def population_rhs(nc, nr):
        return np.array([(physical["R"]*nr-physical["gammaC"])*nc,
                         p_stationary-(physical["gammaR"]+physical["R"]*nc)*nr])
    stationary_error = float(np.max(abs(population_rhs(nc_star, nr_star))))
    epsilon = 1e-6
    numeric_jacobian = np.column_stack((
        (population_rhs(nc_star+epsilon, nr_star)
         -population_rhs(nc_star-epsilon, nr_star))/(2*epsilon),
        (population_rhs(nc_star, nr_star+epsilon)
         -population_rhs(nc_star, nr_star-epsilon))/(2*epsilon)))
    jacobian_error = float(np.max(abs(numeric_jacobian-jacobian_0d(nc_star, nr_star, c))))
    short_times = np.array([0.0, 0.1])
    stationary_psi, _ = solve_0d_reference(
        c, p_stationary, np.sqrt(nc_star), nr_star, short_times)
    frequency = phase_frequency_0d(nc_star, nr_star, p_stationary, c)
    phase_error = float(abs(np.angle(stationary_psi[-1]/stationary_psi[0])+frequency*0.1))
    near_pump = 1.01*physical["Pth0D"]
    sl = stuart_landau(near_pump, c)
    exact_nc = stationary_0d(near_pump, c)["condensed"][0]
    sl_relative_error = float(abs(sl["mu"]/sl["beta"]-exact_nc)/exact_nc)
    growth_below = float(max(np.linalg.eigvals(jacobian_0d(
        *stationary_0d(0.9*physical["Pth0D"], c)["empty"], c)).real))
    growth_above = float(max(np.linalg.eigvals(jacobian_0d(
        *stationary_0d(1.1*physical["Pth0D"], c)["empty"], c)).real))
    report = {
        "parseval_relative_error": parseval,
        "plane_wave_max_abs_error": plane_wave,
        "uniform_2d_vs_independent_0d_max_abs_error": uniform_error,
        "reservoir_exact_max_abs_error": reservoir_error,
        "stationary_0d_max_rhs_residual": stationary_error,
        "jacobian_0d_max_abs_error": jacobian_error,
        "phase_frequency_0d_abs_error_rad": phase_error,
        "stuart_landau_near_threshold_relative_error": sl_relative_error,
        "empty_branch_growth_below_per_ps": growth_below,
        "empty_branch_growth_above_per_ps": growth_above,
        "PASS": (parseval < 1e-12 and plane_wave < 1e-11
                 and uniform_error < 1e-7 and reservoir_error < 1e-8
                 and stationary_error < 1e-12 and jacobian_error < 1e-8
                 and phase_error < 1e-8 and sl_relative_error < 0.02
                 and growth_below < 0 < growth_above),
    }
    return report


def _linear_fit(times, log_numbers):
    x, y = np.asarray(times), np.asarray(log_numbers)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y-(slope*x+intercept)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y-np.mean(y))**2))
    r2 = 1-ss_res/ss_tot if ss_tot > 1e-24 else 0.0
    stderr = np.sqrt(ss_res/max(len(x)-2, 1)
                     /max(np.sum((x-np.mean(x))**2), 1e-30))/2
    return float(slope/2), float(r2), float(stderr)


def linear_growth_rate(config, *, duration_ps=None, seed=None,
                       initial_state=None, return_state=False):
    """Frozen-branch growth; optional exact continuation of a normalized field.

    duration_ps is the absolute end time. With return_state=True return
    (diagnostics, state); the state retains the field and logarithmic norms.
    """
    import copy
    from model import initial_fields, make_pump

    c = copy.deepcopy(config)
    if seed is not None:
        c["initial"]["random_seed"] = int(seed)
    c["initial"]["mode"] = "physical_startup"
    check_config(c)
    if c["model"]["gammaR_per_ps"] <= 0:
        raise ValueError("Frozen zero branch requires gammaR>0")
    grid = make_grid(c)
    pump = make_pump(c, grid)
    absorber = make_absorber(c, grid) if c["space"]["absorbing_layer"] else None
    signature = config_hash(c)
    if initial_state is None:
        psi, _ = initial_fields(c, grid, pump)
        if not np.any(psi):
            raise ValueError("Linear seed is identically zero")
        psi /= np.sqrt(np.sum(abs(psi)**2)*grid.da)
        times, logs = [0.0], [0.0]
    else:
        if initial_state["config_sha256"] != signature:
            raise ValueError("Linear continuation configuration changed")
        psi = np.array(initial_state["psi"], dtype=complex, copy=True)
        times = list(np.asarray(initial_state["times_ps"], dtype=float))
        logs = list(np.asarray(initial_state["log_numbers"], dtype=float))
        if psi.shape != grid.shape or len(times) != len(logs):
            raise ValueError("Invalid linear continuation state")
    dt = c["time"]["dt_ps"]
    window = c["threshold"]["linear_fit_window_ps"]
    duration = duration_ps or 3*window
    steps = round(duration/dt)
    sample_every = max(1, round(1.0/dt))
    half = kinetic_half_factor(grid, c, dt)
    start_step = round(times[-1]/dt)
    if initial_state is not None and (steps-start_step)*dt < window-1e-9:
        raise ValueError("Linear continuation needs at least one fit window")
    accumulated_log = float(logs[-1])
    previous_mode = psi.copy()
    mode_lags = sorted({max(1, round(window/5)), max(1, round(window/2)), round(window)})
    mode_sample_steps = {round((duration-lag)/dt): lag for lag in mode_lags}
    lag_modes = {mode_sample_steps[start_step]: psi.copy()} if start_step in mode_sample_steps else {}
    mode_sample_step = round((duration-window)/dt)
    for step in range(start_step+1, steps+1):
        psi = linear_step(psi, pump, c, half, absorber)
        if step % sample_every == 0:
            number = float(np.sum(abs(psi)**2)*grid.da)
            if not np.isfinite(number) or number <= 0:
                return {"status": "UNRESOLVED", "reason": "linear norm invalid"}
            accumulated_log += np.log(number)
            psi /= np.sqrt(number)
            times.append(step*dt)
            logs.append(accumulated_log)
            if step == mode_sample_step:
                previous_mode = psi.copy()
            if step in mode_sample_steps:
                lag_modes[mode_sample_steps[step]] = psi.copy()
    t, y = np.asarray(times), np.asarray(logs)
    early = (t >= duration-2*window-1e-9) & (t <= duration-window+1e-9)
    late = t >= duration-window-1e-9
    if early.sum() < 3 or late.sum() < 3:
        raise ValueError("Insufficient samples for two growth windows")
    sigma_early, r2_early, err_early = _linear_fit(t[early], y[early])
    sigma_late, r2_late, err_late = _linear_fit(t[late], y[late])
    criteria = c["threshold"]
    agreement = abs(sigma_late-sigma_early) <= max(
        criteria["sigma_agreement_rel"]*abs(sigma_late),
        criteria["sigma_agreement_abs_per_ps"])
    reliable = (r2_early >= criteria["linear_fit_R2_min"]
                and r2_late >= criteria["linear_fit_R2_min"]
                and agreement and abs(sigma_late) > 2*err_late)
    overlap = abs(np.vdot(previous_mode, psi))*grid.da
    overlaps = {str(lag): float(abs(np.vdot(field, psi))*grid.da)
                for lag, field in sorted(lag_modes.items())}
    next_field = linear_step(psi, pump, c, half, absorber)
    projection = np.vdot(psi, next_field)*grid.da
    step_residual = np.linalg.norm(next_field-projection*psi)/np.linalg.norm(next_field)
    result = {
        "status": ("UNSTABLE" if sigma_late > 0 else "STABLE") if reliable else "UNRESOLVED",
        "sigma_early_per_ps": sigma_early, "sigma_late_per_ps": sigma_late,
        "sigma_early_stderr": err_early, "sigma_late_stderr": err_late,
        "fit_R2_early": r2_early, "fit_R2_late": r2_late,
        "sigma_agreement": bool(agreement), "mode_overlap": float(overlap),
        "mode_status": "CONVERGED" if overlap >= 0.95 else "MODE_UNRESOLVED",
        "duration_ps": duration, "seed": c["initial"]["random_seed"],
        "P0": c["pump"]["P0"], "grid": grid.shape,
        "mode_overlaps": overlaps,
        "multi_lag_mode_status": ("CONVERGED" if len(overlaps) == len(mode_lags)
                                  and min(overlaps.values()) >= 0.95 else "MODE_UNRESOLVED"),
        "eigenstep_residual_per_ps": float(step_residual/dt),
    }
    if return_state:
        return result, {"psi": psi.copy(), "times_ps": t, "log_numbers": y,
                        "config_sha256": signature}
    return result


def find_threshold_2d(config, *, save=True, result_root=None,
                      duration_ps=None, max_bisections=20, resume_dir=None):
    """Bracket a linear threshold. Returns UNRESOLVED rather than guessing."""
    import copy
    import json
    from pathlib import Path

    check_config(config)
    if config["pump"]["kind"] not in ("gaussian", "ring"):
        raise ValueError("Threshold is defined for gaussian and ring pumps")
    if resume_dir is not None:
        if not save:
            raise ValueError("Threshold resume requires saved points")
        path = Path(resume_dir)
        if load_run(path)["config"] != config:
            raise ValueError("Threshold resume configuration changed")
        points_file = path/"threshold_points.json"
        points = json.loads(points_file.read_text(encoding="utf-8")) if points_file.exists() else []
        event(path, "INFO", "THRESHOLD_RESUMED", {"existing_points": len(points)})
    else:
        path = create_run(config, result_root or RESULTS_DIR) if save else None
        points = []
    def evaluate(power, seed=None):
        c = copy.deepcopy(config)
        c["pump"]["P0"] = float(power)
        actual_seed = int(c["initial"]["random_seed"] if seed is None else seed)
        existing = next((item for item in points
                         if np.isclose(item["P0"], power, rtol=0, atol=1e-12)
                         and item.get("seed") == actual_seed), None)
        if existing is not None:
            return existing
        base_duration = duration_ps or 3*c["threshold"]["linear_fit_window_ps"]
        attempts = []
        for multiplier in (1, 2, 4):
            result = linear_growth_rate(c, duration_ps=base_duration*multiplier, seed=seed)
            attempts.append({"duration_ps": result.get("duration_ps"),
                             "status": result["status"],
                             "sigma_late_per_ps": result.get("sigma_late_per_ps")})
            if result["status"] != "UNRESOLVED":
                break
        result["attempts"] = attempts
        points.append(result)
        if path:
            (path/"threshold_points.json").write_text(
                json.dumps(points, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            log(path, f"P0={power:.7g} sigma={result.get('sigma_late_per_ps')} "
                        f"status={result['status']}")
        return result
    for power in config["threshold"]["initial_P0_scan"]:
        evaluate(power)
    classified = sorted((r for r in points if r["status"] in ("STABLE", "UNSTABLE")),
                        key=lambda r: r["P0"])
    brackets = [(a, b) for a, b in zip(classified, classified[1:])
                if a["status"] == "STABLE" and b["status"] == "UNSTABLE"]
    if not brackets:
        outcome = {"status": "NOT_BRACKETED", "points": points}
    else:
        low, high = brackets[0]
        lo, hi = low["P0"], high["P0"]
        for _ in range(max_bisections):
            if (hi-lo)/((hi+lo)/2) <= config["threshold"]["target_bracket_rel"]:
                break
            middle = evaluate((lo+hi)/2)
            if middle["status"] == "UNRESOLVED":
                break
            if middle["status"] == "STABLE":
                lo = middle["P0"]
            else:
                hi = middle["P0"]
        width = (hi-lo)/((hi+lo)/2)
        status = "PRELIMINARY" if width <= config["threshold"]["target_bracket_rel"] else "THRESHOLD_UNRESOLVED"
        if status == "PRELIMINARY":
            second_seed = int(config["initial"]["random_seed"])+1
            for power in (lo, hi):
                repeated = evaluate(power, second_seed)
                expected = "STABLE" if power == lo else "UNSTABLE"
                if repeated["status"] != expected:
                    status = "THRESHOLD_UNRESOLVED"
        outcome = {"status": status, "P0_lo": lo, "P0_hi": hi,
                   "P0_crit_est": (lo+hi)/2, "bracket_width_rel": width,
                   "requires_grid_domain_confirmation": True, "points": points}
    if path:
        write_summary(path, {"run_status": "RUN_COMPLETED",
                                     "threshold": outcome,
                                     "config_sha256": config_hash(config)},
                              {"linear threshold": ("PASS" if outcome["status"] == "PRELIMINARY"
                                                    else "WARN", outcome["status"]),
                               "grid/domain confirmation": ("NOT_TESTED", "run separately")})
        outcome["run_dir"] = str(path)
    return outcome


def validate_threshold_2d(config, threshold, *, duration_ps=None):
    """Check the preliminary bracket on a finer grid and larger domain."""
    import copy
    import json
    from pathlib import Path

    if threshold.get("status") not in ("PRELIMINARY", "GRID_DOMAIN_CONFIRMED"):
        return {"status": "NOT_TESTED", "reason": "no reliable preliminary bracket"}
    lo, hi = threshold["P0_lo"], threshold["P0_hi"]
    base = config["space"]
    cases = {
        "refined_grid": {"Nx": 2*base["Nx"], "Ny": 2*base["Ny"]},
        "expanded_domain": {
            "Lx_um": 1.5*base["Lx_um"], "Ly_um": 1.5*base["Ly_um"],
            "Nx": int(round(1.5*base["Nx"])), "Ny": int(round(1.5*base["Ny"])),
        },
    }
    results = {}
    for name, changes in cases.items():
        c = copy.deepcopy(config)
        c["space"].update(changes)
        points = []
        for power in (lo, hi):
            c["pump"]["P0"] = power
            base_duration = duration_ps or 3*c["threshold"]["linear_fit_window_ps"]
            for multiplier in (1, 2, 4):
                estimate = linear_growth_rate(c, duration_ps=base_duration*multiplier)
                if estimate["status"] != "UNRESOLVED":
                    break
            points.append(estimate)
        results[name] = points
    confirmed = all(
        points[0]["status"] == "STABLE" and points[1]["status"] == "UNSTABLE"
        for points in results.values())
    report = {"status": "GRID_DOMAIN_CONFIRMED" if confirmed else "THRESHOLD_UNRESOLVED",
              "P0_lo": lo, "P0_hi": hi, "validation_points": results}
    if threshold.get("run_dir"):
        path = Path(threshold["run_dir"])
        (path/"threshold_validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        saved = load_run(path)
        summary = saved["summary"] or {}
        summary["threshold_validation"] = report
        write_summary(path, summary, {
            "linear threshold": ("PASS", threshold["status"]),
            "grid/domain confirmation": (
                "PASS" if confirmed else "WARN", report["status"]),
            "nonlinear confirmation": ("NOT_TESTED", "run separately"),
        })
    return report


def confirm_threshold_nonlinear(config, threshold, *, result_root=None):
    """Separate physical-startup runs safely below and above a linear bracket."""
    import copy

    if "P0_lo" not in threshold or "P0_hi" not in threshold:
        return {"status": "NOT_TESTED", "reason": "no bracket"}
    runs = {}
    for label, power in (("below", 0.8*threshold["P0_lo"]),
                         ("above", 1.2*threshold["P0_hi"])):
        c = copy.deepcopy(config)
        c["pump"]["P0"] = power
        c["initial"]["mode"] = "physical_startup"
        runs[label] = simulate_2d(c, result_root=result_root)
    return assess_threshold_nonlinear(
        config, threshold, runs["below"], runs["above"])


def assess_threshold_nonlinear(config, threshold, below_run, above_run):
    """Reassess saved nonlinear evidence after extending either run."""
    import csv
    import json
    from pathlib import Path

    runs = {"below": load_run(below_run) if not isinstance(below_run, dict)
            else below_run,
            "above": load_run(above_run) if not isinstance(above_run, dict)
            else above_run}
    for label, expected in (("below", 0.8*threshold["P0_lo"]),
                            ("above", 1.2*threshold["P0_hi"])):
        actual = runs[label]["config"]
        if (not np.isclose(actual["pump"]["P0"], expected)
                or actual["initial"]["mode"] != "physical_startup"):
            raise ValueError(f"{label} run does not match the threshold protocol")
    lower = runs["below"]["summary"]
    upper = runs["above"]["summary"]
    above_edge = upper["spatial_diagnostics"].get("edge_mass_fraction")
    above_edge_peak = upper["spatial_diagnostics"].get("edge_peak_ratio")
    above_high_k = upper["spatial_diagnostics"].get("high_k_fraction")
    # A changing condensate can still be long lived. Check that its signal
    # remains resolved throughout the final stationarity window; the linear
    # bracket, rather than a brightness cutoff, defines the threshold.
    grid = make_grid(runs["above"]["config"])
    with np.load(runs["above"]["snapshots"][0]) as initial:
        initial_nc = float(np.sum(abs(initial["psi"])**2)*grid.da)
    nref = number_reference(config, grid, initial_nc)
    window = float(config["checks"]["stationarity_window_ps"])
    with (runs["above"]["path"]/"diagnostics.csv").open(newline="") as handle:
        history = [(float(row["t_ps"]), float(row["NC"]))
                   for row in csv.DictReader(handle)
                   if float(row["t_ps"]) >= upper["time_ps"]-window-1e-8
                   and float(row["t_ps"]) <= upper["time_ps"]+1e-8]
    persistence = {
        "window_ps": window, "samples": len(history),
        "initial_NC": initial_nc,
        "minimum_NC": min((nc for _, nc in history), default=None),
        "signal_floor_NC": config["checks"]["signal_min_rel"]*nref,
        "status": "NOT_TESTED",
    }
    if (len(history) >= 2
            and history[-1][0]-history[0][0] >= window-1e-8
            and upper["stationarity"].get("dense_points", 0)
            >= round(window/config["save"]["stationarity_reference_every_ps"])+1):
        persistence["status"] = (
            "PERSISTENT" if persistence["minimum_NC"]
            > max(initial_nc, persistence["signal_floor_NC"])
            else "DECAYING_OR_UNRESOLVED")
    quality_pass = (
        above_edge is not None
        and above_edge_peak is not None
        and above_high_k is not None
        and above_edge <= config["checks"]["edge_mass_warn"]
        and above_edge_peak <= config["checks"]["edge_peak_warn"]
        and above_high_k <= config["checks"]["high_k_warn"]
        and upper["balance_relative_residual"] <= config["checks"]["balance_warn_rel"])
    status = "SUPPORTED" if (
        lower["run_status"] == "RUN_COMPLETED"
        and upper["run_status"] == "RUN_COMPLETED"
        and lower["stationarity"]["stationarity_status"] == "DECAYED"
        and (upper["stationarity"]["stationarity_status"] in
             ("STATIONARY_MODE", "TIME_PERIODIC")
             or persistence["status"] == "PERSISTENT")
        and quality_pass
    ) else "UNRESOLVED"
    report = {"status": status, "below": str(runs["below"]["path"]),
              "above": str(runs["above"]["path"]),
              "P0_below": 0.8*threshold["P0_lo"],
              "P0_above": 1.2*threshold["P0_hi"],
              "below_stationarity": lower["stationarity"]["stationarity_status"],
              "above_stationarity": upper["stationarity"]["stationarity_status"],
              "below_final_time_ps": lower["time_ps"],
              "above_final_time_ps": upper["time_ps"],
              "above_NC": upper["observables"]["NC"],
              "above_edge_mass_fraction": above_edge,
              "above_edge_peak_ratio": above_edge_peak,
              "above_high_k_fraction": above_high_k,
              "above_persistence": persistence,
              "above_balance_relative_residual": upper["balance_relative_residual"]}
    if threshold.get("run_dir"):
        path = Path(threshold["run_dir"])
        (path/"nonlinear_confirmation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        saved = load_run(path)
        summary = saved["summary"] or {}
        summary["nonlinear_confirmation"] = report
        write_summary(path, summary, {
            "linear threshold": ("PASS", threshold["status"]),
            "grid/domain confirmation": (
                "PASS" if summary.get("threshold_validation", {}).get("status")
                == "GRID_DOMAIN_CONFIRMED" else "NOT_TESTED", "see threshold_validation.json"),
            "nonlinear confirmation": ("PASS" if status == "SUPPORTED" else "WARN", status),
        })
    return report


def compare_runs(reference, candidate):
    """Compare completed runs after aligning space and global field phase."""

    a = load_run(reference) if not isinstance(reference, dict) else reference
    b = load_run(candidate) if not isinstance(candidate, dict) else candidate
    if a["summary"]["run_status"] != "RUN_COMPLETED" or b["summary"]["run_status"] != "RUN_COMPLETED":
        return {"status": "NOT_TESTED", "reason": "incomplete run"}
    ca, cb = a["config"], b["config"]
    if any(ca[key] != cb[key] for key in ("model", "pump", "initial")):
        raise ValueError("Runs use different physics or initial conditions")
    if not np.isclose(ca["time"]["t_final_ps"], cb["time"]["t_final_ps"]):
        raise ValueError("Runs end at different physical times")
    ga, gb = make_grid(ca), make_grid(cb)
    with np.load(a["snapshots"][-1]) as da, np.load(b["snapshots"][-1]) as db:
        pa, ra = da["psi"], da["nR"]
        pb, rb = db["psi"], db["nR"]
    ja = current(pa, ga, ca)
    jb = current(pb, gb, cb)
    ja_integral = float(np.sum(np.hypot(*ja))*ga.da)
    jb_integral = float(np.sum(np.hypot(*jb))*gb.da)
    if np.isclose(ga.lx, gb.lx) and np.isclose(ga.ly, gb.ly):
        if gb.shape[0] % ga.shape[0] or gb.shape[1] % ga.shape[1]:
            raise ValueError("Fine grid is not integer refinement")
        sy, sx = gb.shape[0]//ga.shape[0], gb.shape[1]//ga.shape[1]
        pb, rb = pb[::sy, ::sx], rb[::sy, ::sx]
    elif np.isclose(ga.dx, gb.dx) and np.isclose(ga.dy, gb.dy):
        oy = (gb.shape[0]-ga.shape[0])//2
        ox = (gb.shape[1]-ga.shape[1])//2
        if oy < 0 or ox < 0:
            raise ValueError("Candidate domain must enclose reference")
        pb, rb = pb[oy:oy+ga.shape[0], ox:ox+ga.shape[1]], rb[oy:oy+ga.shape[0], ox:ox+ga.shape[1]]
    else:
        raise ValueError("Unsupported grid/domain comparison")
    obs_a, obs_b = a["summary"]["observables"], b["summary"]["observables"]
    core_radius = min(ga.lx, ga.ly)/4
    x0, y0 = ca["pump"]["center_um"]
    core = ((ga.x[None, :]-x0)**2+(ga.y[:, None]-y0)**2
            <= core_radius**2)
    metrics = {
        "NC_relative": abs(obs_a["NC"]-obs_b["NC"])/max(abs(obs_b["NC"]), 1e-30),
        "NR_relative": abs(obs_a["NR"]-obs_b["NR"])/max(abs(obs_b["NR"]), 1e-30),
        "density_map_relative": relative_change(abs(pa)**2, abs(pb)**2, ga),
        "reservoir_map_relative": relative_change(ra, rb, ga),
        "field_phase_aligned_relative": phase_aligned_change(pa, pb, ga),
        "current_magnitude_integral_relative": abs(ja_integral-jb_integral)/max(jb_integral, 1e-30),
        "core_radius_um": core_radius,
        "core_density_map_relative": relative_change(
            abs(pa)**2*core, abs(pb)**2*core, ga),
        "core_reservoir_map_relative": relative_change(ra*core, rb*core, ga),
        "core_field_phase_aligned_relative": phase_aligned_change(
            pa*core, pb*core, ga),
    }
    sa, sb = a["summary"]["spatial_diagnostics"], b["summary"]["spatial_diagnostics"]
    for key in ("r_rms_um", "j_max_um_inv_ps_inv", "l_z_over_hbar_NC"):
        if sa.get(key) is not None and sb.get(key) is not None:
            if key == "l_z_over_hbar_NC" and max(abs(sa[key]), abs(sb[key])) < 1e-8:
                continue
            metrics[key+"_relative"] = abs(sa[key]-sb[key])/max(abs(sb[key]), 1e-12)
    if sa.get("r_rms_um") is not None and sb.get("r_rms_um") is not None:
        metrics["mean_r2_relative"] = abs(sa["r_rms_um"]**2-sb["r_rms_um"]**2)/max(sb["r_rms_um"]**2, 1e-30)
    # Same late-time number history window in every comparison, when available.
    def history(run):
        with (run["path"]/"diagnostics.csv").open(newline="") as stream:
            return {float(row["t_ps"]): float(row["NC"]) for row in csv.DictReader(stream)}
    ah, bh = history(a), history(b)
    end = ca["time"]["t_final_ps"]
    times = np.array(sorted(t for t in ah.keys() & bh.keys() if end-100 <= t <= end))
    if len(times) >= 2 and np.isclose(times[-1]-times[0], 100):
        av, bv = np.array([ah[t] for t in times]), np.array([bh[t] for t in times])
        mean = np.trapezoid(bv, times)/100
        metrics["last_100ps_NC_relative_rmse"] = float(np.sqrt(
            np.trapezoid((av-bv)**2, times)/100)/max(abs(mean), 1e-30))
        metrics["history_window_ps"] = [float(times[0]), float(times[-1])]
    return {"status": "COMPARED", "reference": str(a["path"]),
            "candidate": str(b["path"]), **metrics}


def convergence_study(config, *, grids=(256, 512, 1024),
                      time_steps=(0.05, 0.025, 0.0125),
                      domains=((128.0, 512), (192.0, 768)),
                      result_root=None, resume_dir=None, reuse_cases=None,
                      domain_time_step=None, continue_incomplete=False,
                      max_workers=1):
    """Grid/time/domain suite with checked reuse and checkpoint continuation.

    Independent cases may run concurrently; each has its own output directory.
    domain_time_step is shared by both domain cases, independently of the grid
    comparison step. No physical parameters or initial conditions are changed.
    """
    import copy
    import json
    from pathlib import Path

    check_config(config)
    # The continuous seed in model.initial_fields is sampled at the same
    # physical coordinates on every grid and domain.
    if resume_dir is None:
        root = create_run(config, result_root or RESULTS_DIR)
        previous = {}
    else:
        root = Path(resume_dir)
        if load_run(root)["config"] != config:
            raise ValueError("Convergence suite configuration changed")
        manifest = root/"convergence_cases.json"
        previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    if reuse_cases:
        for name, path in reuse_cases.items():
            if name in previous and Path(previous[name]) != Path(path):
                raise ValueError(f"Conflicting saved case {name}")
            previous[name] = str(path)
    lock = Lock()
    cases = {}
    def run_case(name, changes):
        c = copy.deepcopy(config)
        for section, values in changes.items():
            c[section].update(values)
        check_config(c)
        if name in previous:
            result = load_run(previous[name])
            if result["config"] != c:
                raise ValueError(f"Convergence case {name} configuration changed")
            if result["summary"] is None or result["summary"]["run_status"] == "RUN_INCOMPLETE":
                result = simulate_2d(c, resume_dir=result["path"])
        else:
            def initialized(path):
                # Persist immediately: even termination before the first summary
                # must leave the suite able to resume the initialized checkpoint.
                with lock:
                    previous[name] = str(path)
                    _write_json(root/"convergence_cases.json", previous)
            result = simulate_2d(c, result_root=root/"cases", on_initialized=initialized)
        while result["summary"] is None or result["summary"]["run_status"] == "RUN_INCOMPLETE":
            if result["summary"] is not None and not continue_incomplete:
                break
            before = (result["summary"] or {}).get("step", -1)
            result = simulate_2d(c, resume_dir=result["path"])
            if result["summary"]["step"] <= before:
                raise RuntimeError(f"No progress while resuming {name}")
        with lock:
            previous[name] = str(result["path"])
            _write_json(root/"convergence_cases.json", previous)
            cases[name] = result
            log(root, f"{name}: {result['summary']['run_status']} {result['path']}")
            print(f"{name}: {result['summary']['run_status']} {result['path']}", flush=True)
        return result
    base_n = grids[1]
    tasks = {f"grid_{n}": {"space": {"Nx": n, "Ny": n}} for n in grids}
    tasks.update({f"dt_{dt:g}": {"space": {"Nx": base_n, "Ny": base_n},
                               "time": {"dt_ps": dt}} for dt in time_steps})
    tasks.update({f"domain_{length:g}": {
        "space": {"Lx_um": length, "Ly_um": length, "Nx": n, "Ny": n},
        "time": {"dt_ps": domain_time_step or config["time"]["dt_ps"]}}
        for length, n in domains})
    # Deduplicate identical grid/time/domain configurations, including aliases.
    unique, aliases, hashes = {}, {}, {}
    for name, changes in tasks.items():
        c = copy.deepcopy(config)
        for section, values in changes.items():
            c[section].update(values)
        digest = config_hash(c)
        if digest in hashes:
            aliases[name] = hashes[digest]
        else:
            hashes[digest] = name
            unique[name] = changes
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_case, name, changes) for name, changes in unique.items()]
        for future in futures:
            future.result()
    for name, source in aliases.items():
        cases[name] = cases[source]
    comparisons = []
    for a, b in zip(grids, grids[1:]):
        comparisons.append({"comparison": f"grid_{a}_vs_{b}",
                            **compare_runs(cases[f"grid_{a}"], cases[f"grid_{b}"])})
    for a, b in zip(time_steps, time_steps[1:]):
        comparisons.append({"comparison": f"dt_{a:g}_vs_{b:g}",
                            **compare_runs(cases[f"dt_{a:g}"], cases[f"dt_{b:g}"])})
    if len(domains) >= 2:
        comparisons.append({"comparison": "domain_expansion",
                            **compare_runs(cases[f"domain_{domains[0][0]:g}"],
                                           cases[f"domain_{domains[1][0]:g}"])})
    complete = all(r["status"] == "COMPARED" for r in comparisons)
    temporal = [r for r in comparisons if r["comparison"].startswith("dt_")
                and r["status"] == "COMPARED"]
    time_order = {}
    if len(temporal) >= 2 and len(time_steps) >= 3:
        ratio = time_steps[0]/time_steps[1]
        for key in ("NC_relative", "NR_relative", "density_map_relative"):
            coarse_difference = temporal[0].get(key, 0)
            fine_difference = temporal[1].get(key, 0)
            time_order[key] = (float(np.log(coarse_difference/fine_difference)/np.log(ratio))
                               if coarse_difference > 0 and fine_difference > 0
                               and ratio > 1 else None)
    integral_keys = ("NC_relative", "NR_relative", "mean_r2_relative",
                     "current_magnitude_integral_relative")
    violations = [{"comparison": r["comparison"], "metric": key, "value": r[key]}
                  for r in comparisons if r["status"] == "COMPARED"
                  for key in integral_keys if key in r and r[key] >= 0.005]
    for r in comparisons:
        if r["status"] == "COMPARED":
            for key, tolerance in (("core_density_map_relative", 0.01),
                                   ("last_100ps_NC_relative_rmse", 0.005)):
                if r.get(key, 0) >= tolerance:
                    violations.append({"comparison": r["comparison"], "metric": key,
                                       "value": r[key]})
    outcome = {"cases": {k: str(cases[k]["path"]) for k in tasks},
               "comparisons": comparisons,
               "time_ps": float(config["time"]["t_final_ps"]),
               "reference_stationarity_status": (
                   cases[f"grid_{base_n}"]["summary"]["stationarity"]
                   ["stationarity_status"]),
               "status": "COMPLETED" if complete else "NOT_TESTED",
               "assessment": ("NOT_TESTED" if not complete else
                              "WARN" if violations else "PASS"),
               "integral_tolerance_rel": 0.005,
               "protocol": {"grids": list(grids), "time_steps_ps": list(time_steps),
                            "domains": list(domains),
                            "domain_dt_ps": domain_time_step or config["time"]["dt_ps"]},
               "stationarity_by_case": {k: v["summary"]["stationarity"]["stationarity_status"]
                                        for k, v in cases.items()},
               "balance_by_time_step": {str(dt): cases[f"dt_{dt:g}"]["summary"]
                                        ["balance_relative_residual"] for dt in time_steps},
               "observed_time_order": time_order,
               "violations": violations}
    outcome["final_state_assessment"] = (
        "NOT_TESTED" if not complete or any(
            value == "NOT_TESTED" for value in outcome["stationarity_by_case"].values())
        else outcome["assessment"] if all(
            value == "STATIONARY_MODE" for value in outcome["stationarity_by_case"].values())
        else "WARN")
    (root/"convergence.json").write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_summary(root, {"run_status": "RUN_COMPLETED" if outcome["status"] == "COMPLETED"
                                 else "RUN_INCOMPLETE", "convergence": outcome},
                          {"grid/time/domain": (outcome["assessment"],
                                                outcome["status"])})
    outcome["run_dir"] = str(root)
    return outcome


def run_practical_checks():
    """Three reproducible checks; call from the project directory. Reuse/resume saved runs."""
    import copy, hashlib, json, subprocess, sys
    from pathlib import Path
    from plotting import write_report_summary

    checks_file = Path("results/analysis_data/practical_checks.json")
    report = json.loads(checks_file.read_text()) if checks_file.exists() else {"cases": {}}
    current_sources = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                       for name in ("model.py", "numerics.py", "plotting.py")}
    # Preserve the source fingerprints recorded when the saved simulations started.
    report.setdefault("source_sha256", current_sources)
    report["verification_source_sha256"] = current_sources
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           capture_output=True, text=True)
    print(tests.stdout, flush=True)
    report["tests"] = {"returncode": tests.returncode, "output": tests.stdout + tests.stderr}
    checks_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if tests.returncode:
        raise RuntimeError("Tests failed; production checks were not started")

    domain = json.loads(Path("results/analysis_data/domain_comparisons_large.json").read_text())
    cases = [
        ("Кольцо: сетка 192→384", domain["Кольцо"]["reference"], "grid", 192),
        ("Пятно: шаг 0.025→0.0125 пс", domain["Пятно"]["reference"], "dt", 0.0125),
        ("Кольцо: шаг 0.05→0.025 пс", domain["Кольцо"]["reference"], "dt", 0.025),
    ]
    for label, reference_path, change, value in cases:
        base = load_run(reference_path)
        config = copy.deepcopy(base["config"])
        if change == "grid":
            config["space"].update(Nx=value, Ny=value)
        else:
            config["time"]["dt_ps"] = value
        config["save"]["snapshot_every_ps"] = 100.0
        old = report["cases"].get(label, {})
        if old and old["config_sha256"] != config_hash(config):
            raise ValueError("Stored check configuration differs; inspect it before rerunning")
        if old:
            run = load_run(old["run_path"])
        else:
            run = None
        print("START", label, old.get("run_path", "new run"), flush=True)
        if run is None:
            # simulator initializes and persists fields before propagation.
            # Record its path at initialization, so an interrupted cell can resume.
            def record_initialized(path):
                report["cases"][label] = {"run_path": str(path), "reference_path": reference_path,
                                          "config_sha256": config_hash(config)}
                checks_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            run = simulate_2d(config, on_initialized=record_initialized)
        elif run["summary"] is None or run["summary"]["run_status"] == "RUN_INCOMPLETE":
            run = simulate_2d(config, resume_dir=run["path"])
        elif run["summary"]["run_status"] != "RUN_COMPLETED":
            raise RuntimeError("Failed run requires inspection")
        item = report["cases"][label]
        item["run_path"] = str(run["path"])
        item["run_status"] = run["summary"]["run_status"]
        checks_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if item["run_status"] != "RUN_COMPLETED":
            print("RUN_INCOMPLETE: rerun this cell to continue from checkpoint", flush=True)
            break
        coarse, fine = (run, base) if change == "grid" else (base, run)
        comparison = compare_runs(coarse, fine)
        # Independently compare the saved late-time number histories, not just one field.
        import csv
        def number_history(saved_run):
            with (saved_run["path"]/"diagnostics.csv").open(newline="") as stream:
                return {float(row["t_ps"]): float(row["NC"]) for row in csv.DictReader(stream)}
        old_numbers, new_numbers = number_history(base), number_history(run)
        end = run["summary"]["time_ps"]
        sample_times = np.array(sorted(t for t in old_numbers.keys() & new_numbers.keys()
                                      if end-100 <= t <= end))
        if len(sample_times) < 2 or not np.isclose(sample_times[-1]-sample_times[0], 100):
            raise ValueError("Incomplete common 100-ps history")
        reference_numbers = np.array([old_numbers[t] for t in sample_times])
        checked_numbers = np.array([new_numbers[t] for t in sample_times])
        mean_number = np.trapezoid(reference_numbers, sample_times)/100
        comparison["last_100ps_NC_relative_rmse"] = float(np.sqrt(
            np.trapezoid((checked_numbers-reference_numbers)**2, sample_times)/100)/mean_number)
        comparison["history_window_ps"] = [float(sample_times[0]), float(sample_times[-1])]
        item.update(comparison=comparison, time_ps=run["summary"]["time_ps"],
                    wall_seconds=run["summary"]["elapsed_wall_s"],
                    stationarity=run["summary"]["stationarity"],
                    grid=config["space"], dt_ps=config["time"]["dt_ps"])
        quality = all(r["summary"]["balance_relative_residual"] <= r["config"]["checks"]["balance_warn_rel"]
                      and r["summary"]["spatial_diagnostics"]["edge_mass_fraction"] <= r["config"]["checks"]["edge_mass_warn"]
                      and r["summary"]["spatial_diagnostics"]["edge_peak_ratio"] <= r["config"]["checks"]["edge_peak_warn"]
                      and r["summary"]["spatial_diagnostics"]["high_k_fraction"] <= r["config"]["checks"]["high_k_warn"]
                      for r in (base, run))
        item["assessment"] = "PASS" if (quality and comparison["NC_relative"] < 0.005
                                            and comparison["core_density_map_relative"] < 0.01
                                            and comparison["last_100ps_NC_relative_rmse"] < 0.005) else "WARN"
        checks_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("DONE", label, item["assessment"], "dNC=", comparison["NC_relative"],
              "density=", comparison["core_density_map_relative"],
              run["summary"]["stationarity"]["stationarity_status"], flush=True)

    sources = json.loads(Path("results/report_figures/figure_sources.json").read_text())
    write_report_summary("docs/diagnostic_summary.md",
        gaussian_run=sources["gaussian_run"], ring_run=sources["ring_run"],
        gaussian_threshold=sources["gaussian_threshold"], ring_threshold=sources["ring_threshold"],
        convergence=sources["convergence"], domain_comparisons=domain,
        final_state_checks="results/analysis_data/final_state_checks.json",
        validation_report="results/analysis_data/validation_report.json",
        practical_checks=report)
    print(checks_file, flush=True)
    return report


def run_final_convergence_checks(*, max_workers=2):
    """Finish the stationary spot's full v3 suite, reusing compatible saved runs."""
    import copy
    from plotting import write_report_summary

    tests = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True)
    print(tests.stdout, flush=True)
    if tests.returncode:
        raise RuntimeError("Tests failed; final convergence suite was not started")
    test_report = {"returncode": tests.returncode, "output": tests.stdout+tests.stderr}
    verified_sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                        for name in ("model.py", "numerics.py", "plotting.py", "test_project.py")}
    sources_file = ROOT/"results/report_figures/figure_sources.json"
    sources = json.loads(sources_file.read_text())
    main = load_run(sources["gaussian_run"])
    config = copy.deepcopy(main["config"])
    config["time"]["dt_ps"] = 0.05
    checks_file = ROOT/"results/analysis_data/final_convergence_checks.json"
    previous = json.loads(checks_file.read_text()) if checks_file.exists() else {}
    final = json.loads((ROOT/"results/analysis_data/final_state_checks.json").read_text())
    domain = json.loads((ROOT/"results/analysis_data/domain_comparisons_large.json").read_text())
    reuse = {"grid_512": final["шаг времени 0.05→0.025 пс"]["reference"],
             "dt_0.025": str(main["path"]),
             "domain_192": domain["Пятно"]["reference"],
             "domain_256": str(main["path"])}
    resume = previous.get("run_dir")
    if resume is None:
        resume = str(create_run(config))
        _write_json(checks_file, {"run_dir": resume, "status": "RUN_INCOMPLETE"})
    outcome = convergence_study(config, grids=(256, 512, 1024),
        time_steps=(0.05, 0.025, 0.0125), domains=((192.0, 384), (256.0, 512)),
        domain_time_step=0.025, reuse_cases=reuse, resume_dir=resume,
        continue_incomplete=True, max_workers=max_workers)
    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str) and value.startswith(str(ROOT)+os.sep):
            return str(Path(value).relative_to(ROOT))
        return value
    outcome = portable(outcome)
    outcome.update(tests=test_report, verification_source_sha256=verified_sources)
    _write_json(checks_file, outcome)
    study_path = str(Path(outcome["run_dir"])/"convergence.json")
    _write_json(study_path, outcome)
    manifest_file = Path(outcome["run_dir"])/"convergence_cases.json"
    _write_json(manifest_file, portable(json.loads(manifest_file.read_text())))
    suite = load_run(outcome["run_dir"])
    suite_summary = portable(suite["summary"])
    suite_summary["convergence"] = outcome
    _write_json(suite["path"]/"summary.json", suite_summary)
    for path in set(outcome["cases"].values()):
        run = load_run(path)
        summary = run["summary"]
        spatial, limits = summary["spatial_diagnostics"], run["config"]["checks"]
        quality = (summary["balance_relative_residual"] <= limits["balance_warn_rel"]
                   and all(spatial.get(metric) is not None and spatial[metric] <= limits[limit]
                           for metric, limit in (("edge_mass_fraction", "edge_mass_warn"),
                                                 ("edge_peak_ratio", "edge_peak_warn"),
                                                 ("high_k_fraction", "high_k_warn"))))
        validation = outcome["final_state_assessment"]
        physics = "PASS" if validation == "PASS" and quality else "WARN"
        summary["convergence_validation"] = {"status": validation, "source": study_path}
        if summary.get("latest_checkpoint"):
            summary["latest_checkpoint"] = portable(summary["latest_checkpoint"])
        summary["physics_validation_status"] = (
            "PHYSICS_VALIDATED" if physics == "PASS" else "PHYSICS_VALIDATION_WARN")
        _write_json(run["path"]/"summary.json", summary)
        event(run["path"], "INFO", "CONVERGENCE_VALIDATED", summary["convergence_validation"])
    sources["convergence"] = str(Path(outcome["run_dir"])/"convergence.json")
    _write_json(sources_file, sources)
    write_report_summary(ROOT/"docs/diagnostic_summary.md",
        gaussian_run=sources["gaussian_run"], ring_run=sources["ring_run"],
        gaussian_threshold=sources["gaussian_threshold"], ring_threshold=sources["ring_threshold"],
        convergence=outcome, domain_comparisons=domain,
        final_state_checks=final,
        validation_report=ROOT/"results/analysis_data/validation_report.json",
        practical_checks=ROOT/"results/analysis_data/practical_checks.json")
    return outcome


def density_periodicity(times, densities, nc, nr, *, minimum_cycles=2):
    """Period candidate from density; validate recurrence on original core pixels.

    Counts and unnormalised density must return too: detrending is used only
    to find candidates, never to make a transient pass the recurrence checks.
    No TIME_PERIODIC classification of the complex field is assigned here.
    """
    from scipy.signal import detrend, fftconvolve, find_peaks
    from scipy.optimize import minimize_scalar

    t = np.asarray(times, float)
    n = np.asarray(densities)
    nc, nr = np.asarray(nc, float), np.asarray(nr, float)
    if len(t) < 20 or not np.allclose(np.diff(t), np.diff(t)[0]):
        return {"status": "INSUFFICIENT_DATA", "period_ps": None}
    dt = float(t[1]-t[0])
    mean_density=n.mean(axis=0)
    variation=float(np.max(np.linalg.norm((n-mean_density).reshape(len(t),-1),axis=1)) /
                    max(np.linalg.norm(mean_density),1e-30))
    if variation < 0.005 and np.ptp(nc)/max(nc.mean(),1e-30) < 0.005 and np.ptp(nr)/max(nr.mean(),1e-30) < 0.005:
        return {"status":"NO_SIGNIFICANT_DENSITY_OSCILLATION","period_ps":None,
                "density_variation_rel":variation}
    # The full-resolution density remains available for the final check.
    features = n[:, ::4, ::4].reshape(len(t), -1).astype(float)
    features /= max(np.linalg.norm(features.mean(axis=0)), 1e-30)
    features = np.column_stack((features, nc/max(nc.mean(), 1e-30),
                                nr/max(nr.mean(), 1e-30)))
    fluct = detrend(features, axis=0, type="linear")
    energy = np.sum(fluct**2, axis=1)
    variance = float(np.mean(energy))
    if variance < 1e-10:
        return {"status": "NO_RESOLVED_OSCILLATION", "period_ps": None}
    ac = fftconvolve(fluct, fluct[::-1], axes=0)[len(t)-1:].sum(axis=1)
    cumulative = np.r_[0, np.cumsum(energy)]
    lag = np.arange(len(t))
    denominator = np.sqrt((cumulative[-1]-cumulative[lag])*cumulative[len(t)-lag])
    ac /= np.maximum(denominator, 1e-30)
    upper = int((len(t)-1)/minimum_cycles)
    peaks, _ = find_peaks(ac[:upper+1], height=0.6, prominence=0.1)
    peaks = [int(k) for k in peaks if k >= 4]
    if not peaks:
        return {"status": "IRREGULAR_OR_TOO_SLOW", "period_ps": None,
                "longest_testable_period_ps": (t[-1]-t[0])/minimum_cycles}
    k = peaks[0]
    def pairs(period, data):
        shift = period/dt
        indices = np.arange(int(np.ceil(shift)), len(t))
        earlier = indices-shift
        left = np.floor(earlier).astype(int)
        alpha = earlier-left
        alpha = alpha.reshape((-1,)+(1,)*(data.ndim-1))
        return data[indices], (1-alpha)*data[left]+alpha*data[np.minimum(left+1,len(t)-1)]
    def objective(period):
        a, b = pairs(period, features)
        return float(np.mean((a-b)**2))
    fit = minimize_scalar(objective, bounds=(max(3*dt,(k-1.5)*dt),
                                               min(upper*dt,(k+1.5)*dt)), method="bounded")
    period = float(fit.x)
    a, b = pairs(period, n)
    density_error = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))
    totals = n.sum(axis=(1,2))
    shape = n/np.maximum(totals[:,None,None],1e-30)
    a, b = pairs(period, shape)
    shape_error = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))
    errors = {}
    for key, values in (("NC", nc), ("NR", nr)):
        a, b = pairs(period, values)
        errors[key+"_recurrence_rel"] = float(np.sqrt(np.mean((a-b)**2))/max(abs(b.mean()),1e-30))
    amplitudes=[]
    middle=len(t)//2
    for a,b in ((0,middle+1),(middle,len(t))):
        local=t[a:b]-np.mean(t[a:b])
        design=np.column_stack((np.ones(len(local)),local,
                                np.cos(2*np.pi*t[a:b]/period),np.sin(2*np.pi*t[a:b]/period)))
        coefficients=np.linalg.lstsq(design,features[a:b],rcond=None)[0]
        amplitudes.append(float(np.linalg.norm(coefficients[2:4])))
    amplitude_ratio=amplitudes[1]/max(amplitudes[0],1e-30)
    passed = (density_error < 0.01 and shape_error < 0.01
              and all(v < 0.005 for v in errors.values()) and 0.8 <= amplitude_ratio <= 1.25)
    return {"status": "DENSITY_PERIODIC_SUPPORTED" if passed else "PERIODIC_COMPONENT_WITH_DRIFT",
            "period_ps": period, "cycles": float((t[-1]-t[0])/period),
            "autocorrelation_peak": float(ac[k]),
            "amplitude_ratio_last_first":amplitude_ratio,
            "density_recurrence_rel": density_error,
            "shape_recurrence_rel": shape_error, **errors}


def density_maxima(density, x, y, *, relative_cutoff=0.2, max_peaks=12):
    """Separated local maxima, with a plateau represented by one pixel."""
    from scipy.ndimage import maximum_filter, label
    n = np.asarray(density)
    if n.max() <= 0:
        return []
    mask = (n == maximum_filter(n, size=5, mode="constant")) & (n >= relative_cutoff*n.max())
    mask[[0,-1],:] = False
    mask[:,[0,-1]] = False
    labels, count = label(mask)
    peaks = []
    for index in range(1,count+1):
        iy, ix = np.where(labels == index)
        best = int(np.argmax(n[iy,ix]))
        peaks.append([float(x[ix[best]]),float(y[iy[best]]),float(n[iy[best],ix[best]])])
    return sorted(peaks,key=lambda p:p[2],reverse=True)[:max_peaks]


def run_ring_dynamics_experiment():
    """Continue the report ring without changing its archived 1000-ps run.

    A single rolling NPZ contains dense populations, full-resolution central
    densities for the latest 500 ps, and peak tracks. JSON records every
    100-ps window; propagation resumes from its own checkpoints.
    """
    import copy
    from scipy.optimize import linear_sum_assignment
    from plotting import plot_ring_dynamics

    data_dir = ROOT/"results/analysis_data"
    manifest = data_dir/"ring_dynamics.json"
    archive = data_dir/"ring_dynamics.npz"
    report = json.loads(manifest.read_text()) if manifest.exists() else {}
    source = load_run(report.get("source_run", json.loads(
        (ROOT/"results/report_figures/figure_sources.json").read_text())["ring_run"]))
    if not report:
        config = copy.deepcopy(source["config"])
        config["time"]["t_final_ps"] = 1500.0
        path = create_run(config, root=Path("results/runs"))
        checkpoint = latest_checkpoint(source["path"])
        shutil.copy2(checkpoint, path/"checkpoints"/checkpoint.name)
        shutil.copy2(source["snapshots"][-1],path/"snapshots"/source["snapshots"][-1].name)
        shutil.copy2(source["path"]/"diagnostics.csv",path/"diagnostics.csv")
        pending = copy.deepcopy(source["summary"])
        pending.update(run_status="RUN_INCOMPLETE",requested_time_ps=1500.0,
                       config_sha256=config_hash(config))
        _write_json(path/"summary.json",pending)
        source_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report = {"source_run": str(source["path"]),"run_dir":str(path),
                  "source_checkpoint_sha256":source_digest,"source_time_ps":1000.0,
                  "source_wall_s":source["summary"]["elapsed_wall_s"],
                  "core_half_width_um":40.0,"density_dtype":"float32",
                  "dense_interval_ps":1.0,"windows":[],"status":"RUN_INCOMPLETE"}
        _write_json(manifest,report)
        event(path,"INFO","CONTINUED_FROM_ARCHIVED_RUN",report)
    path = Path(report["run_dir"])
    saved = load_run(path)
    config = saved["config"]
    grid = make_grid(config)
    ix = np.flatnonzero(abs(grid.x) <= 40)
    iy = np.flatnonzero(abs(grid.y) <= 40)
    x, y = grid.x[ix], grid.y[iy]
    if archive.exists():
        with np.load(archive) as z:
            history = z["history"].tolist()
            frame_times = z["density_times_ps"].tolist()
            frames = list(z["density_core_um2_inv"])
            tracks = json.loads(str(z["peaks_json"]))
    else:
        with (source["path"]/"diagnostics.csv").open(newline="") as stream:
            old_rows = list(csv.DictReader(stream))
        history = [[float(row[key]) for key in ("t_ps","NC","NR")]
                   for row in old_rows if 500 <= float(row["t_ps"]) <= 1000]
        frame_times,frames,tracks = [],[],[]
        for snapshot in source["snapshots"]:
            with np.load(snapshot) as z:
                t = float(z["time_ps"])
                if 500 <= t <= 1000:
                    frame_times.append(t)
                    frames.append(np.asarray(abs(z["psi"][np.ix_(iy,ix)])**2,dtype=np.float32))
        report["initial_windows"] = [{"window_ps":[float(row["t_ps"])-100,float(row["t_ps"])],
            "NC_span_rel":float(row["NC_window_span"]),"NR_span_rel":float(row["NR_window_span"]),
            "density_reference_change_rel":float(row["density_reference_change_max"]),
            "stationarity_status":row["stationarity_status"],
            "periodicity":{"status":"UNDERSAMPLED_20PS_FRAMES","period_ps":None}}
            for row in old_rows if float(row["t_ps"]) in (600.,700.,800.,900.,1000.)]
    next_id = max((int(p["id"]) for row in tracks for p in row["peaks"]),default=-1)+1
    def persist():
        np.savez_compressed(archive,history=np.asarray(history),density_times_ps=frame_times,
            density_core_um2_inv=np.asarray(frames,dtype=np.float32),
            x_um=x,y_um=y,peaks_json=np.array(json.dumps(tracks)))
        _write_json(manifest,report)
    def observe(t,psi,nr,obs,stationarity):
        nonlocal next_id,history,frame_times,frames,tracks
        # Discard analysis after the restored checkpoint before replaying it.
        history = [row for row in history if row[0] < t]
        keep = [i for i,old_t in enumerate(frame_times) if old_t < t and old_t >= t-500]
        frame_times = [frame_times[i] for i in keep]
        frames = [frames[i] for i in keep]
        tracks = [row for row in tracks if row["time_ps"] < t]
        n = np.asarray(abs(psi[np.ix_(iy,ix)])**2,dtype=np.float32)
        history.append([float(t),float(obs["NC"]),float(obs["NR"])])
        frame_times.append(float(t));frames.append(n)
        peaks = density_maxima(n,x,y)
        current_peaks = [{"x_um":p[0],"y_um":p[1],"density":p[2],"id":None} for p in peaks]
        if tracks and t-tracks[-1]["time_ps"] <= 1.01 and current_peaks:
            old = tracks[-1]["peaks"]
            if old:
                distances = np.linalg.norm(np.array([[p['x_um'],p['y_um']] for p in old])[:,None,:]
                    -np.array([[p['x_um'],p['y_um']] for p in current_peaks])[None,:,:],axis=2)
                a,b=linear_sum_assignment(distances)
                for i,j in zip(a,b):
                    if distances[i,j] <= 2.0:
                        current_peaks[j]["id"] = old[i]["id"]
        for peak in current_peaks:
            if peak["id"] is None:
                peak["id"] = next_id;next_id += 1
        tracks.append({"time_ps":float(t),"peaks":current_peaks})
        if t > 1000 and np.isclose(t%100,0):
            selection=np.flatnonzero(np.asarray(frame_times) >= t-100)
            nt=np.asarray([frames[i] for i in selection])
            ht=np.asarray(history);ht=ht[(ht[:,0]>=t-100)&(ht[:,0]<=t)]
            shapes=nt/np.maximum(nt.sum(axis=(1,2))[:,None,None],1e-30)
            shape_change=float(max(np.linalg.norm(frame-shapes[0])/max(np.linalg.norm(shapes[0]),1e-30)
                                   for frame in shapes))
            item={"window_ps":[float(t-100),float(t)],
                  "NC_span_rel":float(np.ptp(ht[:,1])/ht[:,1].mean()),
                  "NR_span_rel":float(np.ptp(ht[:,2])/ht[:,2].mean()),
                  "N_total_span_rel":float(np.ptp(ht[:,1]+ht[:,2])/(ht[:,1]+ht[:,2]).mean()),
                  "NC_endpoint_change_rel":float((ht[-1,1]-ht[0,1])/ht[0,1]),
                  "NR_endpoint_change_rel":float((ht[-1,2]-ht[0,2])/ht[0,2]),
                  "shape_change_rel":shape_change,
                  "shape_frame_20ps_change_max":float(max(np.linalg.norm(shapes[i]-shapes[i-20])/
                          max(np.linalg.norm(shapes[i-20]),1e-30) for i in range(20,len(shapes)))),
                  "core_mass_fraction":float(n.sum()*grid.da/obs["NC"]),
                  "stationarity":stationarity,
                  "periodicity":density_periodicity(np.asarray(frame_times)[selection],nt,ht[:,1],ht[:,2])}
            report["windows"]=[w for w in report["windows"] if w["window_ps"][1]<t]
            report["windows"].append(item)
            persist()
            print("RING WINDOW",item["window_ps"],"dNC",item["NC_span_rel"],
                  "dNR",item["NR_span_rel"],"dNtotal",item["N_total_span_rel"],"shape",shape_change,
                  "shape20ps",item["shape_frame_20ps_change_max"],
                  "periodicity",item["periodicity"],flush=True)
    for target in (1500.0,2000.0):
        saved=load_run(path)
        if saved["summary"]["time_ps"] > target:
            continue
        if saved["summary"]["time_ps"] < target:
            config=copy.deepcopy(saved["config"]);config["time"]["t_final_ps"]=target
            report.update(status='RUN_INCOMPLETE',requested_time_ps=target)
            persist()
            saved=simulate_2d(config,resume_dir=path,on_observation=observe)
            if saved["summary"]["run_status"] != "RUN_COMPLETED":
                report["status"]=saved["summary"]["run_status"];persist()
                return report
        ht=np.asarray(history)
        aligned=np.array([np.flatnonzero(np.isclose(ht[:,0],t))[0] for t in frame_times])
        global_period=density_periodicity(frame_times,np.asarray(frames),ht[aligned,1],ht[aligned,2],minimum_cycles=3)
        report.update(status="RUN_COMPLETED",time_ps=saved["summary"]["time_ps"],
            final_stationarity=saved["summary"]["stationarity"],last_500ps_periodicity=global_period,
            extension_wall_s=saved["summary"]["elapsed_wall_s"]-report["source_wall_s"])
        persist()
        plot_ring_dynamics(report,archive)
        if target == 1500:
            last=report["windows"][-2:]
            stationary=len(last)==2 and all(w['stationarity'].get('stationarity_status')=='STATIONARY_MODE' for w in last)
            # Recurrence alone is insufficient: neighbouring halves must agree.
            supported=global_period['status']=='DENSITY_PERIODIC_SUPPORTED'
            if supported:
                mid=len(frame_times)//2
                halves=[density_periodicity(frame_times[a:b],np.asarray(frames)[a:b],
                         ht[aligned[a:b],1],ht[aligned[a:b],2])
                        for a,b in ((0,mid+1),(mid,len(frame_times)))]
                supported=all(h['status']=='DENSITY_PERIODIC_SUPPORTED' for h in halves)
                if supported:
                    supported=abs(halves[0]['period_ps']-halves[1]['period_ps'])/global_period['period_ps'] < 0.05
                report['period_half_checks']=halves;persist()
            if stationary or supported:
                report['stop_reason']='STATIONARY' if stationary else 'DENSITY_PERIODIC_SUPPORTED'
                persist();break
            print('Ring remains unsettled or periodicity unconfirmed: extending to 2000 ps',flush=True)
    report.setdefault('stop_reason','REQUESTED_2000PS_LIMIT')
    # Reanalyse the rolling archive as well when loading a completed experiment.
    ht=np.asarray(history)
    ft=np.asarray(frame_times)
    for window in report['windows']:
        start,end=window['window_ps']
        selection=(ft>=start)&(ft<=end)
        if ft[0] <= start and ft[-1] >= end:
            counts=ht[(ht[:,0]>=start)&(ht[:,0]<=end)]
            window['periodicity']=density_periodicity(ft[selection],np.asarray(frames)[selection],
                                                     counts[:,1],counts[:,2])
    stationary=[w for w in report['windows']
                if w['stationarity'].get('stationarity_status')=='STATIONARY_MODE']
    report['first_verified_stationary_window_ps']=stationary[0]['window_ps'] if stationary else None
    source_checkpoint=latest_checkpoint(source['path'])
    digest=hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    if digest != report['source_checkpoint_sha256']:
        raise ValueError('The archived source checkpoint has changed')
    report.setdefault('verification',{})['source_checkpoint_unchanged']=True
    plot_ring_dynamics(report,archive)
    persist()
    return report


def run_final_ring_checks(*, max_workers=2):
    """Two independent refinements of the saved stationary ring at 2000 ps.

    Reuse completed cases; resume interrupted cases from their own checkpoints.
    Starting fields are generated from the same continuous random Fourier sum.
    """
    import copy
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from model import initial_fields

    sources_file = ROOT/'results/report_figures/figure_sources.json'
    sources = json.loads(sources_file.read_text())
    reference = load_run(ROOT/sources['ring_run'])
    base = reference['config']
    if (base['pump']['kind'] != 'ring' or base['space']['Nx'] != 512
            or base['space']['Ny'] != 512 or base['space']['Lx_um'] != 256
            or base['space']['Ly_um'] != 256 or base['time']['dt_ps'] != 0.05
            or base['time']['t_final_ps'] != 2000
            or reference['summary']['run_status'] != 'RUN_COMPLETED'):
        raise ValueError('Expected completed 512², L=256, dt=0.05, t=2000 ring reference')
    report_file = ROOT/'results/analysis_data/final_ring_checks.json'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    ref_snapshot = reference['snapshots'][-1]
    protocol = {'reference': str(reference['path'].relative_to(ROOT)),
        'reference_snapshot_sha256': hashlib.sha256(ref_snapshot.read_bytes()).hexdigest(),
        'reference_config_sha256': config_hash(base), 'time_ps': 2000.,
        'domain_um': [256., 256.], 'seed': base['initial']['random_seed'],
        'cases': {'spatial': {'grid': [256, 256], 'dt_ps': .05},
                  'temporal': {'grid': [512, 512], 'dt_ps': .025}},
        'independent_start_time_ps': 0., 'snapshot_every_ps': 100.,
        'limits': {'NC_relative': .005, 'NR_relative': .005,
                   'density_map_relative': .01, 'Lz_relative': .01,
                   'q_expected': -3, 'stationarity': 'STATIONARY_MODE'},
        'scope': 'Two grid/time refinements at 2000 ps; same domain and seed. No physical perturbation or seed ensemble.'}
    previous = json.loads(report_file.read_text()) if report_file.exists() else {}
    if previous and previous.get('protocol') != protocol:
        raise ValueError('Final ring check protocol changed')
    report = previous or {'status': 'RUNNING', 'protocol': protocol, 'cases': {}, 'comparisons': {}}
    lock = Lock()
    _write_json(report_file, report)
    fine_grid = make_grid(base)
    seed_fine, reservoir_fine = initial_fields(base, fine_grid, make_pump(base, fine_grid))
    origin = json.loads((ROOT/'results/analysis_data/ring_dynamics.json').read_text())['source_run']
    with np.load(ROOT/origin/'snapshots/step_00000000.npz') as initial:
        if not np.array_equal(seed_fine, initial['psi']) or not np.array_equal(reservoir_fine, initial['nR']):
            raise ValueError('Current continuous seed differs from the actual original reference seed')
    def calculate(name, settings):
        c = copy.deepcopy(base)
        c['space']['Nx'], c['space']['Ny'] = settings['grid']
        c['time']['dt_ps'] = settings['dt_ps']
        c['save']['snapshot_every_ps'] = 100.
        grid = make_grid(c)
        seed, reservoir = initial_fields(c, grid, make_pump(c, grid))
        stride = 512//c['space']['Nx']
        if not np.array_equal(seed, seed_fine[::stride, ::stride]) or np.any(reservoir):
            raise ValueError('Continuous seed sampling or empty reservoir changed')
        existing = report['cases'].get(name)
        if existing:
            run = load_run(ROOT/existing)
            if run['config'] != c:
                raise ValueError('Saved final ring case configuration changed')
            if run['summary'] and run['summary']['run_status'] == 'RUN_COMPLETED':
                return run
            if run['summary'] and run['summary']['run_status'] == 'RUN_FAILED':
                raise RuntimeError(f'Previously failed final ring case: {run["path"]}')
            return simulate_2d(c, resume_dir=run['path'])
        def initialized(path):
            with lock:
                report['cases'][name] = str(path.relative_to(ROOT))
                report.setdefault('initial_seed_checks', {})[name] = {
                    'same_physical_coefficients': True, 'max_abs_seed_difference': 0.,
                    'matches_original_reference_snapshot': True, 'reservoir_empty': True}
                _write_json(report_file, report)
        return simulate_2d(c, on_initialized=initialized)
    runs = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(calculate, name, settings): name for name, settings in protocol['cases'].items()}
        for future in as_completed(futures):
            name = futures[future]
            runs[name] = future.result()
            print(f'FINAL_RING {name}: {runs[name]["summary"]["run_status"]}', flush=True)
    with np.load(ref_snapshot) as values:
        ref_psi, ref_nr = values['psi'].copy(), values['nR'].copy()
        if float(values['time_ps']) != 2000:
            raise ValueError('Reference field is not at 2000 ps')
    ref_vortex = vortex_diagnostics(ref_psi, fine_grid, base)
    if ref_vortex['q'] != -3:
        raise ValueError('Reference vortex is not q=-3')
    report['reference_vortex'] = ref_vortex
    report['reference_stationarity'] = reference['summary']['stationarity']
    for name, run in runs.items():
        summary = run['summary']
        if summary['run_status'] != 'RUN_COMPLETED':
            report['comparisons'][name] = {'assessment': 'NOT_TESTED', 'reason': summary['run_status']}
            continue
        grid = make_grid(run['config'])
        with np.load(run['snapshots'][-1]) as values:
            psi, nr = values['psi'].copy(), values['nR'].copy()
            if float(values['time_ps']) != 2000:
                raise ValueError('Candidate field is not at 2000 ps')
        stride = 512//grid.shape[0]
        vortex = vortex_diagnostics(psi, grid, run['config'])
        obs, ref_obs = summary['observables'], reference['summary']['observables']
        nc_error = abs(obs['NC']-ref_obs['NC'])/ref_obs['NC']
        nr_error = abs(obs['NR']-ref_obs['NR'])/ref_obs['NR']
        density_error = relative_change(abs(psi)**2, abs(ref_psi[::stride, ::stride])**2, grid)
        reservoir_error = relative_change(nr, ref_nr[::stride, ::stride], grid)
        lz_error = abs(vortex['Lz_over_hbar_NC']-ref_vortex['Lz_over_hbar_NC'])/abs(ref_vortex['Lz_over_hbar_NC'])
        valid = {c['radius_um']: c['q'] for c in vortex['contours'] if c['status']=='VALID'}
        ref_valid = {c['radius_um']: c['q'] for c in ref_vortex['contours'] if c['status']=='VALID'}
        common = sorted(valid.keys() & ref_valid.keys())
        charge_ok = len(common)>=2 and vortex['q']==-3 and all(valid[r]==ref_valid[r]==-3 for r in common)
        checks = {'populations': nc_error<.005 and nr_error<.005,
                  'density_profile': density_error<.01,
                  'charge': charge_ok,
                  'angular_momentum': lz_error<.01 and abs(vortex['Lz_over_hbar_NC']+3)/3<.01,
                  'stationarity': summary['stationarity']['stationarity_status']=='STATIONARY_MODE'
                                  and reference['summary']['stationarity']['stationarity_status']=='STATIONARY_MODE'}
        assessment = 'PASS' if all(checks.values()) else 'FAIL'
        report['comparisons'][name] = {'assessment': assessment, 'checks': checks,
            'reference': protocol['reference'], 'candidate': str(run['path'].relative_to(ROOT)),
            'NC_relative': float(nc_error), 'NR_relative': float(nr_error),
            'density_map_relative': float(density_error), 'reservoir_map_relative': float(reservoir_error),
            'Lz_relative': float(lz_error), 'common_valid_contours_um': common,
            'vortex': vortex, 'stationarity': summary['stationarity'],
            'elapsed_wall_s': summary['elapsed_wall_s'],
            'balance_relative_residual': summary['balance_relative_residual'],
            'spatial_diagnostics': summary['spatial_diagnostics']}
    report['status'] = 'COMPLETED' if all(r['summary']['run_status']=='RUN_COMPLETED' for r in runs.values()) else 'RUN_INCOMPLETE'
    report['assessment'] = ('PASS' if report['status']=='COMPLETED' and
                            all(c['assessment']=='PASS' for c in report['comparisons'].values()) else
                            'FAIL' if report['status']=='COMPLETED' else 'NOT_TESTED')
    report['additional_calculation_wall_s'] = sum(r['summary']['elapsed_wall_s'] for r in runs.values())
    if hashlib.sha256(ref_snapshot.read_bytes()).hexdigest() != protocol['reference_snapshot_sha256']:
        raise RuntimeError('Reference field changed during validation')
    report['reference_field_unchanged'] = True
    _write_json(report_file, report)
    sources['ring_final_checks'] = str(report_file.relative_to(ROOT))
    _write_json(sources_file, sources)
    for run in [reference, *runs.values()]:
        summary = run['summary']
        summary['final_ring_validation'] = {'status': report['assessment'], 'source': sources['ring_final_checks'],
                                           'scope': 'grid/time refinement at 2000 ps, same seed/domain'}
        _write_json(run['path']/'summary.json', summary)
        event(run['path'], 'INFO', 'FINAL_RING_VALIDATION', summary['final_ring_validation'])
    from plotting import update_report_text
    update_report_text(output_dir=ROOT/'results/report_figures', notes_path=ROOT/'docs/methods.md')
    print(f'FINAL_RING assessment={report["assessment"]}', flush=True)
    return report
