"""FFT split-step propagation and frozen-reservoir linearization."""
from __future__ import annotations

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


def simulate_2d(config, *, result_root=None, resume_dir=None, max_steps=None):
    """Run and persist the full coupled model; max_steps explicitly stops a preview."""
    import json
    from pathlib import Path
    import resource
    import time

    from model import initial_fields, make_pump
    from diagnostics import (StationarityTracker, basic_observables,
                             balance_residual, number_reference,
                             spatial_diagnostics)
    import storage

    storage.check_config(config)
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
        path = storage.create_run(config, root=result_root or storage.RESULTS_DIR)
        psi, nr = initial_fields(config, grid, pump)
        first_step, integral, scale, old_wall = 0, 0.0, 0.0, 0.0
        nc0 = np.sum(abs(psi)**2)*grid.da
        nr0 = np.sum(nr)*grid.da
        initial_total = float(nc0+nr0)
        storage.save_snapshot(path, 0, 0.0, psi, nr, grid)
        storage.save_checkpoint(path, 0, psi, nr, 0.0, 0.0, initial_total, 0.0)
        storage.event(path, "INFO", "RUN_STARTED", {"requested_steps": requested_steps})
        storage.log(path, f"RUN_ID={path.name} config_sha256={storage.config_hash(config)}")
        storage.log(path, f"MODEL K_meV_um2={p['K']:.9g} hbar_meV_ps={p['hbar']:.9g} "
                    f"eta_ps={p['eta']} Pth0D_um2_inv_ps_inv={p['Pth0D']} "
                    f"nRth_um2_inv={p['nRth']} DOI=10.1126/sciadv.adj1589")
        storage.log(path, f"GRID Nx={grid.x.size} Ny={grid.y.size} "
                    f"dx_um={grid.dx} dy_um={grid.dy} dt_ps={dt} "
                    f"steps={requested_steps} pump={config['pump']['kind']} "
                    f"P0={config['pump']['P0']} "
                    f"numerical_absorber={absorber_enabled} "
                    f"gamma_abs_max_per_ps={float(np.max(absorber))}")
    else:
        path = Path(resume_dir)
        old = storage.load_run(path)
        if (old["summary"] and old["summary"]["run_status"] == "RUN_COMPLETED"
                and config["time"]["t_final_ps"] <= old["config"]["time"]["t_final_ps"]):
            raise ValueError("A completed run can only be extended to a later final time")
        if not storage.compatible_for_resume(old["config"], config):
            raise ValueError("Incompatible resume configuration")
        with np.load(storage.latest_checkpoint(path)) as checkpoint:
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
        storage.rewind_artifacts(path, first_step)
        if config != old["config"]:
            import yaml
            (path/"resolved_config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
            meta_file = path/"run_metadata.json"
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta["config_sha256"] = storage.config_hash(config)
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            storage.event(path, "INFO", "FINAL_TIME_EXTENDED",
                          {"old_ps": old["config"]["time"]["t_final_ps"],
                           "new_ps": config["time"]["t_final_ps"]})
        storage.event(path, "INFO", "RUN_RESUMED", {"from_step": first_step})

    nref = number_reference(config, grid, initial_total)
    tracker = StationarityTracker(config, grid, nref)
    obs = basic_observables(psi, nr, pump, grid, config, absorber_arg)
    if tracker_state is None:
        tracker.update(first_step*dt, psi, nr, obs, first_step)
    else:
        tracker.restore_state(tracker_state)
    ptotal = obs["P_total_per_ps"]
    previous_source = (ptotal-p["gammaC"]*obs["NC"]-p["gammaR"]*obs["NR"]
                       -obs["absorber_loss_per_ps"])
    previous_magnitude = (ptotal+p["gammaC"]*obs["NC"]+p["gammaR"]*obs["NR"]
                          +obs["absorber_loss_per_ps"])
    intervals = {key: round(config["save"][key]/dt) for key in config["save"]}
    budget = config["checks"]["max_walltime_per_run_minutes"]*60
    warnings = {}
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
                storage.event(path, "WARN", "WARN_NEGATIVE_RESERVOIR",
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
            if step % intervals["snapshot_every_ps"] == 0 or step == requested_steps:
                storage.save_snapshot(path, step, time_ps, psi, nr, grid)
            if step % intervals["diagnostic_every_ps"] == 0 or step == requested_steps:
                obs = basic_observables(psi, nr, pump, grid, config, absorber_arg)
                extra = spatial_diagnostics(psi, nr, pump, grid, config, nref)
                absolute, relative = balance_residual(
                    initial_total, nc+nr_total, integral, scale)
                row = diagnostic_row(step, time_ps, obs, extra, absolute,
                                     relative, tracker.last)
                storage.append_csv(path, row)
                storage.log(path, "[STEP] "+", ".join(
                    f"{k}={row[k]}" for k in ("t_ps", "NC", "NR",
                    "balance_relative_residual", "edge_mass_fraction",
                    "high_k_fraction", "stationarity_status")))
                if step % round(config["checks"]["stationarity_window_ps"]/dt) == 0:
                    storage.log(path, "[STATIONARITY] "+", ".join(
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
                        storage.event(path, "WARN", code, {"metric": metric, "value": value})
            session_wall = time.perf_counter()-clock
            wall = old_wall+session_wall
            checkpoint_due = step % intervals["checkpoint_every_ps"] == 0 or step == requested_steps
            explicit_stop = max_steps is not None and step-first_step >= max_steps
            if checkpoint_due or explicit_stop:
                storage.save_checkpoint(path, step, psi, nr, integral, scale,
                                        initial_total, wall, tracker)
                if step < requested_steps and session_wall >= budget:
                    storage.event(path, "INFO", "WALLTIME_LIMIT", {"step": step})
                    break
            if explicit_stop and step < requested_steps:
                storage.event(path, "INFO", "EXPLICIT_STEP_LIMIT", {"step": step})
                break
        else:
            status = "RUN_COMPLETED"
    except KeyboardInterrupt:
        status = "RUN_INCOMPLETE"
        storage.event(path, "INFO", "USER_INTERRUPT", {"attempted_step": step})
        with np.load(storage.latest_checkpoint(path)) as checkpoint:
            psi, nr = checkpoint["psi"].copy(), checkpoint["nR"].copy()
            step = int(checkpoint["step"])
            integral = float(checkpoint["balance_integral"])
            scale = float(checkpoint["balance_scale"])
            if "tracker_start_ps" in checkpoint.files:
                tracker.restore_state(checkpoint)
            else:
                tracker.last = {"stationarity_status": "NOT_TESTED"}
        storage.rewind_artifacts(path, step)
    except Exception as exc:
        status = "RUN_FAILED"
        failed_step = step
        storage.event(path, "ERROR", "RUN_FAILED", {"step": failed_step, "message": str(exc)})
        storage.log(path, f"RUN_FAILED step={failed_step} error={exc}")
        with np.load(storage.latest_checkpoint(path)) as checkpoint:
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
        "mean_step_ms": 1000*wall/max(step, 1),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "config_sha256": storage.config_hash(config),
        "observables": obs, "spatial_diagnostics": extra,
        "balance_absolute_residual": absolute, "balance_relative_residual": relative,
        "stationarity": tracker.last,
        "physics_validation_status": (
            "PHYSICS_PENDING_VALIDATION" if
            tracker.last["stationarity_status"] == "STATIONARY_MODE"
            else "NOT_STATIONARY_OR_UNDETERMINED"),
        "warnings": storage.warning_counts(path),
        "latest_checkpoint": str(storage.latest_checkpoint(path)),
        **storage.diagnostic_extrema(path),
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
    storage.write_summary(path, summary, checks)
    summary["artifacts"] = storage.artifact_statistics(path)
    storage.write_summary(path, summary, checks)
    storage.finalize_metadata(path, summary)
    storage.log(path, f"{status} step={step} wall_s={wall:.3f}")
    return storage.load_run(path)


def diagnostic_row(step, time_ps, obs, extra, absolute, relative, stationarity):
    keys = ("NC_window_span", "NR_window_span", "density_change_max",
            "reservoir_change_max", "field_aligned_change_max",
            "current_change_max", "density_reference_change_max",
            "reservoir_reference_change_max", "field_reference_change_max",
            "current_reference_change_max", "current_status",
            "dense_points", "probe_points", "final_log_NC_slope_per_ps")
    return {"step": step, "t_ps": time_ps, **obs, **extra,
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
    from diagnostics import (StationarityTracker, basic_observables,
                             spatial_diagnostics, number_reference)
    import storage

    reports = []
    for n in sizes:
        c = copy.deepcopy(config)
        c["space"]["Nx"] = c["space"]["Ny"] = int(n)
        storage.check_config(c)
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
    from diagnostics import parseval_error
    import storage

    c = copy.deepcopy(config)
    c["space"].update(Lx_um=16.0, Ly_um=16.0, Nx=32, Ny=32)
    c["checks"]["edge_width_um"] = 4.0
    c["run"]["validation_mode"] = "analytic_test"
    storage.check_config(c)
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


def linear_growth_rate(config, *, duration_ps=None, seed=None):
    """Growth of a perturbation around frozen nR=P/gammaR, never a startup run."""
    import copy
    from model import initial_fields, make_pump
    import storage

    c = copy.deepcopy(config)
    if seed is not None:
        c["initial"]["random_seed"] = int(seed)
    c["initial"]["mode"] = "physical_startup"
    storage.check_config(c)
    if c["model"]["gammaR_per_ps"] <= 0:
        raise ValueError("Frozen zero branch requires gammaR>0")
    grid = make_grid(c)
    pump = make_pump(c, grid)
    absorber = make_absorber(c, grid) if c["space"]["absorbing_layer"] else None
    psi, _ = initial_fields(c, grid, pump)
    if not np.any(psi):
        raise ValueError("Linear seed is identically zero")
    psi /= np.sqrt(np.sum(abs(psi)**2)*grid.da)
    dt = c["time"]["dt_ps"]
    window = c["threshold"]["linear_fit_window_ps"]
    duration = duration_ps or 3*window
    steps = round(duration/dt)
    sample_every = max(1, round(1.0/dt))
    half = kinetic_half_factor(grid, c, dt)
    times, logs = [0.0], [0.0]
    accumulated_log = 0.0
    previous_mode = psi.copy()
    mode_sample_step = round((duration-window)/dt)
    for step in range(1, steps+1):
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
    return {
        "status": ("UNSTABLE" if sigma_late > 0 else "STABLE") if reliable else "UNRESOLVED",
        "sigma_early_per_ps": sigma_early, "sigma_late_per_ps": sigma_late,
        "sigma_early_stderr": err_early, "sigma_late_stderr": err_late,
        "fit_R2_early": r2_early, "fit_R2_late": r2_late,
        "sigma_agreement": bool(agreement), "mode_overlap": float(overlap),
        "mode_status": "CONVERGED" if overlap >= 0.95 else "MODE_UNRESOLVED",
        "duration_ps": duration, "seed": c["initial"]["random_seed"],
        "P0": c["pump"]["P0"], "grid": grid.shape,
    }


def find_threshold_2d(config, *, save=True, result_root=None,
                      duration_ps=None, max_bisections=20, resume_dir=None):
    """Bracket a linear threshold. Returns UNRESOLVED rather than guessing."""
    import copy
    import json
    from pathlib import Path
    import storage

    storage.check_config(config)
    if config["pump"]["kind"] not in ("gaussian", "ring"):
        raise ValueError("Threshold is defined for gaussian and ring pumps")
    if resume_dir is not None:
        if not save:
            raise ValueError("Threshold resume requires saved points")
        path = Path(resume_dir)
        if storage.load_run(path)["config"] != config:
            raise ValueError("Threshold resume configuration changed")
        points_file = path/"threshold_points.json"
        points = json.loads(points_file.read_text(encoding="utf-8")) if points_file.exists() else []
        storage.event(path, "INFO", "THRESHOLD_RESUMED", {"existing_points": len(points)})
    else:
        path = storage.create_run(config, result_root or storage.RESULTS_DIR) if save else None
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
            storage.log(path, f"P0={power:.7g} sigma={result.get('sigma_late_per_ps')} "
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
        storage.write_summary(path, {"run_status": "RUN_COMPLETED",
                                     "threshold": outcome,
                                     "config_sha256": storage.config_hash(config)},
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
        import storage
        path = Path(threshold["run_dir"])
        (path/"threshold_validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        saved = storage.load_run(path)
        summary = saved["summary"] or {}
        summary["threshold_validation"] = report
        storage.write_summary(path, summary, {
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
    from diagnostics import number_reference
    import storage

    runs = {"below": storage.load_run(below_run) if not isinstance(below_run, dict)
            else below_run,
            "above": storage.load_run(above_run) if not isinstance(above_run, dict)
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
        saved = storage.load_run(path)
        summary = saved["summary"] or {}
        summary["nonlinear_confirmation"] = report
        storage.write_summary(path, summary, {
            "linear threshold": ("PASS", threshold["status"]),
            "grid/domain confirmation": (
                "PASS" if summary.get("threshold_validation", {}).get("status")
                == "GRID_DOMAIN_CONFIRMED" else "NOT_TESTED", "see threshold_validation.json"),
            "nonlinear confirmation": ("PASS" if status == "SUPPORTED" else "WARN", status),
        })
    return report


def compare_runs(reference, candidate):
    """Compare completed runs after aligning space and global field phase."""
    from diagnostics import phase_aligned_change, relative_change
    import storage

    a = storage.load_run(reference) if not isinstance(reference, dict) else reference
    b = storage.load_run(candidate) if not isinstance(candidate, dict) else candidate
    if a["summary"]["run_status"] != "RUN_COMPLETED" or b["summary"]["run_status"] != "RUN_COMPLETED":
        return {"status": "NOT_TESTED", "reason": "incomplete run"}
    ca, cb = a["config"], b["config"]
    if not np.isclose(ca["time"]["t_final_ps"], cb["time"]["t_final_ps"]):
        raise ValueError("Runs end at different physical times")
    ga, gb = make_grid(ca), make_grid(cb)
    with np.load(a["snapshots"][-1]) as da, np.load(b["snapshots"][-1]) as db:
        pa, ra = da["psi"], da["nR"]
        pb, rb = db["psi"], db["nR"]
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
    return {"status": "COMPARED", "reference": str(a["path"]),
            "candidate": str(b["path"]), **metrics}


def convergence_study(config, *, grids=(256, 512, 1024),
                      time_steps=(0.05, 0.025, 0.0125),
                      domains=((128.0, 512), (192.0, 768)),
                      result_root=None, resume_dir=None):
    """Run the v3 grid/time/domain suite; record incomplete cases honestly."""
    import copy
    import json
    from pathlib import Path
    import storage

    storage.check_config(config)
    # The continuous seed in model.initial_fields is sampled at the same
    # physical coordinates on every grid and domain.
    if resume_dir is None:
        root = storage.create_run(config, result_root or storage.RESULTS_DIR)
        previous = {}
    else:
        root = Path(resume_dir)
        if storage.load_run(root)["config"] != config:
            raise ValueError("Convergence suite configuration changed")
        manifest = root/"convergence_cases.json"
        previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    cases = {}
    def run_case(name, changes):
        c = copy.deepcopy(config)
        for section, values in changes.items():
            c[section].update(values)
        storage.check_config(c)
        if name in previous:
            result = storage.load_run(previous[name])
            if result["config"] != c:
                raise ValueError(f"Convergence case {name} configuration changed")
            if result["summary"]["run_status"] == "RUN_INCOMPLETE":
                result = simulate_2d(c, resume_dir=result["path"])
        else:
            result = simulate_2d(c, result_root=root/"cases")
        previous[name] = str(result["path"])
        (root/"convergence_cases.json").write_text(
            json.dumps(previous, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        cases[name] = result
        storage.log(root, f"{name}: {result['summary']['run_status']} {result['path']}")
        return result
    for n in grids:
        run_case(f"grid_{n}", {"space": {"Nx": n, "Ny": n}})
    base_n = grids[1]
    for dt in time_steps:
        name = f"dt_{dt:g}"
        if dt == config["time"]["dt_ps"] and base_n == config["space"]["Nx"]:
            cases[name] = cases[f"grid_{base_n}"]
        else:
            run_case(name, {"space": {"Nx": base_n, "Ny": base_n},
                            "time": {"dt_ps": dt}})
    for length, n in domains:
        name = f"domain_{length:g}"
        if length == config["space"]["Lx_um"] and n == base_n:
            cases[name] = cases[f"grid_{base_n}"]
        else:
            run_case(name, {"space": {"Lx_um": length, "Ly_um": length,
                                      "Nx": n, "Ny": n}})
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
    integral_keys = ("NC_relative", "NR_relative", "r_rms_um_relative")
    violations = [{"comparison": r["comparison"], "metric": key, "value": r[key]}
                  for r in comparisons if r["status"] == "COMPARED"
                  for key in integral_keys if key in r and r[key] >= 0.005]
    outcome = {"cases": {k: str(v["path"]) for k, v in cases.items()},
               "comparisons": comparisons,
               "time_ps": float(config["time"]["t_final_ps"]),
               "reference_stationarity_status": (
                   cases[f"grid_{base_n}"]["summary"]["stationarity"]
                   ["stationarity_status"]),
               "status": "COMPLETED" if complete else "NOT_TESTED",
               "assessment": ("NOT_TESTED" if not complete else
                              "WARN" if violations else "PASS"),
               "integral_tolerance_rel": 0.005,
               "observed_time_order": time_order,
               "violations": violations}
    outcome["final_state_assessment"] = (
        outcome["assessment"] if outcome["reference_stationarity_status"]
        in ("STATIONARY_MODE", "TIME_PERIODIC") else "NOT_TESTED")
    (root/"convergence.json").write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    storage.write_summary(root, {"run_status": "RUN_COMPLETED" if outcome["status"] == "COMPLETED"
                                 else "RUN_INCOMPLETE", "convergence": outcome},
                          {"grid/time/domain": (outcome["assessment"],
                                                outcome["status"])})
    outcome["run_dir"] = str(root)
    return outcome
