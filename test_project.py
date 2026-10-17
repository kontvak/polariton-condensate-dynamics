# Analytical limits and homogeneous dynamics
import copy

import numpy as np

from model import (make_grid, make_pump, parameters, stationary_0d,
                   jacobian_0d, phase_frequency_0d, solve_0d_reference,
                   stuart_landau)
from numerics import kinetic_half_factor, split_step
from numerics import load_config, check_config


def test_units_threshold_and_stuart_landau():
    config = load_config("validation")
    p = parameters(config)
    assert abs(p["K"]-0.67672862) < 1e-7
    assert abs(p["Pth0D"]-0.4333333333333) < 1e-12
    assert abs(p["nRth"]-8.6666666667) < 1e-9
    pump = 1.2*p["Pth0D"]
    nc, nr = stationary_0d(pump, config)["condensed"]
    np.testing.assert_allclose([nc, nr], [(pump-p["Pth0D"])/p["gammaC"], p["nRth"]])
    assert np.all(np.linalg.eigvals(jacobian_0d(nc, nr, config)).real < 0)
    assert phase_frequency_0d(nc, nr, pump, config) > 0
    sl = stuart_landau(pump, config)
    assert abs(sl["mu"]/sl["beta"]-nc*p["Pth0D"]/pump) < 1e-12


def test_0d_jacobian_threshold_and_rotating_phase():
    config = load_config("validation")
    p = parameters(config)
    for p0, expected_sign in ((0.9, -1), (1.1, 1)):
        pump = p0*p["Pth0D"]
        nc, nr = stationary_0d(pump, config)["empty"]
        eigenvalue = np.max(np.linalg.eigvals(jacobian_0d(nc, nr, config)).real)
        assert np.sign(eigenvalue) == expected_sign
    pump = 1.5*p["Pth0D"]
    nc, nr = stationary_0d(pump, config)["condensed"]
    rhs = lambda a, b: np.array([(p["R"]*b-p["gammaC"])*a,
                                  pump-(p["gammaR"]+p["R"]*a)*b])
    eps = 1e-6
    numeric = np.column_stack(((rhs(nc+eps, nr)-rhs(nc-eps, nr))/(2*eps),
                               (rhs(nc, nr+eps)-rhs(nc, nr-eps))/(2*eps)))
    np.testing.assert_allclose(jacobian_0d(nc, nr, config), numeric,
                               atol=1e-10, rtol=1e-9)
    times = np.array([0.0, 0.2])
    psi, reservoir = solve_0d_reference(config, pump, np.sqrt(nc), nr, times)
    expected = np.sqrt(nc)*np.exp(-1j*phase_frequency_0d(nc, nr, pump, config)*times)
    np.testing.assert_allclose(psi, expected, atol=1e-10)
    np.testing.assert_allclose(reservoir, nr, atol=1e-10)


def test_exact_reservoir_without_condensate():
    config = load_config("validation")
    grid = make_grid(config)
    pump = make_pump(config, grid)
    nr0 = np.full(grid.shape, 2.0)
    psi = np.zeros(grid.shape, dtype=complex)
    nr = nr0.copy()
    dt = 0.05
    half = kinetic_half_factor(grid, config, dt)
    for _ in range(20):
        psi, nr = split_step(psi, nr, pump, config, dt, half)
    expected = pump/config["model"]["gammaR_per_ps"]+(
        nr0-pump/config["model"]["gammaR_per_ps"])*np.exp(-config["model"]["gammaR_per_ps"])
    np.testing.assert_allclose(nr, expected, atol=2e-11, rtol=2e-11)
    assert np.max(abs(psi)) == 0


def test_plane_wave_and_decay():
    config = load_config("validation")
    config = copy.deepcopy(config)
    config["run"]["validation_mode"] = "analytic_test"
    config["model"].update(gC_mev_um2=0, gR_mev_um2=0, R_um2_per_ps=0,
                           gammaC_per_ps=0, gammaR_per_ps=0,
                           background_blueshift=False)
    config["pump"]["kind"] = "off"
    check_config(config)
    grid = make_grid(config)
    x, y = np.meshgrid(grid.x, grid.y)
    kx, ky = grid.kx[3], grid.ky[2]
    initial = np.exp(1j*(kx*x+ky*y))
    psi, nr = split_step(initial, np.zeros(grid.shape), np.zeros(grid.shape),
                         config, 0.05)
    expected = initial*np.exp(-1j*parameters(config)["K"]*(kx*kx+ky*ky)*0.05/parameters(config)["hbar"])
    np.testing.assert_allclose(psi, expected, atol=1e-13, rtol=1e-13)
    config["model"]["gammaC_per_ps"] = 0.2
    psi, nr = split_step(initial, nr, np.zeros(grid.shape), config, 0.05)
    np.testing.assert_allclose(np.sum(abs(psi)**2)/np.sum(abs(initial)**2),
                               np.exp(-0.2*0.05), rtol=1e-11)


def test_uniform_2d_matches_independent_0d_complex_field():
    config = load_config("validation")
    config = copy.deepcopy(config)
    config["pump"]["kind"] = "uniform"
    grid = make_grid(config)
    pump = make_pump(config, grid)
    psi0, nr0 = 0.1+0.03j, 3.0
    psi = np.full(grid.shape, psi0)
    nr = np.full(grid.shape, nr0)
    dt = 0.05
    half = kinetic_half_factor(grid, config, dt)
    for _ in range(40):
        psi, nr = split_step(psi, nr, pump, config, dt, half)
    ref_psi, ref_nr = solve_0d_reference(config, float(pump[0, 0]), psi0, nr0,
                                         np.array([0.0, 2.0]))
    np.testing.assert_allclose(psi, ref_psi[-1], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nr, ref_nr[-1], atol=1e-9, rtol=1e-9)



def test_condensed_eigenvalues_match_jacobian_and_oscillation_frequency():
    from model import condensed_eigenvalues_0d
    c = load_config("validation")
    p = parameters(c)
    for s in (1.0001, 1.05, 1.5, 3.0, 14.0):
        analytic = np.array(condensed_eigenvalues_0d(s, c))
        state = stationary_0d(s*p["Pth0D"], c)["condensed"]
        numeric = np.linalg.eigvals(jacobian_0d(*state, c))
        error = min(np.max(abs(analytic-numeric)), np.max(abs(analytic-numeric[::-1])))
        assert error < 1e-13
        assert np.all(analytic.real < 0)
    lp, lm = condensed_eigenvalues_0d(1.5, c)
    assert lp.imag > 0 and lm.imag < 0
    np.testing.assert_allclose(lp.real, -p["gammaR"]*1.5/2, atol=1e-15)


def test_reduced_models_stationary_density_and_complex_phase():
    from model import reduced_stationary_0d, solve_0d_reduced
    c = load_config("validation")
    p = parameters(c)
    for s in (1.01, 1.2, 1.5, 3.0):
        pump = s*p["Pth0D"]
        full = stationary_0d(pump, c)["condensed"]
        ad = reduced_stationary_0d(pump, c, "adiabatic")
        sl = reduced_stationary_0d(pump, c, "stuart_landau")
        np.testing.assert_allclose(ad, full, atol=1e-14)
        np.testing.assert_allclose(abs(sl[0]-full[0])/full[0], 1-1/s, atol=1e-14)
        for method, state in (("adiabatic", ad), ("stuart_landau", sl)):
            times = np.linspace(0, 5, 21)
            psi, nr = solve_0d_reduced(c, pump, np.sqrt(state[0]), times, method)
            coeff = stuart_landau(pump, c)
            omega = (phase_frequency_0d(*state, pump, c) if method == "adiabatic"
                     else coeff["omega0"]+coeff["nu"]*state[0])
            expected = np.sqrt(state[0])*np.exp(-1j*omega*times)
            np.testing.assert_allclose(psi, expected, atol=2e-10, rtol=2e-10)
            np.testing.assert_allclose(nr, state[1], atol=2e-10)


def test_stuart_landau_transient_matches_exact_logistic_density():
    from model import solve_0d_reduced
    c = load_config("validation")
    p = parameters(c)
    pump = 1.5*p["Pth0D"]
    times = np.linspace(0, 150, 301)
    psi0 = 0.01*np.exp(0.3j)
    psi, _ = solve_0d_reduced(c, pump, psi0, times, "stuart_landau")
    coeff = stuart_landau(pump, c)
    n0 = abs(psi0)**2
    exponential = np.exp(2*coeff["mu"]*times)
    exact = n0*exponential/(1+coeff["beta"]*n0/coeff["mu"]*(exponential-1))
    np.testing.assert_allclose(abs(psi)**2, exact, atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(psi[0], psi0, atol=1e-15)


# Spatial diagnostics and stationarity
import copy
import numpy as np
import pytest

from numerics import StationarityTracker, basic_observables, spatial_diagnostics
from model import make_grid, make_pump
from numerics import load_config, check_config


def test_validation_profiles_and_unknown_keys():
    config = load_config("validation")
    bad = copy.deepcopy(config)
    bad["model"]["gR_mev_um2"] = 0
    with pytest.raises(ValueError):
        check_config(bad)
    bad["run"]["validation_mode"] = "control"
    assert check_config(bad)
    bad["model"]["not_a_parameter"] = 1
    with pytest.raises(ValueError):
        check_config(bad)
    absorber = copy.deepcopy(config)
    absorber["space"].update(absorbing_layer=True, absorber_width_um=0.0,
                             absorber_gamma_max_per_ps=0.5)
    with pytest.raises(ValueError):
        check_config(absorber)


def test_low_signal_and_edge_mask():
    config = load_config("validation")
    grid = make_grid(config)
    pump = make_pump(config, grid)
    empty = spatial_diagnostics(np.zeros(grid.shape), np.zeros(grid.shape),
                                pump, grid, config, 1.0)
    assert empty["high_k_fraction"] is None
    psi = np.zeros(grid.shape, complex)
    psi[0, 0] = 1
    metrics = spatial_diagnostics(psi, np.zeros(grid.shape), pump, grid, config, 1.0)
    assert metrics["edge_mass_fraction"] == 1.0


def test_periodic_profile_is_not_stationary():
    config = load_config("validation")
    config = copy.deepcopy(config)
    config["time"]["dt_ps"] = 0.5
    config["checks"]["stationarity_window_ps"] = 10.0
    config["checks"]["stationarity_lags_ps"] = [2.0, 3.0]
    grid = make_grid(config)
    pump = make_pump(config, grid)
    base = np.ones(grid.shape, complex)
    x = np.cos(2*np.pi*grid.x/grid.lx)[None, :]
    tracker = StationarityTracker(config, grid, 1.0)
    for step in range(21):
        t = step*0.5
        psi = base+0.2*np.sin(2*np.pi*t/2)*x
        nr = np.ones(grid.shape)
        obs = basic_observables(psi, nr, pump, grid, config)
        tracker.update(t, psi, nr, obs, step)
    assert tracker.last["stationarity_status"] != "STATIONARY_MODE"


def test_global_phase_rotation_is_stationary():
    config = load_config("validation")
    config = copy.deepcopy(config)
    config["time"]["dt_ps"] = 0.5
    config["checks"]["stationarity_window_ps"] = 10.0
    config["checks"]["stationarity_lags_ps"] = [2.0, 3.0]
    grid = make_grid(config)
    pump = make_pump(config, grid)
    profile = np.exp(-(grid.x[None, :]**2+grid.y[:, None]**2)/25)
    tracker = StationarityTracker(config, grid, 1.0)
    for step in range(21):
        t = step*0.5
        psi = profile*np.exp(-0.7j*t)
        nr = np.ones(grid.shape)
        tracker.update(t, psi, nr, basic_observables(psi, nr, pump, grid, config), step)
    assert tracker.last["stationarity_status"] == "STATIONARY_MODE"
    assert tracker.last["current_status"] == "NEGLIGIBLE_CURRENT"


def test_low_signal_recovery_is_not_called_decayed():
    config = load_config("validation")
    config = copy.deepcopy(config)
    config["time"]["dt_ps"] = 0.5
    config["checks"]["stationarity_window_ps"] = 10.0
    config["checks"]["stationarity_lags_ps"] = [2.0, 3.0]
    grid = make_grid(config)
    pump = make_pump(config, grid)
    tracker = StationarityTracker(config, grid, 1.0)
    for step in range(21):
        t = step*0.5
        density = 1e-12*np.exp(0.2*t)
        psi = np.full(grid.shape, np.sqrt(density), complex)
        nr = np.ones(grid.shape)
        tracker.update(t, psi, nr, basic_observables(psi, nr, pump, grid, config), step)
    assert tracker.last["stationarity_status"] == "UNDETERMINED"


@pytest.mark.parametrize("period", [20.0, 25.0])
def test_full_window_detects_periodic_population(period):
    config = copy.deepcopy(load_config("validation"))
    config["time"]["dt_ps"] = 1.0
    config["save"]["stationarity_reference_every_ps"] = 1.0
    config["save"]["stationarity_probe_every_ps"] = 5.0
    grid = make_grid(config)
    pump = make_pump(config, grid)
    profile = np.exp(-(grid.x[None, :]**2+grid.y[:, None]**2)/25)
    tracker = StationarityTracker(config, grid, 1.0)
    for step in range(101):
        psi = profile*(1+0.15*np.sin(2*np.pi*step/period))
        nr = np.ones(grid.shape)
        tracker.update(float(step), psi, nr,
                       basic_observables(psi, nr, pump, grid, config), step)
    assert tracker.last["stationarity_status"] != "STATIONARY_MODE"
    assert tracker.last["dense_points"] == 101
    assert all(tracker.last["lag_pair_counts"].values())


def test_moving_profile_with_constant_population_is_not_stationary():
    config = copy.deepcopy(load_config("validation"))
    config["time"]["dt_ps"] = 1.0
    config["save"]["stationarity_reference_every_ps"] = 1.0
    config["save"]["stationarity_probe_every_ps"] = 5.0
    grid = make_grid(config)
    pump = make_pump(config, grid)
    tracker = StationarityTracker(config, grid, 1.0)
    x = 2*np.pi*grid.x[None, :]/grid.lx
    for step in range(101):
        psi = 1+0.2*np.cos(x-2*np.pi*step/25.0)
        psi = np.broadcast_to(psi, grid.shape).astype(complex)
        nr = np.ones(grid.shape)
        tracker.update(float(step), psi, nr,
                       basic_observables(psi, nr, pump, grid, config), step)
    assert tracker.last["NC_window_span"] < 1e-12
    assert tracker.last["stationarity_status"] == "NONSTATIONARY_FIELD"


def test_incomplete_window_is_not_tested():
    config = copy.deepcopy(load_config("validation"))
    grid = make_grid(config)
    pump = make_pump(config, grid)
    tracker = StationarityTracker(config, grid, 1.0)
    psi = np.ones(grid.shape, complex)
    nr = np.ones(grid.shape)
    for step in range(11):
        tracker.update(float(step), psi, nr,
                       basic_observables(psi, nr, pump, grid, config), step)
    assert tracker.evaluate()["stationarity_status"] == "NOT_TESTED"


# FFT propagation, seeds and checkpoint continuation
import copy
import numpy as np

from numerics import parseval_error, current
from model import make_absorber, make_grid, make_pump, initial_fields, parameters
from numerics import simulate_2d, convergence_study, local_rk4
from numerics import load_config, check_config, load_run, append_csv


def test_parseval_on_complex_field():
    config = load_config("validation")
    grid = make_grid(config)
    rng = np.random.default_rng(9)
    psi = rng.normal(size=grid.shape)+1j*rng.normal(size=grid.shape)
    assert parseval_error(psi, grid) < 1e-13


def test_spectral_current_axes_and_nyquist_derivative():
    config = load_config("validation")
    grid = make_grid(config)
    xx, yy = np.meshgrid(grid.x, grid.y)
    kx, ky = grid.kx[3], grid.ky[2]
    psi = np.exp(1j*(kx*xx+ky*yy))
    jx, jy = current(psi, grid, config)
    factor = 2*parameters(config)["K"]/parameters(config)["hbar"]
    np.testing.assert_allclose(jx, factor*kx, atol=1e-13)
    np.testing.assert_allclose(jy, factor*ky, atol=1e-13)
    real_nyquist = (-1.0)**np.arange(grid.shape[1])[None, :]
    jx, jy = current(np.broadcast_to(real_nyquist, grid.shape), grid, config)
    assert np.max(abs(jx))+np.max(abs(jy)) < 1e-13


def test_same_physical_seed_on_refined_grid():
    config = load_config("validation")
    config = copy.deepcopy(config)
    config["initial"]["mode"] = "physical_startup"
    coarse = make_grid(config)
    psi1, _ = initial_fields(config, coarse, make_pump(config, coarse))
    config["space"]["Nx"] *= 2
    config["space"]["Ny"] *= 2
    fine = make_grid(config)
    psi2, _ = initial_fields(config, fine, make_pump(config, fine))
    np.testing.assert_allclose(psi1, psi2[::2, ::2], atol=1e-15)


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    config = load_config("validation")
    config = copy.deepcopy(config)
    config["time"]["t_final_ps"] = 1.0
    config["checks"]["max_walltime_per_run_minutes"] = 10
    check_config(config)
    full = simulate_2d(config, result_root=tmp_path/"full")
    partial = simulate_2d(config, result_root=tmp_path/"resume", max_steps=10)
    assert partial["summary"]["run_status"] == "RUN_INCOMPLETE"
    with np.load(partial["summary"]["latest_checkpoint"]) as checkpoint:
        assert "tracker_dense" in checkpoint.files
        assert len(checkpoint["tracker_dense"]) == 2
    resumed = simulate_2d(config, resume_dir=partial["path"])
    assert resumed["summary"]["run_status"] == "RUN_COMPLETED"
    with np.load(full["snapshots"][-1]) as a, np.load(resumed["snapshots"][-1]) as b:
        np.testing.assert_allclose(a["psi"], b["psi"], atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(a["nR"], b["nR"], atol=1e-11, rtol=1e-11)
    assert resumed["summary"]["balance_relative_residual"] < 1e-6
    metadata = load_run(resumed["path"])["metadata"]
    assert metadata["config_sha256"]
    assert metadata["resource_estimate"]["working_memory_bytes"] > 0
    assert metadata["resource_estimate"]["output_bytes"] > 0
    assert metadata["grid"]["kx_nyquist_um_inv"] > 0
    assert "RUN_RESUMED" in (resumed["path"]/"run_events.jsonl").read_text()


def test_convergence_workflow_compares_saved_fields(tmp_path):
    config = copy.deepcopy(load_config("validation"))
    config["time"]["t_final_ps"] = 1.0
    report = convergence_study(
        config, grids=(16, 32, 64),
        time_steps=(0.05, 0.025, 0.0125),
        domains=((32.0, 32), (48.0, 48)), result_root=tmp_path)
    assert report["status"] == "COMPLETED"
    assert report["final_state_assessment"] == "NOT_TESTED"
    assert len(report["comparisons"]) == 5
    assert all(np.isfinite(item["density_map_relative"])
               for item in report["comparisons"])
    repeated = convergence_study(
        config, grids=(16, 32, 64),
        time_steps=(0.05, 0.025, 0.0125),
        domains=((32.0, 32), (48.0, 48)), resume_dir=report["run_dir"])
    assert repeated["cases"] == report["cases"]


def test_completed_run_can_be_extended_without_changing_physics(tmp_path):
    config = copy.deepcopy(load_config("validation"))
    config["time"]["t_final_ps"] = 1.0
    short = simulate_2d(config, result_root=tmp_path/"short")
    config["time"]["t_final_ps"] = 2.0
    extended = simulate_2d(config, resume_dir=short["path"])
    direct = simulate_2d(config, result_root=tmp_path/"direct")
    with np.load(extended["snapshots"][-1]) as a, np.load(direct["snapshots"][-1]) as b:
        np.testing.assert_allclose(a["psi"], b["psi"], atol=1e-11)
        np.testing.assert_allclose(a["nR"], b["nR"], atol=1e-11)
    assert extended["summary"]["run_status"] == "RUN_COMPLETED"
    assert "FINAL_TIME_EXTENDED" in (extended["path"]/"run_events.jsonl").read_text()


def test_diagnostics_csv_adds_new_columns_on_resume(tmp_path):
    import csv
    append_csv(tmp_path, {"step": 1, "NC": 2.0})
    append_csv(tmp_path, {"step": 2, "NC": 3.0, "current_status": "TESTED"})
    with (tmp_path/"diagnostics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["current_status"] == "n/a"
    assert rows[1]["current_status"] == "TESTED"


def test_keyboard_interrupt_records_incomplete_run(tmp_path, monkeypatch):
    import numerics
    config = copy.deepcopy(load_config("validation"))
    config["time"]["t_final_ps"] = 1.0
    actual_step = numerics.split_step
    count = 0
    def interrupted(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise KeyboardInterrupt
        return actual_step(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(numerics, "split_step", interrupted)
        partial = simulate_2d(config, result_root=tmp_path)
    assert partial["summary"]["run_status"] == "RUN_INCOMPLETE"
    assert "USER_INTERRUPT" in (partial["path"]/"run_events.jsonl").read_text()
    completed = simulate_2d(config, resume_dir=partial["path"])
    assert completed["summary"]["run_status"] == "RUN_COMPLETED"


def test_optional_absorber_has_exact_local_decay_and_balances_loss(tmp_path):
    config = copy.deepcopy(load_config("validation"))
    config["run"]["validation_mode"] = "analytic_test"
    config["pump"]["kind"] = "off"
    config["pump"]["d_um"] = 20.0
    config["initial"]["psi_seed_density_um2_inv"] = 1.0
    config["model"].update(gC_mev_um2=0.0, gR_mev_um2=0.0,
                           R_um2_per_ps=0.0, gammaC_per_ps=0.0,
                           gammaR_per_ps=0.0, background_blueshift=False)
    config["space"].update(absorbing_layer=True, absorber_width_um=5.0,
                           absorber_gamma_max_per_ps=0.5)
    config["time"]["t_final_ps"] = 1.0
    assert check_config(config)
    grid = make_grid(config)
    absorber = make_absorber(config, grid)
    assert absorber[grid.shape[0]//2, grid.shape[1]//2] == 0
    assert np.max(absorber) == 0.5
    psi, nr = local_rk4(np.ones(grid.shape, complex), np.zeros(grid.shape),
                        np.zeros(grid.shape), config, 0.05, absorber)
    np.testing.assert_allclose(psi, np.exp(-0.5*absorber*0.05), atol=1e-11)
    run = simulate_2d(config, result_root=tmp_path)
    assert run["summary"]["observables"]["absorber_loss_per_ps"] > 0
    coarse_error = run["summary"]["balance_relative_residual"]
    config["time"]["dt_ps"] = 0.0125
    fine = simulate_2d(config, result_root=tmp_path)
    fine_error = fine["summary"]["balance_relative_residual"]
    assert fine_error < 1e-6
    assert fine_error < coarse_error/8


# Thresholds and research workflows
from numerics import find_threshold_2d, linear_growth_rate
from numerics import load_config


def test_short_linear_probe_does_not_guess_threshold():
    config = load_config("validation")
    config["threshold"]["linear_fit_window_ps"] = 2.0
    config["threshold"]["initial_P0_scan"] = [0.5, 1.5, 4.0]
    point = linear_growth_rate(config, duration_ps=6.0)
    assert point["status"] in ("STABLE", "UNSTABLE", "UNRESOLVED")
    assert point["sigma_late_per_ps"] == point["sigma_late_per_ps"]
    result = find_threshold_2d(config, save=False, duration_ps=6.0,
                               max_bisections=2)
    assert result["status"] in ("NOT_BRACKETED", "THRESHOLD_UNRESOLVED", "PRELIMINARY")
    assert len(result["points"]) >= 3


def test_threshold_resume_reuses_saved_points(tmp_path):
    config = load_config("validation")
    config["threshold"]["linear_fit_window_ps"] = 2.0
    config["threshold"]["initial_P0_scan"] = [0.5, 1.5, 4.0]
    first = find_threshold_2d(config, duration_ps=6.0, max_bisections=0,
                              result_root=tmp_path)
    second = find_threshold_2d(config, duration_ps=6.0, max_bisections=0,
                               resume_dir=first["run_dir"])
    assert len(second["points"]) == len(first["points"])
    assert "THRESHOLD_RESUMED" in (
        tmp_path/next(tmp_path.iterdir()).name/"run_events.jsonl").read_text()


def test_convergence_never_accepts_only_the_reference(tmp_path):
    import json
    from pathlib import Path
    config = copy.deepcopy(load_config("validation"))
    config["time"]["t_final_ps"] = 1.0
    options = dict(grids=(16, 32, 64), time_steps=(0.05, 0.025, 0.0125),
                   domains=((32.0, 32), (48.0, 48)), max_workers=2)
    report = convergence_study(config, result_root=tmp_path, **options)
    # Synthetic fixture: settled reference, but the fine grid is still changing.
    for path in set(report["cases"].values()):
        file = Path(path)/"summary.json"
        summary = json.loads(file.read_text())
        summary["stationarity"]["stationarity_status"] = "STATIONARY_MODE"
        if str(path) == report["cases"]["grid_64"]:
            summary["stationarity"]["stationarity_status"] = "DYNAMIC_OR_UNSETTLED"
        file.write_text(json.dumps(summary))
    repeated = convergence_study(config, resume_dir=report["run_dir"], **options)
    assert repeated["reference_stationarity_status"] == "STATIONARY_MODE"
    assert repeated["final_state_assessment"] == "WARN"


def test_convergence_reuse_rejects_a_different_seed(tmp_path):
    import pytest
    config = copy.deepcopy(load_config("validation"))
    config["time"]["t_final_ps"] = 1.0
    other = copy.deepcopy(config)
    other["initial"]["random_seed"] += 1
    saved = simulate_2d(other, result_root=tmp_path/"other")
    with pytest.raises(ValueError, match="configuration changed"):
        convergence_study(config, grids=(16, 32, 64),
            time_steps=(0.05, 0.025, 0.0125), domains=((32.0, 32), (48.0, 48)),
            reuse_cases={"grid_64": saved["path"]}, result_root=tmp_path/"study")


def test_runtime_recommendation_does_not_stop_a_run(tmp_path):
    config = copy.deepcopy(load_config("validation"))
    config["checks"]["max_walltime_per_run_minutes"] = 1e-12
    result = simulate_2d(config, result_root=tmp_path)
    assert result["summary"]["run_status"] == "RUN_COMPLETED"
    assert result["summary"]["time_ps"] == config["time"]["t_final_ps"]
    assert result["summary"]["runtime_recommendation_exceeded"]
    events = (result["path"]/"run_events.jsonl").read_text()
    assert "RECOMMENDED_RUNTIME_EXCEEDED" in events
    assert "WALLTIME_LIMIT" not in events


def test_density_periodicity_checks_shape_counts_and_drift():
    from numerics import density_periodicity
    t=np.arange(201,dtype=float)
    x,y=np.meshgrid(np.linspace(-8,8,17),np.linspace(-8,8,17))
    angle=np.arctan2(y,x);radial=np.exp(-(np.hypot(x,y)-4)**2/4)
    for period in (20.,25.):
        oscillation=2*np.pi*t/period
        n=radial[None,:,:]*(1+.3*np.cos(2*angle[None,:,:]-oscillation[:,None,None]))
        nc=100+5*np.sin(oscillation);nr=500-2*np.sin(oscillation)
        result=density_periodicity(t,n,nc,nr,minimum_cycles=3)
        assert result['status']=='DENSITY_PERIODIC_SUPPORTED'
        assert abs(result['period_ps']-period)<.05
        drift=1+.004*t
        transient=density_periodicity(t,n*drift[:,None,None],nc*drift,nr,minimum_cycles=3)
        assert transient['status']=='PERIODIC_COMPONENT_WITH_DRIFT'
        assert transient['NC_recurrence_rel']>.005


def test_density_maxima_keeps_opposite_peaks_separate():
    from numerics import density_maxima
    x=np.arange(-10,10.5,.5);y=x.copy();xx,yy=np.meshgrid(x,y)
    n=np.exp(-((xx-4)**2+yy**2))+np.exp(-((xx+4)**2+yy**2))
    peaks=density_maxima(n,x,y)
    assert len(peaks)==2
    assert sorted(p[0] for p in peaks)==[-4,4]


def test_observation_callback_does_not_change_simulation(tmp_path):
    config=copy.deepcopy(load_config('validation'))
    observed=[]
    def callback(t,psi,nr,obs,state):
        observed.append((t,obs['NC']))
    a=simulate_2d(config,result_root=tmp_path/'a')
    b=simulate_2d(config,result_root=tmp_path/'b',on_observation=callback)
    assert observed[0][0]==0 and observed[-1][0]==2
    with np.load(a['snapshots'][-1]) as za,np.load(b['snapshots'][-1]) as zb:
        np.testing.assert_array_equal(za['psi'],zb['psi'])
        np.testing.assert_array_equal(za['nR'],zb['nR'])


def test_damped_small_density_signal_is_not_a_sustained_cycle():
    from numerics import density_periodicity
    t=np.arange(301,dtype=float)
    x,y=np.meshgrid(np.linspace(-8,8,17),np.linspace(-8,8,17))
    radial=np.exp(-(np.hypot(x,y)-4)**2/4);angle=np.arctan2(y,x)
    n=radial[None,:,:]*(1+.01*np.exp(-t[:,None,None]/80)*np.cos(
                      2*angle[None,:,:]-2*np.pi*t[:,None,None]/25))
    result=density_periodicity(t,n,np.full(len(t),100.),np.full(len(t),500.),minimum_cycles=3)
    assert result['status']!='DENSITY_PERIODIC_SUPPORTED'



def test_vortex_charge_moment_and_density_guard():
    from numerics import vortex_diagnostics
    config=copy.deepcopy(load_config('validation'))
    config['space'].update(Nx=128,Ny=128,Lx_um=32.,Ly_um=32.)
    grid=make_grid(config);x,y=np.meshgrid(grid.x,grid.y)
    for charge in (-3,3):
        z=x+1j*np.sign(charge)*y
        psi=z**abs(charge)*np.exp(-(x*x+y*y)/18)
        result=vortex_diagnostics(psi,grid,config,radii_um=(.5,3.,4.,5.),n_angles=512)
        assert result['q']==charge and result['classification']=='VORTEX_CONDENSATE'
        assert result['contours'][0]['status']=='LOW_DENSITY'
        assert abs(result['Lz_over_hbar_NC']-charge)<1e-8
        assert abs(result['Lz_imag_over_hbar_NC'])<1e-10
        assert all(abs(c['charge_from_current']-charge)<.03 for c in result['contours'][1:])
        gauged=vortex_diagnostics(psi*np.exp(1.7j),grid,config,radii_um=(3.,4.),n_angles=1024)
        assert gauged['q']==charge


def test_radial_flow_has_no_topological_charge():
    from numerics import vortex_diagnostics
    config=copy.deepcopy(load_config('validation'))
    config['space'].update(Nx=128,Ny=128,Lx_um=32.,Ly_um=32.)
    grid=make_grid(config);x,y=np.meshgrid(grid.x,grid.y);r2=x*x+y*y
    psi=np.exp(-r2/18+.03j*r2)
    result=vortex_diagnostics(psi,grid,config,radii_um=(2.,3.,4.))
    assert result['q']==0 and result['classification']=='ZERO_NET_CHARGE'
    assert abs(result['Lz_over_hbar_NC'])<1e-10
    assert result['contours'][0]['mean_radial_current_um_inv_ps_inv']>0


def test_linear_mode_continuation_matches_uninterrupted():
    config = load_config("validation")
    config["threshold"]["linear_fit_window_ps"] = 2.0
    _, state = linear_growth_rate(config, duration_ps=6.0, return_state=True)
    resumed, resumed_state = linear_growth_rate(config, duration_ps=12.0,
                                                initial_state=state, return_state=True)
    full, full_state = linear_growth_rate(config, duration_ps=12.0, return_state=True)
    np.testing.assert_array_equal(resumed_state["psi"], full_state["psi"])
    np.testing.assert_array_equal(resumed_state["log_numbers"], full_state["log_numbers"])
    assert resumed == full
    changed = copy.deepcopy(config)
    changed["pump"]["P0"] += 0.1
    with pytest.raises(ValueError, match="configuration changed"):
        linear_growth_rate(changed, duration_ps=12.0, initial_state=state)
