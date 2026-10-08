import copy
import numpy as np
import pytest

from diagnostics import StationarityTracker, basic_observables, spatial_diagnostics
from model import make_grid, make_pump
from storage import load_config, check_config


def test_validation_profiles_and_unknown_keys():
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
    config = copy.deepcopy(load_config("configs/validation.yaml"))
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
    config = copy.deepcopy(load_config("configs/validation.yaml"))
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
    config = copy.deepcopy(load_config("configs/validation.yaml"))
    grid = make_grid(config)
    pump = make_pump(config, grid)
    tracker = StationarityTracker(config, grid, 1.0)
    psi = np.ones(grid.shape, complex)
    nr = np.ones(grid.shape)
    for step in range(11):
        tracker.update(float(step), psi, nr,
                       basic_observables(psi, nr, pump, grid, config), step)
    assert tracker.evaluate()["stationarity_status"] == "NOT_TESTED"
