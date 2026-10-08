import copy

import numpy as np

from model import (make_grid, make_pump, parameters, stationary_0d,
                   jacobian_0d, phase_frequency_0d, solve_0d_reference,
                   stuart_landau)
from numerics import kinetic_half_factor, split_step
from storage import load_config, check_config


def test_units_threshold_and_stuart_landau():
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
