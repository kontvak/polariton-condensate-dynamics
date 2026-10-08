import copy
import numpy as np

from diagnostics import parseval_error, current
from model import make_absorber, make_grid, make_pump, initial_fields, parameters
from numerics import simulate_2d, convergence_study, local_rk4
from storage import load_config, check_config, load_run, append_csv


def test_parseval_on_complex_field():
    config = load_config("configs/validation.yaml")
    grid = make_grid(config)
    rng = np.random.default_rng(9)
    psi = rng.normal(size=grid.shape)+1j*rng.normal(size=grid.shape)
    assert parseval_error(psi, grid) < 1e-13


def test_spectral_current_axes_and_nyquist_derivative():
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
    config = load_config("configs/validation.yaml")
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
    assert load_run(resumed["path"])["metadata"]["config_sha256"]
    assert "RUN_RESUMED" in (resumed["path"]/"run_events.jsonl").read_text()


def test_convergence_workflow_compares_saved_fields(tmp_path):
    config = copy.deepcopy(load_config("configs/validation.yaml"))
    config["time"]["t_final_ps"] = 1.0
    report = convergence_study(
        config, grids=(16, 32, 64),
        time_steps=(0.05, 0.025, 0.0125),
        domains=((32.0, 32), (48.0, 48)), result_root=tmp_path)
    assert report["status"] == "COMPLETED"
    assert len(report["comparisons"]) == 5
    assert all(np.isfinite(item["density_map_relative"])
               for item in report["comparisons"])
    repeated = convergence_study(
        config, grids=(16, 32, 64),
        time_steps=(0.05, 0.025, 0.0125),
        domains=((32.0, 32), (48.0, 48)), resume_dir=report["run_dir"])
    assert repeated["cases"] == report["cases"]


def test_completed_run_can_be_extended_without_changing_physics(tmp_path):
    config = copy.deepcopy(load_config("configs/validation.yaml"))
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
    config = copy.deepcopy(load_config("configs/validation.yaml"))
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
    config = copy.deepcopy(load_config("configs/validation.yaml"))
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
