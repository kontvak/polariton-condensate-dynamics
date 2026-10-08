"""Static scientific figures made from saved numerical arrays."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import current, spectrum
from model import (make_grid, make_pump, parameters, stationary_0d,
                   jacobian_0d, solve_0d_reference, stuart_landau)
from storage import load_run


def build_0d_data(config, directory):
    """Compute the inexpensive 0D reference once and save its raw arrays."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = parameters(config)
    powers = np.linspace(0.5, 3.0, 151)
    nc, nr, growth = [], [], []
    for p0 in powers:
        pump = p0*p["Pth0D"]
        branches = stationary_0d(pump, config)
        state = branches["condensed"] or branches["empty"]
        nc.append(state[0])
        nr.append(state[1])
        growth.append(float(np.max(np.linalg.eigvals(
            jacobian_0d(*branches["empty"], config)).real)))
    np.savez(directory/"F1_0d_data.npz", P0=powers, nC=np.array(nc),
             nR=np.array(nr), empty_branch_growth=np.array(growth))
    times = np.linspace(0, 200, 401)
    output = {"t_ps": times}
    for p0 in (0.8, 1.05, 1.2, 1.5):
        psi, reservoir = solve_0d_reference(
            config, p0*p["Pth0D"], 0.01+0j, 0.0, times)
        label = f"{p0:g}".replace(".", "p")
        output[f"nC_{label}"], output[f"nR_{label}"] = abs(psi)**2, reservoir
        if p0 > 1:
            sl = stuart_landau(p0*p["Pth0D"], config)
            n0 = 0.01**2
            growth = np.exp(2*sl["mu"]*times)
            output[f"nC_SL_{label}"] = n0*growth/(
                1+(sl["beta"]*n0/sl["mu"])*(growth-1))
    np.savez(directory/"F2_0d_data.npz", **output)
    return directory


def _run(run):
    return load_run(run) if not isinstance(run, dict) else run


def _latest(run):
    run = _run(run)
    if not run["snapshots"]:
        raise FileNotFoundError(f"No snapshots in {run['path']}")
    with np.load(run["snapshots"][-1]) as values:
        return run, values["psi"].copy(), values["nR"].copy(), float(values["time_ps"])


def _save(fig, out, name):
    path = out/f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _figure_0d(data_dir, out):
    files = []
    with np.load(data_dir/"F1_0d_data.npz") as a:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(a["P0"], a["nC"], label="конденсат")
        axes[0].plot(a["P0"], a["nR"], label="резервуар")
        axes[0].axvline(1, color="k", ls="--", lw=1)
        axes[0].set(xlabel="P₀", ylabel="Плотность, мкм⁻²", title="0D стационарные ветви")
        axes[0].legend()
        axes[1].plot(a["P0"], a["empty_branch_growth"])
        axes[1].axhline(0, color="k", lw=1)
        axes[1].axvline(1, color="k", ls="--", lw=1)
        axes[1].set(xlabel="P₀", ylabel="Макс. Re λ, пс⁻¹",
                    title="Устойчивость нулевой ветви")
        files.append(_save(fig, out, "F1_0d_threshold"))
    with np.load(data_dir/"F2_0d_data.npz") as a:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for label in ("0p8", "1p05", "1p2", "1p5"):
            axes[0].plot(a["t_ps"], a[f"nC_{label}"], label=f"P₀={label.replace('p', '.')}")
            axes[1].plot(a["t_ps"], a[f"nR_{label}"], label=f"P₀={label.replace('p', '.')}")
            if f"nC_SL_{label}" in a.files:
                axes[0].plot(a["t_ps"], a[f"nC_SL_{label}"], ls="--", alpha=0.7,
                             label=f"SL {label.replace('p', '.')}")
        axes[0].set(xlabel="t, пс", ylabel="nC, мкм⁻²",
                    title="Конденсат: сплошная — полный 0D, пунктир — SL")
        axes[1].set(xlabel="t, пс", ylabel="nR, мкм⁻²", title="Резервуар")
        axes[0].legend(fontsize=8, ncol=2)
        fig.text(0.5, -0.02,
                 "Полная модель: nR(0)=0; SL: адиабатический резервуар. "
                 "Начальные условия резервуара различаются.",
                 ha="center", fontsize=8)
        files.append(_save(fig, out, "F2_0d_dynamics"))
    return files


def _pair_figures(gaussian_run, ring_run, out, *,
                  gaussian_spectrum_runs=(), ring_spectrum_runs=()):
    gauss, gp, gn, gt = _latest(gaussian_run)
    ring, rp, rn, rt = _latest(ring_run)
    pairs = ((gauss, gp, gn, gt), (ring, rp, rn, rt))
    provenance = "\n".join(
        f"{run['path'].name}: P₀={run['config']['pump']['P0']}, "
        f"N={run['config']['space']['Nx']}, "
        f"L={run['config']['space']['Lx_um']} мкм, "
        f"absorber={run['config']['space']['absorbing_layer']}, "
        f"status={run['summary']['stationarity']['stationarity_status']}, "
        f"edge={run['summary']['spatial_diagnostics'].get('edge_mass_fraction')}"
        for run, _psi, _nr, _time in pairs)
    arrays = []
    for run, psi, nr, time_ps in pairs:
        grid = make_grid(run["config"])
        p = parameters(run["config"])
        pump = make_pump(run["config"], grid)
        bg = p["eta"]*pump if p["background_blueshift"] else 0
        potential = p["gC"]*abs(psi)**2+p["gR"]*(nr+bg)
        arrays.append((pump, nr, abs(psi)**2, potential))
    labels = ("P, мкм⁻² пс⁻¹", "nR, мкм⁻²", "nC, мкм⁻²", "V, мэВ")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    for col in range(4):
        vmax = max(float(np.max(row[col])) for row in arrays)
        for row in range(2):
            run, psi, nr, time_ps = pairs[row]
            grid = make_grid(run["config"])
            image = axes[row, col].imshow(arrays[row][col], origin="lower",
                extent=(-grid.lx/2, grid.lx/2, -grid.ly/2, grid.ly/2),
                vmin=0, vmax=vmax, cmap="viridis")
            axes[row, col].set_xlim(-40, 40)
            axes[row, col].set_ylim(-40, 40)
            if row == 0:
                axes[row, col].set_title(labels[col], fontsize=11)
            if row == 1:
                axes[row, col].set_xlabel("x, мкм")
            if col == 0:
                axes[row, col].set_ylabel(
                    ("Пятно" if row == 0 else "Кольцо")+"\ny, мкм")
            fig.colorbar(image, ax=axes[row, col], shrink=0.75)
    fig.suptitle("F3. Накачка, резервуар, плотность и потенциал", fontsize=14)
    fig.text(0.5, -0.035, provenance, ha="center", fontsize=8)
    files = [_save(fig, out, "F3_density_reservoir_potential")]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    max_current = 0.0
    for run, psi, _nr, _time in pairs:
        jx0, jy0 = current(psi, make_grid(run["config"]), run["config"])
        max_current = max(max_current, float(np.max(np.hypot(jx0, jy0))))
    for row, (run, psi, nr, time_ps) in enumerate(pairs):
        grid = make_grid(run["config"])
        density = abs(psi)**2
        jx, jy = current(psi, grid, run["config"])
        stride = max(1, grid.shape[0]//40)
        axes[row, 0].imshow(density, origin="lower",
            extent=(-grid.lx/2, grid.lx/2, -grid.ly/2, grid.ly/2), cmap="magma")
        axes[row, 0].quiver(grid.x[::stride], grid.y[::stride],
            jx[::stride, ::stride], jy[::stride, ::stride], color="cyan",
            scale=max(max_current/5, 1e-30), scale_units="xy")
        phase = np.ma.masked_where(density < 1e-3*np.max(density), np.angle(psi))
        im = axes[row, 1].imshow(phase, origin="lower",
            extent=(-grid.lx/2, grid.lx/2, -grid.ly/2, grid.ly/2),
            cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[row, 0].set(title=f"{run['config']['pump']['kind']}: плотность и ток",
                         xlabel="x, мкм", ylabel="y, мкм")
        axes[row, 1].set(title="Фаза (маска 0.1% от максимума)",
                         xlabel="x, мкм", ylabel="y, мкм")
        fig.colorbar(im, ax=axes[row, 1], shrink=0.75)
        for ax in axes[row]:
            ax.set_xlim(-40, 40)
            ax.set_ylim(-40, 40)
    fig.suptitle("F4. Плотность, ток и фаза", fontsize=14)
    fig.text(0.5, -0.035, provenance, ha="center", fontsize=8)
    files.append(_save(fig, out, "F4_current_phase"))

    def spectrum_source(source):
        run, psi, _nr, time_ps = _latest(source)
        return run, psi, time_ps
    spectrum_rows = [
        [(gauss, gp, gt)]+[spectrum_source(source)
                            for source in gaussian_spectrum_runs],
        [(ring, rp, rt)]+[spectrum_source(source)
                           for source in ring_spectrum_runs],
    ]
    columns = max(map(len, spectrum_rows))
    fig, axes = plt.subplots(2, columns, figsize=(5*columns, 8),
                             squeeze=False, layout="constrained")
    maximum = max(float(np.max(abs(spectrum(
        psi, make_grid(run["config"])))**2))
        for row in spectrum_rows for run, psi, _time in row)
    vmax = np.log10(max(maximum, 1e-20))
    vmin = vmax-8
    for row, entries in enumerate(spectrum_rows):
        for col in range(columns):
            ax = axes[row, col]
            if col >= len(entries):
                ax.axis("off")
                continue
            run, psi, time_ps = entries[col]
            grid = make_grid(run["config"])
            power = np.fft.fftshift(abs(spectrum(psi, grid))**2)
            kx, ky = np.fft.fftshift(grid.kx), np.fft.fftshift(grid.ky)
            image = ax.imshow(np.log10(np.maximum(power, 1e-30)), origin="lower",
                extent=(kx[0], kx[-1], ky[0], ky[-1]), cmap="magma",
                vmin=vmin, vmax=vmax)
            ax.set(xlabel="kx, мкм⁻¹", ylabel="ky, мкм⁻¹",
                   title=f"{'Пятно' if row == 0 else 'Кольцо'}: "
                         f"P₀={run['config']['pump']['P0']:g}, t={time_ps:g} пс\n"
                         f"N={grid.shape[0]}, L={grid.lx:g} мкм")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
    fig.colorbar(image, ax=axes, shrink=0.7, label="log₁₀(|ψ̃|² / мкм²)")
    fig.suptitle("F5. Импульсные спектры (общая цветовая шкала)", fontsize=14)
    fig.text(0.5, -0.035, "Исходные прогоны: "+", ".join(
        run["path"].name for row in spectrum_rows for run, _psi, _time in row),
        ha="center", fontsize=7)
    files.append(_save(fig, out, "F5_momentum_spectra"))
    return files


def _threshold_figure(results, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    any_data = False
    colors = {"gaussian": "tab:blue", "ring": "tab:orange"}
    for label, source in results:
        if source is None:
            continue
        data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
        bracket = data.get("threshold", data) if isinstance(data, dict) else {}
        points = data if isinstance(data, list) else bracket.get("points", [])
        color = colors[label]
        for status, marker in (("STABLE", "o"), ("UNSTABLE", "^"),
                               ("UNRESOLVED", "x")):
            selected = [p for p in points if p.get("status") == status
                        and p.get("sigma_late_per_ps") is not None]
            if selected:
                ax.errorbar([p["P0"] for p in selected],
                            [p["sigma_late_per_ps"] for p in selected],
                            yerr=[2*p.get("sigma_late_stderr", 0) for p in selected],
                            fmt=marker, color=color, capsize=2,
                            label=f"{label}: {status}")
                any_data = True
        if "P0_lo" in bracket and "P0_hi" in bracket:
            status = data.get("threshold_validation", {}).get("status", bracket["status"])
            ax.axvspan(bracket["P0_lo"], bracket["P0_hi"], color=color,
                       alpha=0.12, label=f"{label}: {status}")
    if not any_data:
        plt.close(fig)
        return None
    ax.axhline(0, color="k", lw=1)
    ax.axvline(1, color="grey", ls="--", label="0D ориентир")
    ax.set(xlabel="P₀", ylabel="Линейный рост σ, пс⁻¹",
           title="2D линейный порог (замороженный резервуар)")
    ax.legend(fontsize=7)
    return _save(fig, out, "F6_linear_threshold")


def _convergence_figure(source, out):
    if source is None:
        return None
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    comparisons = [r for r in data["comparisons"] if r["status"] == "COMPARED"]
    if not comparisons:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(comparisons))
    for key, label in (("NC_relative", "NC"), ("NR_relative", "NR"),
                       ("r_rms_um_relative", "r rms"),
                       ("density_map_relative", "карта nC")):
        if all(key in r for r in comparisons):
            ax.semilogy(x, [max(r[key], 1e-15) for r in comparisons],
                        "o-", label=label)
    ax.axhline(0.005, color="k", ls="--", lw=1, label="0.5% ориентир")
    labels = [r["comparison"].replace("_vs_", " → ").replace("_", "\n")
              for r in comparisons]
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set(xlabel="Сетка / шаг времени / область", ylabel="Относительное отличие",
           title=f"Сходимость при t={data.get('time_ps', '?')} пс; "
                 f"{data.get('reference_stationarity_status', 'статус не указан')}")
    ax.legend()
    return _save(fig, out, "F7_convergence")


def make_figures(config, *, data_dir, output_dir,
                 gaussian_run=None, ring_run=None,
                 gaussian_threshold=None, ring_threshold=None,
                 convergence=None, gaussian_spectrum_runs=(),
                 ring_spectrum_runs=()):
    """Re-export only figures supported by existing stored numerical data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    paths = []
    if (data_dir/"F1_0d_data.npz").exists() and (data_dir/"F2_0d_data.npz").exists():
        paths.extend(_figure_0d(data_dir, out))
    if gaussian_run is not None and ring_run is not None:
        paths.extend(_pair_figures(
            gaussian_run, ring_run, out,
            gaussian_spectrum_runs=gaussian_spectrum_runs,
            ring_spectrum_runs=ring_spectrum_runs))
    threshold = _threshold_figure(
        (("gaussian", gaussian_threshold), ("ring", ring_threshold)), out)
    if threshold is not None:
        paths.append(threshold)
    convergence_plot = _convergence_figure(convergence, out)
    if convergence_plot is not None:
        paths.append(convergence_plot)
    return paths


def write_report_summary(output_path, *, gaussian_run=None, ring_run=None,
                         gaussian_threshold=None, ring_threshold=None,
                         convergence=None, domain_comparisons=None,
                         validation_report=None):
    """Build a factual PASS/WARN/FAIL/NOT_TESTED table from saved results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    times = []
    domain_comparisons = domain_comparisons or {}
    if validation_report is None:
        rows.append(("0D и точные случаи FFT", "NOT_TESTED", "отчёт не указан"))
    else:
        data = (validation_report if isinstance(validation_report, dict) else
                json.loads(Path(validation_report).read_text()))
        rows.append(("0D и точные случаи FFT",
                     "PASS" if data.get("PASS") else "FAIL",
                     f"0D/2D={data.get('uniform_2d_vs_independent_0d_max_abs_error', float('nan')):.3g}, "
                     f"Parseval={data.get('parseval_relative_error', float('nan')):.3g}"))
    for label, source in (("Пятно", gaussian_run), ("Кольцо", ring_run)):
        if source is None:
            for check in ("завершение", "стационарность", "баланс", "граница", "высокие k"):
                rows.append((label+": "+check, "NOT_TESTED", "прогон не указан"))
            continue
        run = _run(source)
        summary = run["summary"]
        checks = run["config"]["checks"]
        status = summary["run_status"]
        rows.append((label+": завершение",
                     "PASS" if status == "RUN_COMPLETED" else
                     "FAIL" if status == "RUN_FAILED" else "WARN", status))
        mode = summary["stationarity"]["stationarity_status"]
        rows.append((label+": стационарность",
                     "PASS" if mode == "STATIONARY_MODE" else
                     "NOT_TESTED" if mode == "NOT_TESTED" else "WARN", mode))
        balance = summary["balance_relative_residual"]
        rows.append((label+": баланс",
                     "PASS" if balance <= checks["balance_warn_rel"] else
                     "FAIL" if balance > 10*checks["balance_warn_rel"] else "WARN",
                     f"{balance:.3g}"))
        edge = summary["spatial_diagnostics"].get("edge_mass_fraction")
        edge_peak = summary["spatial_diagnostics"].get("edge_peak_ratio")
        rows.append((label+": край области",
                     "NOT_TESTED" if edge is None or edge_peak is None else
                     "PASS" if edge <= checks["edge_mass_warn"] and
                     edge_peak <= checks["edge_peak_warn"] else "WARN",
                     "n/a" if edge is None or edge_peak is None else
                     f"mass={edge:.3g}, peak={edge_peak:.3g}"))
        high_k = summary["spatial_diagnostics"].get("high_k_fraction")
        rows.append((label+": высокие k",
                     "NOT_TESTED" if high_k is None else
                     "PASS" if high_k <= checks["high_k_warn"] else "WARN",
                     "n/a" if high_k is None else f"{high_k:.3g}"))
        comparison = domain_comparisons.get(label)
        rows.append((label+": размер области",
                     "NOT_TESTED" if comparison is None else
                     "PASS" if comparison.get("status") == "COMPARED" and
                     comparison.get("NC_relative", 1) < 0.005 and
                     comparison.get("core_density_map_relative", 1) < 0.01
                     else "WARN",
                     "n/a" if comparison is None else
                     f"ΔNC={comparison.get('NC_relative', float('nan')):.3g}, "
                     f"ΔnC(core)={comparison.get('core_density_map_relative', float('nan')):.3g}"))
        times.append((run["path"].name, summary["time_ps"],
                      summary["elapsed_wall_s"], status))
    for label, source in (("Пятно", gaussian_threshold), ("Кольцо", ring_threshold)):
        if source is None:
            rows.append((label+": линейный порог", "NOT_TESTED", "нет данных"))
            rows.append((label+": нелинейное подтверждение", "NOT_TESTED", "нет данных"))
            continue
        data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
        threshold = data.get("threshold", data)
        validation = data.get("threshold_validation", {})
        nonlinear = data.get("nonlinear_confirmation", {})
        state = validation.get("status", threshold.get("status", "NOT_TESTED"))
        rows.append((label+": линейный порог",
                     "PASS" if state == "GRID_DOMAIN_CONFIRMED" else "WARN", state))
        nonlinear_state = nonlinear.get("status", "NOT_TESTED")
        rows.append((label+": нелинейное подтверждение",
                     "PASS" if nonlinear_state == "SUPPORTED" else
                     "NOT_TESTED" if nonlinear_state == "NOT_TESTED" else "WARN",
                     nonlinear_state))
    if convergence is None:
        rows.append(("Сходимость", "NOT_TESTED", "серия не указана"))
    else:
        data = convergence if isinstance(convergence, dict) else json.loads(
            Path(convergence).read_text())
        rows.append(("Сходимость", data.get("assessment", "NOT_TESTED"),
                     f"{data.get('status', 'NOT_TESTED')}, "
                     f"t={data.get('time_ps', '?')} пс, "
                     f"{data.get('reference_stationarity_status', 'статус не указан')}"))
        rows.append(("Сходимость конечного режима",
                     data.get("final_state_assessment", "NOT_TESTED"),
                     "проверяется на установившемся состоянии"))
        seen = {run_id for run_id, _time_ps, _wall, _state in times}
        for path in data.get("cases", {}).values():
            run = _run(path)
            if run["path"].name not in seen and run["summary"] is not None:
                summary = run["summary"]
                times.append((run["path"].name, summary["time_ps"],
                              summary["elapsed_wall_s"], summary["run_status"]))
                seen.add(run["path"].name)
    lines = ["# Проверки проекта", "",
             "| Проверка | Статус | Фактическое значение |", "|---|---|---|"]
    lines.extend(f"| {check} | {state} | {value} |" for check, state, value in rows)
    lines.extend(["", "## Время физических расчётов", "",
                  "| run_id | Время модели, пс | Реальное время, с | Статус |",
                  "|---|---:|---:|---|"])
    lines.extend(f"| {run_id} | {time_ps:g} | {wall:.1f} | {state} |"
                 for run_id, time_ps, wall, state in times)
    output_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return output_path
