"""Static scientific figures made from saved numerical arrays."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numerics import current, spectrum
from model import (make_grid, make_pump, parameters, stationary_0d,
                   jacobian_0d, solve_0d_reference, stuart_landau,
                   condensed_eigenvalues_0d, reduced_stationary_0d, solve_0d_reduced)
from numerics import load_run


def _run(run):
    return load_run(run) if not isinstance(run, dict) else run


def _latest(run):
    run = _run(run)
    if "psi" in run:
        return run, run["psi"], run["nR"], float(run["time_ps"])
    if not run["snapshots"]:
        raise FileNotFoundError(f"No snapshots in {run['path']}")
    with np.load(run["snapshots"][-1]) as values:
        return run, values["psi"].copy(), values["nR"].copy(), float(values["time_ps"])


# Match the compact typography used in dispersion_curve figures.
REPORT_STYLE = {
    "figure.dpi": 120, "font.size": 9, "font.family": "DejaVu Sans",
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.linewidth": 0.8,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.h_pad": 0.04,
}


REPORT_WIDTH_IN = 160 / 25.4


def _save(fig, out, name):
    """PNG at exactly 160 mm width; preserve the font sizes at report scale."""
    from matplotlib.text import Text

    fig.set_size_inches(REPORT_WIDTH_IN, fig.get_size_inches()[1])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend([ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for tick in axis.get_major_ticks():
                if min(limits) <= tick.get_loc() <= max(limits):
                    texts.extend([tick.label1, tick.label2])
            texts.append(axis.get_offset_text())
    for legend in [*fig.legends, *(ax.get_legend() for ax in fig.axes)]:
        if legend is not None:
            texts.extend(legend.findobj(Text))
    for artist in texts:
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1
                or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1):
            raise ValueError(f"Clipped figure text in {name}: {artist.get_text()}")
    visible_boxes = [(artist, artist.get_window_extent(renderer)) for artist in texts
                     if artist.get_visible() and artist.get_text()]
    for i, (first, a) in enumerate(visible_boxes):
        for second, b in visible_boxes[i+1:]:
            if first is second:
                continue
            if min(a.x1,b.x1)-max(a.x0,b.x0)>2 and min(a.y1,b.y1)-max(a.y0,b.y0)>2:
                raise ValueError(f"Overlapping figure text in {name}: {first.get_text()} / {second.get_text()}")
    path = out / f"{name}.png"
    fig.savefig(path, dpi=300, facecolor="white", pil_kwargs={"optimize": True})
    plt.close(fig)
    return path


def _extent(grid):
    # The FFT grid coordinates are pixel centres, including x = -L/2.
    return (grid.x[0]-grid.dx/2, grid.x[-1]+grid.dx/2,
            grid.y[0]-grid.dy/2, grid.y[-1]+grid.dy/2)


def _figure_0d(config, out):
    """Analytic 0D branches; no saved legacy image is needed."""
    # The branch is independent of the space/time settings in the supplied config.
    p = parameters(config)
    powers = np.linspace(0.5, 3.0, 151)
    densities, growth = [], []
    for power in powers:
        branches = stationary_0d(power * p["Pth0D"], config)
        densities.append(branches["condensed"] or branches["empty"])
        growth.append(np.max(np.linalg.eigvals(jacobian_0d(*branches["empty"], config)).real))
    nc, nr = np.array(densities).T
    fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH_IN, 3.05), layout="constrained")
    axes[0].plot(powers, nc, label="Condensate")
    axes[0].plot(powers, nr, label="Reservoir")
    axes[0].set(ylabel=r"Density ($\mu$m$^{-2}$)")
    axes[0].set_title("(a) Stationary densities", loc="left")
    axes[0].legend()
    axes[1].plot(powers, growth, color="C0")
    axes[1].axhline(0, color="0.35", lw=1)
    axes[1].set(ylabel=r"Largest $\mathrm{Re}\,\lambda$ (ps$^{-1}$)")
    axes[1].set_title("(b) Condensate-free state\nstability", loc="left")
    for ax in axes:
        ax.axvline(1, color="0.4", ls="--", lw=1)
        ax.set_xlabel(r"Normalized pump, $s=P/P_{\mathrm{th}}$")
    return _save(fig, out, Path(STAGE2_IMAGES[1]).stem)


def _pair_figures(gaussian_run, ring_run, out, *,
                  gaussian_spectrum_runs=(), figure_numbers=(5,6,7)):
    pairs = (_latest(gaussian_run), _latest(ring_run))
    arrays = []
    for run, psi, nr, _time in pairs:
        grid = make_grid(run["config"])
        p = parameters(run["config"])
        pump = make_pump(run["config"], grid)
        bg = p["eta"]*pump if p["background_blueshift"] else 0
        arrays.append((pump, nr, abs(psi)**2, p["gC"]*abs(psi)**2+p["gR"]*(nr+bg)))

    def map_axes(ax, title, *, bottom, left, limit=40):
        ax.grid(False)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_xticks([-limit, 0, limit]); ax.set_yticks([-limit, 0, limit])
        ax.set_title(title, loc="left")
        if bottom:
            ax.set_xlabel(r"$x$ ($\mu$m)")
        else:
            ax.tick_params(labelbottom=False)
        if left:
            ax.set_ylabel(r"$y$ ($\mu$m)")
        else:
            ax.tick_params(labelleft=False)

    files=[]
    if 5 in figure_numbers:
        # Two geometry rows, four observables; zoom makes the pump cores readable.
        fig = plt.figure(figsize=(REPORT_WIDTH_IN, 4.0), layout="constrained")
        gs = fig.add_gridspec(3, 4, height_ratios=(1, 1, 0.09), wspace=0.04)
        titles = ("Pump", "Reservoir", "Condensate", "Mean-field\nenergy shift")
        units = (r"$P$ ($\mu$m$^{-2}$ ps$^{-1}$)",
                 r"$n_R$ ($\mu$m$^{-2}$)", r"$n_C$ ($\mu$m$^{-2}$)", r"$V_{\mathrm{int}}$ (meV)")
        for col, (title, unit) in enumerate(zip(titles, units)):
            vmax = max(float(np.max(a[col])) for a in arrays)
            for row, geometry in enumerate(("Spot", "Ring")):
                ax = fig.add_subplot(gs[row, col])
                grid = make_grid(pairs[row][0]["config"])
                im = ax.imshow(arrays[row][col], origin="lower", extent=_extent(grid),
                               vmin=0, vmax=vmax, cmap="viridis", interpolation="nearest")
                letter = f"({chr(97+row*4+col)})"
                map_axes(ax, letter+" "+title if row==0 else letter,
                         bottom=row==1, left=col==0, limit=20)
                if col==0:
                    ax.set_ylabel(geometry+f" ({pairs[row][3]:g} ps)\n"+r"$y$ ($\mu$m)")
                ax.title.set_fontsize(9); ax._left_title.set_fontsize(9)
            cax = fig.add_subplot(gs[2, col]); cax.grid(False)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_ticks([0, vmax/2, vmax])
            from matplotlib.ticker import FuncFormatter
            cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.2g}"))
            cb.set_label(unit, fontsize=8)
        files.append(_save(fig, out, Path(STAGE2_IMAGES[5]).stem))

    if 6 in figure_numbers:
        fig = plt.figure(figsize=(REPORT_WIDTH_IN, 7.0), layout="constrained")
        gs = fig.add_gridspec(4, 2, height_ratios=(1, 1, 0.09, 0.07))
        max_density = max(float(np.max(abs(psi)**2)) for _, psi, _, _ in pairs)
        max_current = max(float(np.max(np.hypot(*current(
            psi, make_grid(run["config"]), run["config"])))) for run, psi, _, _ in pairs)
        from numerics import vortex_diagnostics
        ring_check=vortex_diagnostics(pairs[1][1],make_grid(pairs[1][0]['config']),pairs[1][0]['config'])
        phase_cmap = plt.get_cmap("twilight").copy(); phase_cmap.set_bad("0.94")
        arrow_reference = 0.5 if max_current >= 0.5 else max_current/2
        for row, (run, psi, _nr, _time) in enumerate(pairs):
            grid = make_grid(run["config"]); density = abs(psi)**2
            jx, jy = current(psi, grid, run["config"])
            density_ax, phase_ax = [fig.add_subplot(gs[row, col]) for col in range(2)]
            density_im = density_ax.imshow(density, origin="lower", extent=_extent(grid),
                cmap="magma", vmin=0, vmax=max_density, interpolation="nearest")
            stride = max(1, round(5.0/grid.dx))
            ix = np.flatnonzero(abs(grid.x) <= 37)[::stride]
            iy = np.flatnonzero(abs(grid.y) <= 37)[::stride]
            sampled_jx, sampled_jy = jx[np.ix_(iy, ix)], jy[np.ix_(iy, ix)]
            weak = np.hypot(sampled_jx, sampled_jy) < 0.05*max_current
            vectors = density_ax.quiver(grid.x[ix], grid.y[iy],
                np.ma.masked_where(weak, sampled_jx), np.ma.masked_where(weak, sampled_jy),
                color="white", edgecolors="black", linewidths=0.5,
                scale=max(max_current/5, 1e-30), scale_units="xy", angles="xy",
                width=0.007, headwidth=3.5, minlength=0)
            phase = np.ma.masked_where(density < 1e-3*np.max(density), np.angle(psi))
            phase_im = phase_ax.imshow(phase, origin="lower", extent=_extent(grid),
                cmap=phase_cmap, vmin=-np.pi, vmax=np.pi, interpolation="nearest")
            geometry = "Spot" if row==0 else "Ring"
            map_axes(density_ax, f"({chr(97+row*2)}) {geometry}: current", bottom=row==1, left=True)
            charge=f" ($q={ring_check['q']}$)" if row==1 and ring_check['q'] is not None else ''
            map_axes(phase_ax, f"({chr(98+row*2)}) {geometry}: phase"+charge, bottom=row==1, left=False)
            if row==1:
                from matplotlib.patches import Circle
                for radius in (6.,9.):
                    phase_ax.add_patch(Circle(run['config']['pump']['center_um'],radius,
                        fill=False,edgecolor='white',linestyle='--',linewidth=.65,alpha=.8))
        key_ax = fig.add_subplot(gs[2, :]); key_ax.set_axis_off()
        fig.canvas.draw()
        arrow_um = 5*arrow_reference/max(max_current, 1e-30)
        arrow_pixels = density_ax.transData.transform((arrow_um, 0))[0]-density_ax.transData.transform((0, 0))[0]
        arrow_fraction = arrow_pixels/key_ax.bbox.width
        key_ax.annotate("", xy=(0.28+arrow_fraction, 0.5), xytext=(0.28, 0.5),
            xycoords="axes fraction", arrowprops={"arrowstyle":"-|>", "color":"black", "lw":1.2})
        key_ax.text(0.30+arrow_fraction, 0.5,
            rf"Current: ${arrow_reference:g}\;\mu\mathrm{{m}}^{{-1}}\mathrm{{ps}}^{{-1}}$",
            transform=key_ax.transAxes, va="center", fontsize=8)
        for col, (im, label) in enumerate(((density_im, r"$n_C$ ($\mu$m$^{-2}$)"),
                                         (phase_im, r"Phase, $\arg\psi$ (rad)"))):
            cax = fig.add_subplot(gs[3, col]); cax.grid(False)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label(label)
            if col==1:
                cb.set_ticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
            else:
                cb.ax.locator_params(nbins=4)
        files.append(_save(fig, out, Path(STAGE2_IMAGES[6]).stem))

    if 7 in figure_numbers:
        def spectrum_source(source):
            run, psi, _nr, time_ps = _latest(source)
            return run, psi, time_ps
        # Compare spot domains at the same time; the ring uses its final state.
        # A ring spectrum from 1000 ps cannot be compared with one from 2000 ps.
        entries=[('Spot',pairs[0][0],pairs[0][1],pairs[0][3])]
        entries += [('Spot',*spectrum_source(p)) for p in gaussian_spectrum_runs]
        entries += [('Ring',pairs[1][0],pairs[1][1],pairs[1][3])]
        fig=plt.figure(figsize=(REPORT_WIDTH_IN,3.5),layout='constrained')
        gs=fig.add_gridspec(2,len(entries),height_ratios=(1,.08))
        maximum=max(float(np.max(abs(spectrum(psi,make_grid(run['config'])))**2))
                    for _,run,psi,_ in entries)
        vmax=np.log10(max(maximum,1e-20));vmin=vmax-8
        for col,(geometry,run,psi,time_ps) in enumerate(entries):
            ax=fig.add_subplot(gs[0,col]);grid=make_grid(run['config'])
            power=np.fft.fftshift(abs(spectrum(psi,grid))**2)
            kx,ky=np.fft.fftshift(grid.kx),np.fft.fftshift(grid.ky)
            dkx,dky=kx[1]-kx[0],ky[1]-ky[0]
            im=ax.imshow(np.log10(np.maximum(power,1e-30)),origin='lower',
                extent=(kx[0]-dkx/2,kx[-1]+dkx/2,ky[0]-dky/2,ky[-1]+dky/2),
                cmap='magma',vmin=vmin,vmax=vmax,interpolation='nearest')
            ax.set_title(f"({chr(97+col)}) {geometry}, {time_ps:g} ps\n"+rf"$L={grid.lx:g}\,\mu$m",loc='left',fontsize=9)
            ax.set(xlim=(-4,4),ylim=(-4,4),aspect='equal',xlabel=r'$k_x$ ($\mu$m$^{-1}$)')
            ax.set_xticks([-4,0,4]);ax.set_yticks([-4,0,4]);ax.grid(False)
            if col==0:ax.set_ylabel(r'$k_y$ ($\mu$m$^{-1}$)')
            else:ax.tick_params(labelleft=False)
        cax=fig.add_subplot(gs[1,:]);cax.grid(False)
        cb=fig.colorbar(im,cax=cax,orientation='horizontal')
        cb.set_label(r'$\log_{10}[I_k/(1\,\mu\mathrm{m}^2)]$')
        files.append(_save(fig,out,Path(STAGE2_IMAGES[7]).stem))

    return files


def write_report_summary(output_path, *, gaussian_run=None, ring_run=None,
                         gaussian_threshold=None, ring_threshold=None,
                         convergence=None, domain_comparisons=None,
                         final_state_checks=None, validation_report=None, practical_checks=None):
    """Build a factual PASS/WARN/FAIL/NOT_TESTED table from saved results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    times = []
    domain_comparisons = domain_comparisons or {}
    if final_state_checks is not None and not isinstance(final_state_checks, dict):
        final_state_checks = json.loads(Path(final_state_checks).read_text())
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
        extension = data.get("additional_scan")
        if extension:
            points = [p for p in threshold["points"] if p.get("scan_extension_date") == extension["date"]]
            reliable = all(p["status"] in ("STABLE", "UNSTABLE") for p in points)
            modes = sum(p.get("mode_status") == "CONVERGED" for p in points)
            rows.append((label+": дополнительные точки рис. 08",
                         "PASS" if reliable and len(points) == len(extension["requested_P0"]) else "WARN",
                         f"{len(points)} точек, {extension['wall_s']:.1f} с; "
                         f"знак роста проверен, моды установились в {modes}/{len(points)} точек"))
        mode_check = data.get("mode_validation")
        if mode_check:
            tests = mode_check.get("tests")
            if tests and label == "Пятно":
                rows.append(("Тесты после проверки линейных мод",
                             "PASS" if tests["returncode"] == 0 else "FAIL", tests["result"]))
            cases = mode_check["cases"]
            settled = sum(case["status"] == "SETTLED" for case in cases.values())
            details = "; ".join(f"P0={power}: {case['duration_ps']:g} пс" for power, case in cases.items())
            rows.append((label+": установление линейных мод",
                         "PASS" if mode_check["status"] == "SETTLED" else "WARN",
                         f"{settled}/{len(cases)}; {details}; {mode_check.get('wall_s', 0):.1f} с"))
        nonlinear_state = nonlinear.get("status", "NOT_TESTED")
        rows.append((label+": нелинейное подтверждение",
                     "PASS" if nonlinear_state == "SUPPORTED" else
                     "NOT_TESTED" if nonlinear_state == "NOT_TESTED" else "WARN",
                     nonlinear_state))
        persistence = nonlinear.get("above_persistence", {})
        persisted = persistence.get("status", "NOT_TESTED")
        rows.append((label+": долгоживущий сигнал",
                     "PASS" if persisted == "PERSISTENT" else
                     "NOT_TESTED" if persisted == "NOT_TESTED" else "WARN",
                     "n/a" if persistence.get("minimum_NC") is None else
                     f"min NC={persistence['minimum_NC']:.3g} за "
                     f"{persistence['window_ps']:g} пс"))
    if convergence is None:
        rows.append(("Сходимость", "NOT_TESTED", "серия не указана"))
    else:
        data = convergence if isinstance(convergence, dict) else json.loads(
            Path(convergence).read_text())
        tests = data.get("tests")
        if tests is not None:
            rows.append(("Автоматические тесты серии сходимости", "PASS" if tests["returncode"] == 0 else "FAIL",
                         tests["output"].strip().splitlines()[-1]))
        rows.append(("Сходимость", data.get("assessment", "NOT_TESTED"),
                     f"{data.get('status', 'NOT_TESTED')}, "
                     f"t={data.get('time_ps', '?')} пс, "
                     f"{data.get('reference_stationarity_status', 'статус не указан')}"))
        rows.append(("Полная серия сходимости конечного режима",
                     data.get("final_state_assessment", "NOT_TESTED"),
                     "три сетки, три шага и два размера области; проверено установление всех вариантов"))
        for comparison in data.get("comparisons", []):
            if comparison.get("status") != "COMPARED":
                rows.append((comparison["comparison"], "NOT_TESTED", "нет завершённой пары"))
                continue
            errors = [comparison.get(k, 0) for k in ("NC_relative", "NR_relative", "mean_r2_relative",
                                                    "current_magnitude_integral_relative")]
            passed = (max(errors) < 0.005 and comparison.get("core_density_map_relative", 0) < 0.01
                      and comparison.get("last_100ps_NC_relative_rmse", 0) < 0.005)
            rows.append((comparison["comparison"], "PASS" if passed else "WARN",
                         f"ΔNC={comparison['NC_relative']:.3g}, ΔNR={comparison['NR_relative']:.3g}, "
                         f"ΔnC(core)={comparison['core_density_map_relative']:.3g}, "
                         f"RMSE NC(100 пс)={comparison.get('last_100ps_NC_relative_rmse', float('nan')):.3g}"))
        order = data.get("observed_time_order", {}).get("density_map_relative")
        if order is not None:
            rows.append(("Порядок временной сходимости плотности",
                         "PASS" if 1.7 <= order <= 2.3 else "WARN", f"p={order:.3f}; Strang: p≈2"))
        balance = data.get("balance_by_time_step", {})
        steps = data.get("protocol", {}).get("time_steps_ps", [])
        if len(steps) == 3 and all(str(dt) in balance for dt in steps):
            values = [balance[str(dt)] for dt in steps]
            rows.append(("Баланс при уменьшении шага", "PASS" if values[0] > values[1] > values[2]
                         and values[2] <= 1e-6 else "WARN",
                         " → ".join(f"{v:.3g}" for v in values)))
        seen = {run_id for run_id, _time_ps, _wall, _state in times}
        for path in data.get("cases", {}).values():
            run = _run(path)
            if run["path"].name not in seen and run["summary"] is not None:
                summary = run["summary"]
                times.append((run["path"].name, summary["time_ps"],
                              summary["elapsed_wall_s"], summary["run_status"]))
                seen.add(run["path"].name)
    for name, comparison in (final_state_checks or {}).items():
        if comparison.get("status") != "COMPARED":
            rows.append(("Пятно: " + name, "NOT_TESTED", "сравнение не завершено"))
            continue
        reference = _run(comparison["reference"])
        candidate = _run(comparison["candidate"])
        settled = all(run["summary"]["stationarity"]["stationarity_status"]
                      == "STATIONARY_MODE" for run in (reference, candidate))
        nc_error = comparison["NC_relative"]
        density_error = comparison["core_density_map_relative"]
        rows.append(("Пятно: " + name,
                     "PASS" if settled and nc_error < 0.005 and density_error < 0.01
                     else "WARN",
                     f"t={candidate['summary']['time_ps']:g} пс, "
                     f"ΔNC={nc_error:.3g}, ΔnC(core)={density_error:.3g}"))
        seen = {run_id for run_id, _time_ps, _wall, _state in times}
        for run in (reference, candidate):
            if run["path"].name not in seen:
                summary = run["summary"]
                times.append((run["path"].name, summary["time_ps"],
                              summary["elapsed_wall_s"], summary["run_status"]))
                seen.add(run["path"].name)
    if practical_checks is not None:
        practical = (practical_checks if isinstance(practical_checks, dict) else
                     json.loads(Path(practical_checks).read_text()))
        for label, item in practical.get("cases", {}).items():
            comparison = item.get("comparison", {})
            state = ("PASS" if comparison.get("status") == "COMPARED"
                     and comparison.get("NC_relative", 1) < 0.005
                     and comparison.get("core_density_map_relative", 1) < 0.01
                     and comparison.get("last_100ps_NC_relative_rmse", 0) < 0.005
                     else "WARN" if comparison else "NOT_TESTED")
            rows.append((label + " (дополнительная проверка)", state,
                         f"t={item.get('time_ps', '?')} пс, "
                         f"ΔNC={comparison.get('NC_relative', float('nan')):.3g}, "
                         f"ΔnC(core)={comparison.get('core_density_map_relative', float('nan')):.3g}, "
                         f"RMSE NC(100 пс)={comparison.get('last_100ps_NC_relative_rmse', float('nan')):.3g}"))
            if item.get("wall_seconds") is not None:
                checked_run = _run(item["run_path"])
                spatial = checked_run["summary"]["spatial_diagnostics"]
                limits = checked_run["config"]["checks"]
                edge_warning = (spatial.get("edge_mass_fraction", 0) > limits["edge_mass_warn"]
                                or spatial.get("edge_peak_ratio", 0) > limits["edge_peak_warn"])
                if edge_warning:
                    rows.append((label + ": край области", "WARN",
                                 f"mass={spatial['edge_mass_fraction']:.3g}, "
                                 f"peak={spatial['edge_peak_ratio']:.3g}"))
                if item.get("assessment") == "WARN" and state == "PASS" and not edge_warning:
                    rows.append((label + ": общий контроль", "WARN",
                                 "уточнение прошло, есть ограничения области/баланса/спектра"))
                times.append((Path(item["run_path"]).name, item["time_ps"],
                              item["wall_seconds"], item["run_status"]))
    ring_check_file = output_path.parent.parent/'analysis_data/final_ring_checks.json'
    if ring_check_file.exists():
        ring_checks = json.loads(ring_check_file.read_text())
        rows.append(("Конечный вихрь кольца: сетка и шаг при 2000 пс",
                     ring_checks.get("assessment", "NOT_TESTED"), ring_checks["status"]))
        for name, item in ring_checks.get("comparisons", {}).items():
            label = "Кольцо 2000 пс: " + ("сетка 256→512" if name == "spatial" else "шаг 0.05→0.025 пс")
            if "checks" not in item:
                rows.append((label, "NOT_TESTED", item.get("reason", "расчёт ещё не завершён")))
                continue
            for key, text, value in (
                ("populations", "населённости", f"ΔNC={item['NC_relative']:.3g}, ΔNR={item['NR_relative']:.3g}"),
                ("density_profile", "профиль плотности", f"L2={item['density_map_relative']:.3g}"),
                ("charge", "заряд", f"q={item['vortex']['q']}, корректных общих контуров: {len(item['common_valid_contours_um'])}"),
                ("angular_momentum", "угловой момент", f"Lz/(ħNC)={item['vortex']['Lz_over_hbar_NC']:.9g}, Δ={item['Lz_relative']:.3g}"),
                ("stationarity", "стационарность", item['stationarity']['stationarity_status'])):
                rows.append((label+": "+text, "PASS" if item['checks'][key] else "FAIL", value))
            times.append((Path(item['candidate']).name, 2000., item['elapsed_wall_s'], "RUN_COMPLETED"))
        tests = ring_checks.get("tests")
        if tests:
            rows.append(("Тесты перед проверкой конечного вихря", "PASS" if tests['returncode']==0 else "FAIL", tests['result']))
    lines = ["# Проверки проекта", "",
             "| Проверка | Статус | Фактическое значение |", "|---|---|---|"]
    lines.extend(f"| {check} | {state} | {value} |" for check, state, value in rows)
    lines.extend(["", "## Время физических расчётов", "",
                  "| run_id | Время модели, пс | Реальное время, с | Статус |",
                  "|---|---:|---:|---|"])
    lines.extend(f"| {run_id} | {time_ps:g} | {wall:.1f} | {state} |"
                 for run_id, time_ps, wall, state in times)
    experiment=output_path.parent.parent/'analysis_data/ring_dynamics.json'
    if experiment.exists():
        report=json.loads(experiment.read_text())
        if report.get('time_ps'):
            lines.extend(_ring_dynamics_summary_lines(report))
    output_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return output_path



def _ring_dynamics_summary_lines(report):
    """Research observations, kept separate from the archived 1000-ps source."""
    first=report.get('first_verified_stationary_window_ps')
    last=report['windows'][-1]
    period=report['last_500ps_periodicity']
    lines=['', '## Дополнительный эксперимент: динамика кольца', '',
           f"Продолжение 1000→{report['time_ps']:g} пс из сохранённого полного состояния; "
           '512², L=256 мкм, Δx=0.5 мкм, Δt=0.05 пс. Затравка не менялась. '
           'Исходный прогон при 1000 пс сохранён; панели кольца на рисунках 05–07 используют 2000 пс.', '',
           '| Проверка | Статус | Результат |', '|---|---|---|',
           f"| Продолжение | {'PASS' if report['status']=='RUN_COMPLETED' else 'WARN'} | "
           f"{report['status']}, t={report['time_ps']:g} пс |",
           f"| Установление | {'PASS' if report['final_stationarity']['stationarity_status']=='STATIONARY_MODE' else 'WARN'} | "
           f"{report['final_stationarity']['stationarity_status']}; первое полное окно {first} пс |",
           f"| Периодичность плотности, последние 500 пс | INFO | {period['status']}; устойчивый период не подтверждён |"]
    tests=report.get('verification',{}).get('tests')
    if tests:
        lines.append(f"| Тесты после добавления эксперимента | {'PASS' if tests['returncode']==0 else 'FAIL'} | {tests['output'].strip().splitlines()[-1]} |")
    if report.get('verification',{}).get('source_checkpoint_unchanged'):
        lines.append('| Исходная контрольная точка | PASS | SHA256 совпадает с сохранённым до продолжения |')
    diagnostics=report.get('verification',{}).get('final_diagnostics',{})
    for label,metric in diagnostics.items():
        lines.append(f"| Продолжение: {label} | {metric['status']} | {metric['value']} |")
    lines += ['', 'Размах = (max−min)/mean. Форма: s=nC/∫core nC dA; '
              'максимум ||s(t)−s(t₀)||₂/||s(t₀)||₂ за окно; последний столбец формы '
              'сравнивает все пары кадров через 20 пс. Значения ниже в процентах. '
              'Центральные 80×80 мкм содержат не менее '
              f"{100*min(w['core_mass_fraction'] for w in report['windows']):.3f}% NC.", '',
              '| Окно, пс | Размах NC, % | Размах NR, % | Размах NC+NR, % | Изменение формы, % | Через 20 пс, % | Режим | Периодичность |',
              '|---|---:|---:|---:|---:|---:|---|---|']
    for w in report['windows']:
        start,end=w['window_ps']
        numbers=' | '.join(f"{100*w[k]:.6g}" for k in
                         ('NC_span_rel','NR_span_rel','N_total_span_rel','shape_change_rel','shape_frame_20ps_change_max'))
        lines.append(f"| {start:g}–{end:g} | {numbers} | {w['stationarity']['stationarity_status']} | {w['periodicity']['status']} |")
    early=report.get('initial_peak_motion',[])
    if early:
        speed=np.median([p['angular_speed_rad_per_ps'] for p in early])
        displacement=[abs(p['angular_displacement_rad'])*180/np.pi for p in early]
        lines += ['', f"1000–1200 пс: {len(early)} отслеженных пиков, медианная угловая скорость "
                  f"{speed:.4g} рад/пс; перемещение по углу {min(displacement):.1f}…{max(displacement):.1f}°. "
                  'Отрицательный угол означает движение по часовой стрелке при обычных осях x,y.']
    modulation=report.get('angular_modulation')
    if modulation:
        lines += ['', f"Угловая компонента m={modulation['dominant_m']}: "
                  f"Aₘ=2|∫nC exp(imθ)dA|/∫nC dA уменьшается с {100*modulation['at_source_1000_ps']:.3g}% "
                  f"при 1000 пс до {100*modulation['initial']:.3g}% при {modulation['window_ps'][0]:g} пс "
                  f"и {100*modulation['final']:.3g}% при {modulation['window_ps'][1]:g} пс. "
                  'Максимумы переходной структуры движутся; при исчезающем контрасте их '
                  'позиции на сетке не служат доказательством устойчивого вращения.']
    lines += ['', 'IRREGULAR_OR_TOO_SLOW: согласованный период не найден в доступном окне '
              '(два цикла в 100 пс, три в 500 пс). NO_SIGNIFICANT_DENSITY_OSCILLATION: '
              'отклонение плотности от среднего <0.5%, размахи NC/NR <0.5%; '
              'это проверенный малый сигнал. Для исходных 500–1000 пс карты сохранены лишь '
              'через 20 пс: они показывают движение, но не дают надёжной проверки периода. '
              'Это ограничение старых данных, не пропущенная проверка продолжения.', '',
              f"Реальное время продолжения: {report['extension_wall_s']:.1f} с "
              f"({report['extension_wall_s']/60:.2f} мин). Прогон: `{report['run_dir']}`. "
              'Вывод по наблюдаемому интервалу: затухающая переходная динамика с выходом '
              'на стационарный режим в заданных допусках; устойчивый цикл не подтверждён. '
              'Это вывод для данной затравки и параметров.', '',
              'Массивы: `results/analysis_data/ring_dynamics.npz`; все окна и методы диагностики — '
              '`ring_dynamics.json`. Рисунок 09: населённости и карты при 1000/1200/1500/2000 пс.']
    vortex=report.get('vortex_diagnostics')
    if vortex:
        lines += ['', '## Циркуляция кольца при 2000 пс', '',
                  f"**{vortex['classification']}**: q={vortex['q']}; "
                  f"Lz/(ħ NC)={vortex['Lz_over_hbar_NC']:.9f}; "
                  f"Lz={vortex['Lz_mev_ps']:.6f} мэВ·пс; NC={vortex['NC']:.6f}.", '',
                  '| Радиус, мкм | min nC / max nC | q по фазе | q по току | Статус |',
                  '|---:|---:|---:|---:|---|']
        for c in vortex['contours']:
            lines.append(f"| {c['radius_um']:g} | {c['minimum_density_rel']:.4f} | {c['q']} | {c['charge_from_current']:.6f} | {c['status']} |")
        lines += ['', 'Обход против часовой стрелки; отрицательный заряд соответствует циркуляции по часовой стрелке. '
                  'Контуры проходят через высокую плотность (порог 5% максимума); число угловых отсчётов '
                  '512/1024/2048 даёт одинаковый заряд. Lz вычислен по всему полю FFT-производными; '
                  f"Im Lz/(ħ NC)={vortex['Lz_imag_over_hbar_NC']:.3g}. "
                  'Таким образом, затухает угловая модуляция плотности, а вихревая циркуляция сохраняется. '
                  'Осесимметричная плотность не означает отсутствие потока.']
    return lines



STAGE2_IMAGES = {
    1: "01_homogeneous_threshold.png",
    2: "02_condensed_state_stability.png",
    3: "03_matched_initial_dynamics.png",
    4: "04_reduced_model_accuracy.png",
    5: "05_spatial_profiles.png",
    6: "06_particle_flow_and_phase.png",
    7: "07_momentum_distributions.png",
    8: "08_localized_condensation_thresholds.png",
    9: "09_ring_transient_evolution.png",
}

REPORT_TITLES = {
    1: "Stationary densities and condensate-free state stability",
    2: "Condensed-state stability and relaxation",
    3: "Homogeneous condensate–reservoir dynamics",
    4: "Accuracy of reduced models",
    5: "Spatial profiles under localized pumping",
    6: "Particle current and condensate phase",
    7: "Momentum distributions and spot domain comparison",
    8: "Linear instability thresholds of localized pumps",
    9: "Transient evolution of the ring condensate",
}


def build_stage2_0d_data(config, directory, *, t_final_ps=1600.0, sample_dt_ps=0.5):
    """Save all stage-2 homogeneous data and metrics in one reusable NPZ."""
    import time
    import hashlib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"stage2_0d_data.npz"
    error_powers=np.array([1.05,1.075,1.1,1.125,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5])
    protocol = {"version": 3, "error_powers":error_powers.tolist(),
                "model": config["model"], "t_final_ps": t_final_ps,
                "sample_dt_ps": sample_dt_ps, "psi0_real": 0.01, "psi0_imag": 0.0,
                "model_source_sha256": hashlib.sha256(
                    Path(__file__).with_name("model.py").read_bytes()).hexdigest()}
    previous={};previous_meta={}
    if path.exists():
        with np.load(path) as saved:
            old_protocol=json.loads(str(saved['protocol_json']))
            if old_protocol == protocol:
                return path
            common=lambda data:{k:v for k,v in data.items() if k not in ('version','error_powers')}
            if common(old_protocol)==common(protocol):
                previous={k:saved[k].copy() for k in saved.files}
                previous_meta=json.loads(str(saved['metadata_json']))
    started = time.perf_counter()
    p = parameters(config)
    times = np.linspace(0, t_final_ps, round(t_final_ps/sample_dt_ps)+1)
    powers = np.array([0.8, 1.05, 1.2, 1.5])
    branch_s = np.linspace(1.0001, 3, 601)
    lp, lm = condensed_eigenvalues_0d(branch_s, config)
    nc_ss = np.array([stationary_0d(s*p["Pth0D"], config)["condensed"][0]
                      for s in branch_s])
    nc_ad = np.array([reduced_stationary_0d(s*p["Pth0D"], config, "adiabatic")[0]
                      for s in branch_s])
    nc_sl = np.array([reduced_stationary_0d(s*p["Pth0D"], config, "stuart_landau")[0]
                      for s in branch_s])
    output = {"protocol_json": json.dumps(protocol), "s_branch": branch_s,
              "lambda_plus": lp, "lambda_minus": lm, "nc_ss_full": nc_ss,
              "nc_ss_ad": nc_ad, "nc_ss_sl": nc_sl,
              "eps_ss_ad": abs(nc_ad-nc_ss)/nc_ss,
              "eps_ss_sl": abs(nc_sl-nc_ss)/nc_ss, "times_ps": times,
              "powers": powers, "error_powers":error_powers, "config_json": json.dumps(config)}
    view_ends = {0.8: 300.0, 1.05: 1600.0, 1.2: 800.0, 1.5: 500.0}
    metadata = {"initial_density_um2_inv": 1e-4, "initial_phase_rad": 0.0,
                "reservoir_initialization": "P/(gammaR + R*nC0)",
                "dynamic_error_window_ps": [0.0, t_final_ps],
                "stationarity_window_ps": 100.0, "error_powers":error_powers.tolist(),
                "computed_powers":[],"reused_powers":[], "cases": {}}
    eigen_error = 0.0
    for s, plus, minus in zip(branch_s[::30], lp[::30], lm[::30]):
        state = stationary_0d(s*p["Pth0D"], config)["condensed"]
        numeric = np.linalg.eigvals(jacobian_0d(*state, config))
        analytic = np.array([plus, minus])
        eigen_error = max(eigen_error, min(float(np.max(abs(analytic-numeric))),
                                           float(np.max(abs(analytic-numeric[::-1])))))

    def density_status(nc, nr, end, target):
        tail = (times >= end-100-1e-10) & (times <= end+1e-10)
        if end < 100 or np.count_nonzero(tail) < 2:
            return "NOT_TESTED"
        if target[0] == 0:
            return "DECAYED" if nc[tail][-1] < 1e-6*p["gammaR"]/p["R"] else "TRANSIENT"
        checks = (np.ptp(nc[tail])/target[0],
                  np.ptp(nr[tail])/max(abs(target[1]), 1e-30),
                  abs(nc[tail][-1]-target[0])/target[0],
                  abs(nr[tail][-1]-target[1])/max(abs(target[1]), 1e-30))
        return "STATIONARY_DENSITIES" if max(checks) < 0.005 else "TRANSIENT"

    for s in np.unique(np.r_[powers,error_powers]):
        key=f"{s:g}"
        label = key.replace(".", "p")
        names=[f'{kind}_{method}_{label}' for kind in ('psi','nr') for method in ('full','ad','sl')]
        if key in previous_meta.get('cases',{}) and all(name in previous for name in names):
            for name in names:output[name]=previous[name]
            metadata['cases'][key]=previous_meta['cases'][key]
            metadata['reused_powers'].append(float(s))
            continue
        case_started=time.perf_counter()
        pump = s*p["Pth0D"]
        nr0 = pump/(p["gammaR"]+p["R"]*1e-4)
        psi, nr = solve_0d_reference(config, pump, 0.01+0j, nr0, times)
        fields = {"full": (psi, nr),
                  "ad": solve_0d_reduced(config, pump, 0.01+0j, times, "adiabatic"),
                  "sl": solve_0d_reduced(config, pump, 0.01+0j, times, "stuart_landau")}
        targets = {"full": (stationary_0d(pump, config)["condensed"] or
                            stationary_0d(pump, config)["empty"]),
                   "ad": reduced_stationary_0d(pump, config, "adiabatic"),
                   "sl": reduced_stationary_0d(pump, config, "stuart_landau")}
        case = {"pump_um2_inv_ps_inv": float(pump), "nr0_um2_inv": float(nr0),
                "view_end_ps": min(view_ends.get(float(s),t_final_ps), t_final_ps),
                "end_status": {}, "view_status": {}, "dynamic_errors": {}}
        for method, (field, reservoir) in fields.items():
            output[f"psi_{method}_{label}"] = field
            output[f"nr_{method}_{label}"] = reservoir
            nc = abs(field)**2
            case["end_status"][method] = density_status(nc, reservoir, t_final_ps, targets[method])
            case["view_status"][method] = density_status(nc, reservoir, case["view_end_ps"], targets[method])
        for method in ("ad", "sl"):
            difference = abs(fields[method][0])**2-abs(psi)**2
            rmse = np.sqrt(np.trapezoid(difference**2, times)/(times[-1]-times[0]))
            resolved = s > 1 and case["end_status"]["full"] == "STATIONARY_DENSITIES"
            case["dynamic_errors"][method] = {
                "rmse_absolute_um2_inv": float(rmse),
                "relative": float(rmse/targets["full"][0]) if resolved else None,
                "normalization": "analytic full stationary density" if resolved else "absolute density RMSE"}
        metadata["cases"][key] = case
        metadata['computed_powers'].append(float(s))
        print(f"0D s={s:g}: {time.perf_counter()-case_started:.2f} s, "
              f"reference={case['end_status']['full']}",flush=True)
    metadata["checks"] = {"eigenvalue_max_abs_residual_per_ps": eigen_error,
        "adiabatic_stationary_max_relative_error": float(np.max(output["eps_ss_ad"])),
        "SL_stationary_formula_max_abs_residual": float(np.max(abs(output["eps_ss_sl"]-(1-1/branch_s))))}
    metadata['last_update_wall_s']=time.perf_counter()-started
    metadata["elapsed_wall_s"] = previous_meta.get('elapsed_wall_s',0)+metadata['last_update_wall_s']
    output["metadata_json"] = json.dumps(metadata)
    np.savez_compressed(path, **output)
    return path


def _stage2_0d_figures(data_path, out, *, figure_numbers=(2,3,4)):
    from matplotlib.lines import Line2D

    paths = []
    with np.load(data_path) as a:
        meta = json.loads(str(a["metadata_json"]))
        times=a['times_ps'];config=json.loads(str(a['config_json']));p=parameters(config)
        if 2 in figure_numbers:
            fig = plt.figure(figsize=(REPORT_WIDTH_IN, 3.95), layout="constrained")
            gs = fig.add_gridspec(2, 2, height_ratios=(1, 0.17))
            axes = [fig.add_subplot(gs[0, col]) for col in range(2)]
            s = a["s_branch"]
            plus, minus = a["lambda_plus"], a["lambda_minus"]
            for value, color, name in ((plus, "C0", "+"), (minus, "C1", "-")):
                # In the oscillatory region the real parts coincide exactly.
                # A wider translucent blue line remains visible around the orange one.
                axes[0].plot(s, value.real, color=color, alpha=0.65 if name == "+" else 0.8,
                             lw=2.7 if name == "+" else 1.3,
                             label=rf"$\mathrm{{Re}}\,\lambda_{name}$")
                axes[0].plot(s, value.imag, color=color, ls="--", alpha=0.8,
                             label=rf"$\mathrm{{Im}}\,\lambda_{name}$")
            complex_s = s[abs(plus.imag) > 1e-12]
            if complex_s.size:
                axes[0].axvspan(complex_s[0], complex_s[-1], color="C0", alpha=0.05)
            overlapping = np.flatnonzero((abs(plus.imag) > 1e-12) & (s >= 0.95) & (s <= 3.05))
            if overlapping.size:
                index = overlapping[np.argmin(abs(s[overlapping]-2.0))]
                axes[0].annotate(r"$\mathrm{Re}\,\lambda_+=\mathrm{Re}\,\lambda_-$",
                                 xy=(s[index], plus.real[index]), xytext=(1.32, -0.14),
                                 fontsize=8, color="0.25",
                                 arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 0.8})
            axes[0].axvline(1, color="0.4", ls=":", lw=1)
            axes[0].axhline(0, color="0.6", lw=0.7)
            axes[0].set(xlim=(0.95, 3.05), ylim=(-0.17, 0.15),
                        xlabel=r"Normalized pump, $s=P/P_{\mathrm{th}}$",
                        ylabel=r"Eigenvalue (ps$^{-1}$)")
            axes[0].set_title("(a) Branch eigenvalues", loc="left")
            legend_ax = fig.add_subplot(gs[1, 0]); legend_ax.set_axis_off()
            legend_ax.legend(*axes[0].get_legend_handles_labels(), loc="center", ncol=2, frameon=False)
            times = a["times_ps"]
            selected = times <= meta["cases"]["1.5"]["view_end_ps"]
            nc, nr = abs(a["psi_full_1p5"][selected])**2, a["nr_full_1p5"][selected]
            config = json.loads(str(a["config_json"]))
            p = parameters(config)
            steady_state = stationary_0d(1.5*p["Pth0D"], config)["condensed"]
            axes[1].plot(nc, nr, color="C0", lw=1.3)
            axes[1].plot(nc[0], nr[0], "o", color="C2", ms=5, label="Initial state")
            axes[1].plot(*steady_state, "o", color="C1", ms=5.5, label="Steady state")
            length = np.r_[0, np.cumsum(np.hypot(np.diff(nc)/steady_state[0],
                                                np.diff(nr)/steady_state[1]))]
            for fraction in (0.2, 0.5, 0.75):
                index = min(int(np.searchsorted(length, fraction*length[-1])), len(nc)-12)
                stop = min(index+10, len(nc)-1)
                axes[1].annotate("", xy=(nc[stop], nr[stop]), xytext=(nc[index], nr[index]),
                                 arrowprops={"arrowstyle": "->", "color": "C0", "lw": 1.2})
            axes[1].set(xlabel=r"$n_C$ ($\mu$m$^{-2}$)", ylabel=r"$n_R$ ($\mu$m$^{-2}$)")
            axes[1].set_title("(b) Relaxation at $s=1.5$", loc="left")
            legend_ax = fig.add_subplot(gs[1, 1]); legend_ax.set_axis_off()
            legend_ax.legend(*axes[1].get_legend_handles_labels(), loc="center", frameon=False)
            paths.append(_save(fig, out, Path(STAGE2_IMAGES[2]).stem))

        if 3 in figure_numbers:
            fig = plt.figure(figsize=(REPORT_WIDTH_IN, 4.7), layout="constrained")
            gs = fig.add_gridspec(4, 4, height_ratios=(0.12, 1, 1, 0.1), width_ratios=(1.15, 1, 1, 1))
            styles = (("full", "C0", "-", "Full model"),
                      ("ad", "C1", "--", "Adiabatic"),
                      ("sl", "C2", "-.", "Stuart–Landau"))
            legend_ax = fig.add_subplot(gs[0, :]); legend_ax.set_axis_off()
            legend_ax.legend(handles=[Line2D([], [], color=c, ls=ls, label=title)
                for _, c, ls, title in styles], loc="center", ncol=3, frameon=False)
            for col, power in enumerate(a["powers"]):
                key = f"{power:g}"; label = key.replace(".", "p")
                end = meta["cases"][key]["view_end_ps"]; selected = times<=end
                axes = [fig.add_subplot(gs[row+1, col]) for row in range(2)]
                for method, color, style, _title in styles:
                    if power<=1 and method=="sl":
                        continue
                    axes[0].plot(times[selected], abs(a[f"psi_{method}_{label}"][selected])**2, color=color, ls=style)
                    axes[1].plot(times[selected], a[f"nr_{method}_{label}"][selected], color=color, ls=style)
                for row, ax in enumerate(axes):
                    ax.set_xlim(0, end)
                    ax.set_title(f"({chr(97+row*4+col)}) "+rf"$s={power:g}$", loc="left", fontsize=9)
                    ax.set_xticks([0, end/2, end])
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    if col==0:
                        ax.set_ylabel(r"$n_C$ ($\mu$m$^{-2}$)" if row==0 else r"$n_R$ ($\mu$m$^{-2}$)")
                if power<1:
                    axes[0].set_yscale("log")
                    from matplotlib.ticker import LogLocator, NullFormatter
                    axes[0].yaxis.set_major_locator(LogLocator(base=10, numticks=4))
                    axes[0].yaxis.set_minor_formatter(NullFormatter())
                    low_reservoir = meta["cases"][key]["pump_um2_inv_ps_inv"]/p["gammaR"]
                    axes[1].set_ylim(low_reservoir-0.001, low_reservoir+0.001)
                    axes[1].set_yticks([low_reservoir-0.001, low_reservoir, low_reservoir+0.001])
                    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
                    from matplotlib.ticker import FormatStrFormatter
                    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            note_ax = fig.add_subplot(gs[3, :]); note_ax.set_axis_off()
            note_ax.text(0.5, 0.5,
                         "Stuart–Landau reservoir: approximate reconstruction, no separate evolution equation.",
                         ha="center", va="center", fontsize=8, transform=note_ax.transAxes)
            fig.supxlabel(r"Time, $t$ (ps)", fontsize=9)
            paths.append(_save(fig, out, Path(STAGE2_IMAGES[3]).stem))

        if 4 in figure_numbers:
            paths.append(_reduced_accuracy_figure(data_path,out))
    return paths


def _reduced_accuracy_figure(data_path,out):
    """Figure 4 only: analytic stationary error and sampled transient errors."""
    with np.load(data_path) as a:
        s=a['s_branch'];meta=json.loads(str(a['metadata_json']))
        fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH_IN, 3.3), layout="constrained")
        axes[0].plot(s, 100*a["eps_ss_ad"], color="C1", ls="--", label="Adiabatic (steady state)")
        axes[0].plot(s, 100*a["eps_ss_sl"], color="C2", label="Stuart–Landau")
        axes[0].set(xlabel=r"Normalized pump, $s$", ylabel="Stationary density error (%)",
                    xlim=(1, 3))
        axes[0].set_title("(a) Stationary density error", loc="left")
        axes[0].legend(loc="upper left", fontsize=8)
        powers = a['error_powers'] if 'error_powers' in a.files else a['powers'][a['powers']>1]
        for method, color, marker, title in (("ad", "C1", "o", "Adiabatic"),
                ("sl", "C2", "s", "Stuart–Landau")):
            errors = [meta["cases"][f"{s:g}"]["dynamic_errors"][method]["relative"] for s in powers]
            if any(error is None for error in errors):
                raise ValueError("The dynamic relative-error reference is not settled; use absolute errors")
            axes[1].plot(powers, 100*np.array(errors), marker+"-", color=color, ms=4, label=title)
        axes[1].set(xlabel=r"Normalized pump, $s$", ylabel="Normalized density RMSE (%)")
        axes[1].set_xticks([1.05,1.2,1.35,1.5])
        axes[1].set_title("(b) Transient density error", loc="left")
        axes[1].legend(loc="upper left", fontsize=8)
        return _save(fig, out, Path(STAGE2_IMAGES[4]).stem)


def _stage2_threshold_figure(sources, out):
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(REPORT_WIDTH_IN, 5.6), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=(0.13, 1, 1))
    legend_ax = fig.add_subplot(gs[0, :]); legend_ax.set_axis_off()
    legend_ax.legend(handles=[Line2D([], [], color="C0", marker="o", ls="", label="STABLE"),
        Line2D([], [], color="C1", marker="^", ls="", label="UNSTABLE"),
        Line2D([], [], color="0.35", ls=":", label=r"Homogeneous: $P_0=1$")],
        loc="center", ncol=3, frameon=False)
    for col, (key, title) in enumerate((("gaussian_threshold", "Spot"), ("ring_threshold", "Ring"))):
        entry = sources[key]
        bracket = (entry if isinstance(entry, dict) else
                   json.loads(Path(entry).read_text()))["threshold"]
        lo, hi = bracket["P0_lo"], bracket["P0_hi"]
        points = bracket["points"]
        zoom = (lo-0.04*(lo+hi)/2, hi+0.04*(lo+hi)/2)
        for row in range(2):
            ax = fig.add_subplot(gs[row+1, col])
            selected = points if row==0 else [p for p in points if zoom[0]<=p["P0"]<=zoom[1]]
            for status, marker, color in (("STABLE", "o", "C0"), ("UNSTABLE", "^", "C1"), ("UNRESOLVED", "x", "0.5")):
                group = [p for p in selected if p["status"]==status and p.get("sigma_late_per_ps") is not None]
                if group:
                    ax.errorbar([p["P0"] for p in group], [p["sigma_late_per_ps"] for p in group],
                        yerr=[2*p.get("sigma_late_stderr", 0) for p in group],
                        fmt=marker, color=color, ms=4, capsize=2)
            ax.axvspan(lo, hi, color="0.5", alpha=0.18)
            if row==0:
                ax.axvline(1, color="0.35", ls=":", lw=1)
                ax.set_xlim(0, max(hi*1.15, max(p["P0"] for p in points)*1.05))
            else:
                padding = 0.02*(zoom[1]-zoom[0])
                ax.set_xlim(zoom[0]-padding, zoom[1]+padding)
            ax.axhline(0, color="0.35", lw=1)
            ax.set_xlabel(r"Normalized pump, $P_0$")
            ax.set_ylabel(r"Growth rate, $\sigma$ (ps$^{-1}$)")
            ax.set_title(f"({chr(97+row*2+col)}) {title}: "+("full scan" if row==0 else "threshold zoom"), loc="left")
            if row==1:
                ax.text(0.04, 0.96, rf"$P_{{0,c}}\in[{lo:.8g},\,{hi:.8g}]$", transform=ax.transAxes,
                    va="top", fontsize=8, bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.9})
                # Leave room for the bracket annotation without hiding any fitted point.
                ax.margins(y=0.30)
    return _save(fig, out, Path(STAGE2_IMAGES[8]).stem)


def make_stage2_report(config, *, data_dir="results/analysis_data",
                       output_dir="results/report_figures", source_dir=None,
                       notes_path="docs/methods.md"):
    """Render the report from saved fields; no 2D propagation."""
    out = Path(output_dir)
    source_dir = Path(source_dir) if source_dir is not None else out
    sources = json.loads((source_dir/"figure_sources.json").read_text())
    for key in ("gaussian_run", "ring_run"):
        if _run(sources[key])["config"]["model"] != config["model"]:
            raise ValueError("Report coefficients differ from the saved 2D source")
    out.mkdir(parents=True, exist_ok=True)
    data_path = build_stage2_0d_data(config, data_dir)
    with plt.rc_context(REPORT_STYLE):
        paths = [_figure_0d(config, out)]
        paths.extend(_stage2_0d_figures(data_path, out))
        paths.extend(_pair_figures(sources["gaussian_run"], sources["ring_run"], out,
            gaussian_spectrum_runs=sources["gaussian_spectrum_runs"]))
        paths.append(_stage2_threshold_figure(sources, out))
    from PIL import Image
    sizes = {}
    for path in paths:
        with Image.open(path) as im:
            sizes[path.name] = [round(n/dpi*25.4, 1) for n,dpi in zip(im.size, im.info["dpi"])]
    sources.update(figure_sizes_mm=sizes, data_dir=str(data_dir), png_dpi=300, width_mm=160,
                   titles={STAGE2_IMAGES[n]: title for n,title in REPORT_TITLES.items() if n<=8})
    supplement = Path(data_dir)/"ring_dynamics.json"
    if supplement.exists():
        experiment=json.loads(supplement.read_text())
        if experiment.get('time_ps'):
            paths.extend(plot_ring_dynamics(experiment,supplement.with_suffix('.npz'),output_dir=out))
            supplement.write_text(json.dumps(experiment,ensure_ascii=False,indent=2)+'\n')
            sources['ring_dynamics']=str(supplement)
            sources['ring_dynamics_run']=experiment['run_dir']
            for path in paths[-1:]:
                with Image.open(path) as im:
                    sources['figure_sizes_mm'][path.name]=[round(n/dpi*25.4,1) for n,dpi in zip(im.size,im.info['dpi'])]
            sources['titles'][STAGE2_IMAGES[9]]=REPORT_TITLES[9]
    (out/"figure_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2)+"\n")
    def saved(name):
        file = Path(data_dir)/name
        return file if file.exists() else None
    domain = saved("domain_comparisons_large.json")
    write_report_summary(Path(notes_path).parent/"diagnostic_summary.md", gaussian_run=sources["gaussian_run"],
        ring_run=sources["ring_run"], gaussian_threshold=sources["gaussian_threshold"],
        ring_threshold=sources["ring_threshold"], convergence=sources["convergence"],
        domain_comparisons=json.loads(domain.read_text()) if domain else None,
        final_state_checks=saved("final_state_checks.json"), validation_report=saved("validation_report.json"),
        practical_checks=saved("practical_checks.json"))
    _write_stage2_notes(config, sources, data_path, out, Path(notes_path))
    return sorted(paths)


def update_report_text(*, output_dir="results/report_figures", notes_path="docs/methods.md"):
    """Refresh methods and diagnostics from saved arrays, without rendering PNGs."""
    from numerics import load_config
    out = Path(output_dir)
    sources = json.loads((out/"figure_sources.json").read_text())
    data = Path(sources["data_dir"])
    domain = data/"domain_comparisons_large.json"
    write_report_summary(Path(notes_path).parent/"diagnostic_summary.md",
        gaussian_run=sources["gaussian_run"], ring_run=sources["ring_run"],
        gaussian_threshold=sources["gaussian_threshold"], ring_threshold=sources["ring_threshold"],
        convergence=sources["convergence"], domain_comparisons=json.loads(domain.read_text()),
        final_state_checks=data/"final_state_checks.json", validation_report=data/"validation_report.json",
        practical_checks=data/"practical_checks.json")
    _write_stage2_notes(load_config(), sources, data/"stage2_0d_data.npz", out, Path(notes_path))


def _write_stage2_notes(config, sources, data_path, out, notes_path):
    """Compact code audit: equations, methods, grids and figure sources."""
    import os
    def link(path):
        return Path(os.path.relpath(Path(path).absolute(), notes_path.parent.absolute())).as_posix()

    with np.load(data_path) as a:
        meta = json.loads(str(a["metadata_json"]))
        protocol = json.loads(str(a["protocol_json"]))
        branch = a["s_branch"]
        branch_info = f"{branch[0]:g}–{branch[-1]:g}, {branch.size} точка"
    p = parameters(config)
    threshold_durations = sorted({point["duration_ps"]
        for key in ("gaussian_threshold", "ring_threshold")
        for point in json.loads(Path(sources[key]).read_text())["threshold"]["points"]
        if "duration_ps" in point})
    duration_text = "/".join(f"{duration:g}" for duration in threshold_durations)
    lines = [
        "# Как рассчитаны рисунки 01–09", "",
        "Краткая сверка с `model.py`, `numerics.py` и `plotting.py`. "
        "Номера относятся к PNG в `results/report_figures/`; экспорт 160 мм, 300 dpi, шрифты 8–10 pt. "
        "Ниже — алгоритмы и параметры, без обсуждения физических результатов.", "",
        "## Общая модель и единицы", "",
        "Единицы: мкм, пс, мэВ; nC и nR — мкм⁻²; P — мкм⁻²·пс⁻¹. "
        f"ħ={p['hbar']:.10g} мэВ·пс; m/me={config['model']['mass_over_me']:g}; "
        f"K=ħ²/(2m)={p['K']:.9g} мэВ·мкм²; "
        f"gC={p['gC']:g}, gR={p['gR']:g} мэВ·мкм²; R={p['R']:g} мкм²/пс; "
        f"γC={p['gammaC']:g}, γR={p['gammaR']:g} пс⁻¹; η={p['eta']:g} пс. "
        "Во всех используемых прогонах фоновый сдвиг ηP включён.", "",
        r"$$\partial_t\psi=\frac{iK}{\hbar}\nabla^2\psi+\left[\frac{Rn_R-\gamma_C}{2}-\frac{i}{\hbar}(g_C|\psi|^2+g_R(n_R+\eta P))\right]\psi,\qquad \partial_t n_R=P-(\gamma_R+R|\psi|^2)n_R.$$", "",
        r"$n_C=|\psi|^2$, $P=P_0P_{\rm th}f(r)$, $P_{\rm th}=\gamma_C\gamma_R/R$. "
        r"Пятно: $f=e^{-r^2/d^2}$; кольцо: $f=e^{-(r-r_0)^2/d^2}$; "
        "d=2.5 мкм, r0=10 мкм, центр (0,0). Код: `model.parameters`, `pump_profile`, `numerics.local_rhs`.", "",
        "## Рис. 01 — Stationary densities and condensate-free state stability", "",
        r"0D, без интегрирования: при $s=P/P_{\rm th}\le1$: $(n_C,n_R)=(0,P/\gamma_R)$; "
        r"при $s>1$: $(n_C^*,n_R^*)=((P-P_{\rm th})/\gamma_C,\gamma_C/R)$. "
        r"На нулевой ветви $\lambda_{\max}=\max(\gamma_C(s-1),-\gamma_R)$ "
        "получается через `np.linalg.eigvals(jacobian_0d(...))`. "
        "151 равномерная точка s=0.5–3 (Δs=1/60); сетки x,t нет. "
        "Код: `stationary_0d`, `jacobian_0d`, `plotting._figure_0d`; массивы вычисляются аналитически при экспорте.", "",
        "## Рис. 02 — Condensed-state stability and relaxation", "",
        r"$$J=\begin{pmatrix}Rn_R-\gamma_C&Rn_C\\-Rn_R&-\gamma_R-Rn_C\end{pmatrix},\qquad "
        r"\lambda_\pm=-\frac{\gamma_Rs}{2}\pm\frac12\sqrt{\gamma_R^2s^2-4\gamma_C\gamma_R(s-1)}.$$", "",
        f"Собственные значения плотности и резервуара: s={branch_info}; "
        "комплексный квадратный корень, Re и Im показаны отдельно. "
        "Глобальная фазовая мода в этот 2×2 Якоби не входит. Фазовая траектория (nC,nR) — "
        "численное решение полной 0D-модели при s=1.5 за 0–500 пс; стрелки идут по соседним отсчётам. "
        "Steady state — стационарное решение открытой системы. "
        "Код: `condensed_eigenvalues_0d`, `solve_0d_reference`, `plotting._stage2_0d_figures`.", "",
        "## Рис. 03 — Homogeneous condensate–reservoir dynamics", "",
        r"Полная модель — общие уравнения без $\nabla^2$. Адиабатическая — замена "
        r"$n_R=P/(\gamma_R+Rn_C)$ в уравнении поля. Стюарт–Ландау (SL):", "",
        r"$$\dot\psi=[\mu-\beta n_C-i(\omega_0+\nu n_C)]\psi,\quad "
        r"\mu=\tfrac12(RP/\gamma_R-\gamma_C),\quad\beta=PR^2/(2\gamma_R^2),\quad "
        r"\omega_0=g_R(P/\gamma_R+\eta P)/\hbar,\quad\nu=(g_C-g_RPR/\gamma_R^2)/\hbar.$$", "",
        r"В SL резервуар восстановлен как $n_R=(P/\gamma_R)(1-Rn_C/\gamma_R)$; "
        r"это разложение при $Rn_C/\gamma_R\ll1$, а не отдельное уравнение резервуара. "
        "Подпись на рисунке: Stuart–Landau reservoir: approximate reconstruction, no separate evolution equation. "
        "Код: `solve_0d_reference`, `solve_0d_reduced`, `stuart_landau`.", "",
        f"Общий протокол рис. 02–04: ψ(0)=0.01+0i, nC(0)=10⁻⁴, "
        "nR(0)=P/(γR+R·10⁻⁴) в полной модели. "
        "DOP853 для Re ψ, Im ψ и (в полной модели) nR; rtol=2e−11, atol=2e−13. "
        f"Время 0–{protocol['t_final_ps']:g} пс; сохранение через {protocol['sample_dt_ps']:g} пс "
        "— шаг выборки, внутренний шаг интегратора адаптивный. Пространственной сетки нет. "
        "На рис. 03 s=0.8/1.05/1.2/1.5; окна соответственно 300/1600/800/500 пс. "
        "SL при s=0.8 не показан. Все новые 0D-массивы: "
        f"[{data_path.name}]({link(data_path)}). На рисунке сверху nC, снизу nR; накачка возрастает слева направо.", "",
        "## Рис. 04 — Accuracy of reduced models", "",
        r"Стационарная ошибка $\varepsilon_{\rm ss}=|n_C^{\rm approx,*}-n_C^*|/n_C^*$: "
        r"адиабатическая — 0 аналитически: стационарные ветви совпадают, в коде "
        "используется одна формула (это не невязка интегрирования). "
        r"SL — $1-1/s$, поскольку $n_C^{\rm SL,*}=n_C^*/s$. "
        "Использована та же сетка s, что на рис. 02.", "",
        r"$$\varepsilon_{\rm dyn}=\frac{1}{n_C^*}\sqrt{\frac1T\int_0^T(n_C^{\rm approx}(t)-n_C^{\rm full}(t))^2dt},\qquad T=1600\;\mathrm{ps}.$$", "",
        "Интеграл — `np.trapezoid`, Δt выборки=0.5 пс; на графике 100ε, "
        "для динамической ошибки — 12 отдельных расчётов: "
        "s=1.05/1.075/1.1/1.125/1.15/1.2/1.25/1.3/1.35/1.4/1.45/1.5; "
        "линии соединяют рассчитанные точки. Нормировка по аналитической плотности полной модели; "
        "ниже порога сохраняется только абсолютная RMSE. Перед нормировкой проверены "
        "размах и отклонение обеих плотностей за последние 100 пс (<0.5%). "
        "Код: `plotting.build_stage2_0d_data`, `_reduced_accuracy_figure`.", "",
        "## Рис. 05–07 — общие 2D-прогоны", "",
        "Метод: расщепление Стрэнга — кинетика Δt/2 → совместный локальный RK4 "
        "для ψ,nR на Δt → кинетика Δt/2. Кинетический множитель "
        r"$e^{-iKk^2\Delta t/(2\hbar)}$, NumPy FFT/IFFT; общий порядок по времени 2. "
        r"$\Delta x=L_x/N_x$, $k_x=2\pi\,\mathrm{fftfreq}(N_x,\Delta x)$, аналогично y; "
        "периодические границы, поглощение выключено. Код: `make_grid`, `split_step`, `local_rk4`.", "",
        "Начало: nR=0; ψ=√(10⁻⁶ f)·Σ₂₄exp[i(kx x+ky y+φ)]/√24, "
        "kx,ky ~ N(0,1 мкм⁻²), φ ~ U(0,2π), seed=20261008. "
        "10⁻⁶ — коэффициент затравки, локальная плотность зависит от суммы. Код: `initial_fields`.", "",
        "| Панели | P0 | Nx×Ny | Lx×Ly, мкм | Δx=Δy, мкм | Δt, пс | Последний снимок, пс | Источник |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    selected = [("Пятно, 05–07", sources["gaussian_run"]),
                ("Кольцо, 05–07", sources["ring_run"])]
    selected += [("Пятно, малая область 07", path) for path in sources["gaussian_spectrum_runs"]]
    for label, path in selected:
        run = _run(path)
        c = run["config"]
        g = make_grid(c)
        lines.append(f"| {label} | {c['pump']['P0']:g} | {g.x.size}×{g.y.size} | "
                     f"{g.lx:g}×{g.ly:g} | {g.dx:g} | {c['time']['dt_ps']:g} | "
                     f"{run['summary']['time_ps']:g} | [{run['path'].name}]({link(run['path']/'resolved_config.yaml')}) |")
    lines += ["",
        "Снимки сохранены через 20 пс; diagnostics.csv — через 10 пс. Интегральные "
        "наблюдаемые вычисляются через 1 пс и используются в памяти для проверки окна; "
        "поля для лагов — через 5 пс, сравнение с эталоном окна — через 1 пс. "
        "PNG 05–07 строятся из последних сохранённых полей; новых 2D-расчётов при экспорте нет.", "",
        "**05 Spatial profiles under localized pumping:** P, nR, nC=|ψ|², Vint=gC nC+gR(nR+ηP). "
        "Mean-field energy shift включает собственное взаимодействие gC nC. "
        "Центральные 40×40 мкм (увеличение относительно 06); сверху пятно, снизу кольцо; общая шкала каждого столбца. Код: `plotting._pair_figures`.", "",
        r"**06 Particle current and condensate phase:** $\mathbf j=(2K/\hbar)\operatorname{Im}(\psi^*\nabla\psi)$; "
        "производные через IFFT(ik·FFT ψ), множитель k у моды Найквиста обнулён. "
        "Фаза arg ψ скрыта при nC<10⁻³ max nC; стрелки через ~5 мкм, скрыты "
        "при |j|<5% общего максимума; область 80×80 мкм, единый масштаб стрелок. "
        "Стрелки белые с чёрной обводкой для контраста на светлой и тёмной плотности. "
        "На фазе кольца показаны контуры r=6 и 9 мкм, q=−3. Код: `numerics.current`, `plotting._pair_figures`.", "",
        r"**07 Momentum distributions and spot domain comparison:** $\widetilde\psi=\Delta x\Delta y\,\mathrm{FFT2}(\psi)$; "
        r"$I_k=|\widetilde\psi|^2$, $\log_{10}(I_k/(1\,\mu\mathrm m^2))$, `fftshift`, "
        "общая абсолютная шкала на 8 порядков, без нормировки по максимуму каждой панели. "
        "Чёрный цвет означает Iк ниже max Iк·10⁻⁸; слабые хвосты не обрезаны в массиве. "
        "Видимая внешняя кромка задаётся цветовой шкалой и не означает строгого исчезновения спектра. "
        "Отображается kx,ky=−4…4 мкм⁻¹; нижний предел аргумента log10 — 10⁻³⁰. "
        "Пятно: L=256 и 192 мкм при t=500 пс, одинаковых Δx=0.5 мкм и Δt. "
        "Кольцо: только L=256 мкм при t=2000 пс; старый спектр L=192 при 1000 пс "
        "убран из рисунка, поскольку сравнение областей требует одного времени. "
        "Пятно при L=192 имеет краевой WARN; основные панели L=256 проходят контроль краёв. "
        "Код: `numerics.spectrum`, `plotting._pair_figures`.", "",
        "## Рис. 08 — Linear instability thresholds of localized pumps", "",
        r"Замороженная нулевая ветвь $n_{R0}=P/\gamma_R$; в уравнении ψ "
        "убраны gC|ψ|² и обратное действие на резервуар. Кинетика — FFT; локальный "
        "линейный шаг — точная экспонента; расщепление Стрэнга. Δt=0.05 пс. "
        "Основная сетка 128², L=128 мкм, Δx=1 мкм; проверка концов порога: "
        "256² при L=128 (Δx=0.5) и 192² при L=192 (Δx=1).", "",
        "Затравка как выше, нормирована до ∫|ψ|²dA=1. Каждую 1 пс поле "
        "перенормируется, логарифмы норм суммируются. Считается "
        r"$\sigma=\tfrac12\,d\ln N_{\rm lin}/dt$ через `np.polyfit` "
        "в двух последних соседних окнах по 100 пс. Длительности сохранённых точек "
        f"{duration_text} пс. Надёжность: R²≥0.99 в обоих окнах, разница σ≤max(0.1|σlate|,10⁻⁴), "
        "|σlate|>2stderr. Знак σ задаёт STABLE/UNSTABLE; иначе UNRESOLVED. "
        "Перекрытие мод ≥0.95 записывается отдельно, в критерий знака не входит. "
        "Верхние панели показывают всю сохранённую сканируемую выборку и ориентир P0=1; "
        "нижние увеличивают окрестность порога, серая полоса — [P0lo,P0hi]. "
        "Порог уточняется бисекцией до относительной ширины <1%, с повтором концов "
        "для seed=20261009. Бары — ±2stderr σ, не ошибка сетки. "
        "Код: `linear_step`, `linear_growth_rate`, `_linear_fit`, `find_threshold_2d`, "
        "`validate_threshold_2d`, `plotting._stage2_threshold_figure`.", "",
    ]
    for label, key in (("Пятно", "gaussian_threshold"), ("Кольцо", "ring_threshold")):
        source = Path(sources[key])
        threshold = json.loads(source.read_text())["threshold"]
        count=len({point['P0'] for point in threshold['points']})
        width=2*(threshold['P0_hi']-threshold['P0_lo'])/(threshold['P0_hi']+threshold['P0_lo'])
        lines.append(f"{label}: P0∈[{threshold['P0_lo']:.8g},{threshold['P0_hi']:.8g}]; "
                     f"{count} значений накачки, {len(threshold['points'])} сохранённых точек с повторными затравками; "
                     f"относительная ширина интервала {100*width:.3f}%. "
                     f"[Точки и проверки]({link(source)}).")
        extension = json.loads(source.read_text()).get("additional_scan")
        if extension:
            powers = "/".join(f"{power:g}" for power in extension["completed_P0"])
            lines.append(f"Дополнительные точки: P0={powers}; тот же `resolved_config.yaml` "
                         f"с заменой P0; `linear_growth_rate`, 300 пс, при ненадёжном знаке "
                         f"повтор 600/1200 пс. Время первичного сканирования {extension['wall_s']:.1f} с.")
            if any(point.get("mode_status") == "MODE_UNRESOLVED" for point in threshold["points"]
                   if point.get("scan_extension_date") == extension["date"]):
                lines.append("При MODE_UNRESOLVED подтверждён знак роста в указанных окнах, "
                             "но не установление собственной моды; σ не гарантирована как асимптотическая.")
        mode_check = json.loads(source.read_text()).get("mode_validation")
        if mode_check:
            cases = "; ".join(f"P0={power}, t={case['duration_ps']:g} пс: {case['status']}"
                              for power, case in mode_check["cases"].items())
            lines.append("Продлённые линейные расчёты: " + cases + ". "
                         f"Время проверок {mode_check.get('wall_s', 0):.1f} с; "
                         f"[поля и история норм]({link(Path(mode_check['source_arrays']))}).")
    if any(json.loads(Path(sources[key]).read_text()).get("mode_validation")
           for key in ("gaussian_threshold", "ring_threshold")):
        lines.append("Установление: |⟨ψ(t−τ),ψ(t)⟩|≥0.95 при τ=20/50/100 пс "
                     "(нормы равны 1, общая фаза исключена модулем), на двух последовательных "
                     "временах проверки; изменение σ≤max(0.01|σ|,10⁻⁴ пс⁻¹) и прежние проверки знака. "
                     "Сначала та же затравка заново до 1200 пс (старое линейное поле не сохранялось), "
                     "затем точное продолжение поля и логарифмов норм; сетка и Δt прежние. "
                     "Невязка шага ||Uψ−⟨ψ,Uψ⟩ψ||/(||Uψ||Δt) записана отдельно, без порога принятия. "
                     "Последняя продлённая σ заменяет короткую на рис. 08; оригинал и попытки "
                     "в original_short_probe/mode_check_attempts summary. Продолжение проверено "
                     "на совпадение с непрерывным расчётом (`test_linear_mode_continuation_matches_uninterrupted`).")
    study_path = Path(sources["convergence"])
    study = json.loads(study_path.read_text())
    protocol = study.get("protocol", {})
    case_configs = {name: _run(path)["config"] for name, path in study["cases"].items()}
    grid_names = sorted((name for name in case_configs if name.startswith("grid_")),
                        key=lambda name: int(name.split("_")[1]))
    grid_cases = [case_configs[name] for name in grid_names]
    reference = grid_cases[len(grid_cases)//2]
    grids = "/".join(str(c["space"]["Nx"])+"²" for c in grid_cases)
    steps = "/".join(f"{c['time']['dt_ps']:g}" for name, c in case_configs.items() if name.startswith("dt_"))
    domains = "/".join(f"{c['space']['Lx_um']:g}" for name, c in case_configs.items() if name.startswith("domain_"))
    domain_case = next(c for name, c in case_configs.items() if name.startswith("domain_"))
    lines += ["", "## Проверки сходимости", "",
        f"[Полная серия]({link(study_path)}): пятно P0={reference['pump']['P0']:g}, "
        f"t={study['time_ps']:g} пс. Сетки {grids} при L={reference['space']['Lx_um']:g} мкм, "
        f"Δt={reference['time']['dt_ps']:g} пс; шаги {steps} пс при {reference['space']['Nx']}². "
        f"Области L={domains} мкм при одинаковом Δx="
        f"{domain_case['space']['Lx_um']/domain_case['space']['Nx']:g} мкм "
        f"и Δt={domain_case['time']['dt_ps']:g} пс. Физика и непрерывная затравка одинаковы.", "",
        "Сравнение: относительные ошибки NC, NR, ⟨r²⟩ относительно центра плотности, "
        "∫|j|dA, плотности/резервуара и поля после устранения общей фазы. Тонкая сетка "
        "выбирается в узлах грубой; для областей сравнивается центральный круг Lмал/4. "
        "Интегральный допуск 0.5%, плотность 1%; RMSE NC за последние 100 пс — 0.5%. "
        "Код: `compare_runs`, `convergence_study`, `run_final_convergence_checks`.", "",
        f"Серия: **{study['assessment']}**; конечный режим: "
        f"**{study.get('final_state_assessment', 'NOT_TESTED')}**. "
        "Принятие конечного режима требует проверки установления всех вариантов, "
        "а не только эталона. Численные значения и время — в общей диагностике.", "",
        "`NOT_TESTED` означает отсутствие достаточных данных для конкретной проверки. "
        "Отдельный прогон сам не проверяет сходимость: после полной серии в его summary "
        "записывается `convergence_validation` со ссылкой на исследование. "
        "Стационарность требует полного окна 100 пс, плотной истории и лагов 20/35 пс; "
        "недостаточный сигнал ограничивает краевые/спектральные проверки. "
        "Исходное кольцо при 1000 пс — переходное; на рисунках 05–07 используется "
        "стационарный режим при 2000 пс. Первые проверки кольца относились к 1000 пс; "
        "проверка области по-прежнему относится к этому времени.", "",
    ]
    ring_checks_path = Path(data_path).parent/"final_ring_checks.json"
    if ring_checks_path.exists():
        ring_checks = json.loads(ring_checks_path.read_text())
        lines += ["## Конечный вихрь кольца: проверка при 2000 пс", "",
            "`run_final_ring_checks`: два независимых запуска от t=0, без продолжения эталона: "
            "А — 256², Δx=1 мкм, Δt=0.05 пс; Б — 512², Δx=0.5 мкм, Δt=0.025 пс. "
            "Оба до 2000 пс, L=256 мкм; эталон 512²/0.05 пс/2000 пс. Модель, P0=3.16875, "
            "непрерывная затравка seed=20261008 и nR(0)=0 прежние. Совпадение затравки с "
            "начальным снимком исходного эталона и в узлах грубой сетки проверяется точно. "
            "Метод Strang/RK4, полные снимки через 100 пс, плотная диагностика и лаги стационарности прежние.", "",
            "Ошибки NC,NR нормированы на эталон (<0.5%); плотность — L2-норма разности по всей области "
            "(<1%), для А эталон берётся в каждом втором узле без интерполяции; Lz/(ħNC) — относительная "
            "разница с эталоном (<1%). q=−3 должен совпасть на минимум двух общих корректных контурах "
            "r=4/6/7.5/9/12 мкм. Заряд и момент — `vortex_diagnostics` как ниже. Каждый запуск "
            "должен самостоятельно иметь STATIONARY_MODE в последнем плотном окне 100 пс.", ""]
        for name, item in ring_checks.get("comparisons", {}).items():
            if "NC_relative" in item:
                lines.append(f"{name}: **{item['assessment']}**, ΔNC={100*item['NC_relative']:.4g}%, "
                             f"ΔNR={100*item['NR_relative']:.4g}%, ΔnC={100*item['density_map_relative']:.4g}%, "
                             f"q={item['vortex']['q']}, Lz/(ħNC)={item['vortex']['Lz_over_hbar_NC']:.9g}, "
                             f"twall={item['elapsed_wall_s']:.1f} с.")
        lines += [f"Итог: **{ring_checks.get('assessment', 'NOT_TESTED')}**. "
                  f"[Конфигурации, поля, контуры и проверки]({link(ring_checks_path)}). "
                  "Это проверка чувствительности найденного состояния к двум уточнениям при одной "
                  "затравке и области; физические возмущения и универсальность выбора заряда не исследовались.", ""]
    practical_path = Path(data_path).parent/"practical_checks.json"
    if practical_path.exists():
        practical = json.loads(practical_path.read_text())
        lines += ["", "## Дополнительные проверки из ноутбука", "",
                  "`numerics.run_practical_checks` (ячейка `DO_PRACTICAL_CHECKS`) повторяет тесты и сравнивает три варианта "
                  "с сохранёнными эталонами при L=192 мкм. Модель, накачка и затравка "
                  "совпадают; меняется только сетка или шаг. Метод — прежний Strang/RK4. "
                  "Снимки новых прогонов через 100 пс, остальная диагностика прежняя. "
                  "Сравнение — `compare_runs` при одинаковом конечном времени: "
                  "число частиц, L2-норма карты плотности в центральном круге L/4: "
                  r"$\varepsilon_N=|N_a-N_b|/N_b$, "
                  r"$\varepsilon_n=\|n_a-n_b\|_{2,core}/\|n_b\|_{2,core}$. "
                  "Для сеток тонкое поле выбирается в узлах грубого без интерполяции. "
                  "Допуски 0.5% и 1%; также RMSE числа частиц за последние 100 пс "
                  "(трапеции по diagnostics.csv с выборкой 10 пс, нормировка на среднее эталона; "
                  "допуск 0.5%). Дополнительно проверяются баланс, край и высокие k. "
                  "Для кольца прохождение этой проверки не означает стационарности. "
                  f"[Параметры и результаты]({link(practical_path)}).", "",
                  "| Вариант | Сетка | L, мкм | Δt, пс | t, пс |",
                  "|---|---|---:|---:|---:|"]
        for label, item in practical.get("cases", {}).items():
            run = _run(item["run_path"])
            c = run["config"]
            lines.append(f"| {label} | {c['space']['Nx']}² | {c['space']['Lx_um']:g} | "
                         f"{c['time']['dt_ps']:g} | {c['time']['t_final_ps']:g} |")
    experiment_file = Path(data_path).parent/"ring_dynamics.json"
    if experiment_file.exists():
        experiment = json.loads(experiment_file.read_text())
        if experiment.get("time_ps"):
            lines[0] = "# Как рассчитаны рисунки 01–09"
            lines[2] = lines[2].replace("Номера относятся", "Рисунки 01–08 и дополнительный эксперимент 09. Номера относятся")
            lines += ["", "## Рис. 09 — временная динамика кольца", "",
                f"Продолжение из точной контрольной точки 1000 пс до {experiment['time_ps']:g} пс; "
                "исходный прогон не изменён. Сетка 512², L=256 мкм, Δx=0.5 мкм, Δt=0.05 пс; "
                "тот же Strang/RK4, поле complex128 и резервуар float64. "
                "Населённости и центральная плотность сохраняются через 1 пс; "
                "плотность в архиве float32 (без влияния на интегратор), квадрат 80×80 мкм, "
                "хранится скользящий хвост 500 пс. Полные поля — в снимках через 20 пс. "
                "В исходном интервале 500–1000 пс населённости доступны через 10 пс, карты через 20 пс.", "",
                "В каждом окне 100 пс: размах NC, NR и NC+NR, делённый на среднее; "
                "изменение нормированной формы s=nC/∫core nC dA: max ||s(t)−s(t0)||₂/||s(t0)||₂, "
                "также сравнение кадров через 20 пс. Стационарность — прежние полные критерии.", "",
                "Период: автокорреляция после линейного detrend используется только для поиска "
                "кандидата (пик ≥0.6, выраженность ≥0.1; минимум два цикла в 100 пс, три в 500 пс). "
                "Период уточняется минимизацией ошибки возврата с линейной интерполяцией между 1-пс кадрами. "
                "Подтверждение: исходная плотность и нормированная форма возвращаются с RMS-ошибкой <1%, "
                "NC и NR — <0.5%; дрейф при этой проверке не вычитается. "
                "Амплитуда гармонической компоненты (sin/cos + линейный фон, least squares) "
                "в двух половинах должна иметь отношение 0.8…1.25. "
                "При максимальном отклонении плотности от среднего <0.5% и размахах NC/NR <0.5% "
                "выводится отсутствие значимой осцилляции, а не фиктивный период. Для остановки при 1500 пс "
                "нужно установление двух последних окон или возврат плотности в обеих половинах "
                "последних 500 пс с согласованием периода <5%. Это проверка периодичности плотности, "
                "а не доказательство TIME_PERIODIC комплексного поля. Длинные периоды при недостатке циклов остаются неподтверждёнными.", "",
                "Максимумы: maximum_filter 5×5, уровень ≥20% максимума; плато — один пик. "
                "Соответствие соседних кадров — венгерский алгоритм, скачок ≤2 мкм за 1 пс. "
                "Угол каждого долгоживущего пика разворачивается отдельно. Дополнительно "
                "Aₘ=2|∫core nC exp(imθ)dA|/∫core nC dA измеряет угловую деформацию; "
                "m выбирается по максимальной амплитуде исходного кадра 1000 пс среди m=1…12. "
                "Движение максимума при исчезающем контрасте не равно сильной динамике всей плотности.", "",
                "09: NC и NR при 0–2000 пс, карты при 1000/1200/1500/2000 пс; "
                "карты 40×40 мкм, одна абсолютная цветовая шкала. У NC/NR отдельные вертикальные "
                "шкалы, пунктир отмечает времена карт. До 500 пс история добавлена из diagnostics.csv "
                "исходного прогона; населённости при t=0 вычислены из начального снимка. "
                "Плотности взяты из полных снимков, "
                "с шагом исходной сетки, без интерполяции изображения. "
                "Код: `run_ring_dynamics_experiment`, `density_periodicity`, `density_maxima`, `plot_ring_dynamics`. "
                f"[Окна и итог]({link(experiment_file)}); [массивы]({link(experiment_file.with_suffix('.npz'))}).", "",
                "## Заряд и угловой момент при 2000 пс", "",
                r"$q=(2\pi)^{-1}\sum_j\arg[\psi_{j+1}\psi_j^*]$ на замкнутых окружностях "
                "r=4/6/7.5/9/12 мкм вокруг центра накачки, обход против часовой стрелки. "
                "Билинейная интерполяция комплексного поля, не фазы; 1024 отсчёта, "
                "проверка 512/2048. Контур допустим при min nC≥5% max nC, "
                "max |Δφ|<π/2 и ошибке целочисленного заряда <10⁻⁶. Также считается "
                "qj=∮(j/nC)·dl/[2π(2K/ħ)] как проверка направления и силы циркуляции.", "",
                r"$L_z=\hbar\operatorname{Re}\sum\psi^*[-i(x\partial_y-y\partial_x)]\psi\,\Delta x\Delta y$ "
                "по всей области, координаты относительно центра накачки. Производные FFT "
                "как у тока; приводятся Lz в мэВ·пс, Lz/(ħ NC) и мнимая невязка. "
                "Код: `field_gradients`, `vortex_diagnostics`; результаты в том же `ring_dynamics.json` "
                "и общей диагностике. Тесты: аналитические вихри q=±3 и радиальный поток q=0.", ""]
    notes_path.write_text("\n".join(lines), encoding="utf-8")


def _ring_transient_figure(history, maps, out):
    """Figure 9 from population samples and four saved density maps."""
    end = float(history[-1, 0])
    map_times = tuple(t for t, *_ in maps)
    with plt.rc_context(REPORT_STYLE):
        fig=plt.figure(figsize=(REPORT_WIDTH_IN,4.65),layout='constrained')
        gs=fig.add_gridspec(3,4,height_ratios=(1,.9,.075))
        for col,key in enumerate(('C','R')):
            ax=fig.add_subplot(gs[0,col*2:col*2+2])
            ax.plot(history[:,0],history[:,col+1],color=('tab:blue','tab:orange')[col])
            ax.set_title(f"({chr(97+col)}) "+('Condensate population' if col==0 else 'Reservoir population'),loc='left')
            ax.set_ylabel(rf'$N_{key}$');ax.set_xlabel('Time, $t$ (ps)')
            ax.set_xlim(0,end);ax.set_xticks([0,500,1000,1500,end])
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            for t in map_times:ax.axvline(t,color='0.55',linestyle=':',linewidth=.65,zorder=0)
        vmax=max(n.max() for _,n,_,_ in maps)
        for col,(t,n,xx,yy) in enumerate(maps):
            ax=fig.add_subplot(gs[1,col]);dx,dy=xx[1]-xx[0],yy[1]-yy[0]
            im=ax.imshow(n,origin='lower',extent=(xx[0]-dx/2,xx[-1]+dx/2,yy[0]-dy/2,yy[-1]+dy/2),
                cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
            ax.set_title(f"({chr(99+col)}) {t:g} ps",loc='left',fontsize=9)
            ax.set_xlabel(r'$x$ ($\mu$m)');ax.set_xticks([-15,0,15]);ax.set_yticks([-15,0,15]);ax.grid(False)
            if col==0:ax.set_ylabel(r'$y$ ($\mu$m)')
            else:ax.tick_params(labelleft=False)
        cax=fig.add_subplot(gs[2,:]);cax.grid(False)
        fig.colorbar(im,cax=cax,orientation='horizontal',label=r'Condensate density, $n_C$ ($\mu$m$^{-2}$)')
        paths=[_save(fig,out,Path(STAGE2_IMAGES[9]).stem)]
    return paths


def plot_ring_dynamics(report, archive, *, output_dir="results/report_figures"):
    """One report figure: population histories and four saved density maps."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    with np.load(archive) as z:
        history=z['history'];times=z['density_times_ps'];density=z['density_core_um2_inv']
        x,y=z['x_um'],z['y_um'];tracks=json.loads(str(z['peaks_json']))
    source=_run(report['source_run']);continued=_run(report['run_dir'])
    # Prepend the original run's early history; the continuation archive
    # starts at 500 ps. The t=0 populations come from the saved initial field.
    if history[0, 0] > 0:
        import csv
        grid = make_grid(source['config'])
        with np.load(source['snapshots'][0]) as initial:
            if float(initial['time_ps']) != 0:
                raise ValueError('Figure 9 requires the saved t=0 snapshot')
            early = [[0., float(np.sum(abs(initial['psi'])**2)*grid.da),
                      float(np.sum(initial['nR'])*grid.da)]]
        with (source['path']/'diagnostics.csv').open(newline='') as stream:
            early.extend([float(row[key]) for key in ('t_ps', 'NC', 'NR')]
                         for row in csv.DictReader(stream)
                         if 0 < float(row['t_ps']) < history[0, 0])
        history = np.vstack((early, history))
    if not np.all(np.diff(history[:, 0]) > 0):
        raise ValueError('Population history times must be strictly increasing')
    end=report['time_ps']
    map_times=(1000.,1200.,1500.,end)
    maps=[]
    for t in map_times:
        run=source if t<=1000 else continued
        step=round(t/run['config']['time']['dt_ps'])
        with np.load(run['path']/'snapshots'/f'step_{step:08d}.npz') as z:
            xx,yy=z['x_um'],z['y_um'];ix=np.flatnonzero(abs(xx)<=20);iy=np.flatnonzero(abs(yy)<=20)
            maps.append((t,abs(z['psi'][np.ix_(iy,ix)])**2,xx[ix],yy[iy]))
    paths = _ring_transient_figure(history, maps, out)
    # Track strong early peaks separately from the weak late modulation.
    def measure_motion(start,stop):
        by_id={}
        for row in tracks:
            if not start <= row['time_ps'] <= stop:continue
            for p in row['peaks']:
                by_id.setdefault(p['id'],[]).append([row['time_ps'],p['x_um'],p['y_um']])
        motion=[]
        for ident,points in by_id.items():
            a=np.array(points)
            if len(a)<50 or np.min(np.hypot(a[:,1],a[:,2])) < 2.5:continue
            angles=np.unwrap(np.arctan2(a[:,2],a[:,1]))
            slope,intercept=np.polyfit(a[:,0],angles,1)
            fitted=slope*a[:,0]+intercept
            variance=np.sum((angles-angles.mean())**2)
            motion.append({'track_id':ident,'time_window_ps':[float(a[0,0]),float(a[-1,0])],
                'angular_speed_rad_per_ps':float(slope),'angular_displacement_rad':float(angles[-1]-angles[0]),
                'net_displacement_um':float(np.linalg.norm(a[-1,1:]-a[0,1:])),
                'radius_mean_um':float(np.hypot(a[:,1],a[:,2]).mean()),
                'linear_angle_R2':float(1-np.sum((angles-fitted)**2)/variance) if variance>0 else None})
        return sorted(motion,key=lambda m:m['time_window_ps'][1]-m['time_window_ps'][0],reverse=True)
    report['peak_motion']=measure_motion(end-500,end)
    report['initial_peak_motion']=measure_motion(1000,1200)
    # Determine the dominant angular modulation from the original density,
    # rather than guessing its symmetry from the brightest global maximum.
    with np.load(source['snapshots'][-1]) as z:
        ix=np.flatnonzero(abs(z['x_um'])<=40);iy=np.flatnonzero(abs(z['y_um'])<=40)
        source_density=abs(z['psi'][np.ix_(iy,ix)])**2
    theta=np.arctan2(y[:,None],x[None,:])
    source_amplitudes={m:float(2*abs(np.sum(source_density*np.exp(1j*m*theta)))/source_density.sum())
                       for m in range(1,13)}
    dominant=max(source_amplitudes,key=source_amplitudes.get)
    moment=np.einsum('tyx,yx->t',density,np.exp(1j*dominant*theta),optimize=True)
    amplitude=2*abs(moment)/np.maximum(density.sum(axis=(1,2)),1e-30)
    orientation=np.unwrap(np.angle(moment))/dominant
    slope,intercept=np.polyfit(times,orientation,1)
    report['angular_modulation']={'definition':'Am=2 abs(integral nC exp(i m theta) dA) / integral nC dA',
        'dominant_m':int(dominant),'source_amplitudes':source_amplitudes,
        'window_ps':[float(times[0]),float(times[-1])],
        'at_source_1000_ps':source_amplitudes[dominant],
        'initial':float(amplitude[0]),'final':float(amplitude[-1]),
        'maximum':float(amplitude.max()),'pattern_angular_speed_rad_per_ps':float(slope),
        'orientation_displacement_rad':float(orientation[-1]-orientation[0])}
    from numerics import vortex_diagnostics
    with np.load(continued['snapshots'][-1]) as z:
        psi=z['psi']
    grid=make_grid(continued['config'])
    vortex=vortex_diagnostics(psi,grid,continued['config'])
    vortex['sampling_checks']=[{'n_angles':count,
        'q':vortex_diagnostics(psi,grid,continued['config'],n_angles=count)['q']}
        for count in (512,1024,2048)]
    vortex['source_snapshot']=str(continued['snapshots'][-1]);vortex['time_ps']=end
    report['vortex_diagnostics']=vortex
    report['figures']=[str(p) for p in paths]
    registry=out/'figure_sources.json'
    if registry.exists():
        from PIL import Image
        sources=json.loads(registry.read_text())
        sources['ring_dynamics']=str(Path(archive).with_suffix('.json'))
        sources['ring_dynamics_run']=report['run_dir']
        for name in ('09_ring_population_dynamics.png','10_ring_density_evolution.png'):
            sources['titles'].pop(name,None);sources['figure_sizes_mm'].pop(name,None)
        sources['titles'][STAGE2_IMAGES[9]]=REPORT_TITLES[9]
        for p in paths:
            with Image.open(p) as im:
                sources['figure_sizes_mm'][p.name]=[round(n/dpi*25.4,1) for n,dpi in zip(im.size,im.info['dpi'])]
        registry.write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    return paths


def make_spatial_density_gifs(output_dir="results/report_figures", *, fps=5):
    """Animate figure 5 densities: one GIF per pump geometry.

    Reservoir and condensate panels share physical time. Each panel has a
    fixed linear scale over the entire saved trajectory (spot: 0–500 ps;
    ring: 0–2000 ps). No interpolation or new simulation is performed.
    """
    from PIL import Image

    if fps <= 0:
        raise ValueError("fps must be positive")
    out = Path(output_dir)
    sources = json.loads((out / "figure_sources.json").read_text())
    dynamics = json.loads(Path(sources["ring_dynamics"]).read_text())

    def frames(paths):
        result = {}
        for path in paths:
            for snapshot in load_run(path)["snapshots"]:
                with np.load(snapshot) as data:
                    x, y = data["x_um"], data["y_um"]
                    ix, iy = np.flatnonzero(abs(x) <= 20), np.flatnonzero(abs(y) <= 20)
                    selection = np.ix_(iy, ix)
                    reservoir = data["nR"][selection].copy()
                    condensate = abs(data["psi"][selection])**2
                    if not (np.isfinite(reservoir).all() and np.isfinite(condensate).all()):
                        raise ValueError(f"Nonfinite field in {snapshot}")
                    dx, dy = x[1]-x[0], y[1]-y[0]
                    extent = (x[ix[0]]-dx/2, x[ix[-1]]+dx/2,
                              y[iy[0]]-dy/2, y[iy[-1]]+dy/2)
                    result[float(data["time_ps"])] = (reservoir, condensate, extent)
        return result

    spot = frames([sources["gaussian_run"]])
    ring = frames([dynamics["source_run"], sources["ring_run"]])
    # Fixed palette prevents frame-dependent colour quantisation/flicker.
    palette = Image.new("P", (1, 1))
    rgb = np.vstack((plt.get_cmap("viridis")(np.linspace(0, 1, 224))[:, :3],
                     np.repeat(np.linspace(0, 1, 32)[:, None], 3, axis=1)))
    palette.putpalette(np.round(255*rgb).astype("uint8").ravel().tolist())
    out.mkdir(parents=True, exist_ok=True)
    files = []
    with plt.rc_context(REPORT_STYLE):
        for name, geometry, collection in (("spot", "Gaussian spot", spot),
                                            ("ring", "Ring pump", ring)):
            times = sorted(collection)
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.1), dpi=110,
                                     layout="constrained")
            images = []
            first = collection[times[0]]
            for column, (ax, label, symbol) in enumerate(zip(
                    axes, ("Reservoir", "Condensate"), ("R", "C"))):
                vmax = max(float(frame[column].max()) for frame in collection.values())
                im = ax.imshow(first[column], origin="lower", extent=first[2],
                               cmap="viridis", vmin=0, vmax=vmax,
                               interpolation="nearest")
                images.append(im)
                ax.grid(False)
                ax.set(xlim=(-20, 20), ylim=(-20, 20), aspect="equal",
                       xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)")
                ax.set_title(label, fontsize=11)
                ax.set_xticks([-20, 0, 20]); ax.set_yticks([-20, 0, 20])
                cb = fig.colorbar(im, ax=ax, shrink=.78, pad=.03)
                cb.set_label(r"$n_" + symbol + r"$ ($\mu$m$^{-2}$)")
            rendered = []
            try:
                for t in times:
                    for column, im in enumerate(images):
                        im.set_data(collection[t][column])
                    fig.suptitle(f"{geometry} · t = {t:g} ps", fontsize=13)
                    fig.canvas.draw()
                    rgb_frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
                    rendered.append(rgb_frame.quantize(palette=palette, dither=Image.Dither.NONE))
                durations = [max(10, round(1000/fps/10)*10)] * len(rendered)
                durations[0], durations[-1] = round(4*1000/fps), round(18*1000/fps)
                file = out / f"05_{name}_evolution.gif"
                rendered[0].save(file, save_all=True, append_images=rendered[1:],
                                 duration=durations, loop=0, disposal=2, optimize=False)
                files.append(file)
            finally:
                plt.close(fig)
                for frame in rendered:
                    frame.close()
    return files


REPORT_DATA = Path(__file__).resolve().parent / "results/analysis_data/report_data.npz"


def load_report_case(name="ring", data_file=REPORT_DATA):
    """Load an archived final field without the original run directory."""
    data_file = Path(data_file)
    metadata = json.loads(data_file.with_name("report_metadata.json").read_text())
    case = metadata["cases"][name]
    with np.load(data_file, allow_pickle=False) as saved:
        return {"path": data_file, "config": case["config"],
                "psi": saved[name + "_psi"].copy(),
                "nR": saved[name + "_nR"].copy(), "time_ps": case["time_ps"]}


def make_report(*, data_file=REPORT_DATA, output_dir="results/report_figures"):
    """Redraw nine report PNGs from portable data, without propagation.

    The saved fields retain their original precision, grids and physical units.
    This function does not regenerate the research notes or GIFs.
    """
    data_file, out = Path(data_file), Path(output_dir)
    metadata = json.loads(data_file.with_name("report_metadata.json").read_text())
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported report archive version")
    out.mkdir(parents=True, exist_ok=True)
    spot = load_report_case("gaussian", data_file)
    ring = load_report_case("ring", data_file)
    spectrum_cases = [load_report_case(name, data_file)
                      for name in metadata["gaussian_spectrum_cases"]]
    with np.load(data_file, allow_pickle=False) as saved:
        history = saved["ring_history"].copy()
        maps = [(t, saved[f"ring_map_{i}"].copy(), saved["ring_map_x"].copy(),
                 saved["ring_map_y"].copy())
                for i, t in enumerate(metadata["ring_map_times_ps"])]
    with plt.rc_context(REPORT_STYLE):
        paths = [_figure_0d(spot["config"], out)]
        paths.extend(_stage2_0d_figures(data_file, out))
        paths.extend(_pair_figures(spot, ring, out,
                                  gaussian_spectrum_runs=spectrum_cases))
        paths.append(_stage2_threshold_figure(metadata["thresholds"], out))
        paths.extend(_ring_transient_figure(history, maps, out))
    return paths


def export_report_data(*, source_file="results/report_figures/figure_sources.json",
                       data_file=REPORT_DATA):
    """Package existing results for Git; require the full local research archive.

    No simulation or interpolation is performed. Provenance, reference
    configurations and recorded checks go in the adjacent JSON file.
    """
    import csv
    import hashlib

    source_file, data_file = Path(source_file), Path(data_file)
    sources = json.loads(source_file.read_text())
    data_dir = Path(sources["data_dir"])
    with np.load(data_dir / "stage2_0d_data.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    cases = {"gaussian": sources["gaussian_run"], "ring": sources["ring_run"]}
    spectrum_names = []
    for i, path in enumerate(sources["gaussian_spectrum_runs"]):
        name = f"gaussian_spectrum_{i}"
        cases[name] = path
        spectrum_names.append(name)
    metadata = {"format_version": 1, "units": "um, ps, meV",
                "cases": {}, "gaussian_spectrum_cases": spectrum_names,
                "source_registry": sources, "checks": {}, "thresholds": {}}
    for name, path in cases.items():
        run, psi, nr, time_ps = _latest(path)
        snapshot = run["snapshots"][-1]
        arrays[name + "_psi"], arrays[name + "_nR"] = psi, nr
        metadata["cases"][name] = {
            "config": run["config"], "time_ps": time_ps,
            "source_run": str(run["path"]), "source_snapshot": str(snapshot),
            "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "summary": run["summary"]}
    for key in ("gaussian_threshold", "ring_threshold"):
        metadata["thresholds"][key] = json.loads(Path(sources[key]).read_text())
    metadata["convergence"] = json.loads(Path(sources["convergence"]).read_text())
    for file in sorted(data_dir.glob("*.json")):
        if file.name != "report_metadata.json":
            metadata["checks"][file.stem] = json.loads(file.read_text())
    experiment = metadata["checks"]["ring_dynamics"]
    with np.load(data_dir / "ring_dynamics.npz", allow_pickle=False) as saved:
        history = saved["history"].copy()
    source, continued = _run(experiment["source_run"]), _run(experiment["run_dir"])
    if history[0, 0] > 0:
        grid = make_grid(source["config"])
        with np.load(source["snapshots"][0], allow_pickle=False) as initial:
            if float(initial["time_ps"]) != 0:
                raise ValueError("Missing initial snapshot")
            early = [[0., float(np.sum(abs(initial["psi"])**2)*grid.da),
                      float(np.sum(initial["nR"])*grid.da)]]
        with (source["path"] / "diagnostics.csv").open(newline="") as stream:
            early.extend([float(row[key]) for key in ("t_ps", "NC", "NR")]
                         for row in csv.DictReader(stream)
                         if 0 < float(row["t_ps"]) < history[0, 0])
        history = np.vstack((early, history))
    if not np.all(np.diff(history[:, 0]) > 0):
        raise ValueError("Population history must have increasing times")
    arrays["ring_history"] = history
    map_times = [1000., 1200., 1500., float(experiment["time_ps"])]
    metadata["ring_map_times_ps"] = map_times
    for i, t in enumerate(map_times):
        run = source if t <= 1000 else continued
        step = round(t / run["config"]["time"]["dt_ps"])
        with np.load(run["path"] / "snapshots" / f"step_{step:08d}.npz",
                     allow_pickle=False) as snapshot:
            x, y = snapshot["x_um"], snapshot["y_um"]
            ix, iy = np.flatnonzero(abs(x) <= 20), np.flatnonzero(abs(y) <= 20)
            arrays[f"ring_map_{i}"] = abs(snapshot["psi"][np.ix_(iy, ix)])**2
            arrays["ring_map_x"], arrays["ring_map_y"] = x[ix].copy(), y[iy].copy()
    data_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_file, **arrays)
    metadata["archive_sha256"] = hashlib.sha256(data_file.read_bytes()).hexdigest()
    metadata["scope"] = ("Portable figure data and recorded research checks. "
                         "Original run paths are provenance; full time-series "
                         "snapshots and checkpoints remain in local results/runs.")
    data_file.with_name("report_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return data_file
