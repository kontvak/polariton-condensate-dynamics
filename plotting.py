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


# Match the compact typography used in dispersion_curve figures.
REPORT_STYLE = {
    "figure.dpi": 120, "font.size": 9, "font.family": "DejaVu Sans",
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.linewidth": 0.8,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.w_pad": 0.04,
    "figure.constrained_layout.h_pad": 0.06,
}


REPORT_TITLES = {
    "F1": "Homogeneous condensation threshold",
    "F2": "Condensate and reservoir relaxation",
    "F3": "Spatial structure under localized pumping",
    "F4": "Particle flow and condensate phase",
    "F5": "Momentum distributions and domain-size comparison",
    "F6": "Linear condensation thresholds for localized pumps",
    "F7": "Numerical convergence and finite-domain effects",
}

def _save(fig, out, name, caption):
    """Export a PNG and append its English caption to the common document."""
    path = out/f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06,
                facecolor="white", pil_kwargs={"optimize": True})
    number = name.split("_")[0]
    with (out/"figure_captions.md").open("a", encoding="utf-8") as handle:
        handle.write(f"## Figure {number[1:]}. {REPORT_TITLES[number]}\n\n{caption}\n\n")
    plt.close(fig)
    return path


def _extent(grid):
    # The FFT grid coordinates are pixel centres, including x = -L/2.
    return (grid.x[0]-grid.dx/2, grid.x[-1]+grid.dx/2,
            grid.y[0]-grid.dy/2, grid.y[-1]+grid.dy/2)


def _map_axes(ax, *, row, col, label, limit=40):
    ax.grid(False)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.set_xticks([-limit, 0, limit])
    ax.set_yticks([-limit, 0, limit])
    ax.set_title(label, loc="left")
    if row == 1:
        ax.set_xlabel(r"$x$ ($\mu$m)")
    else:
        ax.tick_params(labelbottom=False)
    if col == 0:
        geometry = "Gaussian spot" if row == 0 else "Ring pump"
        ax.set_ylabel(geometry+"\n"+r"$y$ ($\mu$m)")
    else:
        ax.tick_params(labelleft=False)


def _case_caption(run, time_ps):
    c = run["config"]
    mode = run["summary"]["stationarity"]["stationarity_status"]
    state = {"STATIONARY_MODE": "stationary mode",
             "DYNAMIC_OR_UNSETTLED": "time-dependent or unsettled state",
             "DECAYED": "decayed signal", "TIME_PERIODIC": "time-periodic state",
             "NONSTATIONARY_FIELD": "nonstationary field",
             "UNDETERMINED": "low or unresolved signal",
             "NOT_TESTED": "stationarity not evaluated"}.get(mode, mode)
    return (f"P0 = {c['pump']['P0']:g}, t = {time_ps:g} ps, "
            f"N = {c['space']['Nx']} × {c['space']['Ny']}, "
            f"L = {c['space']['Lx_um']:g} × {c['space']['Ly_um']:g} µm, "
            f"Δt = {c['time']['dt_ps']:g} ps; {state}")


def _figure_0d(data_dir, out):
    from matplotlib.lines import Line2D

    files = []
    with np.load(data_dir/"F1_0d_data.npz") as a:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
        axes[0].plot(a["P0"], a["nC"], label="Condensate")
        axes[0].plot(a["P0"], a["nR"], label="Reservoir")
        axes[0].axvline(1, color="0.4", ls="--", lw=1)
        axes[0].set(xlabel=r"Normalized pump, $P_0$",
                    ylabel=r"Density ($\mu$m$^{-2}$)")
        axes[0].set_title("(a) Steady-state densities", loc="left")
        axes[0].legend()
        axes[1].plot(a["P0"], a["empty_branch_growth"], color="tab:blue")
        axes[1].axhline(0, color="0.3", lw=1)
        axes[1].axvline(1, color="0.4", ls="--", lw=1)
        axes[1].set(xlabel=r"Normalized pump, $P_0$",
                    ylabel=r"Largest $\mathrm{Re}\,\lambda$ (ps$^{-1}$)")
        axes[1].set_title("(b) Empty-state stability", loc="left")
        files.append(_save(fig, out, "F1_0d_threshold",
            "Homogeneous condensation threshold. (a) Physical steady-state "
            "condensate and reservoir densities. (b) Largest real part of the "
            "density–reservoir Jacobian eigenvalues on the empty branch. "
            "P0 = P/Pth,0D; the dashed line marks P0 = 1. "
            "These homogeneous results do not define the threshold of a localized pump."))
    with np.load(data_dir/"F2_0d_data.npz") as a:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
        for i, label in enumerate(("0p8", "1p05", "1p2", "1p5")):
            color = f"C{i}"
            power_label = label.replace("p", ".")
            axes[0].plot(a["t_ps"], a[f"nC_{label}"], color=color)
            axes[1].plot(a["t_ps"], a[f"nR_{label}"], color=color,
                         label=power_label)
            if f"nC_SL_{label}" in a.files:
                axes[0].plot(a["t_ps"], a[f"nC_SL_{label}"], ls="--", color=color)
        axes[0].set(xlabel=r"Time, $t$ (ps)",
                    ylabel=r"Condensate density, $n_C$ ($\mu$m$^{-2}$)")
        axes[1].set(xlabel=r"Time, $t$ (ps)",
                    ylabel=r"Reservoir density, $n_R$ ($\mu$m$^{-2}$)")
        axes[0].set_title("(a) Condensate dynamics", loc="left")
        axes[1].set_title("(b) Reservoir dynamics", loc="left")
        axes[0].legend(handles=[
            Line2D([], [], color="0.25", label="Full 0D model"),
            Line2D([], [], color="0.25", ls="--", label="Stuart–Landau")],
            loc="upper left")
        axes[1].legend(title=r"$P_0$", loc="lower right", ncol=2)
        files.append(_save(fig, out, "F2_0d_dynamics",
            "Homogeneous condensate and reservoir dynamics for P0 = 0.8, 1.05, "
            "1.2 and 1.5 (the same colours in both panels). Solid curves: independent "
            "integration of the full 0D model with nR(0) = 0 and nC(0) = 10⁻⁴ µm⁻². "
            "Dashed curves: Stuart–Landau approximation above threshold. Its "
            "adiabatic reservoir assumption differs from the full-model startup; "
            "the approximation is intended for the near-threshold regime."))
    return files


def _pair_figures(gaussian_run, ring_run, out, *,
                  gaussian_spectrum_runs=(), ring_spectrum_runs=()):
    gauss, gp, gn, gt = _latest(gaussian_run)
    ring, rp, rn, rt = _latest(ring_run)
    pairs = ((gauss, gp, gn, gt), (ring, rp, rn, rt))
    cases = ("Top row: Gaussian spot ("+_case_caption(gauss, gt)+"). "
             "Bottom row: ring pump ("+_case_caption(ring, rt)+"). ")
    arrays = []
    for run, psi, nr, _time in pairs:
        grid = make_grid(run["config"])
        p = parameters(run["config"])
        pump = make_pump(run["config"], grid)
        bg = p["eta"]*pump if p["background_blueshift"] else 0
        energy = p["gC"]*abs(psi)**2+p["gR"]*(nr+bg)
        arrays.append((pump, nr, abs(psi)**2, energy))

    fig = plt.figure(figsize=(10.2, 5.9), layout="constrained")
    gs = fig.add_gridspec(3, 4, height_ratios=(1, 1, 0.075), hspace=0.06)
    titles = ("Pump rate", "Reservoir density", "Condensate density", "Interaction energy")
    units = (r"$P$ ($\mu$m$^{-2}$ ps$^{-1}$)",
             r"$n_R$ ($\mu$m$^{-2}$)", r"$n_C$ ($\mu$m$^{-2}$)",
             r"$V_{\mathrm{int}}$ (meV)")
    for col in range(4):
        vmax = max(float(np.max(row[col])) for row in arrays)
        for row in range(2):
            ax = fig.add_subplot(gs[row, col])
            grid = make_grid(pairs[row][0]["config"])
            im = ax.imshow(arrays[row][col], origin="lower", extent=_extent(grid),
                           vmin=0, vmax=vmax, cmap="viridis", interpolation="nearest",
                           rasterized=True)
            _map_axes(ax, row=row, col=col,
                      label=f"({chr(97+row*4+col)}) {titles[col]}")
        cax = fig.add_subplot(gs[2, col])
        cax.grid(False)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.locator_params(nbins=3)
        cb.set_label(units[col])
    files = [_save(fig, out, "F3_density_reservoir_potential",
        "Real-space pump and condensate profiles. "+cases+
        "Columns show the pump rate, reservoir density, condensate density and "
        "interaction-induced energy shift Vint = gC nC + gR(nR + ηP). "
        "Each column uses the same colour scale for both geometries. "
        "The displayed region is the central 80 × 80 µm; the full numerical domain "
        "has periodic boundaries and no absorbing layer.")]

    fig = plt.figure(figsize=(7.2, 7.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 0.065), hspace=0.04)
    max_density = max(float(np.max(abs(psi)**2)) for _, psi, _, _ in pairs)
    max_current = max(float(np.max(np.hypot(*current(
        psi, make_grid(run["config"]), run["config"])))) for run, psi, _, _ in pairs)
    phase_cmap = plt.get_cmap("twilight").copy()
    phase_cmap.set_bad("0.94")
    arrow_reference = 0.5 if max_current >= 0.5 else max_current/2
    for row, (run, psi, _nr, _time) in enumerate(pairs):
        grid = make_grid(run["config"])
        density = abs(psi)**2
        jx, jy = current(psi, grid, run["config"])
        density_ax, phase_ax = [fig.add_subplot(gs[row, col]) for col in range(2)]
        density_im = density_ax.imshow(density, origin="lower", extent=_extent(grid),
            cmap="magma", vmin=0, vmax=max_density, interpolation="nearest", rasterized=True)
        # Sample the vector field at a fixed physical spacing within the crop.
        stride = max(1, round(3.0/grid.dx))
        ix = np.flatnonzero(abs(grid.x) <= 37)[::stride]
        iy = np.flatnonzero(abs(grid.y) <= 37)[::stride]
        sampled_jx, sampled_jy = jx[np.ix_(iy, ix)], jy[np.ix_(iy, ix)]
        weak_current = np.hypot(sampled_jx, sampled_jy) < 0.02*max_current
        vectors = density_ax.quiver(grid.x[ix], grid.y[iy],
            np.ma.masked_where(weak_current, sampled_jx),
            np.ma.masked_where(weak_current, sampled_jy), color="cyan",
            scale=max(max_current/5, 1e-30), scale_units="xy", angles="xy",
            width=0.005, headwidth=3.5, minlength=0)
        density_ax.quiverkey(vectors, 0.74, 0.08, arrow_reference,
            rf"${arrow_reference:g}\;\mu\mathrm{{m}}^{{-1}}\mathrm{{ps}}^{{-1}}$",
            labelpos="N", coordinates="axes", color="cyan", labelcolor="white",
            fontproperties={"size": 7})
        phase = np.ma.masked_where(density < 1e-3*np.max(density), np.angle(psi))
        phase_im = phase_ax.imshow(phase, origin="lower", extent=_extent(grid),
            cmap=phase_cmap, vmin=-np.pi, vmax=np.pi, interpolation="nearest", rasterized=True)
        _map_axes(density_ax, row=row, col=0,
                  label=f"({chr(97+row*2)}) Density and current")
        _map_axes(phase_ax, row=row, col=1,
                  label=f"({chr(98+row*2)}) Condensate phase")
    for col, (im, label) in enumerate(((density_im, r"$n_C$ ($\mu$m$^{-2}$)"),
                                     (phase_im, r"Phase, $\arg\psi$ (rad)"))):
        cax = fig.add_subplot(gs[2, col])
        cax.grid(False)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(label)
        if col == 1:
            cb.set_ticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
        else:
            cb.ax.locator_params(nbins=4)
    files.append(_save(fig, out, "F4_current_phase",
        "Condensate flow and phase. "+cases+
        "Left: density with particle-current vectors; both rows use the same "
        "density and arrow scales. Right: condensate phase, masked where "
        "nC < 10⁻³ max(nC) in each run. Grey areas have unresolved phase. "
        "Arrows are sampled every 3 µm; currents below 2% of the shared maximum "
        "are omitted. The displayed region is 80 × 80 µm."))

    def spectrum_source(source):
        run, psi, _nr, time_ps = _latest(source)
        return run, psi, time_ps
    spectrum_rows = [
        [(gauss, gp, gt)]+[spectrum_source(source) for source in gaussian_spectrum_runs],
        [(ring, rp, rt)]+[spectrum_source(source) for source in ring_spectrum_runs],
    ]
    columns = max(map(len, spectrum_rows))
    fig = plt.figure(figsize=(min(10.2, 3.6*columns+0.6), 6.0), layout="constrained")
    gs = fig.add_gridspec(2, columns+1, width_ratios=[1]*columns+[0.05])
    maximum = max(float(np.max(abs(spectrum(psi, make_grid(run["config"])))**2))
                  for row in spectrum_rows for run, psi, _time in row)
    vmax = np.log10(max(maximum, 1e-20))
    vmin = vmax-8
    caption_cases = []
    for row, entries in enumerate(spectrum_rows):
        for col, (run, psi, time_ps) in enumerate(entries):
            ax = fig.add_subplot(gs[row, col])
            grid = make_grid(run["config"])
            power = np.fft.fftshift(abs(spectrum(psi, grid))**2)
            kx, ky = np.fft.fftshift(grid.kx), np.fft.fftshift(grid.ky)
            dkx, dky = kx[1]-kx[0], ky[1]-ky[0]
            im = ax.imshow(np.log10(np.maximum(power, 1e-30)), origin="lower",
                extent=(kx[0]-dkx/2, kx[-1]+dkx/2, ky[0]-dky/2, ky[-1]+dky/2),
                cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest", rasterized=True)
            geometry = "Gaussian spot" if row == 0 else "Ring pump"
            letter = chr(97+row*columns+col)
            ax.set_title(f"({letter}) {geometry}, "+rf"$L={grid.lx:g}\,\mu$m", loc="left")
            ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_yticks([-4, -2, 0, 2, 4])
            ax.grid(False)
            if row == 1:
                ax.set_xlabel(r"$k_x$ ($\mu$m$^{-1}$)")
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(r"$k_y$ ($\mu$m$^{-1}$)")
            else:
                ax.tick_params(labelleft=False)
            caption_cases.append(f"({letter}) {geometry}: "+_case_caption(run, time_ps)+".")
    cax = fig.add_subplot(gs[:, -1])
    cax.grid(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}[|\widetilde\psi|^2/(\mu\mathrm{m}^2)]$")
    files.append(_save(fig, out, "F5_momentum_spectra",
        "Momentum-space condensate density. All panels use the same absolute "
        "logarithmic colour scale and Fourier convention. "+" ".join(caption_cases)+
        " The smaller-domain panels are comparison runs; their boundary diagnostics "
        "must be checked before interpreting differences as physical effects."))
    return files


def _threshold_figure(results, out):
    available = [(label, source) for label, source in results if source is not None]
    if not available:
        return None
    fig, axes = plt.subplots(1, len(available), figsize=(7.2, 3.2),
                             squeeze=False, layout="constrained")
    any_data = False
    captions = []
    for ax, (label, source), letter in zip(axes[0], available, "ab"):
        data = source if isinstance(source, (dict, list)) else json.loads(Path(source).read_text())
        bracket = data.get("threshold", data) if isinstance(data, dict) else {}
        points = data if isinstance(data, list) else bracket.get("points", [])
        if points:
            powers = [p["P0"] for p in points]
            margin = max(0.05*(max(powers)-min(powers)), 0.05)
            ax.set_xlim(min(powers)-margin, max(powers)+margin)
        if "P0_lo" in bracket and "P0_hi" in bracket:
            lo, hi = bracket["P0_lo"], bracket["P0_hi"]
            mid = (lo+hi)/2
            ax.set_xlim(lo-0.25*mid, hi+0.25*mid)
            ax.axvspan(lo, hi, color="0.5", alpha=0.18)
            ax.text(0.04, 0.96, rf"$P_{{0,c}}\in[{lo:.6g},\,{hi:.6g}]$",
                    transform=ax.transAxes, va="top", fontsize=8)
            captions.append(f"{label}: P0,c ∈ [{lo:.6g}, {hi:.6g}].")
        for status, marker, color, legend in (("STABLE", "o", "tab:blue", "Decaying"),
                ("UNSTABLE", "^", "tab:orange", "Growing"),
                ("UNRESOLVED", "x", "0.5", "Unresolved")):
            selected = [p for p in points if p.get("status") == status
                        and p.get("sigma_late_per_ps") is not None
                        and ax.get_xlim()[0] <= p["P0"] <= ax.get_xlim()[1]]
            if selected:
                ax.errorbar([p["P0"] for p in selected],
                    [p["sigma_late_per_ps"] for p in selected],
                    yerr=[2*p.get("sigma_late_stderr", 0) for p in selected],
                    fmt=marker, ms=4, color=color, capsize=2, label=legend)
                any_data = True
        ax.axhline(0, color="0.3", lw=1)
        geometry = "Gaussian spot" if label == "gaussian" else "Ring pump"
        ax.set_title(f"({letter}) {geometry}", loc="left")
        ax.set_xlabel(r"Normalized pump, $P_0$")
        ax.set_ylabel(r"Linear growth rate, $\sigma$ (ps$^{-1}$)")
        ax.legend(loc="lower right", fontsize=7)
    if not any_data:
        plt.close(fig)
        return None
    return _save(fig, out, "F6_linear_threshold",
        "Localized-pump condensation thresholds from linear stability of the "
        "empty branch, with the reservoir frozen at nR = P/γR. "
        "P0 normalizes the pump amplitude by the homogeneous threshold. "
        "Symbols distinguish decaying, growing and unresolved perturbations; "
        "error bars show two standard errors of the late-time growth-rate fit. "
        "Shaded bands mark the refined threshold brackets. "+" ".join(captions)+
        " Only the neighbourhood of each threshold is displayed; complete scans "
        "and grid/domain validation are stored with the numerical results.")


def _convergence_figure(source, out):
    if source is None:
        return None
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    comparisons = [r for r in data["comparisons"] if r["status"] == "COMPARED"]
    if not comparisons:
        return None
    groups = [("Spatial resolution", [r for r in comparisons if r["comparison"].startswith("grid_")]),
              ("Time-step refinement", [r for r in comparisons if r["comparison"].startswith("dt_")]),
              ("Domain size", [r for r in comparisons if r["comparison"] == "domain_expansion"])]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharey=True, layout="constrained")
    for ax, (title, entries), letter in zip(axes, groups, "abc"):
        x = np.arange(len(entries))
        for key, label, marker in (("NC_relative", r"$N_C$", "o"),
                ("NR_relative", r"$N_R$", "s"),
                ("r_rms_um_relative", r"$r_{\mathrm{rms}}$", "^"),
                ("density_map_relative", r"$n_C$ map", "D")):
            if entries and all(key in r for r in entries):
                ax.semilogy(x, [max(r[key], 1e-15) for r in entries],
                            marker+"-", ms=4, label=label)
        ax.axhline(0.005, color="0.35", ls="--", lw=1, label="0.5% reference")
        labels = []
        for entry in entries:
            name = entry["comparison"]
            if name.startswith("grid_"):
                labels.append(name.removeprefix("grid_").replace("_vs_", " → "))
            elif name.startswith("dt_"):
                labels.append(name.removeprefix("dt_").replace("_vs_", " → "))
            else:
                a, b = _run(entry["reference"]), _run(entry["candidate"])
                labels.append(f"{a['config']['space']['Lx_um']:g} → "
                              f"{b['config']['space']['Lx_um']:g}")
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.35, max(len(entries)-1, 0)+0.35)
        ax.set_title(f"({letter}) {title}", loc="left")
    axes[0].set(ylabel="Relative difference", xlabel="Grid refinement")
    axes[1].set_xlabel(r"$\Delta t$ refinement (ps)")
    axes[2].set_xlabel(r"Domain width ($\mu$m)")
    axes[2].legend(loc="lower left", fontsize=7)
    time_ps = data.get("time_ps", "unknown")
    mode = data.get("reference_stationarity_status", "NOT_TESTED")
    state = {"STATIONARY_MODE": "stationary mode",
             "DYNAMIC_OR_UNSETTLED": "transient or time-dependent state"}.get(mode, mode)
    reference_path = data.get("cases", {}).get("grid_512", comparisons[0]["candidate"])
    reference_config = _run(reference_path)["config"]
    geometry = "Gaussian spot" if reference_config["pump"]["kind"] == "gaussian" else "ring pump"
    case = f"Representative case: {geometry}, P0 = {reference_config['pump']['P0']:g}. "
    return _save(fig, out, "F7_convergence",
        f"Numerical convergence at t = {time_ps} ps. "+case+"(a) Spatial refinement at "
        "fixed domain width and time step. (b) Time-step refinement on a fixed "
        "grid. (c) Domain expansion at fixed spatial spacing. Curves show relative "
        "differences in condensate number NC, reservoir number NR, condensate "
        "rms radius and the density map. The dashed line is the 0.5% reference "
        f"for integral observables. The reference run is a {state}. "
        "The same continuous initial field is sampled at identical physical coordinates "
        "in all runs. "
        "This series tests the state at the specified time; it does not establish "
        "convergence of a later stationary regime. Differences above the target "
        "are retained in the figure and diagnostic report.")


def make_figures(config, *, data_dir, output_dir,
                 gaussian_run=None, ring_run=None,
                 gaussian_threshold=None, ring_threshold=None,
                 convergence=None, gaussian_spectrum_runs=(),
                 ring_spectrum_runs=()):
    """Export report-ready PNG figures and one English caption document."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    paths = []
    (out/"figure_captions.md").write_text(
        "# Figure captions\n\nPNG figures are exported at 300 dpi.\n\n", encoding="utf-8")
    with plt.rc_context(REPORT_STYLE):
        if (data_dir/"F1_0d_data.npz").exists() and (data_dir/"F2_0d_data.npz").exists():
            paths.extend(_figure_0d(data_dir, out))
        if gaussian_run is not None and ring_run is not None:
            paths.extend(_pair_figures(gaussian_run, ring_run, out,
                gaussian_spectrum_runs=gaussian_spectrum_runs,
                ring_spectrum_runs=ring_spectrum_runs))
        threshold = _threshold_figure(
            (("gaussian", gaussian_threshold), ("ring", ring_threshold)), out)
        if threshold is not None:
            paths.append(threshold)
        convergence_plot = _convergence_figure(convergence, out)
        if convergence_plot is not None:
            paths.append(convergence_plot)
    def source_path(source):
        if source is None:
            return None
        if isinstance(source, dict) and "path" in source:
            return str(source["path"])
        return source if isinstance(source, (dict, list)) else str(source)
    sources = {"gaussian_run": source_path(gaussian_run), "ring_run": source_path(ring_run),
               "gaussian_spectrum_runs": [source_path(p) for p in gaussian_spectrum_runs],
               "ring_spectrum_runs": [source_path(p) for p in ring_spectrum_runs],
               "gaussian_threshold": source_path(gaussian_threshold),
               "ring_threshold": source_path(ring_threshold), "convergence": source_path(convergence),
               "data_dir": str(data_dir), "png_dpi": 300}
    (out/"figure_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2)+"\n")
    return paths


def write_report_summary(output_path, *, gaussian_run=None, ring_run=None,
                         gaussian_threshold=None, ring_threshold=None,
                         convergence=None, domain_comparisons=None,
                         final_state_checks=None, validation_report=None):
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
        rows.append(("Сходимость", data.get("assessment", "NOT_TESTED"),
                     f"{data.get('status', 'NOT_TESTED')}, "
                     f"t={data.get('time_ps', '?')} пс, "
                     f"{data.get('reference_stationarity_status', 'статус не указан')}"))
        rows.append(("Полная серия сходимости конечного режима",
                     data.get("final_state_assessment", "NOT_TESTED"),
                     "три сетки и три шага на установившемся состоянии"))
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
