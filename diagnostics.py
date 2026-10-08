"""Physical observables, numerical warnings and stationarity classification."""
from __future__ import annotations

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


def current(psi, grid, config):
    p = parameters(config)
    fft = np.fft.fft2(psi)
    # An even FFT grid has no unique signed derivative at the Nyquist mode.
    kx, ky = grid.kx.copy(), grid.ky.copy()
    if kx.size % 2 == 0:
        kx[kx.size//2] = 0.0
    if ky.size % 2 == 0:
        ky[ky.size//2] = 0.0
    dxpsi = np.fft.ifft2(1j*kx[None, :]*fft)
    dypsi = np.fft.ifft2(1j*ky[:, None]*fft)
    factor = 2*p["K"]/p["hbar"]
    return factor*np.imag(np.conj(psi)*dxpsi), factor*np.imag(np.conj(psi)*dypsi)


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
