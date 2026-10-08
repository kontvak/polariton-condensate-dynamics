"""Scalar polariton condensate with one reservoir (um, ps, meV)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

HBAR = 0.6582119569  # meV ps
HBAR2_OVER_2ME = 3.80998212e-5  # meV um^2
DOI = "10.1126/sciadv.adj1589"


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    k2: np.ndarray
    dx: float
    dy: float
    lx: float
    ly: float

    @property
    def shape(self):
        return self.y.size, self.x.size

    @property
    def da(self):
        return self.dx * self.dy


def make_grid(config):
    s = config["space"]
    nx, ny = int(s["Nx"]), int(s["Ny"])
    lx, ly = float(s["Lx_um"]), float(s["Ly_um"])
    dx, dy = lx/nx, ly/ny
    x = (np.arange(nx)-nx//2)*dx
    y = (np.arange(ny)-ny//2)*dy
    kx = 2*np.pi*np.fft.fftfreq(nx, dx)
    ky = 2*np.pi*np.fft.fftfreq(ny, dy)
    return Grid(x, y, kx, ky, ky[:, None]**2+kx[None, :]**2,
                dx, dy, lx, ly)


def parameters(config):
    m = config["model"]
    gc, gr, r = (float(m[k]) for k in
                 ("gammaC_per_ps", "gammaR_per_ps", "R_um2_per_ps"))
    return {
        "hbar": HBAR,
        "K": HBAR2_OVER_2ME/float(m["mass_over_me"]),
        "gC": float(m["gC_mev_um2"]), "gR": float(m["gR_mev_um2"]),
        "R": r, "gammaC": gc, "gammaR": gr,
        "eta": float(m["eta_times_gammaR"])/gr if gr > 0 else None,
        "Pth0D": gc*gr/r if r > 0 else None,
        "nRth": gc/r if r > 0 else None,
        "tauC": 1/gc if gc > 0 else None,
        "tauR": 1/gr if gr > 0 else None,
        "background_blueshift": bool(m["background_blueshift"]),
    }


def pump_profile(config, grid):
    p = config["pump"]
    kind = p["kind"]
    if kind == "off":
        return np.zeros(grid.shape)
    if kind == "uniform":
        return np.ones(grid.shape)
    x0, y0 = map(float, p["center_um"])
    d = float(p["d_um"])
    radius2 = (grid.x[None, :]-x0)**2+(grid.y[:, None]-y0)**2
    if kind == "gaussian":
        return np.exp(-radius2/d**2)
    if kind == "ring":
        return np.exp(-((np.sqrt(radius2)-float(p["r0_um"]))/d)**2)
    raise ValueError(f"Unknown pump kind: {kind}")


def make_pump(config, grid):
    f = pump_profile(config, grid)
    if config["pump"]["kind"] == "off":
        return f
    pth = parameters(config)["Pth0D"]
    if pth is None:
        raise ValueError("P0 requires R>0; use pump=off for zero-R tests")
    return float(config["pump"]["P0"])*pth*f


def make_absorber(config, grid):
    """Optional numerical loss near periodic boundaries, zero in the interior."""
    if not config["space"]["absorbing_layer"]:
        return np.zeros(grid.shape, dtype=float)
    width = float(config["space"]["absorber_width_um"])
    maximum = float(config["space"]["absorber_gamma_max_per_ps"])
    distance_x = grid.lx/2-np.abs(grid.x)
    distance_y = grid.ly/2-np.abs(grid.y)
    ramp_x = np.sin(0.5*np.pi*np.clip((width-distance_x)/width, 0, 1))**2
    ramp_y = np.sin(0.5*np.pi*np.clip((width-distance_y)/width, 0, 1))**2
    return maximum*np.maximum(ramp_x[None, :], ramp_y[:, None])


def initial_fields(config, grid, pump):
    i = config["initial"]
    mode = i["mode"]
    amplitude = np.sqrt(float(i["psi_seed_density_um2_inv"]))
    x0, y0 = map(float, config["pump"]["center_um"])
    x, y = grid.x[None, :]-x0, grid.y[:, None]-y0
    if mode == "deterministic_test":
        width = max(float(config["pump"]["d_um"]), 1.0)
        psi = amplitude*np.exp(-(x*x+y*y)/(2*width*width))*np.exp(1j*(0.2*x+0.1*y))
        nr = np.full(grid.shape, float(i.get("reservoir_initial_um2_inv", 0.0)))
    elif mode in ("physical_startup", "preloaded_reservoir"):
        rng = np.random.default_rng(int(i["random_seed"]))
        corr = float(i["seed_correlation_um"])
        field = np.zeros(grid.shape, dtype=np.complex128)
        # One continuous random Fourier sum is sampled on all grids.
        for kx, ky, phase in zip(rng.normal(0, 1/corr, 24),
                                rng.normal(0, 1/corr, 24),
                                rng.uniform(0, 2*np.pi, 24)):
            field += np.exp(1j*(kx*x+ky*y+phase))/np.sqrt(24)
        psi = amplitude*np.sqrt(pump_profile(config, grid))*field
        nr = (np.zeros(grid.shape) if mode == "physical_startup"
              else pump/float(config["model"]["gammaR_per_ps"]))
    else:
        raise ValueError(f"Unknown initial mode: {mode}")
    return np.asarray(psi, dtype=np.complex128), np.asarray(nr, dtype=np.float64)


def stationary_0d(pump, config):
    p = parameters(config)
    if p["Pth0D"] is None or p["gammaR"] == 0:
        raise ValueError("0D stationary branches require R>0 and gammaR>0")
    return {
        "empty": (0.0, pump/p["gammaR"]),
        "condensed": ((pump-p["Pth0D"])/p["gammaC"], p["nRth"])
        if pump > p["Pth0D"] else None,
    }


def jacobian_0d(nc, nr, config):
    p = parameters(config)
    r = p["R"]
    return np.array([[r*nr-p["gammaC"], r*nc],
                     [-r*nr, -p["gammaR"]-r*nc]])


def phase_frequency_0d(nc, nr, pump, config):
    p = parameters(config)
    bg = p["eta"]*pump if p["background_blueshift"] else 0.0
    return (p["gC"]*nc+p["gR"]*(nr+bg))/p["hbar"]


def stuart_landau(pump, config):
    p = parameters(config)
    if p["gammaR"] <= 0:
        raise ValueError("Stuart-Landau requires gammaR>0")
    r, gr = p["R"], p["gammaR"]
    return {
        "mu": 0.5*(r*pump/gr-p["gammaC"]),
        "beta": pump*r*r/(2*gr*gr),
        "omega0": p["gR"]*(pump/gr+(p["eta"]*pump if p["background_blueshift"] else 0))/p["hbar"],
        "nu": (p["gC"]-p["gR"]*pump*r/gr**2)/p["hbar"],
    }


def solve_0d_reference(config, pump, psi0, nr0, times):
    """Independent adaptive ODE reference, including complex phase."""
    p = parameters(config)
    bg = p["eta"]*pump if p["background_blueshift"] else 0.0
    def rhs(_t, state):
        psi = complex(state[0], state[1])
        nr = state[2]
        nc = abs(psi)**2
        growth = (p["R"]*nr-p["gammaC"])/2
        energy = p["gC"]*nc+p["gR"]*(nr+bg)
        dpsi = (growth-1j*energy/p["hbar"])*psi
        return [dpsi.real, dpsi.imag, pump-(p["gammaR"]+p["R"]*nc)*nr]
    result = solve_ivp(rhs, (float(times[0]), float(times[-1])),
                       [psi0.real, psi0.imag, nr0], t_eval=times,
                       rtol=2e-11, atol=2e-13, method="DOP853")
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[0]+1j*result.y[1], result.y[2]
