"""Seasonal Lorenz-84 with a slow reservoir; monthly output as an xr.Dataset.

Dependencies (Python >= 3.10): numpy, scipy, xarray
    conda install -c conda-forge numpy scipy xarray

Time is in model years: 365 days/year, 12 equal months/year. This is an
idealized clock, not a Gregorian calendar. All four states are nondimensional.

Equations
---------
    F(t) = F0 + A*sin(2*pi*t + phase) + alpha*q(t) + beta*h
    dx/dt = 73*(-y*y - z*z - a*x + a*F(t))
    dy/dt = 73*(x*y - b*x*z - y + G)
    dz/dt = 73*(x*z + b*x*y - z)
    dh/dt = (q(t) - h)/tau_o

The factor 73 converts the standard five-day Lorenz-84 time unit to years.
q is prescribed; h is a one-way, linear memory of q. This reservoir extension
is a proposed toy benchmark, not a published physical ocean heat-budget model.
Defaults are starting values; chaos and the response should be assessed for
each forcing/coupling range before using it as an ML benchmark.

Examples
--------
    import numpy as np
    from lorenz84_reservoir import simulate, Parameters, co2_to_forcing

    # Continuous ramp: t is years since the beginning of the OUTPUT period.
    ds = simulate(lambda t: min(t / 100.0, 1.0), n_months=1200)

    # One prescribed, constant forcing value per month; length sets duration.
    ds = simulate(np.linspace(0.0, 1.0, 1200))

    # A CO2 pathway, normalized so a doubling corresponds to q=1.
    co2 = np.linspace(280.0, 560.0, 1200)
    ds = simulate(co2_to_forcing(co2, reference=280.0))

    # Instantaneous states at month centers instead of monthly means.
    ds = simulate(0.0, n_months=120, output="sample")

    # Continuous interpolation of values prescribed at irregular YEAR knots.
    knots = np.array([0.0, 20.0, 60.0, 100.0])
    values = np.array([0.0, 0.4, 1.0, 0.3])
    ds = simulate(lambda t: np.interp(t, knots, values), n_months=1200,
                  forcing_breaks=knots[1:-1])

    ds.to_netcdf("training_data.nc", engine="scipy")

The Dataset has exactly four data variables, x/y/z/h, each with dimension
"time". q, time_start, time_end, and month are coordinates. For output="mean",
q is also a monthly mean. Pass xr.DataArray forcing via its values, or directly:
array inputs use only values; existing coordinate labels are NOT interpreted.

Run this file directly for a ramp experiment:
    python lorenz84_reservoir.py --years 100 --q-final 1 --save training_data.nc

References for the base model and solver:
    https://doi.org/10.1111/j.1600-0870.1984.tb00230.x  (Lorenz, 1984)
    https://doi.org/10.1175/JCLI-D-14-00459.1    (seasonal forcing)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp
import xarray as xr

__all__ = ["Parameters", "co2_to_forcing", "rhs", "simulate"]

DAYS_PER_YEAR = 365.0
ATMOSPHERE_RATE = DAYS_PER_YEAR / 5.0
STATE_NAMES = ("x", "y", "z", "h")


@dataclass(frozen=True)
class Parameters:
    """Model coefficients; tau_o is in years and phase is in radians."""

    a: float = 0.25
    b: float = 4.0
    G: float = 1.0
    F0: float = 9.5
    A: float = 2.0
    alpha: float = -1.0
    beta: float = -2.0
    tau_o: float = 10.0
    phase: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("All model parameters must be finite.")
        if self.a <= 0 or self.tau_o <= 0:
            raise ValueError("a and tau_o must be positive.")


def co2_to_forcing(concentration, reference: float = 280.0):
    """Return q=log2(C/reference), a dimensionless forcing coordinate."""
    c = np.asarray(concentration, dtype=float)
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("reference must be finite and positive.")
    if not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("CO2 concentrations must be finite and positive.")
    q = np.log2(c / reference)
    return float(q) if q.ndim == 0 else q


def rhs(t: float, state, q: float, params: Parameters = Parameters()):
    """Return derivatives [dx, dy, dz, dh] per YEAR for a scalar forcing q."""
    x, y, z, h = state
    p = params
    F = p.F0 + p.A * math.sin(2 * math.pi * t + p.phase)
    F += p.alpha * q + p.beta * h
    return np.array([
        ATMOSPHERE_RATE * (-y*y - z*z - p.a*x + p.a*F),
        ATMOSPHERE_RATE * (x*y - p.b*x*z - y + p.G),
        ATMOSPHERE_RATE * (x*z + p.b*x*y - z),
        (q - h) / p.tau_o,
    ])


def simulate(
    forcing: float | Sequence[float] | Callable[[float], float] = 0.0,
    n_months: int | None = None,
    *,
    params: Parameters | None = None,
    initial_state: Sequence[float] | None = None,
    spinup_years: float = 10.0,
    output: str = "mean",
    forcing_breaks: Sequence[float] = (),
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step_days: float = 1.0,
) -> xr.Dataset:
    """Integrate the four-state model and return monthly x/y/z/h.

    Parameters
    ----------
    forcing : scalar, 1D array-like, or callable
        A scalar specifies constant q. An array specifies one constant value
        per month; n_months defaults to its length. Array/DataArray coordinate
        labels are ignored. A callable receives scalar elapsed years t>=0 and
        must return scalar q(t); supply n_months explicitly in this case.
        For a scalar, n_months defaults to 1200 (100 years).
    n_months : int, optional
        Number of complete equal-length months. Must match an input array.
    params : Parameters, optional
        Model coefficients. See the dataclass defaults.
    initial_state : sequence of four floats, optional
        State at t=-spinup_years, in x/y/z/h order. Default: [1,1,1,q(0)], so
        the reservoir starts exactly equilibrated with the baseline forcing.
        With spinup_years=0 this is the state at the output period's start.
    spinup_years : float
        Integrate from -spinup_years to 0 with q held at q(0), while retaining
        seasonality. This deliberately does NOT extrapolate the forcing into
        the past. Supply a prehistory as part of forcing and then slice the
        resulting Dataset if an evolving prehistory is needed. For a custom
        nonequilibrium h, spin-up must be long relative to tau_o to remove it.
    output : {"mean", "sample"}
        "mean": integrate each state over each month and divide by its duration.
        "sample": instantaneous state at each month center. In both cases,
        the time labels are month centers, in elapsed model years.
    forcing_breaks : sequence of floats
        Known callable forcing discontinuities or interpolation knots, in
        years strictly inside the output period. The integrator restarts there.
        Month boundaries are already handled, including jumps in array inputs.
    rtol, atol : float
        DOP853 local error tolerances. Means use four additional integral
        variables, plus one for q; these are numerical accumulators, not model
        states. Accumulators are reset each month to avoid cancellation drift.
    max_step_days : float
        Maximum internal integration step, independent of output resolution.

    Returns
    -------
    xr.Dataset
        Exactly four data variables, each with dimension time. q is a coordinate
        and uses the same mean/sample convention as the states. time_start and
        time_end give month bounds; month runs from 1 to 12 on the model clock.
        Model settings and the actual start/end states are recorded in attrs.
        Output has no stochastic noise: ensembles require different initial
        atmospheric states. Monthly averaging does not guarantee Markov closure.
    """
    p = Parameters() if params is None else params
    if not isinstance(p, Parameters):
        raise TypeError("params must be a Parameters instance.")
    if output not in ("mean", "sample"):
        raise ValueError("output must be 'mean' or 'sample'.")
    if not math.isfinite(spinup_years) or spinup_years < 0:
        raise ValueError("spinup_years must be finite and nonnegative.")
    if any(not math.isfinite(v) or v <= 0 for v in (rtol, atol, max_step_days)):
        raise ValueError("rtol, atol, and max_step_days must be finite and positive.")

    monthly_q = None
    if callable(forcing):
        if n_months is None:
            raise ValueError("Supply n_months when forcing is a callable.")

        def q_function(t):
            value = np.asarray(forcing(float(t)))
            if value.ndim != 0:
                raise ValueError("forcing(t) must return a scalar.")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite forcing at t={t} years.")
            return value

        q0 = q_function(0.0)
        forcing_kind = "callable"
    else:
        values = np.asarray(forcing, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Forcing values must be finite.")
        if values.ndim == 0:
            q0 = float(values)
            n_months = 1200 if n_months is None else n_months
            forcing_kind = "constant"
        elif values.ndim == 1 and values.size:
            monthly_q = values.copy()
            n_months = len(values) if n_months is None else n_months
            if n_months != len(values):
                raise ValueError("n_months must equal the forcing array length.")
            q0 = float(values[0])
            forcing_kind = "piecewise_constant_monthly"
        else:
            raise ValueError("forcing must be scalar, a nonempty 1D array, or callable.")

        def q_function(t):
            return q0  # Array inputs are assigned explicitly inside the month loop.

    if isinstance(n_months, (bool, np.bool_)) or not isinstance(n_months, (int, np.integer)):
        raise ValueError("n_months must be a positive integer.")
    if n_months <= 0:
        raise ValueError("n_months must be a positive integer.")
    n_months = int(n_months)
    edges = np.arange(n_months + 1, dtype=float) / 12.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    breaks = np.asarray(forcing_breaks, dtype=float)
    if breaks.ndim != 1 or not np.all(np.isfinite(breaks)):
        raise ValueError("forcing_breaks must be a finite 1D sequence.")
    if np.any((breaks <= 0) | (breaks >= edges[-1])):
        raise ValueError("forcing_breaks must lie strictly inside the output period.")
    breaks = np.unique(breaks)

    state = np.array([1.0, 1.0, 1.0, q0] if initial_state is None else initial_state,
                     dtype=float)
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain four finite values: x,y,z,h.")
    original_state = state.copy()
    solver_options = dict(method="DOP853", rtol=rtol, atol=atol,
                          max_step=max_step_days / DAYS_PER_YEAR)
    nfev = 0

    def integrate(fun, left, right, u0, dense=False):
        nonlocal nfev
        result = solve_ivp(fun, (left, right), u0, dense_output=dense, **solver_options)
        nfev += result.nfev
        if not result.success or not np.all(np.isfinite(result.y[:, -1])):
            raise RuntimeError(f"Integration failed on [{left}, {right}]: {result.message}")
        return result

    if spinup_years > 0:
        result = integrate(lambda t, u: rhs(t, u, q0, p), -spinup_years, 0.0, state)
        state = result.y[:, -1].copy()
    output_initial_state = state.copy()

    states = np.empty((n_months, 4))
    q_output = np.empty(n_months)
    for i, (start, stop, center) in enumerate(zip(edges[:-1], edges[1:], centers)):
        fixed_q = float(monthly_q[i]) if monthly_q is not None else None
        # Each month is a separate solve. Thus array-forcing jumps are never
        # interpolated or stepped across by the integrator.
        cuts = np.r_[start, breaks[(breaks > start) & (breaks < stop)], stop]
        u = np.r_[state, np.zeros(5)] if output == "mean" else state.copy()
        sampled = None
        for left, right in zip(cuts[:-1], cuts[1:]):
            # Evaluate the interval's one-sided limits at breakpoints. Point
            # values at a forcing jump do not contribute to the time integrals.
            lower, upper = np.nextafter(left, right), np.nextafter(right, left)

            def dynamics(t, v):
                q = fixed_q if fixed_q is not None else q_function(min(max(t, lower), upper))
                tendency = rhs(t, v[:4], q, p)
                if output == "mean":
                    return np.r_[tendency, v[:4], q]
                return tendency

            result = integrate(dynamics, left, right, u, dense=(output == "sample"))
            if output == "sample" and left <= center <= right:
                sampled = result.sol(center)[:4]
            u = result.y[:, -1].copy()
        state = u[:4].copy()
        if output == "mean":
            states[i] = u[4:8] / (stop - start)
            q_output[i] = u[8] / (stop - start)
        else:
            states[i] = sampled
            q_output[i] = fixed_q if fixed_q is not None else q_function(center)

    descriptions = ["large-scale atmospheric circulation", "wave cosine component",
                    "wave sine component", "slow forcing-adjustment reservoir"]
    cell_method = "time: mean" if output == "mean" else "time: point"
    ds = xr.Dataset(
        data_vars={
            name: ("time", states[:, j], {"long_name": descriptions[j],
                   "units": "1", "cell_methods": cell_method})
            for j, name in enumerate(STATE_NAMES)
        },
        coords={
            "time": ("time", centers, {"long_name": "elapsed model years at month center",
                                        "units": "years"}),
            "time_start": ("time", edges[:-1], {"units": "years"}),
            "time_end": ("time", edges[1:], {"units": "years"}),
            "month": ("time", np.arange(n_months) % 12 + 1,
                      {"long_name": "model month (not a Gregorian month)"}),
            "q": ("time", q_output, {"long_name": "prescribed external forcing",
                                      "units": "1", "cell_methods": cell_method}),
        },
        attrs={
            "model": "Seasonal Lorenz-84 plus one-way slow reservoir",
            "version": "1.0",
            "parameters": json.dumps(asdict(p)),
            "time_convention": "365 days/year; 12 equal months/year; t=0 begins output",
            "seasonality": "sin(2*pi*t+phase); default maximum at t mod 1 = 0.25",
            "output": "monthly_mean" if output == "mean" else "month_center_sample",
            "forcing_input": forcing_kind,
            "forcing_breaks_years": json.dumps(breaks.tolist()),
            "spinup_years": float(spinup_years),
            "spinup_forcing_q": q0,
            "initial_state_before_spinup": json.dumps(original_state.tolist()),
            "state_at_output_start": json.dumps(output_initial_state.tolist()),
            "state_at_output_end": json.dumps(state.tolist()),
            "state_order": "x,y,z,h",
            "solver": "scipy.integrate.solve_ivp / DOP853",
            "rtol": float(rtol), "atol": float(atol),
            "max_step_days": float(max_step_days),
            "rhs_evaluations": int(nfev),
            "note": "Toy benchmark; h is filtered forcing, not physical ocean heat content.",
        },
    )
    return ds


def main():
    """Command-line example: a prescribed linear ramp followed by a plateau."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, default=20, help="output duration (default: 20)")
    parser.add_argument("--spinup", type=float, default=10.0, help="baseline spin-up years")
    parser.add_argument("--q-initial", type=float, default=0.0)
    parser.add_argument("--q-final", type=float, default=1.0)
    parser.add_argument("--ramp-years", type=float, default=None,
                        help="ramp duration; defaults to --years, then q stays constant")
    parser.add_argument("--tau-o", type=float, default=10.0, help="reservoir timescale in years")
    parser.add_argument("--sampling", choices=["mean", "sample"], default="mean")
    parser.add_argument("--save", default="lorenz84_reservoir.nc", help="output NetCDF path")
    args = parser.parse_args()
    ramp_years = float(args.years) if args.ramp_years is None else args.ramp_years
    if args.years <= 0 or not math.isfinite(ramp_years) or ramp_years <= 0:
        parser.error("--years and --ramp-years must be positive.")

    def pathway(t):
        return args.q_initial + (args.q_final - args.q_initial) * min(t / ramp_years, 1.0)

    ds = simulate(pathway, n_months=12 * args.years, spinup_years=args.spinup,
                  params=Parameters(tau_o=args.tau_o), output=args.sampling,
                  forcing_breaks=[ramp_years] if ramp_years < args.years else [])
    ds.attrs.update(example_pathway="linear ramp followed by plateau",
                    q_initial=args.q_initial, q_final=args.q_final, ramp_years=ramp_years)
    ds.to_netcdf(args.save, engine="scipy")
    print(ds)
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
