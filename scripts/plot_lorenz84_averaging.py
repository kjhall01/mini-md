#!/usr/bin/env python3
"""Plot the seasonal Lorenz-84 trajectory and its monthly means together.

Standalone companion to lorenz84_reservoir.py: the same equations and default
parameters, with constant external forcing q and the annual cycle retained.
Dependencies: numpy, scipy, matplotlib

    python plot_lorenz84_averaging.py
    python plot_lorenz84_averaging.py --years 41 --sample-days 0.25 --show
    python plot_lorenz84_averaging.py --data trajectory.npz

Defaults: 10 years of discarded spin-up, 41 plotted years, daily output.
The integrator is adaptive DOP853 with a maximum internal step of one day;
--sample-days controls output spacing, not the integrator timestep. Time uses
365-day years and 12 equal months per year, exactly as in the original code.
Monthly means are integrated over each month, not sampled at its center and
not estimated by grouping calendar days. Both plotted products come from the
same numerical trajectory. No second simulation is run for the means.

The figure shows projections onto x/y/z of the long-run, seasonally forced
trajectory. The complete periodically forced system also includes seasonal
phase. With constant q and initial h=q, the slow reservoir h stays constant.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def rhs(t, state, q=0.0):
    """Original four-variable equations; t in years, states dimensionless."""
    x, y, z, h = state
    a, b, G, F0, A, alpha, beta, tau_o = 0.25, 4.0, 1.0, 9.5, 2.0, -1.0, -2.0, 10.0
    F = F0 + A * np.sin(2 * np.pi * t) + alpha * q + beta * h
    return np.array([
        73.0 * (-y * y - z * z - a * x + a * F),
        73.0 * (x * y - b * x * z - y + G),
        73.0 * (x * z + b * x * y - z),
        (q - h) / tau_o,
    ])


def generate(years=41, spinup_years=10.0, sample_days=1.0, q=0.0,
             max_step_days=1.0, rtol=1e-8, atol=1e-10):
    """Return fine times/states and monthly center times/time-mean states."""
    if not isinstance(years, (int, np.integer)) or years < 1:
        raise ValueError("years must be a positive integer")
    if not np.isfinite(spinup_years) or spinup_years < 0:
        raise ValueError("spinup_years must be finite and nonnegative")
    if not all(np.isfinite(v) and v > 0 for v in
               (sample_days, max_step_days, rtol, atol)):
        raise ValueError("sampling, max step, and tolerances must be positive")
    if not np.isfinite(q):
        raise ValueError("q must be finite")

    options = dict(method="DOP853", rtol=rtol, atol=atol,
                   max_step=max_step_days / 365.0)

    def integrate(fun, interval, initial, dense=False):
        sol = solve_ivp(fun, interval, initial, dense_output=dense, **options)
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError(sol.message)
        return sol

    state = np.array([1.0, 1.0, 1.0, q])
    if spinup_years:
        sol = integrate(lambda t, s: rhs(t, s, q), (-spinup_years, 0.0), state)
        state = sol.y[:, -1]

    # Include the endpoint; the last interval may be shorter than sample_days.
    fine_time = np.r_[np.arange(0.0, float(years), sample_days / 365.0), float(years)]
    fine_state = np.empty((fine_time.size, 4))
    fine_state[0] = state
    edges = np.arange(12 * years + 1) / 12.0
    monthly_time = (edges[:-1] + edges[1:]) / 2.0
    monthly_state = np.empty((12 * years, 4))

    def augmented_rhs(t, augmented):
        physical = augmented[:4]
        # Additional variables accumulate integral(state dt); they do not feed back.
        return np.r_[rhs(t, physical, q), physical]

    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        sol = integrate(augmented_rhs, (left, right),
                        np.r_[state, np.zeros(4)], dense=True)
        state = sol.y[:4, -1]
        monthly_state[i] = sol.y[4:, -1] / (right - left)
        # Half-open assignment ensures every requested output is written once.
        start = np.searchsorted(fine_time, left, side="right")
        stop = np.searchsorted(fine_time, right, side="right")
        if start < stop:
            fine_state[start:stop] = sol.sol(fine_time[start:stop])[:4].T

    return fine_time, fine_state, monthly_time, monthly_state


def plot_overlay(fine_state, monthly_state, years, sample_days, q):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#26364a", "text.color": "#26364a",
                         "axes.edgecolor": "#aab4c0", "axes.titleweight": "bold"})
    fine_color, mean_color = "#587891", "#d34c21"
    fig = plt.figure(figsize=(12, 9.4), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.97, bottom=0.11,
                           top=0.84, wspace=0.25, hspace=0.30)
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax3.plot(*fine_state[:, :3].T, color=fine_color, lw=0.35, alpha=0.22)
    ax3.scatter(*monthly_state[:, :3].T, color=mean_color, s=10,
                alpha=0.9, depthshade=False, linewidths=0)
    ax3.set(xlabel="x", ylabel="y", zlabel="z")
    ax3.set_title("Three-dimensional projection", pad=10)
    ax3.view_init(elev=24, azim=-52)
    ax3.set_box_aspect((1.15, 1, 1))
    ax3.tick_params(labelsize=8, pad=0)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.fill = False

    for cell, (i, j) in zip((grid[0, 1], grid[1, 0], grid[1, 1]),
                            ((0, 1), (0, 2), (1, 2))):
        ax = fig.add_subplot(cell)
        ax.plot(fine_state[:, i], fine_state[:, j],
                color=fine_color, lw=0.35, alpha=0.22, zorder=1)
        ax.scatter(monthly_state[:, i], monthly_state[:, j],
                   color=mean_color, s=11, alpha=0.85, linewidths=0, zorder=3)
        ax.set(xlabel="xyz"[i], ylabel="xyz"[j])
        ax.set_title(f"{'xyz'[i]}–{'xyz'[j]} projection", loc="left", pad=10)
        ax.grid(color="#e5e9ee", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(0.04)

    fig.suptitle("Seasonal Lorenz-84: the effect of monthly averaging",
                 x=0.07, y=0.965, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.07, 0.916,
             f"{years} years after spin-up  •  output every {sample_days:g} day(s)  •  "
             f"constant external forcing q = {q:g}; annual cycle retained",
             fontsize=10.5, color="#526375")
    fig.text(0.07, 0.885,
             "Orange points are time-integrated monthly means of the same trajectory.",
             fontsize=10.5, color="#526375")
    handles = [Line2D([0], [0], color=fine_color, lw=1.5,
                       label="Fine trajectory (instantaneous states)"),
               Line2D([0], [0], marker="o", linestyle="none", color=mean_color,
                       markersize=5, label="Monthly means")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.048),
               ncol=2, frameon=False)
    fig.text(0.5, 0.025,
             "Monthly means need not lie on the instantaneous orbit. All states are dimensionless.",
             ha="center", fontsize=9, color="#607080")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, default=41)
    parser.add_argument("--spinup-years", type=float, default=10.0)
    parser.add_argument("--sample-days", type=float, default=1.0,
                        help="fine-output spacing; 0.25 gives six-hourly output")
    parser.add_argument("--max-step-days", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("lorenz84_averaging.png"))
    parser.add_argument("--data", type=Path, help="optional .npz file of both trajectories")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, state, tm, monthly = generate(args.years, args.spinup_years, args.sample_days,
                                    args.q, args.max_step_days)
    fig = plot_overlay(state, monthly, args.years, args.sample_days, args.q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor="white")
    print(f"Saved {args.output}: {len(t)} fine samples, {len(tm)} monthly means")
    if args.data:
        args.data.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.data, time=t, state=state, monthly_time=tm,
                            monthly_state=monthly, state_names=np.array(list("xyzh")),
                            q=args.q, days_per_year=365.0, months_per_year=12,
                            spinup_years=args.spinup_years, sample_days=args.sample_days,
                            max_step_days=args.max_step_days)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
