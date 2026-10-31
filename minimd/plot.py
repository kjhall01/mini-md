import numpy as np
import matplotlib.pyplot as plt


def scatter_3d(
    ds, *,
    x="x", y="y", z="z", h="h",
    time=None,
    cmap="viridis",
    s=15,
    alpha=0.8,
    figsize=(9, 7),
):
    """Plot three variables, colored by a fourth.

    Each of x/y/z/h may name any variable or coordinate, including "time".
    time: optional coordinate selection, e.g. slice(10, 30).
    Returns (fig, ax).
    """
    if time is not None:
        ds = ds.sel(time=time)

    X, Y, Z, H = [
        np.asarray(ds[name].values).ravel()
        for name in (x, y, z, h)
    ]
    valid = (
        np.isfinite(X) & np.isfinite(Y)
        & np.isfinite(Z) & np.isfinite(H)
    )
    if not valid.any():
        raise ValueError("No finite points in the selected interval.")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    points = ax.scatter(
        X[valid], Y[valid], Z[valid],
        c=H[valid], cmap=cmap, s=s, alpha=alpha,
        depthshade=False,  # Keep color tied to H rather than depth.
    )
    ax.set(xlabel=x, ylabel=y, zlabel=z)
    fig.colorbar(points, ax=ax, label=h, pad=0.1, shrink=0.75)
    fig.tight_layout()

    return fig, ax