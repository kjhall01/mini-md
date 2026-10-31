from minimd import simulate, scatter_3d
import matplotlib.pyplot as plt

ds = simulate(
    0.0,                  # Constant external forcing; seasonality remains.
    n_months=41 * 12 * 10,
    output="mean",
)

fig, ax = scatter_3d(
    ds,
    x="x", y="y", z="z",
    h="month",
    cmap=plt.get_cmap("twilight", 12),  # Cyclic colors for 12 months.
    s=20,
    alpha=0.8,
)

fig.axes[-1].set_yticks(range(1, 13))
ax.set_title("Monthly means colored by model month")
plt.show()