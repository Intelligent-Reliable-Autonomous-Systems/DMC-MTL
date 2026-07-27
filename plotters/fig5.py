import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ferg = [0.009, 0.011]
hybrid = [0.015, 0.019]
deep = [0.062, 0.083]


text = np.asarray([[1.88, 1.90], [1.9, 1.94], [1.9, 1.94], [1.96, 1.93], [2.26, 2.17]])
adapt = np.asarray([[2.08, 2.07], [2.10, 2.11], [2.13, 2.14], [2.15, 2.17], [2.28, 2.30]])

np.asarray([[1.88, 1.90], [1.9, 1.94], [1.9, 1.94], [1.96, 1.93], [2.26, 2.17]]) + np.random.normal(
    size=(5, 2), loc=0.15, scale=0.07
)

fig = plt.figure(figsize=(12, 3))
gs = GridSpec(2, 2, width_ratios=[1, 1], figure=fig)

ax = [
    fig.add_subplot(gs[0, 0]),  # top-left
    fig.add_subplot(gs[1, 0]),  # bottom-left
    fig.add_subplot(gs[:, 1]),  # right column, spans both rows
]

ax[0].plot(text[:, 0][::-1], c="r", marker="s", label="Text (Deep)")
ax[0].plot(text[:, 1][::-1], c="b", marker="o", label="Text (Hybrid)")
ax[1].plot(adapt[:, 0][::-1], c="r", marker="s", linestyle="--", label="Few Season (Deep)")
ax[1].plot(adapt[:, 1][::-1], c="b", marker="o", linestyle="--", label="Few Season (Hybrid)")

ax[0].set_ylim(bottom=1.8, top=2.4)
ax[1].set_ylim(bottom=1.8, top=2.4)
ax[0].set_ylabel(r"RMSE ($^\circ$C)", fontsize=12)
ax[1].set_ylabel(r"RMSE ($^\circ$C)", fontsize=12)

ax[0].set_xticks(
    ticks=[0, 1, 2, 3, 4],
    labels=["Bare", "Minimal", "Standard", "Detailed", "Verbose"],
    fontsize=11,
)
ax[1].set_xticks(
    ticks=[0, 1, 2, 3, 4],
    labels=["1 year", "2 years", "3 years", "4 years", "5 years"],
    fontsize=11,
)

ax[0].legend(loc="upper right")
ax[1].legend(loc="lower left")

ax[0].tick_params(axis="y", labelsize=11)
ax[1].tick_params(axis="y", labelsize=11)

# ax[0].text(0.5, 0.91, "(a)", ha="right", va="top", transform=ax[0].transAxes, fontsize=11, fontweight="bold")
# ax[1].text(0.5, 0.91, "(b)", ha="right", va="top", transform=ax[1].transAxes, fontsize=11, fontweight="bold")

ax[0].set_xlabel("Informativeness of Text Description", fontsize=12)
ax[1].set_xlabel("Years of Historical Data Available", fontsize=12)

ax[2].bar([0.25, 0.75], ferg, width=0.5, hatch=["", "//"], edgecolor="k")
ax[2].bar([1.5, 2], hybrid, width=0.5, hatch=["", "//"], edgecolor="k")
ax[2].bar([2.75, 3.25], deep, width=0.5, hatch=["", "//"], edgecolor="k")
ax[2].set_ylabel("% Unrealistic Predictions", fontsize=12)
ax[2].set_xlabel("Modeling Choice", fontsize=12)
ax[2].set_xticks(
    ticks=[0.5, 1.75, 3],
    labels=["Ferguson Model", "Hybrid Model", "Deep Learning"],
    fontsize=12,
)
ax[2].tick_params(axis="y", labelsize=12)

hatch_legend = [
    Patch(facecolor="white", edgecolor="black", label="Trained on All Regions"),
    Patch(facecolor="white", edgecolor="black", hatch="//", label="Transfered From BCOV+ONNP"),
]

ax[2].legend(handles=hatch_legend, loc="upper left", title="Region")

fig.subplots_adjust(hspace=0.6, wspace=0.35, top=0.95, bottom=0.12, left=0.1, right=0.95)

plt.savefig("q4.png", bbox_inches="tight")
plt.close()
