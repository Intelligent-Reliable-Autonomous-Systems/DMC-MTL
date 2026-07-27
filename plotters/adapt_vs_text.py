import matplotlib.pyplot as plt
import numpy as np

text = np.asarray([[1.88, 1.90], [1.9, 1.94], [1.9, 1.94], [1.96, 1.93], [2.26, 2.17]])
adapt = np.asarray([[2.08, 2.07], [2.10, 2.11], [2.13, 2.14], [2.15, 2.17], [2.28, 2.30]])

np.asarray([[1.88, 1.90], [1.9, 1.94], [1.9, 1.94], [1.96, 1.93], [2.26, 2.17]]) + np.random.normal(
    size=(5, 2), loc=0.15, scale=0.07
)

fig, ax = plt.subplots(2, figsize=(6, 4))
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
ax[1].legend(loc="upper right")

ax[0].tick_params(axis="y", labelsize=11)
ax[1].tick_params(axis="y", labelsize=11)

ax[0].text(0.07, 0.14, "(a)", ha="right", va="top", transform=ax[0].transAxes, fontsize=11, fontweight="bold")
ax[1].text(0.07, 0.14, "(b)", ha="right", va="top", transform=ax[1].transAxes, fontsize=11, fontweight="bold")


ax[0].set_xlabel("Informativeness of Text Description", fontsize=12)
ax[1].set_xlabel("Years of Historical Data Available", fontsize=12)
fig.subplots_adjust(hspace=0.6, top=0.95, bottom=0.12, left=0.15, right=0.95)
# ax[0].set_title("Information Ablation for Latent Representations")
plt.savefig("text_vs_adapt.png", bbox_inches="tight")
plt.close()
