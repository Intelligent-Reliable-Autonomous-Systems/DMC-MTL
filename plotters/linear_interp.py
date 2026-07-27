import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

cab_sauv_heatmap = np.asarray(
    [
        [1.19, 0.51, 1.19, 0.52, 1.22, 0.52, 1.3, 0.52, 1.44, 0.51],
        [1.14, 0.3, 1.14, 0.3, 1.14, 0.3, 1.14, 0.3, 1.14, 0.3],
        [1.2, 0.55, 1.22, 0.51, 1.35, 0.47, 1.57, 0.48, 1.85, 0.51],
        [1.83, 1.05, 1.82, 0.96, 1.87, 0.89, 1.96, 0.84, 2.06, 0.81],
        [1.55, 0.77, 1.55, 0.78, 1.6, 0.79, 1.71, 0.8, 1.86, 0.82],
        [1.17, 0.19, 1.22, 0.22, 1.29, 0.26, 1.42, 0.34, 1.62, 0.39],
        [1.64, 0.49, 1.61, 0.5, 1.63, 0.49, 1.73, 0.48, 1.92, 0.47],
        [1.33, 0.52, 1.36, 0.52, 1.46, 0.51, 1.66, 0.5, 1.91, 0.51],
        [1.53, 1.17, 1.56, 1.18, 1.65, 1.2, 1.81, 1.21, 2.04, 1.21],
        [1.2, 0.36, 1.17, 0.36, 1.18, 0.35, 1.26, 0.33, 1.38, 0.33],
        [1.83, 1.59, 1.78, 1.54, 1.77, 1.49, 1.78, 1.47, 1.83, 1.49],
        [1.04, 0.2, 1.02, 0.26, 1.1, 0.26, 1.22, 0.24, 1.35, 0.19],
    ]
)

merlot_heatmap = np.asarray(
    [
        [1.19, 0.51, 1.21, 0.49, 1.21, 0.49, 1.45, 0.47, 1.64, 0.47],
        [1.14, 0.30, 1.19, 0.36, 1.19, 0.36, 1.54, 0.53, 1.71, 0.62],
        [1.20, 0.55, 1.24, 0.49, 1.24, 0.49, 1.71, 0.51, 2.05, 0.55],
        [1.83, 1.05, 1.78, 0.99, 1.78, 0.99, 1.82, 0.92, 1.93, 0.89],
        [1.55, 0.77, 1.55, 0.77, 1.55, 0.77, 1.55, 0.77, 1.55, 0.77],
        [1.17, 0.19, 1.16, 0.22, 1.16, 0.22, 1.34, 0.27, 1.55, 0.31],
        [1.64, 0.49, 1.61, 0.49, 1.61, 0.49, 1.84, 0.53, 2.13, 0.54],
        [1.33, 0.52, 1.41, 0.53, 1.41, 0.53, 1.91, 0.61, 2.22, 0.68],
        [1.53, 1.17, 1.63, 1.17, 1.63, 1.17, 2.26, 1.15, 2.65, 1.12],
        [1.20, 0.36, 1.20, 0.37, 1.20, 0.37, 1.36, 0.42, 1.48, 0.46],
        [1.83, 1.59, 1.81, 1.54, 1.81, 1.54, 1.89, 1.45, 1.94, 1.45],
        [1.04, 0.20, 1.05, 0.27, 1.05, 0.27, 1.09, 0.25, 1.05, 0.21],
    ]
)
cab_sauv_dist = np.asarray([6.16, 0, 7.15, 6.5, 6.08, 7.3, 8, 6.96, 7.86, 6.66, 6.22, 7.63])


def linear_interp_heatmap(data: np.ndarray, dist: np.ndarray, fname: str) -> None:
    data_mean = data[:, ::2]

    data_increase = data_mean / data_mean[:, 0][:, np.newaxis]

    norm = TwoSlopeNorm(vmin=-75, vcenter=0, vmax=75)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow((data_increase - 1) * 100, cmap="RdYlGn_r", norm=norm)  # _r reverses so red=high, green=low

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RMSE Increase (%)")

    ax.set_yticks(
        np.arange(12),
        labels=[
            "Cabernet_Franc",
            "Cabernet_Sauvignon",
            "Chardonnay",
            "Gewurztraminer",
            "Merlot",
            "Pinot_Blanc",
            "Pinot_Gris",
            "Pinot_Noir",
            "Riesling",
            "Sauvignon_Blanc",
            "Syrah",
            "Zinfandel",
        ],
    )
    ax.set_xticks(np.arange(5), labels=[0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"$\alpha$ Interpolation Coefficient")
    ax.set_title("Interpolation to Merlot Embedding")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    # plt.show()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dist, (data_increase[:, -1] - 1) * 100)
    plt.show()


def main():
    linear_interp_heatmap(merlot_heatmap, cab_sauv_dist, "merlot.png")


if __name__ == "__main__":
    main()
