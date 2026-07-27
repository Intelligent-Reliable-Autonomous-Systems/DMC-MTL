import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

region_heatmap = np.asarray(
    [
        [1.14, 0.35, 1.14, 0.35, 1.47, 0.48, 1.5, 0.48, 1.47, 0.5, 1.37, 0.44, 1.43, 0.47],
        [1.03, 0.3, 1.34, 0.45, 1.03, 0.3, 1.33, 0.41, 1.3, 0.41, 1.29, 0.45, 1.37, 0.5],
        [1.3, 0.3, 1.67, 0.39, 1.66, 0.56, 1.3, 0.3, 1.65, 0.51, 1.93, 0.66, 1.82, 0.7],
        [1.64, 0.5, 1.73, 0.49, 1.61, 0.47, 1.64, 0.5, 1.64, 0.5, 1.72, 0.54, 1.82, 0.61],
        [1.58, 0.33, 1.62, 0.32, 1.65, 0.3, 1.8, 0.45, 1.61, 0.34, 1.58, 0.33, 1.77, 0.56],
        [2.3, 0.57, 2.43, 0.54, 2.56, 0.54, 2.34, 0.55, 2.19, 0.42, 2.29, 0.47, 2.3, 0.57],
    ]
)


def linear_interp_heatmap(data: np.ndarray, fname: str) -> None:
    data_mean = data[:, ::2]

    data_increase = data_mean / data_mean[:, 0][:, np.newaxis]

    norm = TwoSlopeNorm(vmin=-75, vcenter=0, vmax=75)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow((data_increase[:, 1:] - 1) * 100, cmap="RdYlGn_r", norm=norm)  # _r reverses so red=high, green=low

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RMSE Increase (%)")

    ax.set_yticks(
        np.arange(6),
        labels=[
            "BCOV",
            "ONNP",
            "WA",
            "NY",
            "NS",
            "MI",
        ],
    )
    ax.set_xticks(
        np.arange(6),
        labels=[
            "BCOV",
            "ONNP",
            "WA",
            "NY",
            "NS",
            "MI",
        ],
    )
    ax.set_title("Evaluation with Different Regional Embedding")
    print(fname)
    plt.savefig(fname, bbox_inches="tight")

    plt.show()


def main():
    linear_interp_heatmap(region_heatmap, "region.png")


if __name__ == "__main__":
    main()
