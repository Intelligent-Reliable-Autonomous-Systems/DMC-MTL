import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ferg = [0.009, 0.011]
hybrid = [0.015, 0.019]
deep = [0.062, 0.083]


fix, ax = plt.subplots(1, figsize=(6, 3))

ax.bar([0.25, 0.75], ferg, width=0.5, hatch=["", "//"], edgecolor="k")
ax.bar([1.5, 2], hybrid, width=0.5, hatch=["", "//"], edgecolor="k")
ax.bar([2.75, 3.25], deep, width=0.5, hatch=["", "//"], edgecolor="k")
ax.set_ylabel("% Unrealistic Predictions", fontsize=12)
ax.set_xlabel("Modeling Choice", fontsize=12)
ax.set_xticks(
    ticks=[0.5, 1.75, 3],
    labels=["Ferguson Model", "Hybrid Model", "Deep Learning"],
    fontsize=12,
)
plt.yticks(fontsize=12)

hatch_legend = [
    Patch(facecolor="white", edgecolor="black", label="Trained on All Regions"),
    Patch(facecolor="white", edgecolor="black", hatch="//", label="Transfered From BCOV+ONNP"),
]

plt.legend(handles=hatch_legend, loc="upper left", title="Region")

plt.title("Biologically Unrealistic Predictions During Dormancy", fontsize=14)

plt.savefig("biologically_unreal.png", bbox_inches="tight")
plt.close()
