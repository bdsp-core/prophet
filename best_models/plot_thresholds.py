import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("tpr_95_thresholds_best_models.json") as f:
    data = json.load(f)

# Clean up condition names for display
def fmt(name):
    return name.replace("_", " ").title()

conditions = list(data.keys())
labels = [fmt(c) for c in conditions]
roc_aucs = [data[c]["roc_auc"] for c in conditions]
fprs = [data[c]["fpr_at_threshold"] for c in conditions]
tprs = [data[c]["tpr_at_threshold"] for c in conditions]

# Sort by ROC AUC descending
order = np.argsort(roc_aucs)
labels_s = [labels[i] for i in order]
roc_aucs_s = [roc_aucs[i] for i in order]
fprs_s = [fprs[i] for i in order]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Model Performance at ~95% TPR Threshold", fontsize=14, fontweight="bold", y=1.01)

# --- Left: ROC AUC bar chart ---
ax1 = axes[0]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(labels_s)))
bars = ax1.barh(labels_s, roc_aucs_s, color=colors, edgecolor="white", height=0.65)
ax1.set_xlim(0.93, 1.0)
ax1.set_xlabel("ROC AUC", fontsize=11)
ax1.set_title("ROC AUC by Condition", fontsize=12)
ax1.axvline(x=np.mean(roc_aucs_s), color="steelblue", linestyle="--", linewidth=1.2, label=f"Mean {np.mean(roc_aucs_s):.4f}")
ax1.legend(fontsize=9)
for bar, val in zip(bars, roc_aucs_s):
    ax1.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", ha="left", fontsize=8.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Right: FPR operating point scatter ---
ax2 = axes[1]
scatter_colors = plt.cm.tab20(np.linspace(0, 1, len(conditions)))
for i, (c, label) in enumerate(zip(conditions, labels)):
    ax2.scatter(data[c]["fpr_at_threshold"], data[c]["tpr_at_threshold"],
                s=120, color=scatter_colors[i], zorder=3, label=label)
    ax2.annotate(fmt(c).split(" ")[0], (data[c]["fpr_at_threshold"], data[c]["tpr_at_threshold"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=7.5, color=scatter_colors[i])

ax2.axhline(y=0.95, color="gray", linestyle="--", linewidth=1, label="95% TPR target")
ax2.set_xlabel("False Positive Rate", fontsize=11)
ax2.set_ylabel("True Positive Rate", fontsize=11)
ax2.set_title("Operating Point at ~95% TPR Threshold", fontsize=12)
ax2.set_xlim(-0.01, 0.32)
ax2.set_ylim(0.94, 0.96)
ax2.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
out = "tpr_95_dashboard.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
