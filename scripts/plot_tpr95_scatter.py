import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

with open("best_models/tpr_95_thresholds_best_models.json") as f:
    data = json.load(f)

labels = {
    "mild_cognitive_impairment_alzhiemers_disease": "MCI / AD",
    "traumatic_brain_injury": "TBI",
    "epilepsy": "Epilepsy",
    "subarachnoid_hemorrhage": "SAH",
    "ischemic_stroke": "Ischemic Stroke",
    "neuroinfectious_diseases": "NIDX",
    "withdrawal_of_life_sustaining_therapy": "WLST",
    "brain_tumor": "Brain Tumor",
    "congestive_heart_failure": "CHF",
    "cardiac_arrest": "Cardiac Arrest",
    "intracranial_hemorrhage": "ICH",
    "parkinsons_disease": "Parkinson's",
    "subdural_hematoma": "SDH",
    "narcolepsy_any_narcolepsy_vs_others": "Narcolepsy",
}

phenotypes = list(data.keys())
fpr_vals = [data[p]["fpr_at_threshold"] for p in phenotypes]
auc_vals = [data[p]["roc_auc"] for p in phenotypes]
short_labels = [labels.get(p, p) for p in phenotypes]

cmap = plt.get_cmap("tab10")
colors = [cmap(i / len(phenotypes)) for i in range(len(phenotypes))]

fig, ax = plt.subplots(figsize=(9, 6))

for i, (fpr, auc, label, color) in enumerate(zip(fpr_vals, auc_vals, short_labels, colors)):
    ax.scatter(fpr, auc, color=color, s=90, zorder=3, label=label)

ax.set_xlabel("False Positive Rate at 95% Sensitivity Threshold", fontsize=12)
ax.set_ylabel("ROC AUC", fontsize=12)
ax.set_title("95% Sensitivity Threshold: ROC AUC vs FPR", fontsize=14)

ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

ax.set_xlim(left=0)
ax.set_ylim(bottom=min(auc_vals) - 0.01, top=1.002)

ax.grid(True, linestyle="--", alpha=0.4)

ax.legend(
    loc="lower right",
    fontsize=8,
    framealpha=0.85,
    ncol=2,
    title="Phenotype",
    title_fontsize=9,
)

plt.tight_layout()
plt.savefig("scripts/figures/tpr95_scatter.png", dpi=150)
plt.show()
print("Saved to scripts/figures/tpr95_scatter.png")
