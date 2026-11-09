"""Project 02: Data Visualization with Matplotlib & Seaborn (Titanic dataset)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Custom palette & global formatting ----------
NAVY, CORAL, TEAL, GOLD, GREY = "#1f3a5f", "#e4572e", "#2a9d8f", "#e9b44c", "#8d99ae"
SURV = {0: CORAL, 1: TEAL}                      # died = coral, survived = teal
CLASS_PAL = [GOLD, TEAL, NAVY]                   # 1st, 2nd, 3rd
sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 15,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

df = pd.read_csv("titanic_cleaned.csv")
df["Status"] = df["Survived"].map({0: "Died", 1: "Survived"})
status_pal = {"Died": CORAL, "Survived": TEAL}


def finish(fig, name):
    fig.savefig(name)
    plt.close(fig)
    print("saved", name)


# ---------- 1. BAR CHART: survival rate by class and sex ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex",
            palette={"female": CORAL, "male": NAVY}, errorbar=None, ax=ax)
for c in ax.containers:
    ax.bar_label(c, labels=[f"{v.get_height():.0%}" for v in c], padding=3)
ax.set_title("Women in every class were far more likely to survive")
ax.set_xlabel("Passenger class"); ax.set_ylabel("Survival rate")
ax.set_ylim(0, 1.1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.legend(title="", frameon=False)
finish(fig, "viz_1_bar_survival.png")

# ---------- 2. HISTOGRAM: age distribution ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.histplot(data=df, x="Age", hue="Status", palette=status_pal, bins=25,
             multiple="layer", alpha=0.55, kde=True, ax=ax)
ax.set_title("Young children had the best odds; most deaths were adults 20-40")
ax.set_xlabel("Age (years)"); ax.set_ylabel("Passengers")
finish(fig, "viz_2_hist_age.png")

# ---------- 3. BOX PLOT: fare by class ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.boxplot(data=df, x="Pclass", y="Fare", hue="Pclass", palette=CLASS_PAL,
            legend=False, fliersize=3, ax=ax)
ax.set_yscale("log")
ax.set_title("Fares differ sharply by class (log scale)")
ax.set_xlabel("Passenger class"); ax.set_ylabel("Fare (log scale)")
finish(fig, "viz_3_box_fare.png")

# ---------- 4. SCATTER PLOT: age vs fare ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.scatterplot(data=df[df.Fare > 0], x="Age", y="Fare", hue="Status",
                style="Status", palette=status_pal, alpha=0.7, s=60, ax=ax)
ax.set_yscale("log")
ax.set_title("Higher-fare passengers survived more often")
ax.set_xlabel("Age (years)"); ax.set_ylabel("Fare (log scale)")
ax.legend(title="", frameon=False)
finish(fig, "viz_4_scatter_age_fare.png")

# ---------- 5. CORRELATION HEATMAP ----------
cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare",
        "FamilySize", "IsAlone", "HasCabin"]
corr = df[cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, linewidths=0.6, square=True,
            cbar_kws={"shrink": 0.75, "label": "Pearson r"}, ax=ax)
ax.grid(False)
ax.set_title("Correlation matrix: class, cabin and fare track survival")
finish(fig, "viz_5_corr_heatmap.png")

# ---------- 6. HEATMAP: survival by class x age group ----------
df["AgeGroup"] = pd.cut(df["Age"], [0, 12, 18, 35, 60, 100],
                        labels=["Child", "Teen", "Young adult", "Adult", "Senior"])
pivot = df.pivot_table(values="Survived", index="Pclass", columns="AgeGroup",
                       aggfunc="mean", observed=True)
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap=sns.light_palette(TEAL, as_cmap=True),
            linewidths=0.6, cbar_kws={"label": "Survival rate"}, ax=ax)
ax.grid(False)
ax.set_title("Survival rate by class and age group")
ax.set_xlabel(""); ax.set_ylabel("Passenger class")
finish(fig, "viz_6_heatmap_class_age.png")

# ---------- 7. DASHBOARD: the full story ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex",
            palette={"female": CORAL, "male": NAVY}, errorbar=None, ax=axes[0, 0])
axes[0, 0].set_title("Survival rate by class and sex"); axes[0, 0].legend(frameon=False)
axes[0, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); axes[0, 0].set_ylabel("Survival rate")
sns.histplot(data=df, x="Age", hue="Status", palette=status_pal, bins=25,
             alpha=0.55, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Age distribution")
sns.boxplot(data=df, x="Pclass", y="Fare", hue="Pclass", palette=CLASS_PAL,
            legend=False, ax=axes[1, 0])
axes[1, 0].set_yscale("log"); axes[1, 0].set_title("Fare by class (log)")
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, cbar=False, annot_kws={"size": 8}, ax=axes[1, 1])
axes[1, 1].grid(False); axes[1, 1].set_title("Correlation matrix")
fig.suptitle("Titanic: who survived, and why?", fontsize=22, fontweight="bold", y=1.0)
fig.tight_layout()
finish(fig, "viz_0_dashboard.png")
