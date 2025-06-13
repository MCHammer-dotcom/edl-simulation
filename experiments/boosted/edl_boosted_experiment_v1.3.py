# --------------------- edl_boosted_experiment_v1.3.py ---------------------
"""
Boosted-Productivity Experiment (bounded version)
=================================================

Compares four orchestrant-productivity settings in the EDL model:

1. Baseline-Control        – no productivity boost
2. Boost-Operant           – orchestrant boosts operant term only
3. Boost-Operand           – orchestrant boosts operand term only
4. Boost-Both (bounded)    – orchestrant boosts both terms (bounded)

New for ecosim v1.3
-------------------
• Sets a safety cap on the orchestrant stock:  R_cap = 10.0  
• Uses milder, bounded boost strengths:  alpha = 0.05, beta = 0.025  
These are passed to every run_simulation() call via the COMMON dict.

Outputs (unchanged)
-------------------
boosted_value_trajectories.png   – total-value lines
boosted_delta_plot.png           – Δ lines vs Baseline
boosted_decomposition.png        – stacked component areas
boosted_value_violin.png         – final-value violin/box
effect_sizes.csv, group_summary.csv, final_snapshot.csv, param_log.json
"""
# ---------------------------------------------------------------------
import json
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-muted")
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import ecosim_v1_5 as ecosim
# ---------------------------------------------------------------------
OUTDIR = Path(".")

# Common parameters for *all* conditions (incl. new v1.3 controls)
COMMON = dict(
    n_actors=50,
    n_steps=100,
    density=0.05,
    seed=123,
    gamma=0.08,
    lam=0.05,
    alpha=0.05,   # milder operant boost strength
    beta=0.025,   # milder operand boost strength
    R_cap=10.0    # bound the orchestrant stock
)

# Condition-specific boost flags
CONDITIONS = {
    "Baseline-Control": dict(orchestrant_boost_operant=False,
                             orchestrant_boost_operand=False),
    "Boost-Operant":    dict(orchestrant_boost_operant=True,
                             orchestrant_boost_operand=False),
    "Boost-Operand":    dict(orchestrant_boost_operant=False,
                             orchestrant_boost_operand=True),
    "Boost-Both (bounded)": dict(orchestrant_boost_operant=True,
                                 orchestrant_boost_operand=True)
}
# ---------------------------------------------------------------------
def run_condition(name: str, flags: dict) -> pd.DataFrame:
    """Run ecosim for one condition and label the DataFrame."""
    df = ecosim.run_simulation(**COMMON, **flags)
    df["group"] = name
    return df

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby(["group", "time"])
          [["operand_value", "operant_value", "orchestrant_value", "total_value"]]
          .agg(["mean", "std"])
          .reset_index()
    )
    grp.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c
                   for c in grp.columns]
    for comp in ["operand", "operant", "orchestrant", "total"]:
        grp[f"{comp}_SE"] = grp[f"{comp}_value_std"] / np.sqrt(COMMON["n_actors"])
    return grp

def effect_size(a, b) -> float:
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
# ---------------------------------------------------------------------
def plot_trajectories(stats_df, groups):
    plt.figure(figsize=(7.1, 4.5))
    palette = sns.color_palette("Set2", n_colors=len(groups))
    for col, g in zip(palette, groups):
        s = stats_df[stats_df["group"] == g]
        plt.plot(s["time"], s["total_value_mean"], label=g, color=col, linewidth=2)
    plt.xlabel("Time"); plt.ylabel("Mean Total Value")
    plt.title("Total Value Trajectories (bounded boosts)")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUTDIR / "boosted_value_trajectories.png", dpi=300)
    plt.close()

def plot_delta(stats_df, groups):
    ctrl = stats_df[stats_df["group"] == "Baseline-Control"].set_index("time")
    plt.figure(figsize=(7.1, 4.5))
    palette = sns.color_palette("Dark2", n_colors=len(groups)-1)
    for col, g in zip(palette, [g for g in groups if g != "Baseline-Control"]):
        s = stats_df[stats_df["group"] == g].set_index("time")
        delta = s["total_value_mean"] - ctrl["total_value_mean"]
        plt.plot(delta.index, delta, label=f"{g} − Control", linewidth=2, color=col)
    plt.xlabel("Time"); plt.ylabel("Δ Mean Total Value")
    plt.title("Value Gain Over Baseline (bounded)"); plt.grid(alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig(OUTDIR / "boosted_delta_plot.png", dpi=300)
    plt.close()

def plot_decomposition(stats_df, groups):
    fig, axs = plt.subplots(1, len(groups), figsize=(10, 4.5), sharey=True)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for ax, g in zip(axs, groups):
        s = stats_df[stats_df["group"] == g]
        ax.stackplot(
            s["time"],
            s["operand_value_mean"],
            s["operant_value_mean"],
            s["orchestrant_value_mean"],
            labels=["Operand", "Operant", "Orchestrant"],
            colors=colors, alpha=0.9
        )
        ax.set_title(g); ax.set_xlabel("Time")
    axs[0].set_ylabel("Mean Component Value"); axs[0].legend(loc="upper left")
    fig.suptitle("Component Decomposition (bounded)"); fig.tight_layout()
    fig.savefig(OUTDIR / "boosted_decomposition.png", dpi=300); plt.close(fig)

def plot_violin(df_final):
    plt.figure(figsize=(7.1, 4.5))
    sns.violinplot(data=df_final, x="group", y="total_value",
                   palette="Set2", inner="box")
    plt.ylabel("Final V_i(T)"); plt.title("Final Actor Value Distribution")
    plt.tight_layout(); plt.savefig(OUTDIR / "boosted_value_violin.png", dpi=300)
    plt.close()
# ---------------------------------------------------------------------
def main():
    dfs = [run_condition(name, flags) for name, flags in CONDITIONS.items()]
    df_all = pd.concat(dfs, ignore_index=True)

    stats_df = summarise(df_all)
    stats_df.to_csv(OUTDIR / "group_summary.csv", index=False)

    df_final = df_all[df_all["time"] == COMMON["n_steps"] - 1]
    df_final.to_csv(OUTDIR / "final_snapshot.csv", index=False)

    baseline_vals = df_final[df_final["group"] == "Baseline-Control"]["total_value"]
    eff_rows = []
    for g in CONDITIONS:
        if g == "Baseline-Control": continue
        d = effect_size(df_final[df_final["group"] == g]["total_value"], baseline_vals)
        eff_rows.append({"comparison": f"{g} vs Baseline", "cohens_d": d})
    pd.DataFrame(eff_rows).to_csv(OUTDIR / "effect_sizes.csv", index=False)

    samples = [df_final[df_final["group"] == g]["total_value"] for g in CONDITIONS]
    F, p = stats.f_oneway(*samples)

    shares = (
        df_final.groupby("group")[["operand_value", "operant_value", "orchestrant_value"]]
        .mean()
        .apply(lambda r: r / r.sum() * 100, axis=1)
        .round(1)
    )

    with open(OUTDIR / "param_log.json", "w") as f:
        json.dump({"COMMON": COMMON, "CONDITIONS": CONDITIONS}, f, indent=2)

    plot_trajectories(stats_df, list(CONDITIONS.keys()))
    plot_delta(stats_df, list(CONDITIONS.keys()))
    plot_decomposition(stats_df, list(CONDITIONS.keys()))
    plot_violin(df_final)

    print(f"\nEffect sizes vs Baseline:\n{pd.DataFrame(eff_rows)}")
    print(f"\nANOVA on final total values: F={F:.2f}, p={p:.3g}")
    print("\nComponent share (%) at T:\n", shares)
    print("\nNote: Bounded boosts (α=0.05, β=0.025, R_cap=10) implement a "
          "saturating enablement effect, preventing runaway growth while "
          "reflecting EDL’s orchestrant context influences.")

if __name__ == "__main__":
    main()
