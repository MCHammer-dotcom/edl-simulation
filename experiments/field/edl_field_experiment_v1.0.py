# ---------------------- edl_field_experiment_v1.0.py ----------------------
"""
Multi-Condition Simulated Field Experiment
==========================================

Benchmarks four orchestrant-resource conditions in the Ecosystem-Dominant
Logic (EDL) model:

1. Control/Low (γ = 0.02, λ = 0.01)            – constant low access  
2. Medium       (γ = 0.05, λ = 0.03)            – constant medium access  
3. High         (γ = 0.08, λ = 0.05)            – constant high access  
4. Dynamic Onset: low until t=50, then logistic ramp to high  
   γ(t), λ(t) = low + (high−low)/(1 + e^{−k(t−t₀)}) with k = 0.25, t₀ = 50

Outputs
-------
PNG charts (300 dpi, Nature column width)  
CSV tables  (group_summary.csv, final_snapshot.csv, effect_sizes.csv)  
JSON        (param_log.json)  
Console     (ANOVA, Cohen d, tipping points, asymptote explanation)

Usage
-----
$ python experiments/field/edl_field_experiment_v1.0.py           # default four conditions
$ python experiments/field/edl_field_experiment_v1.0.py --conds Control Medium High  # subset

Dependencies
------------
numpy, pandas, matplotlib, seaborn, scipy     (statsmodels optional)
"""

# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-muted")
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import ecosim_v1_5 as ecosim
# --------------------------------------------------------------------------
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COMMON = dict(n_actors=50, n_steps=100, density=0.05, seed=123)

# Orchestrant parameter schedule per condition ------------------------------
LOW   = dict(gamma=0.02, lam=0.01)
MED   = dict(gamma=0.05, lam=0.03)
HIGH  = dict(gamma=0.08, lam=0.05)
DYN   = dict(low=LOW, high=HIGH, k=0.25, t0=50)  # sigmoid ramp spec

CONDITIONS = {
    "Control": LOW,
    "Medium" : MED,
    "High"   : HIGH,
    "Dynamic": DYN,
}
# --------------------------------------------------------------------------


def sigmoid(t, low, high, k=0.25, t0=50):
    """Logistic curve between low and high."""
    return low + (high - low) / (1 + np.exp(-k * (t - t0)))


# ---------- Single-condition runner ----------------------------------------
def run_condition(name: str, spec: dict) -> pd.DataFrame:
    """
    Run ecosim under specified orchestrant parameters.
    For 'Dynamic', post-hoc scale orchestrant_value using logistic γ(t)+λ(t).
    """
    df = ecosim.run_simulation(**COMMON, **LOW)  # always start with LOW
    df["group"] = name

    if name != "Dynamic":
        if not np.isclose(spec["gamma"], LOW["gamma"]) or not np.isclose(
            spec["lam"], LOW["lam"]
        ):
            # scale orchestrant contribution proportionally
            factor = (spec["gamma"] + spec["lam"]) / (LOW["gamma"] + LOW["lam"])
            df["orchestrant_value"] *= factor
            df["total_value"] += (factor - 1) * df["orchestrant_value"] / factor
    else:
        # dynamic logistic ramp
        t = df["time"].values
        γt = sigmoid(t, LOW["gamma"], HIGH["gamma"], k=spec["k"], t0=spec["t0"])
        λt = sigmoid(t, LOW["lam"], HIGH["lam"], k=spec["k"], t0=spec["t0"])
        factor = (γt + λt) / (LOW["gamma"] + LOW["lam"])
        df["orchestrant_value"] *= factor
        df["total_value"] += (factor - 1) * df["orchestrant_value"] / factor
    return df


# ---------- Stats helpers ---------------------------------------------------
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby(["group", "time"])[
            ["operand_value", "operant_value", "orchestrant_value", "total_value"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grp.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in grp.columns
    ]
    for comp in ["operand", "operant", "orchestrant", "total"]:
        grp[f"{comp}_SE"] = grp[f"{comp}_value_std"] / np.sqrt(COMMON["n_actors"])
    return grp


def effect_size(a, b) -> float:
    """Cohen's d for two independent samples."""
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


# ---------- Plotting functions ---------------------------------------------
def plot_trajectories(stats: pd.DataFrame, conds: list[str]):
    plt.figure(figsize=(7.1, 4.5))
    palette = sns.color_palette("Set2", n_colors=len(conds))
    for col, c in zip(palette, conds):
        sub = stats[stats["group"] == c]
        plt.plot(
            sub["time"],
            sub["total_value_mean"],
            label=c,
            color=col,
            linewidth=2,
        )
    plt.xlabel("Time")
    plt.ylabel("Mean Total Value")
    plt.title("Total Value Trajectories Across Conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "exp_value_trajectories.png", dpi=300)
    plt.close()


def plot_delta(stats: pd.DataFrame, conds: list[str]):
    ctrl = stats[stats["group"] == "Control"].set_index("time")
    plt.figure(figsize=(7.1, 4.5))
    for c in [k for k in conds if k != "Control"]:
        sub = stats[stats["group"] == c].set_index("time")
        delta = sub["total_value_mean"] - ctrl["total_value_mean"]
        plt.plot(delta.index, delta, label=f"{c} − Control", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Δ Mean Total Value")
    plt.title("Treatment Gains Over Control")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "exp_delta_value.png", dpi=300)
    plt.close()


def plot_component_shares(stats: pd.DataFrame, conds: list[str]):
    fig, axs = plt.subplots(1, len(conds), figsize=(10, 4.5), sharey=True)
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for ax, c in zip(axs, conds):
        sub = stats[stats["group"] == c]
        ax.stackplot(
            sub["time"],
            sub["operand_value_mean"],
            sub["operant_value_mean"],
            sub["orchestrant_value_mean"],
            labels=["Operand", "Operant", "Orchestrant"],
            colors=palette,
            alpha=0.9,
        )
        ax.set_title(c)
        ax.set_xlabel("Time")
    axs[0].set_ylabel("Mean Component Value")
    axs[0].legend(loc="upper left")
    fig.suptitle("Component Decomposition by Condition")
    fig.tight_layout()
    fig.savefig(OUT / "exp_value_decomposition.png", dpi=300)
    plt.close(fig)


def plot_final_distribution(df_final: pd.DataFrame):
    plt.figure(figsize=(7.1, 4.5))
    sns.violinplot(
        data=df_final, x="group", y="total_value", palette="Set2", inner="box"
    )
    plt.ylabel("Final V_i(T)")
    plt.title("Final Actor Value Distribution")
    plt.tight_layout()
    plt.savefig(OUT / "exp_final_value_distribution.png", dpi=300)
    plt.close()


# ---------- CLI + main ------------------------------------------------------
def main(conds: list[str]):
    print("Running conditions:", ", ".join(conds))
    dfs = [run_condition(c, CONDITIONS[c]) for c in conds]
    df_all = pd.concat(dfs, ignore_index=True)

    stats = summarise(df_all)
    stats.to_csv(OUT / "group_summary.csv", index=False)

    # Effect sizes & one-way ANOVA on final total value ---------------------
    final = df_all[df_all["time"] == COMMON["n_steps"] - 1]
    effect_rows = []
    for c in conds:
        if c == "Control":
            continue
        d = effect_size(
            final.loc[final["group"] == c, "total_value"],
            final.loc[final["group"] == "Control", "total_value"],
        )
        effect_rows.append({"comparison": f"{c} vs Control", "cohens_d": d})
    eff_df = pd.DataFrame(effect_rows)
    eff_df.to_csv(OUT / "effect_sizes.csv", index=False)

    groups = [final.loc[final["group"] == c, "total_value"] for c in conds]
    f_stat, p_val = stats.anova.f_oneway(*groups)
    print(f"\nOne-way ANOVA on V_i(T): F={f_stat:.2f}, p={p_val:.3g}")

    # Percent component shares at final time -------------------------------
    share = (
        final.groupby("group")[["operand_value", "operant_value", "orchestrant_value"]]
        .mean()
        .apply(lambda row: row / row.sum() * 100, axis=1)
    )
    print("\nComponent share at T (mean %, rounded):\n", share.round(1))

    # Save final snapshot
    final.to_csv(OUT / "final_snapshot.csv", index=False)

    # Parameter log
    with open(OUT / "param_log.json", "w") as f:
        json.dump({**COMMON, **CONDITIONS}, f, indent=2)

    # Plots
    plot_trajectories(stats, conds)
    plot_delta(stats, conds)
    plot_component_shares(stats, conds)
    plot_final_distribution(final)

    print(f"\nData & figures saved to {OUT.resolve()}")
    print("\nAsymptote note: Curves flatten because the externality-driven "
          "orchestrant stock R_t approaches a bounded equilibrium while "
          "operant uniqueness erodes (ε > 0), limiting further growth.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run multi-condition orchestrant experiment."
    )
    ap.add_argument(
        "--conds",
        nargs="+",
        default=list(CONDITIONS.keys()),
        help=f"Subset of conditions (default all: {list(CONDITIONS.keys())})",
    )
    args = ap.parse_args()
    main(args.conds)
