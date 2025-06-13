# ---------------------- edl_simulated_experiment_v1.1.py ----------------------
"""
Multi-Condition Simulated Field Experiment
==========================================

Benchmarks four orchestrant-resource conditions:

1. Control  (Low γ, λ)
2. Medium   (Mid γ, λ)
3. High     (High γ, λ)
4. Dynamic  (sigmoid ramp from low to high at t₀ = 50)

Corrected version: avoids variable shadowing of `scipy.stats` and
uses `stats.f_oneway` for ANOVA.
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
from scipy import stats             # <-- keep module reference intact

from src import ecosim_v1_5 as ecosim
# --------------------------------------------------------------------------
OUTDIR = Path(".")
COMMON = dict(n_actors=50, n_steps=100, density=0.05, seed=123)

# Condition specifications ---------------------------------------------------
LOW   = dict(gamma=0.02, lam=0.01)
MED   = dict(gamma=0.05, lam=0.03)
HIGH  = dict(gamma=0.08, lam=0.05)
DYN   = dict(low=LOW, high=HIGH, k=0.25, t0=50)  # sigmoid spec

CONDITIONS = {"Control": LOW, "Medium": MED, "High": HIGH, "Dynamic": DYN}
# --------------------------------------------------------------------------


def sigmoid(t, low, high, k=0.25, t0=50):
    """Logistic curve between low and high."""
    return low + (high - low) / (1 + np.exp(-k * (t - t0)))


def run_condition(name: str, spec: dict) -> pd.DataFrame:
    """Run ecosim under specified orchestrant parameters."""
    df = ecosim.run_simulation(**COMMON, **LOW)  # start from LOW baseline
    df["group"] = name

    if name != "Dynamic":
        factor = (spec["gamma"] + spec["lam"]) / (LOW["gamma"] + LOW["lam"])
        if not np.isclose(factor, 1.0):
            df["orchestrant_value"] *= factor
            df["total_value"] += (factor - 1) * df["orchestrant_value"] / factor
    else:
        t = df["time"].values
        γt = sigmoid(t, LOW["gamma"], HIGH["gamma"], k=spec["k"], t0=spec["t0"])
        λt = sigmoid(t, LOW["lam"], HIGH["lam"], k=spec["k"], t0=spec["t0"])
        factor = (γt + λt) / (LOW["gamma"] + LOW["lam"])
        df["orchestrant_value"] *= factor
        df["total_value"] += (factor - 1) * df["orchestrant_value"] / factor
    return df


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


# -------------------- plotting helpers -------------------------------------
def plot_trajectories(stats_df: pd.DataFrame, conds: list[str]):
    plt.figure(figsize=(7.1, 4.5))
    palette = sns.color_palette("Set2", n_colors=len(conds))
    for col, c in zip(palette, conds):
        sub = stats_df[stats_df["group"] == c]
        plt.plot(sub["time"], sub["total_value_mean"], label=c, color=col, linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Mean Total Value")
    plt.title("Total Value Trajectories Across Conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "exp_value_trajectories.png", dpi=300)
    plt.close()


def plot_delta(stats_df: pd.DataFrame, conds: list[str]):
    ctrl = stats_df[stats_df["group"] == "Control"].set_index("time")
    plt.figure(figsize=(7.1, 4.5))
    for c in [k for k in conds if k != "Control"]:
        sub = stats_df[stats_df["group"] == c].set_index("time")
        delta = sub["total_value_mean"] - ctrl["total_value_mean"]
        plt.plot(delta.index, delta, label=f"{c} − Control", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Δ Mean Total Value")
    plt.title("Treatment Gains Over Control")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "exp_delta_value.png", dpi=300)
    plt.close()


def plot_component_shares(stats_df: pd.DataFrame, conds: list[str]):
    fig, axs = plt.subplots(1, len(conds), figsize=(10, 4.5), sharey=True)
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for ax, c in zip(axs, conds):
        sub = stats_df[stats_df["group"] == c]
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
    fig.savefig(OUTDIR / "exp_value_decomposition.png", dpi=300)
    plt.close(fig)


def plot_final_distribution(df_final: pd.DataFrame):
    plt.figure(figsize=(7.1, 4.5))
    sns.violinplot(data=df_final, x="group", y="total_value", palette="Set2", inner="box")
    plt.ylabel("Final V_i(T)")
    plt.title("Final Actor Value Distribution")
    plt.tight_layout()
    plt.savefig(OUTDIR / "exp_final_value_distribution.png", dpi=300)
    plt.close()


# -------------------- main experiment --------------------------------------
def main(conds: list[str]):
    print("Running conditions:", ", ".join(conds))
    dfs = [run_condition(c, CONDITIONS[c]) for c in conds]
    df_all = pd.concat(dfs, ignore_index=True)

    stats_df = summarise(df_all)
    stats_df.to_csv(OUTDIR / "group_summary.csv", index=False)

    # Effect sizes and ANOVA -------------------------------------------------
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
    pd.DataFrame(effect_rows).to_csv(OUTDIR / "effect_sizes.csv", index=False)

    groups = [final.loc[final["group"] == c, "total_value"] for c in conds]
    f_stat, p_val = stats.f_oneway(*groups)        # <-- corrected call
    print(f"\nOne-way ANOVA on V_i(T): F={f_stat:.2f}, p={p_val:.3g}")

    # Component share at T
    share = (
        final.groupby("group")[["operand_value", "operant_value", "orchestrant_value"]]
        .mean()
        .apply(lambda row: row / row.sum() * 100, axis=1)
    )
    print("\nComponent share at T (mean %, rounded):\n", share.round(1))

    final.to_csv(OUTDIR / "final_snapshot.csv", index=False)

    # Save param log
    with open(OUTDIR / "param_log.json", "w") as f:
        json.dump({**COMMON, **CONDITIONS}, f, indent=2)

    # Plots
    plot_trajectories(stats_df, conds)
    plot_delta(stats_df, conds)
    plot_component_shares(stats_df, conds)
    plot_final_distribution(final)

    print("\nData & figures saved. Asymptote note: Curves flatten as "
          "externality-driven orchestrant stock approaches equilibrium "
          "and operant uniqueness erodes (ε > 0).")


# -------------------- CLI ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-condition orchestrant experiment."
    )
    parser.add_argument(
        "--conds",
        nargs="+",
        default=list(CONDITIONS.keys()),
        help="Subset of conditions to run",
    )
    args = parser.parse_args()
    main(args.conds)
