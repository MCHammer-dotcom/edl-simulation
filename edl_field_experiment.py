# ----------------------- edl_field_experiment.py -----------------------
"""
Simulated Field Experiment: Orchestrant Resource Access
=======================================================

Compares matched Control vs Treatment groups to assess the causal impact
of higher orchestrant parameters (γ, λ) on ecosystem value trajectories.

Outputs
-------
PNG figures:
    • field_value_trajectories.png         – mean ±1 SE total value
    • field_value_decomposition.png        – value-component stacks (both groups)
    • field_delta_value.png                – Δ mean total value (T−C)
    • field_final_value_boxplot.png        – distribution of final actor value
CSV:
    • group_summary.csv                    – mean, std, SE per component × time
    • final_snapshot.csv                   – actor-level variables at T
JSON:
    • param_log.json                       – all experiment parameters
Console:
    • summary stats & tipping-point report
"""
# ----------------------------------------------------------------------
import json
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-muted")
import numpy as np
import pandas as pd
import seaborn as sns

import ecosim  # ecosim.py must reside in the same directory

# ------------------- editable parameters -----------------------------------
params = {
    "gamma_control"   : 0.02,
    "lambda_control"  : 0.01,
    "gamma_treatment" : 0.08,
    "lambda_treatment": 0.05,
    "n_actors"        : 50,
    "n_steps"         : 100,
    "seed"            : 123,
    "density"         : 0.05,
}

OUTDIR = Path(".")
# ---------------------------------------------------------------------------


def run_group(group_name: str, gamma: float, lam: float) -> pd.DataFrame:
    """Execute ecosim with given γ, λ and return annotated DataFrame."""
    df = ecosim.run_simulation(
        n_actors=params["n_actors"],
        n_steps=params["n_steps"],
        density=params["density"],
        seed=params["seed"],
        gamma=gamma,
        lam=lam,
    )
    df["group"] = group_name
    return df


def aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean, std, SEM of each value component by time × group."""
    grp = (
        df.groupby(["group", "time"])[
            ["operand_value", "operant_value", "orchestrant_value", "total_value"]
        ]
        .agg(["mean", "std"])
    )
    grp.columns = ["_".join(col) for col in grp.columns]
    grp = grp.reset_index()
    for comp in ["operand", "operant", "orchestrant", "total"]:
        grp[f"{comp}_SE"] = grp[f"{comp}_value_std"] / np.sqrt(params["n_actors"])
    return grp


def plot_value_trajectories(stats: pd.DataFrame):
    plt.figure(figsize=(7.1, 4.5))
    for grp, style, col in [
        ("Control", "--", "tab:blue"),
        ("Treatment", "-", "tab:orange"),
    ]:
        sub = stats[stats["group"] == grp]
        m = sub["total_value_mean"]
        se = sub["total_SE"]
        plt.plot(sub["time"], m, linestyle=style, color=col, linewidth=2, label=grp)
        plt.fill_between(sub["time"], m - se, m + se, color=col, alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Mean Total Value ±1 SE")
    plt.title("Total Value Over Time: Treatment vs Control")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "field_value_trajectories.png", dpi=300)
    plt.close()


def plot_decomposition(stats: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, grp, colset in zip(
        axs,
        ["Control", "Treatment"],
        [["#1f77b4", "#2ca02c", "#ff7f0e"], ["#8da0cb", "#66c2a5", "#fc8d62"]],
    ):
        sub = stats[stats["group"] == grp]
        ax.stackplot(
            sub["time"],
            sub["operand_value_mean"],
            sub["operant_value_mean"],
            sub["orchestrant_value_mean"],
            labels=["Operand", "Operant", "Orchestrant"],
            colors=colset,
            alpha=0.9,
        )
        ax.set_title(grp)
        ax.set_xlabel("Time")
    axs[0].set_ylabel("Mean Component Value")
    axs[0].legend(loc="upper left")
    fig.suptitle("Value Component Decomposition")
    fig.tight_layout()
    fig.savefig(OUTDIR / "field_value_decomposition.png", dpi=300)
    plt.close(fig)


def plot_delta(stats: pd.DataFrame):
    ctrl = stats[stats["group"] == "Control"].set_index("time")
    trt = stats[stats["group"] == "Treatment"].set_index("time")
    delta = trt["total_value_mean"] - ctrl["total_value_mean"]
    plt.figure(figsize=(7.1, 4.5))
    plt.plot(delta.index, delta.values, color="tab:green", linewidth=2, label="Δ Mean (T−C)")
    # tipping point: first time after which delta stays positive
    tip = None
    for t in delta.index:
        if (delta.loc[t:] > 0).all():
            tip = t
            break
    if tip is not None:
        plt.axvline(tip, color="red", linestyle="--", alpha=0.7)
        ymax = delta.max()
        plt.text(tip + 1, ymax * 0.9, "Tipping Point", color="red", fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Δ Mean Total Value")
    plt.title("Value Gain from Orchestrant Access Over Time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTDIR / "field_delta_value.png", dpi=300)
    plt.close()


def plot_final_boxplot(df_final: pd.DataFrame):
    plt.figure(figsize=(7.1, 4.5))
    sns.boxplot(data=df_final, x="group", y="total_value", palette="Set2")
    plt.ylabel("Final Actor Value V_i(T)")
    plt.title("Final Value Distribution by Group")
    plt.tight_layout()
    plt.savefig(OUTDIR / "field_final_value_boxplot.png", dpi=300)
    plt.close()


def console_summary(df_control: pd.DataFrame, df_treat: pd.DataFrame):
    for name, df in [("Control", df_control), ("Treatment", df_treat)]:
        final = df[df["time"] == params["n_steps"] - 1]
        mean_total = final["total_value"].mean()
        orchestrant_share = (
            final["orchestrant_value"].mean() / mean_total
        )
        adv_mean = final["competitive_advantage"].mean()
        adv_std = final["competitive_advantage"].std()
        print(f"\n{name} Final Stats")
        print("-" * 25)
        print(f"Mean total value       : {mean_total:.2f}")
        print(f"Mean orchestrant share : {orchestrant_share:.2%}")
        print(f"Advantage mean (std)   : {adv_mean:.2f} ({adv_std:.2f})")

    # group-level tipping point when orchestrant > operant (Treatment)
    treat_stats = (
        df_treat.groupby("time")[["orchestrant_value", "operant_value"]]
        .mean()
        .reset_index()
    )
    tipping = treat_stats[
        treat_stats["orchestrant_value"] > treat_stats["operant_value"]
    ]
    if not tipping.empty:
        tp = tipping["time"].iloc[0]
        print(f"\nTipping-point timestep (Orchestrant > Operant in Treatment): {tp}")
    else:
        print("\nNo tipping point where orchestrant exceeded operant.")


def main():
    # 1. Simulate both groups -------------------------------------------------
    df_control = run_group("Control", params["gamma_control"], params["lambda_control"])
    df_treat = run_group(
        "Treatment", params["gamma_treatment"], params["lambda_treatment"]
    )

    # 2. Aggregate stats -----------------------------------------------------
    stats_ctrl = aggregate_stats(df_control)
    stats_trt = aggregate_stats(df_treat)
    stats = pd.concat([stats_ctrl, stats_trt], ignore_index=True)
    stats.to_csv(OUTDIR / "group_summary.csv", index=False)

    # 3. Save final snapshot --------------------------------------------------
    df_final = pd.concat(
        [df_control[df_control["time"] == params["n_steps"] - 1],
         df_treat[df_treat["time"] == params["n_steps"] - 1]],
        ignore_index=True,
    )
    df_final.to_csv(OUTDIR / "final_snapshot.csv", index=False)

    # 4. Parameter log --------------------------------------------------------
    with open(OUTDIR / "param_log.json", "w") as f:
        json.dump(params, f, indent=2)

    # 5. Plots ---------------------------------------------------------------
    plot_value_trajectories(stats)
    plot_decomposition(stats)
    plot_delta(stats)
    plot_final_boxplot(df_final)

    # 6. Console output ------------------------------------------------------
    console_summary(df_control, df_treat)

    print("\nFirst 10 rows of aggregated time-series:")
    print(stats.head(10))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
