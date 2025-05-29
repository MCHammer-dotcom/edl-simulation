# ----------------------- edl_field_experiment.py -----------------------
"""
Field-Experiment Simulation
===========================
Compares Control (low orchestrant access) vs Treatment (high orchestrant access)
using the Ecosystem-Dominant Logic engine.

Outputs
-------
field_value_trajectories.png   – total value (sum) over time
field_delta_value.png          – Δ mean value (Treatment − Control) over time
field_final_value_boxplot.png  – boxplot of final actor values
field_experiment_data.csv      – time-series + final values data
"""
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-muted")
import pandas as pd
import numpy as np

import ecosim                     # ecosim.py must reside in same directory

# ------------------ parameters ----------------------------------------
COMMON_KW = dict(n_actors=50, n_steps=100, density=0.05, seed=456)
CTRL_KW   = dict(gamma=0.1, lam=0.0)
TRT_KW    = dict(gamma=0.5, lam=0.2)

# ------------------ run matched simulations ---------------------------
df_control   = ecosim.run_simulation(**COMMON_KW, **CTRL_KW)
df_treatment = ecosim.run_simulation(**COMMON_KW, **TRT_KW)

# ------------------ aggregate trajectories ----------------------------
sum_ctrl = (df_control.groupby("time")["total_value"].sum()
            .rename("sum_control"))
sum_trt  = (df_treatment.groupby("time")["total_value"].sum()
            .rename("sum_treatment"))
mean_ctrl = (df_control.groupby("time")["total_value"].mean()
             .rename("mean_control"))
mean_trt  = (df_treatment.groupby("time")["total_value"].mean()
             .rename("mean_treatment"))

traj_df = pd.concat([sum_ctrl, sum_trt, mean_ctrl, mean_trt], axis=1).reset_index()
traj_df["delta_mean"] = traj_df["mean_treatment"] - traj_df["mean_control"]

# ------------------ tipping point (first sustained positive) ----------
tp_idx = None
for i in range(len(traj_df)):
    if traj_df.loc[i:, "delta_mean"].min() > 0:
        tp_idx = i
        break

# ------------------ Plot A: total value trajectories -------------------
plt.figure(figsize=(7.1, 4.5))
plt.plot(traj_df["time"], traj_df["sum_control"], label="Control",
         color="tab:blue", linestyle="--", linewidth=2)
plt.plot(traj_df["time"], traj_df["sum_treatment"], label="Treatment",
         color="tab:orange", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Total Value (Σ actors)")
plt.title("Total Value Over Time: Treatment vs Control")
plt.legend()
plt.tight_layout()
plt.savefig("field_value_trajectories.png", dpi=300)

# ------------------ Plot B: Δ mean value -------------------------------
plt.figure(figsize=(7.1, 4.5))
plt.plot(traj_df["time"], traj_df["delta_mean"],
         label="Δ Mean Value (T−C)", color="tab:green", linewidth=2)
if tp_idx is not None:
    tip_t = traj_df.loc[tp_idx, "time"]
    plt.axvline(tip_t, color="red", linestyle="--", alpha=0.7)
    ymax = traj_df["delta_mean"].max()
    plt.text(tip_t + 1, ymax * 0.9, "Tipping Point", color="red", fontsize=9)
plt.xlabel("Time")
plt.ylabel("Δ Mean Value")
plt.title("Value Gain from Orchestrant Access Over Time")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("field_delta_value.png", dpi=300)

# ------------------ Plot C: final value distribution -------------------
final_ctrl = df_control[df_control["time"] == COMMON_KW["n_steps"] - 1] \
             .assign(group="Control")
final_trt  = df_treatment[df_treatment["time"] == COMMON_KW["n_steps"] - 1] \
             .assign(group="Treatment")
final_df = pd.concat([final_ctrl, final_trt], ignore_index=True)

plt.figure(figsize=(7.1, 4.5))
plt.boxplot([final_ctrl["total_value"], final_trt["total_value"]],
            labels=["Control", "Treatment"],
            patch_artist=True,
            boxprops=dict(facecolor="lightgrey"))
plt.ylabel("Final Actor Value V_i(T)")
plt.title("Final Value Distribution: Control vs Treatment")
plt.tight_layout()
plt.savefig("field_final_value_boxplot.png", dpi=300)

# ------------------ Export data & console check ------------------------
export_df = traj_df.copy()
export_df["group"] = "trajectory"
export_df = export_df.melt(id_vars=["time", "group"],
                           value_vars=["sum_control", "sum_treatment",
                                       "mean_control", "mean_treatment",
                                       "delta_mean"],
                           var_name="metric", value_name="value")
final_vals_export = final_df[["time", "actor_id", "total_value", "group"]]
final_vals_export = final_vals_export.rename(columns={"total_value": "value",
                                                      "actor_id": "metric"})
final_vals_export["group"] = final_vals_export["group"] + "_final"

combined = pd.concat([export_df, final_vals_export], ignore_index=True)
combined.to_csv("field_experiment_data.csv", index=False)
print("Data exported to field_experiment_data.csv")

print("\nFirst 10 rows of time-series averages:")
print(traj_df.head(10))

# ----------------------------------------------------------------------
if __name__ == "__main__":
    plt.show()     # show all queued figures interactively
