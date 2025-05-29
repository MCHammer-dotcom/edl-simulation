# -------------------- edl_experiment_extended.py --------------------
"""
Extended EDL Experiment – exploiting ecosim v1.4 capabilities
==============================================================

Conditions (100 actors, 120 periods, density 0.08)

1. Baseline-Control              – simple world
2. Enable-Externalities          – actor types + negative externalities
3. Enable-Tipping                – adds tipping-point penalty
4. Enable-All                    – full model: types, boosts, negativity,
                                   adaptive ρ, tipping

All share:
    γ = 0.08, λ = 0.05, R_cap = 10.0, α = 0.05, β = 0.025
Outputs
-------
Tables:  group_summary.csv, final_snapshot.csv, effect_sizes.csv,
         externality_map.csv
Figures: value_trajectories.png, component_decomposition.png,
         delta_plot.png, violin_plot.png, externality_heatmap.png,
         actor_type_scatter.png, rho_volatility.png
"""
# -------------------------------------------------------------------
import json, itertools
from pathlib import Path
import matplotlib.pyplot as plt; plt.style.use("seaborn-v0_8-muted")
import numpy as np, pandas as pd, seaborn as sns
from scipy import stats
import ecosim                                 # must be v1.4+
# -------------------------------------------------------------------
OUT = Path(".")
COMMON = dict(n_actors=100, n_steps=120, density=0.08, seed=123,
              gamma=0.08, lam=0.05,
              alpha=0.05, beta=0.025, R_cap=10.0)

CONDITIONS = {
    "Baseline-Control": dict(),
    "Enable-Externalities": dict(enable_actor_types=True,
                                 enable_negative_externality=True),
    "Enable-Tipping":      dict(enable_tipping_penalty=True),
    "Enable-All":          dict(enable_actor_types=True,
                                 enable_negative_externality=True,
                                 enable_tipping_penalty=True,
                                 enable_volatility_memory=True,
                                 orchestrant_boost_operant=True,
                                 orchestrant_boost_operand=True)
}
# -------------------- helpers ----------------------------------------------
def run_cond(label, extra):
    df = ecosim.run_simulation(**COMMON, **extra, enable_diagnostics=True)
    df["group"] = label
    return df

def summarise(df):
    g = (df.groupby(["group", "time"])
            [["operand_value","operant_value","orchestrant_value","total_value"]]
            .agg(["mean","std"])
            .reset_index())
    g.columns = ["_".join(c).strip("_") if isinstance(c,tuple) else c for c in g.columns]
    for comp in ["operand","operant","orchestrant","total"]:
        g[f"{comp}_SE"] = g[f"{comp}_value_std"] / np.sqrt(COMMON["n_actors"])
    return g

def cohens_d(a,b):
    return (a.mean()-b.mean())/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)

# -------------------- run experiment ---------------------------------------
dfs = [run_cond(k,v) for k,v in CONDITIONS.items()]
df_all = pd.concat(dfs, ignore_index=True)
stats_df = summarise(df_all); stats_df.to_csv(OUT/"group_summary.csv",index=False)

final = df_all[df_all["time"]==COMMON["n_steps"]-1]
final.to_csv(OUT/"final_snapshot.csv",index=False)

# effect sizes vs control
base = final[final["group"]=="Baseline-Control"]["total_value"]
eff = []
for g in CONDITIONS:
    if g=="Baseline-Control": continue
    eff.append({"comparison":f"{g} vs Control",
                "cohens_d":cohens_d(final[final["group"]==g]["total_value"],base)})
pd.DataFrame(eff).to_csv(OUT/"effect_sizes.csv",index=False)

# externality map
ext_map = df_all.pivot(index="actor_id",columns="time",values="externality_received")
ext_map.to_csv(OUT/"externality_map.csv")

# -------------------- ANOVA & shares ---------------------------------------
samples=[final[final["group"]==g]["total_value"] for g in CONDITIONS]
F,p = stats.f_oneway(*samples)
share=(final.groupby("group")[["operand_value","operant_value","orchestrant_value"]]
       .mean().apply(lambda r:r/r.sum()*100,axis=1)).round(1)

type_means = (final.groupby(["group","actor_type"])["total_value"].mean()
                     .unstack().round(2) if "actor_type" in final else None)

rho_track = df_all.groupby("time")["rho_t"].mean() if "rho_t" in df_all else None
vol_track = df_all.groupby("time")["volatility"].mean() if "volatility" in df_all else None

# -------------------- plotting ---------------------------------------------
palette=sns.color_palette("Set2",len(CONDITIONS))

# total trajectories
plt.figure(figsize=(7.1,4.5))
for col,g in zip(palette,CONDITIONS):
    s=stats_df[stats_df["group"]==g]
    plt.plot(s["time"],s["total_value_mean"],label=g,color=col,linewidth=2)
plt.legend();plt.xlabel("Time");plt.ylabel("Mean Total Value")
plt.title("Value Trajectories");plt.tight_layout()
plt.savefig(OUT/"value_trajectories.png",dpi=300);plt.close()

# component decomposition (stacked, first 4 conditions side-by-side)
fig,axs=plt.subplots(1,len(CONDITIONS),figsize=(12,4.5),sharey=True)
for ax,(g,col) in zip(axs,zip(CONDITIONS,palette)):
    s=stats_df[stats_df["group"]==g]
    ax.stackplot(s["time"],s["operand_value_mean"],s["operant_value_mean"],
                 s["orchestrant_value_mean"],colors=["#1f77b4","#2ca02c","#ff7f0e"])
    ax.set_title(g);ax.set_xlabel("Time")
axs[0].set_ylabel("Mean Component Value");fig.tight_layout()
fig.savefig(OUT/"component_decomposition.png",dpi=300);plt.close(fig)

# delta vs control
ctrl=stats_df[stats_df["group"]=="Baseline-Control"].set_index("time")
plt.figure(figsize=(7.1,4.5))
for g,col in zip([k for k in CONDITIONS if k!="Baseline-Control"],palette[1:]):
    s=stats_df[stats_df["group"]==g].set_index("time")
    plt.plot(s.index,s["total_value_mean"]-ctrl["total_value_mean"],
             label=f"{g}−Control",linewidth=2,color=col)
plt.legend();plt.grid(alpha=.4);plt.xlabel("Time");plt.ylabel("Δ Mean Value")
plt.title("Delta Plot");plt.tight_layout()
plt.savefig(OUT/"delta_plot.png",dpi=300);plt.close()

# violin of final
plt.figure(figsize=(7.1,4.5))
sns.violinplot(data=final,x="group",y="total_value",palette="Set2",inner="box")
plt.ylabel("Final V_i(T)");plt.tight_layout()
plt.savefig(OUT/"violin_plot.png",dpi=300);plt.close()

# heatmap of externalities
plt.figure(figsize=(8,6))
sns.heatmap(ext_map, cmap="coolwarm", center=0, cbar_kws={"label":"Ext received"})
plt.xlabel("Time");plt.ylabel("Actor");plt.tight_layout()
plt.savefig(OUT/"externality_heatmap.png",dpi=300);plt.close()

# scatter value vs actor type (if types enabled)
if type_means is not None:
    plt.figure(figsize=(7.1,4.5))
    sns.boxplot(data=final,x="actor_type",y="total_value",palette="Set3")
    plt.title("Actor Type vs Final Value");plt.tight_layout()
    plt.savefig(OUT/"actor_type_scatter.png",dpi=300);plt.close()

# rho & volatility track
if rho_track is not None:
    fig,ax1=plt.subplots(figsize=(7.1,4.5))
    ax1.plot(rho_track.index,rho_track,label="ρ_t",color="navy")
    ax1.set_ylabel("ρ_t");ax2=ax1.twinx()
    ax2.plot(vol_track.index,vol_track,label="Volatility",color="orange")
    ax2.set_ylabel("Volatility (std E_t)")
    ax1.set_xlabel("Time");fig.tight_layout()
    fig.savefig(OUT/"rho_volatility.png",dpi=300);plt.close(fig)

# -------------------- console summary --------------------------------------
print(f"\nANOVA on final total value: F={F:.2f}, p={p:.3g}")
print("\nCohen’s d vs Control:\n",pd.DataFrame(eff))
print("\nComponent share % at T:\n",share)
if type_means is not None:
    print("\nMean final value by actor type:\n",type_means)
if vol_track is not None:
    print(f"\nVolatility mean={vol_track.mean():.3f}  "
          f"ρ_t range=({rho_track.min():.2f}–{rho_track.max():.2f})")

print("\nAll tables and figures saved to working directory.")
