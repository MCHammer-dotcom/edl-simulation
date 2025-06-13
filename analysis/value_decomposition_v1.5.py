# ---------------------- value_decomposition.py ----------------------
"""
Decomposition of Value Sources Over Time
========================================
Creates publication-quality plots (stacked area OR line) showing how operand,
operant, and orchestrant contributions evolve in the simulation.

Run
---
# Stacked-area (default)
python value_decomposition.py

# Overlaid line plot
python value_decomposition.py --line
"""
# -------------------------------------------------------------------
# AESTHETIC PRESETS
try:
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-muted")          # clean Nature-ish look
    import pandas as pd
    import argparse
    from src import ecosim_v1_5 as ecosim                               # ecosim.py must be in same dir
except ModuleNotFoundError:
    print("Missing package. Install with:\n"
          "    pip install matplotlib pandas networkx")
    raise
# -------------------------------------------------------------------
def plot_decomposition(grouped_df: pd.DataFrame,
                       use_line_plot: bool = False,
                       save_path: str = "value_decomposition.png") -> None:
    """
    Plot operand, operant, orchestrant trajectories.
    Parameters
    ----------
    grouped_df : DataFrame with columns time, operand_value, operant_value, orchestrant_value
    use_line_plot : False for stacked area, True for three overlaid lines
    save_path : PNG path for high-res export
    """
    max_y = (grouped_df[["operand_value",
                         "operant_value",
                         "orchestrant_value"]]
             .sum(axis=1).max())

    plt.rcParams.update({"font.size": 10})
    plt.figure(figsize=(7.1, 4.5))              # ≈180 mm width

    if use_line_plot:
        plt.plot(grouped_df["time"], grouped_df["operand_value"],
                 label="Operand",  color="#1f77b4", linewidth=2)
        plt.plot(grouped_df["time"], grouped_df["operant_value"],
                 label="Operant", color="#2ca02c", linewidth=2)
        plt.plot(grouped_df["time"], grouped_df["orchestrant_value"],
                 label="Orchestrant", color="#ff7f0e", linewidth=2)
    else:
        plt.stackplot(grouped_df["time"],
                      grouped_df["operand_value"],
                      grouped_df["operant_value"],
                      grouped_df["orchestrant_value"],
                      labels=["Operand", "Operant", "Orchestrant"],
                      colors=["#1f77b4", "#2ca02c", "#ff7f0e"],
                      alpha=0.9)

    # tipping-point annotation (when orchestrant > operant)
    overtakes = grouped_df["orchestrant_value"] > grouped_df["operant_value"]
    if overtakes.any():
        tip_idx  = overtakes.idxmax()
        tip_time = grouped_df.loc[tip_idx, "time"]
        plt.axvline(tip_time, color="red", linestyle="--", alpha=0.6)
        plt.text(tip_time + 1, max_y * 0.9, "Tipping Point",
                 color="red", fontsize=9)

    plt.xlabel("Time")
    plt.ylabel("Total Value Component")
    plt.title("Decomposition of Value Sources Over Time")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot operand, operant, orchestrant value decomposition."
    )
    parser.add_argument("--line", action="store_true",
                        help="Use three overlaid line plots instead of stacked area.")
    args = parser.parse_args()
    use_line_plot = args.line

    # 1) Run simulation ------------------------------------------------------
    df = ecosim.run_simulation(n_actors=50, n_steps=100, seed=123)

    # 2) Correct orchestrant value if λ part is missing ----------------------
    gamma = ecosim._DEFAULTS["gamma"]
    lam   = ecosim._DEFAULTS["lam"]
    expected_factor = (gamma + lam) / gamma if gamma else 1.0
    # Test on first row
    if gamma and lam:
        first_row = df.iloc[0]
        approx_R  = first_row["orchestrant_value"] / gamma
        if not np.isclose(first_row["orchestrant_value"],
                          approx_R * (gamma + lam)):
            df["orchestrant_value"] *= expected_factor

    # 3) Aggregate per timestep ---------------------------------------------
    grouped = (
        df.groupby("time")[["operand_value",
                            "operant_value",
                            "orchestrant_value"]]
        .sum()
        .reset_index()
    )

    # 4) Export CSV ----------------------------------------------------------
    csv_path = "decomposition_data.csv"
    grouped.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")
    print(grouped.head())

    # 5) Plot & save ---------------------------------------------------------
    out_name = ("value_decomposition_lines.png"
                if use_line_plot else
                "value_decomposition_stacked.png")

    plot_decomposition(grouped, use_line_plot=use_line_plot, save_path=out_name)
    print(f"Figure saved to {out_name}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np  # needed for np.isclose in orchestrant check
    main()
