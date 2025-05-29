# ---------------------------- demo_plot.py ----------------------------
"""
Quick demo runner for ecosim.py

Features
--------
1. Runs a 50-actor, 100-step simulation (seed=123).
2. Prints the first 10 rows of the DataFrame.
3. Plots the total orchestrant value over time.
4. Optional: save the plot as a PNG (uncomment the line near the end).

Usage
-----
    python demo_plot.py
"""
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import ecosim                     # assumes ecosim.py is in the same folder
except ModuleNotFoundError as e:
    print(
        "Missing a required package. Install with:\n"
        "    pip install matplotlib pandas networkx\n"
        "Also ensure ecosim.py is present in the same directory."
    )
    raise

# ----------------------------------------------------------------------
def run_demo():
    """Run simulation and make the orchestrant-value plot."""
    # 1) Run simulation
    df = ecosim.run_simulation(n_actors=50, n_steps=100, seed=123)

    # 2) Inspect first 10 rows
    print(df.head(10))

    # 3) Aggregate total orchestrant value per timestep
    tot_orc = (
        df.groupby("time")["orchestrant_value"]
        .sum()
        .reset_index()
    )

    # 4) Plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        tot_orc["time"],
        tot_orc["orchestrant_value"],
        label="Total Orchestrant Value",
        color="navy",
        linewidth=2,
    )
    plt.xlabel("Time")
    plt.ylabel("Total Orchestrant Value")
    plt.title("Total Orchestrant Value Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Uncomment the following line to save the plot to disk at 300 dpi
    # plt.savefig("orchestrant_plot.png", dpi=300)

    # If the curve is flat, possible issues:
    #   • eta=0 or rho=1  ⇒ orchestrant stock never grows
    #   • phi/psi/gamma=0 ⇒ orchestrant value not counted

# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_demo()
