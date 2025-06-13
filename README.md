# edl-simulation
Simulation for Ecosystem-Dominant Logic (EDL)

## Running the Demo

The core simulation code now lives in `src/` as `ecosim_v1_5.py`.  The
experiment scripts import it via:

```python
from src import ecosim_v1_5 as ecosim
```

Run the demo plot with:

```bash
python demo_plot.py
```

## Running Experiments

Each experiment script can be executed directly from the repository root.
Examples:

```bash
python edl_boosted_experiment.py
python edl_simulated_experiment.py
python edl_field_experiment.py
python edl_experiment_extended.py
python value_decomposition.py
```

Ensure all required packages (pandas, matplotlib, seaborn, numpy, scipy) are
installed.
