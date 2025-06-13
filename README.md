# edl-simulation
Simulation for Ecosystem-Dominant Logic (EDL)

## Project Layout

```
src/                      # ecosim_v1_5.py engine
experiments/field/        # field experiments
experiments/boosted/      # productivity boost experiment
experiments/extended/     # extended experiment
analysis/                 # value analysis scripts
demos/                    # quick demo scripts
```

All scripts import the engine via:

```python
from src import ecosim_v1_5 as ecosim
```

## Running the Demo

```
python demos/demo_plot.py
```

## Running Experiments

Execute any experiment from the repository root. Examples:

```
python experiments/boosted/edl_boosted_experiment_v1.3.py
python experiments/field/edl_field_experiment_v1.0.py
python experiments/field/edl_simulated_experiment_v1.1.py
python experiments/extended/edl_experiment_extended_v1.5.py
python analysis/value_decomposition_v1.5.py
```

Ensure all required packages (pandas, matplotlib, seaborn, numpy, scipy) are installed.
