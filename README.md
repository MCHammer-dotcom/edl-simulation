# edl-simulation

Simulation for Ecosystem-Dominant Logic (EDL)

## Folder layout

```
src/                     core simulation package
experiments/field/       field experiment scripts
experiments/boosted/     boosted-productivity experiment
experiments/extended/    extended model experiment
analysis/                analysis utilities
demos/                   small runnable demos
```

The core model now lives in `src/ecosim_v1_5.py` and is exposed as the
package `src.ecosim_v1_5`.

## Usage examples

From the repository root run any of the scripts, for example:

```bash
python experiments/field/edl_field_experiment_v1.0.py
python experiments/boosted/edl_boosted_experiment_v1.3.py
python experiments/extended/edl_experiment_extended_v1.5.py
python analysis/value_decomposition_v1.5.py
python demos/demo_plot.py
```

Each script writes its output to the current working directory.
