import argparse
import runpy
from pathlib import Path

SCRIPTS = {
    "extended": "experiments/extended/edl_experiment_extended_v1.5.py",
    "boosted": "experiments/boosted/edl_boosted_experiment_v1.3.py",
    "field": "experiments/field/edl_field_experiment_v1.0.py",
    "simulated": "experiments/field/edl_simulated_experiment_v1.1.py",
}


def run_script(name: str) -> Path:
    path = SCRIPTS.get(name)
    if not path:
        raise ValueError(f"Unknown script '{name}'. Available: {list(SCRIPTS)}")
    out_dir_before = set(Path('outputs').glob('*'))
    runpy.run_path(path)
    out_dir_after = set(Path('outputs').glob('*'))
    new_dirs = [d for d in out_dir_after - out_dir_before if d.is_dir()]
    for d in new_dirs:
        print(f"Created output folder: {d}")
    return new_dirs[0] if new_dirs else None


def main():
    parser = argparse.ArgumentParser(description="Run EDL experiment scripts")
    parser.add_argument("--script", required=True, choices=SCRIPTS.keys(),
                        help="Which experiment to run")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of times to run the script")
    args = parser.parse_args()
    for i in range(args.repeat):
        print(f"\nRun {i+1}/{args.repeat} of {args.script}")
        run_script(args.script)
    print("Done. Check the outputs/ directory for results.")


if __name__ == "__main__":
    main()
