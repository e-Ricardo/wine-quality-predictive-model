"""
main.py
▶️  Run the complete Wine Quality Prediction pipeline.

Usage:
    python main.py               # full pipeline
    python main.py --skip-train  # skip training (use saved models)
"""

import sys
import argparse
import time

def run_step(label, module_path, fn_name=None):
    """Import and run a step module."""
    print(f"\n{'='*60}")
    print(f"  ▶  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    import importlib.util
    spec = importlib.util.spec_from_file_location("step", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"\n  ⏱  {label} completed in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wine Quality ML Pipeline')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training and use existing saved models')
    args = parser.parse_args()

    print("\n" + "🍷 "*20)
    print("  WINE QUALITY PREDICTION — FULL PIPELINE")
    print("🍷 "*20)

    steps = [
        ("Step 1: EDA",               "src/01_eda.py"),
        ("Step 2: Preprocessing",     "src/02_preprocessing.py"),
        ("Step 3: Model Definitions", "src/03_models.py"),
    ]

    if not args.skip_train:
        steps.append(("Step 4: Training", "src/04_train.py"))

    steps += [
        ("Step 5: Evaluation",       "src/05_evaluate.py"),
        ("Step 6: Feature Analysis", "src/06_feature_analysis.py"),
    ]

    total_start = time.time()
    for label, path in steps:
        run_step(label, path)

    print(f"\n{'='*60}")
    print(f"  ✅  ALL STEPS COMPLETE")
    print(f"  ⏱  Total time: {(time.time()-total_start)/60:.1f} minutes")
    print(f"{'='*60}")
    print("\n📁 Check these folders for outputs:")
    print("   figures/  — all plots & charts")
    print("   models/   — saved .keras models")
    print("   data/     — processed numpy arrays\n")
