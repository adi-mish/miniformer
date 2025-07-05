"""
Run full 10-fold leave-one-subject-out cross-validation for gait classification
using the MiniFormer trainer. Splits data, writes per-fold configs, invokes
trainer, captures logs, and summarizes validation accuracies.
"""
import subprocess
import argparse
import json
import re
from pathlib import Path
from split_gait import split_by_subject

def run_cross_validation(data_dir, base_config_path, n_subjects=10, logs_dir=None):
    data_dir = Path(data_dir)
    base_config_path = Path(base_config_path)
    work_dir = Path.cwd()

    # Load base configuration
    with open(base_config_path) as f:
        base_config = json.load(f)

    # Prepare logs directory
    logs_directory = Path(logs_dir) if logs_dir else work_dir / "logs"
    logs_directory.mkdir(parents=True, exist_ok=True)

    accuracies = []

    for subject in range(1, n_subjects + 1):
        print(f"\n{'='*60}\nFold: leave out subject {subject}\n{'='*60}")

        # Split data for this subject
        split_by_subject(leave_out_subject=subject, data_dir=data_dir)

        # determine where splits were written
        # it could be data_dir/splits or data_dir/jsonl/splits
        if (data_dir / "splits").exists():
            split_root = data_dir / "splits"
        else:
            split_root = data_dir / "jsonl" / "splits"

        # Create per-fold config dict
        subject_config = base_config.copy()
        exp_base = base_config.get("experiment_name", "gait_miniformer")
        experiment_name = f"{exp_base}_s{subject}"
        subject_config["experiment_name"] = experiment_name
        subject_config["train_path"] = str(split_root / f"train_s{subject}.jsonl")
        subject_config["val_path"]   = str(split_root / f"val_s{subject}.jsonl")

        # Write the fold-specific config to disk
        subject_cfg_path = base_config_path.parent / f"{base_config_path.stem}_s{subject}.json"
        with open(subject_cfg_path, "w") as cf:
            json.dump(subject_config, cf, indent=2)

        # Launch training subprocess and capture output
        cmd = [
            "python", "-m", "miniformer.train.trainer",
            "--config_json", str(subject_cfg_path)
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Save stdout & stderr to a log file
        log_file = logs_directory / f"{experiment_name}.log"
        with open(log_file, "w") as lf:
            lf.write(result.stdout)
            lf.write("\n" + result.stderr)

        # Pick metric name by task  -----------------------------------------
        task = subject_config["task"]
        if task == "classification":
            pattern = r"val_acc(?:uracy)?[\s=:]+([0-9\.]+)"
            metric_name = "val_accuracy"
        elif task == "regression":
            pattern = r"val_mae[\s=:]+([0-9\.]+)"
            metric_name = "val_mae"
        else:                            # language_modeling
            pattern = r"val_ppl[\s=:]+([0-9\.]+)"
            metric_name = "val_ppl"

        m = re.search(pattern, result.stdout)
        val_metric = float(m.group(1)) if m else None
        accuracies.append((subject, val_metric))
        print(f"Subject {subject} ➜ {metric_name} = {val_metric}\n")

    # Summarize all folds
    print(f"\n{'='*60}\nCross-validation results summary\n{'='*60}")
    for subject, acc in accuracies:
        print(f"Subject {subject}: val_acc = {acc}")
    valid_accs = [a for _, a in accuracies if a is not None]
    if valid_accs:
        mean_acc = sum(valid_accs) / len(valid_accs)
        print(f"Mean validation accuracy: {mean_acc:.4f}")
    else:
        print("No valid accuracies parsed.")

def main():
    parser = argparse.ArgumentParser(
        description="Run 10-way leave-one-subject-out cross-validation"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/home/nonu_mishra/miniformer/data/miniformer/gait",
        help="Path to gait data directory (containing gait_all.jsonl or a jsonl/ subfolder)"
    )
    parser.add_argument(
        "--config", dest="config_path", type=str,
        default="/home/nonu_mishra/miniformer/configs/gait/gait_cfg.json",
        help="Path to base configuration JSON file"
    )
    parser.add_argument(
        "--n_subjects", type=int, default=10,
        help="Number of subjects (folds) to run"
    )
    parser.add_argument(
        "--logs_dir", type=str, default=None,
        help="Optional directory to write per-fold log files"
    )
    args = parser.parse_args()
    run_cross_validation(
        data_dir=args.data_dir,
        base_config_path=args.config_path,
        n_subjects=args.n_subjects,
        logs_dir=args.logs_dir
    )

if __name__ == "__main__":
    main()