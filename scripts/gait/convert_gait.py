#!/usr/bin/env python3
"""
Convert gait CSV data to windowed JSONLines for MiniFormer regression.
Reads input CSV, applies sliding windows, and writes JSONL with sequence inputs and next-angle labels.
"""
import os
import json
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Convert gait.csv to windowed JSONLines for MiniFormer regression."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/home/nonu_mishra/miniformer/data/miniformer/gait/raw/gait.csv",
        help="Path to input gait CSV file."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/home/nonu_mishra/miniformer/data/miniformer/gait/jsonl/gait_all.jsonl",
        help="Path to output JSONL file."
    )
    parser.add_argument(
        "--window-size", "-w",
        type=int,
        default=10,
        help="Number of time steps per input window."
    )
    parser.add_argument(
        "--stride", "-s",
        type=int,
        default=1,
        help="Step size between consecutive windows."
    )
    parser.add_argument(
        "--predict-horizon", "-p",
        type=int,
        default=1,
        help="Steps ahead to predict (1 = next time step)."
    )
    args = parser.parse_args()

    # Read and sort data
    df = pd.read_csv(args.input)
    group_cols = ["subject", "condition", "replication", "leg", "joint"]
    df.sort_values(group_cols + ["time"], inplace=True)

    # Prepare output
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(args.output, 'w') as f_out:
        # Group by each gait sequence
        for _, grp in df.groupby(group_cols):
            angles = grp["angle"].tolist()
            # Slide window
            for start in range(0, len(angles) - args.window_size - args.predict_horizon + 1, args.stride):
                window = grp.iloc[start:start + args.window_size]
                # Sequence of features per time step
                features_seq = window[group_cols + ["time"]].to_dict(orient="records")
                # Predict target angle at horizon
                target_idx = start + args.window_size + args.predict_horizon - 1
                target_angle = float(angles[target_idx])
                record = {"input": features_seq, "value": target_angle}
                f_out.write(json.dumps(record) + "\n")
                count += 1

    print(f"Generated {count} windowed records and wrote to {args.output}")

if __name__ == '__main__':
    main()
