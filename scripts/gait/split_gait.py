#!/usr/bin/env python
import json, random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "../../data/gait"
OUT_DIR  = DATA_DIR / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def split_by_subject(leave_out_subject: int):
    train, val = [], []
    with open(DATA_DIR/"gait_all.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            # `meta.subject` must already be in your JSONL; adapt key if needed
            if rec.get("meta",{}).get("subject") == leave_out_subject:
                val.append(rec)
            else:
                train.append(rec)

    random.shuffle(train)
    for split, lst in [("train", train), ("val", val)]:
        path = OUT_DIR/f"{split}_s{leave_out_subject}.jsonl"
        with open(path, "w") as fo:
            for r in lst:
                fo.write(json.dumps(r) + "\n")
    print(f"Split for subject {leave_out_subject} written to {OUT_DIR}")

if __name__ == "__main__":
    # e.g. leave out subject 10; loop over 1–10 for CV if you like
    split_by_subject(leave_out_subject=10)
