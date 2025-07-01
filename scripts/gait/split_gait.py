import json
import random
from pathlib import Path
from typing import Union

def split_by_subject(
    leave_out_subject: int,
    data_dir: Union[str, Path],
):
    """
    Reads gait_all.jsonl from `data_dir` (or data_dir/jsonl), splits it into
    train/val for the given subject, and writes:
      <jsonl_dir>/splits/train_s{subject}.jsonl
      <jsonl_dir>/splits/val_s{subject}.jsonl
    """
    data_dir = Path(data_dir)
    # auto-detect jsonl folder
    if (data_dir / "gait_all.jsonl").exists():
        jsonl_dir = data_dir
    elif (data_dir / "jsonl" / "gait_all.jsonl").exists():
        jsonl_dir = data_dir / "jsonl"
    else:
        raise FileNotFoundError(
            f"Could not find gait_all.jsonl under {data_dir} or {data_dir/'jsonl'}"
        )

    data_file = jsonl_dir / "gait_all.jsonl"
    out_dir = jsonl_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val = [], []
    with open(data_file, "r") as f:
        for line in f:
            rec = json.loads(line)
            subj = rec.get("meta", {}).get("subject")
            if subj == leave_out_subject:
                val.append(rec)
            else:
                train.append(rec)

    random.shuffle(train)
    for split_name, records in [("train", train), ("val", val)]:
        path = out_dir / f"{split_name}_s{leave_out_subject}.jsonl"
        with open(path, "w") as fo:
            for r in records:
                fo.write(json.dumps(r) + "\n")

    print(f"Split for subject {leave_out_subject} written to {out_dir}")