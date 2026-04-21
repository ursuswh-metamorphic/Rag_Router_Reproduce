"""Split MIRAGE benchmark questions at question level.

The script keeps the original MIRAGE structure:
{
  "dataset_name": {
    "question_id": {...question payload...},
    ...
  },
  ...
}

Outputs three JSON files (train/val/test) and one metadata JSON file.
"""

from __future__ import annotations

import argparse
import json
import random
from math import floor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split MIRAGE benchmark file by question IDs (train/val/test)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mirage.json"),
        help="Input MIRAGE JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where split files will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.30)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-stratify-by-dataset",
        action="store_true",
        help="If set, split all question IDs globally across datasets.",
    )
    return parser.parse_args()


def allocate_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [n * ratio for ratio in ratios]
    base = [floor(value) for value in raw]
    remaining = n - sum(base)

    fractions = sorted(
        ((raw[idx] - base[idx], idx) for idx in range(len(raw))), reverse=True
    )
    for _, idx in fractions[:remaining]:
        base[idx] += 1

    return int(base[0]), int(base[1]), int(base[2])


def split_ids(
    ids: list[str], ratios: tuple[float, float, float], rng: random.Random
) -> tuple[set[str], set[str], set[str]]:
    ids = list(ids)
    rng.shuffle(ids)
    n_train, n_val, n_test = allocate_counts(len(ids), ratios)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val : n_train + n_val + n_test])
    return train_ids, val_ids, test_ids


def main() -> None:
    args = parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    if not args.input.exists():
        raise FileNotFoundError(f"Input MIRAGE file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as infile:
        source = json.load(infile)

    if not isinstance(source, dict):
        raise ValueError("MIRAGE JSON must be a dictionary at top level.")

    for dataset_name, dataset_items in source.items():
        if not isinstance(dataset_items, dict):
            raise ValueError(
                f"Dataset '{dataset_name}' must be a dictionary of question entries."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    train_split: dict[str, dict] = {}
    val_split: dict[str, dict] = {}
    test_split: dict[str, dict] = {}

    split_counts = {
        "datasets": {},
        "questions": {"train": 0, "val": 0, "test": 0, "total": 0},
    }

    if args.no_stratify_by_dataset:
        qid_to_dataset: dict[str, str] = {}
        for dataset_name, dataset_items in source.items():
            for qid in dataset_items.keys():
                unique_key = f"{dataset_name}::{qid}"
                qid_to_dataset[unique_key] = dataset_name

        rng = random.Random(args.seed)
        train_keys, val_keys, test_keys = split_ids(list(qid_to_dataset), ratios, rng)

        grouped = {
            "train": train_keys,
            "val": val_keys,
            "test": test_keys,
        }

        for dataset_name, dataset_items in source.items():
            train_rows = {}
            val_rows = {}
            test_rows = {}
            for qid, payload in dataset_items.items():
                key = f"{dataset_name}::{qid}"
                if key in grouped["train"]:
                    train_rows[qid] = payload
                elif key in grouped["val"]:
                    val_rows[qid] = payload
                else:
                    test_rows[qid] = payload

            train_split[dataset_name] = train_rows
            val_split[dataset_name] = val_rows
            test_split[dataset_name] = test_rows

            split_counts["datasets"][dataset_name] = {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
                "total": len(dataset_items),
            }
    else:
        global_rng = random.Random(args.seed)
        for dataset_name in source:
            dataset_items = source[dataset_name]
            ids = sorted(dataset_items.keys())
            dataset_rng = random.Random(global_rng.randint(0, 2**31 - 1))
            train_ids, val_ids, test_ids = split_ids(ids, ratios, dataset_rng)

            train_rows = {qid: dataset_items[qid] for qid in ids if qid in train_ids}
            val_rows = {qid: dataset_items[qid] for qid in ids if qid in val_ids}
            test_rows = {qid: dataset_items[qid] for qid in ids if qid in test_ids}

            train_split[dataset_name] = train_rows
            val_split[dataset_name] = val_rows
            test_split[dataset_name] = test_rows

            split_counts["datasets"][dataset_name] = {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
                "total": len(dataset_items),
            }

    for dataset_stats in split_counts["datasets"].values():
        split_counts["questions"]["train"] += dataset_stats["train"]
        split_counts["questions"]["val"] += dataset_stats["val"]
        split_counts["questions"]["test"] += dataset_stats["test"]
        split_counts["questions"]["total"] += dataset_stats["total"]

    train_path = args.output_dir / f"{stem}.train.json"
    val_path = args.output_dir / f"{stem}.val.json"
    test_path = args.output_dir / f"{stem}.test.json"
    meta_path = args.output_dir / f"{stem}.split_meta.json"

    with train_path.open("w", encoding="utf-8") as outfile:
        json.dump(train_split, outfile, ensure_ascii=False, indent=2)
    with val_path.open("w", encoding="utf-8") as outfile:
        json.dump(val_split, outfile, ensure_ascii=False, indent=2)
    with test_path.open("w", encoding="utf-8") as outfile:
        json.dump(test_split, outfile, ensure_ascii=False, indent=2)

    meta = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "stratify_by_dataset": not args.no_stratify_by_dataset,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "counts": split_counts,
        "outputs": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }
    with meta_path.open("w", encoding="utf-8") as outfile:
        json.dump(meta, outfile, ensure_ascii=False, indent=2)

    print("MIRAGE split completed.")
    print(
        "Questions: "
        f"train={split_counts['questions']['train']}, "
        f"val={split_counts['questions']['val']}, "
        f"test={split_counts['questions']['test']}, "
        f"total={split_counts['questions']['total']}"
    )
    print(f"Wrote: {train_path}")
    print(f"Wrote: {val_path}")
    print(f"Wrote: {test_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
