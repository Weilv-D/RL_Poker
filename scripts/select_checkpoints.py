#!/usr/bin/env python3
"""Select and prune checkpoints based on eval_gpu JSON output.

Usage:
    python scripts/select_checkpoints.py --eval-json checkpoints/star/eval_results.json --keep 5 --metric mean_score --delete
    python scripts/select_checkpoints.py --run-name star --keep 5 --metric mean_score
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Select/prune checkpoints based on eval results")
    parser.add_argument("--eval-json", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Run name (e.g., star, garlic)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory containing eval_results.json",
    )
    parser.add_argument("--metric", type=str, default="mean_score", choices=["mean_score", "win_rate", "avg_rank"])
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()
    eval_json = args.eval_json
    if eval_json is None:
        if args.run_name:
            eval_json = str(Path("checkpoints") / args.run_name / "eval_results.json")
        elif args.checkpoint_dir:
            eval_json = str(Path(args.checkpoint_dir) / "eval_results.json")
        else:
            raise SystemExit("Provide --eval-json or --run-name or --checkpoint-dir")

    eval_path = Path(eval_json)
    if not eval_path.exists():
        raise SystemExit(f"Eval JSON not found: {eval_path}")

    data = json.loads(eval_path.read_text())

    reverse = args.metric != "avg_rank"  # avg_rank lower is better
    data_sorted = sorted(data, key=lambda x: x.get(args.metric, 0.0), reverse=reverse)

    keep = data_sorted[: args.keep]
    drop = data_sorted[args.keep :]

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(keep, indent=2))

    print(f"Keeping {len(keep)} checkpoints:")
    for item in keep:
        print(f"  {item['checkpoint']} ({args.metric}={item.get(args.metric)})")

    if not args.delete:
        print("\nDry run or delete not requested. No files removed.")
        return

    if args.dry_run:
        print("\nDry run enabled; no files removed.")
        return

    for item in drop:
        ckpt = Path(item["checkpoint"])
        if ckpt.exists():
            ckpt.unlink()
            print(f"Deleted {ckpt}")


if __name__ == "__main__":
    main()
