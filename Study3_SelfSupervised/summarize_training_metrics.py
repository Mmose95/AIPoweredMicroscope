"""Summarize an official DINOv3 JSON-lines metrics file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = []
    with args.metrics.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}") from error
    if not records:
        raise ValueError(f"No metric records found: {args.metrics}")

    numeric = {
        key: [float(record[key]) for record in records if key in record]
        for key in (
            "total_loss",
            "dino_local_crops_loss",
            "dino_global_crops_loss",
            "koleo_loss",
            "ibot_loss",
            "backbone_grad_norm",
        )
    }
    non_finite = {
        key: sum(not math.isfinite(value) for value in values)
        for key, values in numeric.items()
    }
    summary = {
        "status": "passed" if not any(non_finite.values()) else "failed",
        "metrics_file": str(args.metrics.resolve()),
        "logged_records": len(records),
        "first_iteration": records[0].get("iteration"),
        "last_iteration": records[-1].get("iteration"),
        "non_finite_values": non_finite,
        "metrics": {
            key: {
                "first": values[0],
                "last": values[-1],
                "minimum": min(values),
                "maximum": max(values),
            }
            for key, values in numeric.items()
            if values
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

