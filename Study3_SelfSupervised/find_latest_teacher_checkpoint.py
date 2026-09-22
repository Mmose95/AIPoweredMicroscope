"""Print the latest numbered official DINOv3 teacher export in a run directory."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates: list[tuple[int, Path]] = []
    for path in (args.run_dir / "eval").glob("training_*/teacher_checkpoint.pth"):
        try:
            iteration = int(path.parent.name.removeprefix("training_"))
        except ValueError:
            continue
        candidates.append((iteration, path.resolve()))
    if not candidates:
        raise FileNotFoundError(f"No eval/training_*/teacher_checkpoint.pth under {args.run_dir}")
    print(max(candidates)[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
