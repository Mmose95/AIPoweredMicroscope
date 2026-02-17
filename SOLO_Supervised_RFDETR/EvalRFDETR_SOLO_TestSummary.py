#!/usr/bin/env python
"""
EvalRFDETR_SOLO_TestSummary.py

Collect RFDETR validation + test metrics for a set of trained runs.

Assumptions:
- You have a folder EVAL_RUNS_ROOT with one subfolder per run.
- Each run subfolder contains:
    - hpo_record.json  (written by your HPO launcher)
    - results.json     (written by RFDETR with val + test metrics)
- results.json has the structure:
    {
        "class_map": {
            "valid": [...],
            "test":  [...]
        }
    }
  where each split is a list of rows with keys like "class", "map@50", "map@50:95".

The script:
- Scans all run subfolders under EVAL_RUNS_ROOT,
- Extracts val/test AP@50 and AP@50:95,
- Detects whether the backbone is standard or SSL (via ENCODER_CKPT),
- Writes:
    - eval_summary.csv
    - eval_summary.json
  into EVAL_RUNS_ROOT,
- Prints a Markdown table to stdout (easy to paste into notes / draft).

Env vars:
- USER_BASE_DIR     : auto-detected if not set (AAU/SDU /work layout)
- EVAL_RUNS_ROOT   : folder containing run dirs (default: /work/USER/RFDETR_FINAL_EVAL)
- EVAL_OUT_CSV     : optional override for CSV path
- EVAL_OUT_JSON    : optional override for JSON path
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────
# UCloud-friendly path detection
# ─────────────────────────────────────────────

def _detect_user_base() -> str | None:
    import glob, os.path as op
    aau = glob.glob("/work/Member Files:*")
    if aau:
        return op.basename(aau[0])
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None

USER_BASE_DIR = os.environ.get("USER_BASE_DIR") or _detect_user_base() or ""
if USER_BASE_DIR:
    os.environ["USER_BASE_DIR"] = USER_BASE_DIR
WORK_ROOT = Path("/work") / USER_BASE_DIR if USER_BASE_DIR else Path.cwd()


def env_path(name: str, default: Path) -> Path:
    v = os.getenv(name, "").strip()
    return Path(v) if v else default


# Default: /work/USER/RFDETR_FINAL_EVAL
EVAL_RUNS_ROOT = env_path(
    "EVAL_RUNS_ROOT",
    WORK_ROOT / "RFDETR_FINAL_EVAL",
)

# Output files (default in EVAL_RUNS_ROOT, can be overridden via env)
EVAL_OUT_CSV = env_path(
    "EVAL_OUT_CSV",
    EVAL_RUNS_ROOT / "eval_summary.csv",
)
EVAL_OUT_JSON = env_path(
    "EVAL_OUT_JSON",
    EVAL_RUNS_ROOT / "eval_summary.json",
)

# Optional: only keep certain targets (e.g. "Squamous Epithelial Cell")
EVAL_TARGET_FILTER = os.getenv("EVAL_TARGET_FILTER", "").strip()  # empty = no filter


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _pick_split_metrics(results_json: dict, split: str = "test"):
    """
    Extract AP@50 and AP@50:95 for a given split from RFDETR results.json.

    Strategy:
    - Look for results_json["class_map"][split]
    - Prefer the row where class == "all"
    - If not found, fall back to the first row (if any)
    """
    cm = results_json.get("class_map", {})
    rows = cm.get(split, []) or cm.get(split.capitalize(), [])

    if not rows:
        return None, None

    # Prefer 'class' == 'all'
    row = None
    for r in rows:
        if r.get("class") == "all":
            row = r
            break
    if row is None:
        row = rows[0]

    ap50 = row.get("map@50",  None)
    ap5095 = row.get("map@50:95", None)

    ap50 = float(ap50) if ap50 is not None else None
    ap5095 = float(ap5095) if ap5095 is not None else None
    return ap50, ap5095


def collect_metrics(eval_root: Path):
    """
    Scan all direct subdirectories of eval_root, and for each one that has
    hpo_record.json + results.json, collect:

    - run_dir name
    - target class
    - TRAIN_FRACTION
    - whether an ENCODER_CKPT was used (SSL vs baseline)
    - val_AP50, val_mAP50:95 (from results.json or hpo_record.json)
    - test_AP50, test_mAP50:95 (from results.json)
    """
    eval_root = eval_root.resolve()
    print(f"[EVAL] Scanning runs under: {eval_root}")

    if not eval_root.exists():
        raise FileNotFoundError(
            f"EVAL_RUNS_ROOT does not exist: {eval_root}\n"
            f"Set EVAL_RUNS_ROOT to your folder of run directories."
        )

    rows = []

    for sub in sorted(eval_root.iterdir()):
        if not sub.is_dir():
            continue

        hpo_path = sub / "hpo_record.json"
        res_path = sub / "results.json"

        if not hpo_path.exists():
            print(f"[SKIP] {sub.name}: no hpo_record.json")
            continue
        if not res_path.exists():
            print(f"[SKIP] {sub.name}: no results.json")
            continue

        print(f"[EVAL] Reading run dir: {sub.name}")

        # --- parse HPO record ---
        hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
        target = hpo.get("target", "unknown")
        if EVAL_TARGET_FILTER and target != EVAL_TARGET_FILTER:
            print(f"[SKIP] {sub.name}: target={target!r} != filter={EVAL_TARGET_FILTER!r}")
            continue

        train_frac = float(hpo.get("TRAIN_FRACTION", 1.0))
        val_ap50 = hpo.get("val_AP50", None)
        val_ap5095 = hpo.get("val_mAP5095", None)

        # backbone type: SSL if ENCODER_CKPT present and non-empty
        encoder_ckpt = (hpo.get("ENCODER_CKPT")
                        or hpo.get("encoder_ckpt")
                        or hpo.get("pretrained_backbone")
                        or hpo.get("encoder_name"))
        if encoder_ckpt:
            backbone_type = "SSL_backbone"
        else:
            backbone_type = "standard_backbone"

        # --- parse results.json for test & valid metrics ---
        results = json.loads(res_path.read_text(encoding="utf-8"))
        test_ap50, test_ap5095   = _pick_split_metrics(results, split="test")
        valid_ap50, valid_ap5095 = _pick_split_metrics(results, split="valid")

        # fall back to HPO record if val metrics missing in results.json
        if valid_ap50 is None:
            valid_ap50 = val_ap50
        if valid_ap5095 is None:
            valid_ap5095 = val_ap5095

        row = {
            "run_dir": sub.name,
            "target": target,
            "backbone": backbone_type,
            "train_fraction": train_frac,

            "val_AP50": valid_ap50,
            "val_mAP50_95": valid_ap5095,

            "test_AP50": test_ap50,
            "test_mAP50_95": test_ap5095,

            "encoder_ckpt": encoder_ckpt,
            "output_dir": str(sub),
        }
        rows.append(row)

    # sort rows by backbone, then by train_fraction (ascending)
    rows.sort(key=lambda r: (r["backbone"], r["train_fraction"]))
    return rows


def save_csv(rows, out_csv: Path):
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("[WARN] No rows to save; CSV will be empty.")
        return

    fieldnames = [
        "run_dir",
        "target",
        "backbone",
        "train_fraction",
        "val_AP50",
        "val_mAP50_95",
        "test_AP50",
        "test_mAP50_95",
        "encoder_ckpt",
        "output_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[SAVE] Wrote CSV summary → {out_csv}")


def save_json(rows, out_json: Path):
    out_json = out_json.resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_root": str(EVAL_RUNS_ROOT),
        "n_runs": len(rows),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SAVE] Wrote JSON summary → {out_json}")


def print_markdown_table(rows):
    if not rows:
        print("\n[INFO] No runs collected, nothing to print.")
        return

    headers = [
        "run_dir",
        "backbone",
        "train_frac",
        "val_AP50",
        "val_mAP50_95",
        "test_AP50",
        "test_mAP50_95",
    ]

    print("\n### RFDETR test-set summary\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in rows:
        row_vals = [
            r["run_dir"],
            r["backbone"],
            f"{r['train_fraction']:.3f}",
            f"{r['val_AP50']:.3f}"      if r["val_AP50"] is not None else "NA",
            f"{r['val_mAP50_95']:.3f}"  if r["val_mAP50_95"] is not None else "NA",
            f"{r['test_AP50']:.3f}"     if r["test_AP50"] is not None else "NA",
            f"{r['test_mAP50_95']:.3f}" if r["test_mAP50_95"] is not None else "NA",
        ]
        print("| " + " | ".join(row_vals) + " |")


def main():
    print("[WORK_ROOT]", WORK_ROOT)
    print("[EVAL_RUNS_ROOT]", EVAL_RUNS_ROOT)

    rows = collect_metrics(EVAL_RUNS_ROOT)

    save_csv(rows, EVAL_OUT_CSV)
    save_json(rows, EVAL_OUT_JSON)
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
