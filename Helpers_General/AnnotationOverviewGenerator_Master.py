# -*- coding: utf-8 -*-
"""
Create a single Excel workbook with 4 sheets:
  - Master          (merged, review-preserving; EXACT columns)
  - All_Frames      (latest backup)
  - Class_Totals    (latest backup)
  - Summary         (latest backup)

Rules for Master (exact columns):
  Frame, Squamous Epithelial Cell, Leucocyte, Cylindrical Epithelial Cell,
  Total Objects, Reviewed, Annotator, Comments

- Scans BASE_DIR recursively for:
    * 'Annotated_Frames_Only.csv'  (source of counts + possible review fields)
    * 'All_Frames.csv', 'Class_Totals.csv', 'Summary.csv'  (latest-only for info sheets)
    * existing 'Master_Review*.xlsx/.csv' (review fields preferred over backups)
- Auto-detects CSV delimiter (comma/semicolon)
- Numeric counts come ONLY from backup CSVs (newest per frame)
- Review fields (Reviewed/Annotator/Comments) come from newest non-empty across
  prior Masters (priority) then backups.

Output:
  BASE_DIR / Master_Review_YYYY-MM-DD_HHMM.xlsx    (with sheets above)

Usage:
  python build_master_workbook.py
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd

# =============== CONFIG =================
BASE_DIR = Path(r"D:\PHD\PhdData\CellScanData\Annotation_Backups\Quality Assessment Backups")

CSV_BACKUP_OVERVIEW = "Annotated_Frames_Only.csv"
CSV_ALL_FRAMES = "All_Frames.csv"
CSV_CLASS_TOTALS = "Class_Totals.csv"
CSV_SUMMARY = "Summary.csv"

CLASS_COLS = ["Squamous Epithelial Cell", "Leucocyte", "Cylindrical Epithelial Cell"]
REVIEW_COLS = ["Reviewed", "Annotator", "Comments"]
MASTER_COLS = ["Frame"] + CLASS_COLS + ["Total Objects"] + REVIEW_COLS

# Optional: set to frame name to print trace
DEBUG_FRAME = ""  # e.g. "Sample14 - BF.2_3_patch_x1280_y0.tif"


# ========================================


def ts_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def norm_str(s: pd.Series) -> pd.Series:
    """Normalize strings: strip quotes/space and map literal 'nan' -> ''."""
    return (
        s.astype(str)
        .str.replace(r'^\s*"(.*)"\s*$', r"\1", regex=True)
        .str.replace(r"^\s*'(.*)'\s*$", r"\1", regex=True)
        .str.strip()
        .mask(lambda x: x.str.lower().eq("nan"), "")
    )


def is_empty(s: pd.Series) -> pd.Series:
    return norm_str(s).eq("")


def find_files(root: Path, pattern: str):
    files = [p for p in root.rglob(pattern) if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)  # oldest -> newest
    return files


def read_csv_generic(path: Path) -> pd.DataFrame:
    """Robust CSV reader (handles comma/semicolon)."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")


def read_overview_csv(path: Path) -> pd.DataFrame:
    df = read_csv_generic(path)

    # Ensure exact schema (create missing)
    for c in MASTER_COLS:
        if c not in df.columns:
            df[c] = "" if c in REVIEW_COLS else 0
    df = df[MASTER_COLS].copy()

    df["Frame"] = norm_str(df["Frame"])
    for c in CLASS_COLS + ["Total Objects"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in REVIEW_COLS:
        df[c] = norm_str(df[c])

    df["__SourceCSV"] = str(path)
    df["__SourceMTime"] = datetime.fromtimestamp(path.stat().st_mtime)
    df["__SourceType"] = "backup_csv"
    return df


def read_master_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        try:
            df = pd.read_excel(path, sheet_name=0)
        except Exception:
            df = pd.read_excel(path)
    else:
        df = read_csv_generic(path)

    for c in MASTER_COLS:
        if c not in df.columns:
            df[c] = "" if c in REVIEW_COLS else 0
    df = df[MASTER_COLS].copy()

    df["Frame"] = norm_str(df["Frame"])
    for c in REVIEW_COLS:
        df[c] = norm_str(df[c])
    for c in CLASS_COLS + ["Total Objects"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["__SourceCSV"] = str(path)
    df["__SourceMTime"] = datetime.fromtimestamp(path.stat().st_mtime)
    df["__SourceType"] = "prior_master"
    return df


def pick_latest_counts(all_df: pd.DataFrame) -> pd.DataFrame:
    csv_only = all_df[all_df["__SourceType"] == "backup_csv"].copy()
    csv_only = csv_only.sort_values("__SourceMTime")  # oldest -> newest
    latest = csv_only.drop_duplicates(subset=["Frame"], keep="last").reset_index(drop=True)
    return latest


def pick_latest_nonempty(all_df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = all_df[~is_empty(all_df[col])].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Frame", col])

    # Prior masters win over backups when both have values; then newest mtime
    prio = {"backup_csv": 1, "prior_master": 2}
    rows["__prio"] = rows["__SourceType"].map(prio).fillna(0).astype(int)
    rows = rows.sort_values(["__prio", "__SourceMTime"])  # lowest→highest, oldest→newest
    best = rows.drop_duplicates(subset=["Frame"], keep="last")[["Frame", col]].reset_index(drop=True)
    return best


def read_latest_info_table(base: Path, basename: str) -> tuple[pd.DataFrame, Path | None]:
    files = find_files(base, basename)
    if not files:
        return pd.DataFrame(), None
    latest_path = files[-1]
    df = read_csv_generic(latest_path)
    # keep raw columns; just normalize potential strings
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = norm_str(df[c])
    return df, latest_path


def main():
    # ---------- discover inputs ----------
    backup_overviews = find_files(BASE_DIR, CSV_BACKUP_OVERVIEW)
    prior_masters = find_files(BASE_DIR, "Master_Review*.xlsx") + find_files(BASE_DIR, "Master_Review*.csv")

    print("== Backup overview CSVs (oldest → newest) ==")
    if backup_overviews:
        for p in backup_overviews:
            print(f"- {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M:%S}  {p}")
    else:
        print("No Annotated_Frames_Only.csv found.");
        return

    if prior_masters:
        print("\n== Prior Master files (oldest → newest) ==")
        for p in prior_masters:
            print(f"- {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M:%S}  {p}")
    else:
        print("\n== No prior Master files found ==")

    # ---------- read everything for MASTER ----------
    parts = [read_overview_csv(p) for p in backup_overviews]
    for p in prior_masters:
        try:
            parts.append(read_master_file(p))
        except Exception as e:
            print(f"Skip prior master {p}: {e}")
    all_df = pd.concat(parts, ignore_index=True)

    # optional debug
    if DEBUG_FRAME:
        dbg = all_df[all_df["Frame"] == DEBUG_FRAME][
            ["Frame", "Reviewed", "Annotator", "Comments", "__SourceType", "__SourceCSV", "__SourceMTime"]
        ].sort_values("__SourceMTime")
        print(f"\n== DEBUG: occurrences of '{DEBUG_FRAME}' ==")
        if dbg.empty:
            print("  (none)")
        else:
            for _, r in dbg.iterrows():
                print(f"{r['__SourceMTime']} | {r['__SourceType']:<12} | {r['__SourceCSV']}"
                      f" | Reviewed='{r['Reviewed']}' Annotator='{r['Annotator']}' Comments='{r['Comments']}'")

    # newest counts from backups
    latest = pick_latest_counts(all_df)

    # newest non-empty review fields (masters win)
    for col in REVIEW_COLS:
        best = pick_latest_nonempty(all_df, col)
        if not best.empty:
            latest = latest.merge(best, on="Frame", how="left", suffixes=("", f"__{col}_best"))
            src = f"{col}__{col}_best"
            latest[col] = latest[col].where(~is_empty(latest[col]), norm_str(latest[src]))
            if src in latest.columns:
                latest.drop(columns=[src], inplace=True)
        else:
            if col not in latest.columns:
                latest[col] = ""

    master_df = latest[MASTER_COLS].copy()

    # ---------- read latest info sheets ----------
    all_frames_df, af_path = read_latest_info_table(BASE_DIR, CSV_ALL_FRAMES)
    class_totals_df, ct_path = read_latest_info_table(BASE_DIR, CSV_CLASS_TOTALS)
    summary_df, summ_path = read_latest_info_table(BASE_DIR, CSV_SUMMARY)

    print("\n== Info sheets selected ==")
    print(f"- All_Frames:   {af_path if af_path else '(not found)'}")
    print(f"- Class_Totals: {ct_path if ct_path else '(not found)'}")
    print(f"- Summary:      {summ_path if summ_path else '(not found)'}")

    # ---------- write workbook with 4 sheets ----------
    out_xlsx = BASE_DIR / f"Master_Review_{ts_name()}.xlsx"
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xl:
        # Master
        master_df.to_excel(xl, sheet_name="Master", index=False)
        w = xl.sheets["Master"];
        w.autofilter(0, 0, len(master_df), len(MASTER_COLS) - 1);
        w.freeze_panes(1, 1)

        # Info sheets (only if present)
        if not all_frames_df.empty:
            all_frames_df.to_excel(xl, sheet_name="All_Frames", index=False)
            xl.sheets["All_Frames"].autofilter(0, 0, len(all_frames_df), max(0, all_frames_df.shape[1] - 1))
        if not class_totals_df.empty:
            class_totals_df.to_excel(xl, sheet_name="Class_Totals", index=False)
            xl.sheets["Class_Totals"].autofilter(0, 0, len(class_totals_df), max(0, class_totals_df.shape[1] - 1))
        if not summary_df.empty:
            summary_df.to_excel(xl, sheet_name="Summary", index=False)
            xl.sheets["Summary"].autofilter(0, 0, len(summary_df), max(0, summary_df.shape[1] - 1))

    print(f"\n[OK] Workbook written: {out_xlsx}")
    print("Sheets: Master, All_Frames, Class_Totals, Summary")
    print("Done.")


if __name__ == "__main__":
    main()
