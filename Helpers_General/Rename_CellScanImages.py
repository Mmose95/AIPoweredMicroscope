import os, re, csv, time, traceback
from pathlib import Path

# ==== CONFIG ==========================================================
ROOT = Path(r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned")
SAMPLE_MIN, SAMPLE_MAX = 67, 67   # adjust if you want to limit
SUBDIR_PREFIX = "2025"
EXTS = {".tif", ".tiff"}
DRY_RUN = False                    # True = preview only
# ======================================================================

sample_dir_re = re.compile(r"^Sample\s*(\d+)$", re.IGNORECASE)

def suffix_from(stem: str) -> str:
    m = re.search(r"\s-\s(.+)$", stem)
    return m.group(1) if m else stem

def ensure_unique(dirpath: Path, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    i, candidate = 1, filename
    while (dirpath / candidate).exists():
        candidate = f"{base} ({i}){ext}"
        i += 1
    return candidate

def rename_in_dir(dirpath: Path, sample_num: int, changes: list):
    for f in dirpath.iterdir():
        if f.is_file() and f.suffix.lower() in EXTS:
            keep = suffix_from(f.stem)
            target_name = f"Sample{sample_num} - {keep}{f.suffix}"
            # build the *desired* name (without forcing uniqueness),
            # actual collision is handled in the apply phase
            if f.name != target_name:
                changes.append((str(f), str(dirpath / target_name)))

def get_target_dir(sample_dir: Path) -> Path:
    sub = next((d for d in sample_dir.iterdir() if d.is_dir() and d.name.startswith(SUBDIR_PREFIX)), None)
    return sub if sub else sample_dir

# ---------- Phase 1: gather planned renames ----------
changes, errors = [], []
for sample_dir in sorted((p for p in ROOT.iterdir() if p.is_dir()), key=lambda x: x.name.lower()):
    m = sample_dir_re.match(sample_dir.name)
    if not m:
        continue
    n = int(m.group(1))
    if not (SAMPLE_MIN <= n <= SAMPLE_MAX):
        continue
    try:
        target_dir = get_target_dir(sample_dir)
        rename_in_dir(target_dir, n, changes)
    except Exception as e:
        errors.append([str(sample_dir), repr(e), traceback.format_exc()])

print(f"[RENAME] Planned: {len(changes)}")
for old, new in changes[:20]:
    print(f"  {old} -> {new}")
if len(changes) > 20:
    print(f"  ...and {len(changes)-20} more")

ts = time.strftime('%Y%m%d_%H%M%S')
log_path = ROOT / f"rename_log_{ts}.csv"
with open(log_path, "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["old_path","new_path"]); w.writerows(changes)
print(f"Rename log: {log_path}")

if errors:
    err_path = ROOT / f"errors_{ts}.csv"
    with open(err_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp); w.writerow(["where","error","traceback"]); w.writerows(errors)
    print(f"Errors: {err_path}")

# ---------- Phase 1: apply safely ----------
if not DRY_RUN:
    applied = skipped = 0
    for old, new in changes:
        old_p, new_p = Path(old), Path(new)
        # re-check at execution time
        if new_p.exists():
            print(f"[SKIP] target exists: {new_p}")
            skipped += 1
            continue
        try:
            old_p.rename(new_p)
            applied += 1
        except Exception as e:
            print(f"[ERROR] rename {old_p} -> {new_p}: {e}")
    print(f"[RENAME] Applied: {applied}, Skipped: {skipped}")
else:
    print("DRY_RUN=True — no files were renamed.")

# ---------- Phase 2: cleanup names like '(1)', '(2)', '(1) (1)', etc. ----------
# Match any stem with trailing one or more ' (number)' chunks
dup_pattern = re.compile(r"\s+\(\d+\)(?:\s+\(\d+\))*$", re.IGNORECASE)

def cleanup_candidates_in_dir(dirpath: Path):
    items = []
    for f in dirpath.iterdir():
        if not f.is_file() or f.suffix.lower() not in EXTS:
            continue
        if dup_pattern.search(f.stem):
            clean_stem = dup_pattern.sub("", f.stem)
            target = f.with_name(clean_stem + f.suffix)
            # Add regardless of current existence; we'll check at apply time
            if f.name != target.name:
                items.append((str(f), str(target)))
    return items

cleanup_changes = []
for sample_dir in (p for p in ROOT.iterdir() if p.is_dir()):
    m = sample_dir_re.match(sample_dir.name)
    if not m:
        continue
    n = int(m.group(1))
    if not (SAMPLE_MIN <= n <= SAMPLE_MAX):
        continue
    dir_to_check = get_target_dir(sample_dir)
    cleanup_changes.extend(cleanup_candidates_in_dir(dir_to_check))

print(f"[CLEANUP] Planned: {len(cleanup_changes)}")
for old, new in cleanup_changes[:20]:
    print(f"  {old} -> {new}")
if len(cleanup_changes) > 20:
    print(f"  ...and {len(cleanup_changes)-20} more")

cleanup_log = ROOT / f"cleanup_namefix_log_{ts}.csv"
with open(cleanup_log, "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["old_path","new_path"]); w.writerows(cleanup_changes)
print(f"Cleanup log: {cleanup_log}")

# ---------- Phase 2: apply safely ----------
if not DRY_RUN:
    c_renamed = c_skipped = 0
    for old, new in cleanup_changes:
        old_p, new_p = Path(old), Path(new)
        # re-check right now
        if new_p.exists():
            print(f"[SKIP] cleanup target exists: {new_p}")
            c_skipped += 1
            continue
        try:
            old_p.rename(new_p)
            c_renamed += 1
        except Exception as e:
            print(f"[ERROR] cleanup rename {old_p} -> {new_p}: {e}")
    print(f"[CLEANUP] Renamed: {c_renamed}, Skipped: {c_skipped}")
else:
    print("DRY_RUN=True — no cleanup renames executed.")
