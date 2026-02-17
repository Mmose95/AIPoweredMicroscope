# Generate multiple CSV files (UTF-8 with BOM) instead of an .xlsx, so Excel preserves names exactly.
# Files:
#  - Annotated_Frames_Only.csv
#  - Class_Totals.csv
#  - Summary.csv
#  - All_Frames.csv
import json
import csv
from collections import defaultdict, Counter
from pathlib import Path

src = Path("D:\PHD\PhdData\CellScanData\Annotation_Backups/Quality Assessment Backups/11-02-2026/annotations/instances_default.json")
with open(src, "r", encoding="utf-8") as f:
    coco = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
img_id_to_name = {im["id"]: im["file_name"] for im in coco.get("images", [])}
class_names = list(cat_id_to_name.values())

per_image_counts = defaultdict(Counter)
for a in coco.get("annotations", []):
    per_image_counts[a["image_id"]][cat_id_to_name[a["category_id"]]] += 1

# Build full per-frame table
rows_all = []
for img_id, fname in img_id_to_name.items():
    counts = {cls: per_image_counts.get(img_id, Counter()).get(cls, 0) for cls in class_names}
    total_objs = sum(counts.values())
    row = {"Frame": fname, **counts, "Total Objects": total_objs, "Reviewed": "", "Annotator": "MM", "Comments": ""}
    rows_all.append(row)

# Sort like before
rows_all = sorted(rows_all, key=lambda r: (-r["Total Objects"], r["Frame"]))

# Filtered (annotated only)
rows_annot = [r for r in rows_all if r["Total Objects"] > 0]

# Totals and summary
totals = {cls: 0 for cls in class_names}
for r in rows_all:
    for cls in class_names:
        totals[cls] += r.get(cls, 0)

summary_rows = [
    {"Metric": "# Images (all)", "Value": len(rows_all)},
    {"Metric": "# Images (with annotations)", "Value": len(rows_annot)},
    {"Metric": "Total Annotations", "Value": sum(totals.values())},
]

# Helper to write CSV with BOM
def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(col, "") for col in header])

# Write CSVs
base = src.parent
paths = {}

# Annotated Frames Only
header_overview = ["Frame"] + class_names + ["Total Objects", "Reviewed", "Annotator", "Comments"]
p1 = base / "Annotated_Frames_Only.csv"
write_csv(p1, header_overview, rows_annot)
paths["annotated"] = p1.as_posix()

# Class Totals
p2 = base / "Class_Totals.csv"
write_csv(p2, ["Class", "Total Count"], [{"Class": cls, "Total Count": totals[cls]} for cls in class_names])
paths["totals"] = p2.as_posix()

# Summary
p3 = base / "Summary.csv"
write_csv(p3, ["Metric", "Value"], summary_rows)
paths["summary"] = p3.as_posix()

# All Frames
p4 = base / "All_Frames.csv"
write_csv(p4, header_overview, rows_all)
paths["all"] = p4.as_posix()

paths
