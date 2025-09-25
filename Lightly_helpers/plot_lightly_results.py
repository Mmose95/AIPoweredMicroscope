# Read and summarize the uploaded JSONL metrics file, display a clean table,
# generate a few quick charts (if the columns exist), and save a neat CSV copy.

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ace_tools_open import display_dataframe_to_user

src = Path("C:/Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\outputLightly\metrics.jsonl")
assert src.exists(), f"File not found: {src}"

rows = []
with src.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            rows.append(obj)
        except Exception as e:
            # Skip non-JSON lines
            continue

df = pd.DataFrame(rows)

# Standardize common column names if present
# Prefer 'step' over 'global_step' or 'global_step_int', etc.
if 'global_step' in df.columns and 'step' not in df.columns:
    df.rename(columns={'global_step': 'step'}, inplace=True)
if 'global_step_int' in df.columns and 'step' not in df.columns:
    df.rename(columns={'global_step_int': 'step'}, inplace=True)
if 'epoch' not in df.columns:
    # try to infer epoch from other keys
    for k in df.columns:
        if k.lower().endswith('epoch'):
            df.rename(columns={k: 'epoch'}, inplace=True)
            break

# Try to coerce numerics where possible
for c in df.columns:
    if df[c].dtype == 'object':
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            # leave as is (likely strings / nested)
            pass

# Sort by step/epoch if available
sort_cols = [c for c in ['step', 'epoch'] if c in df.columns]
if sort_cols:
    df.sort_values(sort_cols, inplace=True)

# Select a concise set of columns to show (if present)
candidate_cols = [
    'step', 'epoch',
    'train_loss', 'dino_loss', 'ibot_loss', 'koleo_loss',
    'lr', 'learning_rate',
    'steps_per_sec', 'eta_min', 'it/s', 'data_wait',
    'time', 'train/acc', 'grad_norm'
]
present_cols = [c for c in candidate_cols if c in df.columns]

# If none of the expected columns are present, just show all
if not present_cols:
    present_cols = df.columns.tolist()

# Build a "clean" df for display
clean = df[present_cols].copy()

# Save to CSV for download
csv_path = Path("C:/Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\outputLightly\metrics_parsed.csv")
clean.to_csv(csv_path, index=False)

# Basic summary stats
summary = {}
if 'step' in df.columns:
    summary['min_step'] = int(df['step'].min())
    summary['max_step'] = int(df['step'].max())
    summary['n_steps_logged'] = df['step'].nunique()
if 'epoch' in df.columns and pd.api.types.is_numeric_dtype(df['epoch']):
    summary['min_epoch'] = int(df['epoch'].min())
    summary['max_epoch'] = int(df['epoch'].max())
    summary['n_epochs_logged'] = df['epoch'].nunique()
for loss_key in ['train_loss', 'dino_loss', 'ibot_loss', 'koleo_loss']:
    if loss_key in df.columns and pd.api.types.is_numeric_dtype(df[loss_key]):
        summary[f'{loss_key}_min'] = float(df[loss_key].min())
        summary[f'{loss_key}_max'] = float(df[loss_key].max())

# Create charts if relevant columns exist
fig_paths = []

def save_plot(x, y, xlabel, ylabel, title, fname):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    outp = Path("C:/Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\outputLightly/") / fname
    plt.tight_layout()
    plt.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close()
    fig_paths.append(str(outp))

# Train loss vs step
if 'step' in df.columns:
    for key in ['train_loss', 'dino_loss', 'ibot_loss']:
        if key in df.columns and pd.api.types.is_numeric_dtype(df[key]):
            save_plot(df['step'], df[key], "Step", key.replace("_", " ").title(), f"{key.replace('_',' ').title()} vs Step", f"{key}_vs_step.png")

# Steps per second (or it/s) vs step
rate_key = None
for k in ['steps_per_sec', 'it/s']:
    if k in df.columns and pd.api.types.is_numeric_dtype(df[k]):
        rate_key = k
        break
if rate_key and 'step' in df.columns:
    save_plot(df['step'], df[rate_key], "Step", rate_key, f"{rate_key} vs Step", f"{rate_key}_vs_step.png")

# Data wait vs step
if 'data_wait' in df.columns and 'step' in df.columns and pd.api.types.is_numeric_dtype(df['data_wait']):
    save_plot(df['step'], df['data_wait'], "Step", "data_wait", "Data wait vs Step", "data_wait_vs_step.png")

# Display the dataframe to the user as an interactive table
display_dataframe_to_user("Lightly metrics (parsed)", clean)

# Prepare a compact textual summary to print as cell output
print("=== Parsed metrics summary ===")
print(f"Rows parsed: {len(df)}")
if summary:
    for k, v in summary.items():
        print(f"{k}: {v}")
print("\nSaved a neat CSV to:", csv_path)
if fig_paths:
    print("Saved figures:")
    for p in fig_paths:
        print(" -", p)
