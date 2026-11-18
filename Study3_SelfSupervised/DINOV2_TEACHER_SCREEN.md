# Full detector screening

Run `bash Study3_SelfSupervised/run_dinov2_screen_ucloud.sh` after the native704
job initializer, with six GPUs and the existing data/output storage mounted.

This launches original20, long20, long50, long100, long150 and long200 in
parallel. Each uses RF-DETR Large's published detector weights, replaces only
the encoder with the selected DINOv2 teacher, and fine-tunes both together.
The standard two-class classifier is initialized for the supervised labels.

All 507 training patches and 233 validation patches from full_native2528 are
used, at resolution 2528. Test data is not evaluated. Settings match the
previous full-patch own-SSL detector: seed 0, batch 1, accumulation 16,
learning rate 5e-5, encoder default learning rate 1.5e-4, weight decay 0.0007,
gradient checkpointing, no multiscale, scale jitter or random resize padding.
The initial screening budget is 10 detector epochs, not the original 50.
These are fine-tuning screens, not frozen linear probes.

Outputs: Native704_20261110/dinov2_teacher_screen_large_full_10ep, with one
log and subfolder per teacher and a final leaderboard.json. Existing frozen
probes remain untouched. Completed runs are skipped; interrupted folders
are rejected to prevent accidental overwrites. RF-DETR saves last.ckpt with
training state each epoch plus interval checkpoints every five epochs, so
selected runs can subsequently be extended with an explicit resume setup.

Compare original20 with its previous epoch-10 result (~0.447 EMA mAP) as a
sanity check, not an exact numerical acceptance threshold. Select candidates
using validation curves and both classes; extend promising candidates to
30–40 epochs before making a final checkpoint choice. Do not select using
test performance.

## Extend to 40 epochs with a random control

With seven visible GPUs run:

```bash
cd /work/projects/myproj
git pull --ff-only && bash Study3_SelfSupervised/run_dinov2_screen40_ucloud.sh
```

The six teachers resume from the original epoch-10 last.ckpt, restoring model,
optimizer, scheduler and EMA training state. They run 30 additional epochs,
for 40 total. The random DINOv2 encoder control uses the same stock pretrained
detector components and two-class head adaptation, and trains for 40 epochs
from scratch. All encoders remain trainable. This isolates the practical
benefit of own SSL initialization within this pretrained detector setup; it
does not remove supervision learned by the stock detector.

Outputs go to dinov2_teacher_screen_large_full_40ep. The original ten-epoch
outputs are only read. Each extended teacher keeps metrics_first10.csv and
merges those rows into metrics.csv after training; while running, metrics.csv
contains only resumed epochs. All seven have the same supervised epoch
budget and step learning-rate settings (drop at epoch 100, no warmup).
Completed outputs are skipped. Incomplete output directories cause a clear
error instead of being overwritten. A final leaderboard compares all 40
validation epochs. Seed is fixed at 0; small score differences from this
single-seed experiment should not be treated as conclusive.
