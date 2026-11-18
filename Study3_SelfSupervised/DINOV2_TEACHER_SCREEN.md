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
