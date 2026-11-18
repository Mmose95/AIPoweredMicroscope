# Native D: feasibility first, then 50 SSL epochs

Use the existing `init_native704_ucloud.sh` as the UCloud job initializer.
Mount `40xSSLTrainingPatches` and persistent output storage. All original tiles
from TRAIN specimens are eligible, not only annotated tiles. Validation/test
specimen IDs are checked by the existing manifest builder.

From `/work/projects/myproj`, after pulling the changes:

```bash
export DINOV2_SSL_GPUS=2
export DINOV2_SSL_GLOBAL_BATCH=8
bash Study3_SelfSupervised/run_native_d_ssl_ucloud.sh prepare
bash Study3_SelfSupervised/run_native_d_ssl_ucloud.sh benchmark
```

Preparation audits the complete mounted training pool once and caches portable
weights. It reads images but does not copy/modify them. Eight preparation threads
are CPU workers, not GPU jobs. Near-black scan fill gets zero probability;
all other images retain positive probability. Specimens are uniform, and images
within each specimen use 20% uniform plus 80% proxy-weighted draws. The proxy
scores are 1/.5/.25/.05 for varied/sparse/low-contrast/very-low-signal images.
Annotations do not determine eligibility or spatial anchors.

The benchmark runs 10 warmup plus40 measured optimizer updates, including
teacher/student forward, backward, EMA, iBOT and data loading. It prints actual
tensor dimensions and writes `benchmark-g2-b8/benchmark.json`, including peak
allocated/reserved memory, images/sec and estimated epoch/run hours. This is an
estimate: I/O and saving can add time. No feasibility result is claimed until it
runs on UCloud. For an OOM, try global batch4 with the same two GPUs, rerun the
benchmark, and retain that batch for training. Every GPU needs at least2 images.
For GPU-count comparisons hold global batch fixed and benchmark each count.

After reviewing the benchmark:

```bash
bash Study3_SelfSupervised/run_native_d_ssl_ucloud.sh train
```

Training is DINOv2 Small/16 **from random weights**, not public adaptation.
It uses50 epochs,5 warmup epochs, AdamW peak LR1e-4, the existing DINO+iBOT+KoLeo
loss, and EMA teacher. It saves teachers at10,20,30,40,50 and optimizer state
each epoch. An epoch is floor(pool size/global batch)*global batch weighted
draws WITH replacement; it does not guarantee a visit to each file.
The effective batch8 differs from the older batch32 experiments and must be
reported; this is a candidate-recipe trial, not an isolated resolution ablation.
Set NATIVE_D_EPOCHS=100 **before benchmark and training** to plan100 instead;
do not change the schedule midway and call it an identical continuation.

The two globals are native768px, eight locals native384px, from1024px regions.
Component-balanced dark-structure guidance with jitter chooses the region;
empty structure masks fall back to uniform locations. Local crops are inside
both overlapping global windows. Guidance can still select debris. Crop pixels
are not resized. Flips/90-degree rotations, brightness/contrast .9–1.1 and
saturation .95–1.05 are applied independently; no blur/hue/solarization.
ImageNet normalization follows. Dynamic global masks contain2304 tokens.

Monitor:

```bash
source ./dinov3_ucloud_env.sh
tail -f "$OUTPUT_ROOT/Native704_20261110/ssl_native_d_768_384_v1/training/metrics.jsonl"
```

Epoch/run time estimates update every10 steps. Use `resume` instead of `train`
after interruption, with the same paths, batch, GPU count, epochs and options.
Resume is epoch-boundary recovery, not bitwise replay of interrupted draws.
Benchmark outputs use fresh directories; change NATIVE_D_ROOT for a fresh trial.

## Short checkpoint screening, after SSL

Ensure `full_native2528` train/valid prepared detector data is mounted in the
existing output tree. Then:

```bash
bash Study3_SelfSupervised/run_native_d_ssl_ucloud.sh probe
```

This screens teachers10,30,50 plus a RANDOM-backbone control on available GPUs,
queuing if fewer than4 are visible. Each is a10-epoch RF-DETR Small detector
screen with frozen encoder, the same150 training-patch subset and all validation
patches, trainable detector parameters and stock detector initialization.
This is NOT a strict linear probe and low absolute AP is possible. Frozen
features and eventual fine-tuned performance need not rank checkpoints equally.
Use results as an initial diagnostic; confirm promising checkpoints with a
matched trainable RF-DETR Large screen before committing to final selection.
Never select using test data. Logs and summary.json are under
`ssl_native_d_768_384_v1/frozen_detector_screen`.

Neither the launcher nor this setup has started a GPU training job locally.
