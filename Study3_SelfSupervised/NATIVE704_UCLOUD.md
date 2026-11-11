# Corrected 40× four-arm experiment on UCloud

Use `init_native704_ucloud.sh` as the initializer for each GPU job. Use a GPU
image with local Python 3.11 or Conda in `/opt/conda`. The initializer builds
the pinned environment on job-local storage and verifies the CUDA runtime.
In the terminal, source the generated `dinov3_ucloud_env.sh` from the checkout.

Use the same persistent `OUTPUT_ROOT` in all jobs. Set `DINOV3_OUTPUT_ROOT` in
the initializer if the shared member folder is mounted under a different layout.
`CORRECTED_PATCH_ROOT` points to
the flat directory of corrected annotated parent JPEGs. `SSL_IMAGE_ROOT` points
to the original full level-0 image pool, with one folder per specimen. It defaults
to `IMAGE_ROOT` if configured with `DINOV3_IMAGE_ROOT` in the initializer.
For the public arm and public
adaptation, set `STUDY3_PUBLIC_SSL_WEIGHTS` to the official DINOv3-S/16
LVD-1689M checkpoint.

From the checkout, run the preparation command once. This step uses Python and
Pillow and can run on a CPU job or at the start of the scratch job.

```bash
export CORRECTED_PATCH_ROOT='/path/to/corrected/images'
bash Study3_SelfSupervised/run_native704_ucloud.sh prepare
```

After preparation, run one command in each detector job. The third job can start
SSL while the tiles are being generated.

| Job | Command |
|---|---|
| Scratch detector | `bash Study3_SelfSupervised/run_native704_ucloud.sh scratch` |
| Public SSL detector | `bash Study3_SelfSupervised/run_native704_ucloud.sh public_ssl` |
| Two domain SSL checkpoints, sequentially | `bash Study3_SelfSupervised/run_native704_ucloud.sh ssl` |

The two SSL recipes can also run as separate jobs using `ssl-scratch` and
`ssl-adapt`. Build the shared SSL inventory in one job first if launching these
concurrently. The manifest builder refuses specimen overlap between supervised
splits and selects only the new training specimen folders. The default schedule
is 20 epochs from scratch and 10 epochs of adaptation from public weights.
The detector launchers for `own_data_ssl` and `public_domain_ssl` require the
corresponding completed checkpoint and its checksum record.

All four arms use RF-DETR Large at fixed 704 × 704 with the same DINOv3-S/16
backbone architecture, two classes, 50 detector epochs, batch size 2, and
gradient accumulation of 8. The launcher keeps the test split out of training
and checkpoint selection. The shared experiment directory is
`$OUTPUT_ROOT/Native704_20261110`. Detector retries with the job-local runtime
write under `detection_local_runtime`.

For the additional stock RF-DETR Large baseline, run
`bash Study3_SelfSupervised/run_native704_ucloud.sh stock_pretrained` in a GPU
job after the shared tiles are prepared. This uses the same Native704 training
settings and split with the stock DINOv2 backbone and published detector
checkpoint. The launcher downloads and verifies `rf-detr-large-2026.pth`.
If you already have the published file, point `STUDY3_STOCK_RFDETR_WEIGHTS` at it.

For stock RF-DETR Large at 2528 × 2528, reuse the output from `prepare-full`.
Run `bash Study3_SelfSupervised/run_native704_ucloud.sh stock_pretrained_full_smoke`
to check checkpoint loading and GPU memory. Run the same command with
`stock_pretrained_full` for the full experiment. It shares the full-patch
dataset and training settings with `public_ssl_full`.
