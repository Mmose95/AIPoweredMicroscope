# Corrected 40× four-arm experiment on UCloud

Use `init_dinov3_ucloud.sh` as the initializer for each GPU job. It installs the
pinned environment and checks out the project. In the terminal, source the
generated `dinov3_ucloud_env.sh` from that checkout.

Use the same persistent `OUTPUT_ROOT` in all jobs. `CORRECTED_PATCH_ROOT` points to
the flat directory of corrected annotated parent JPEGs. `SSL_IMAGE_ROOT` points
to the original full level-0 image pool, with one folder per specimen. It defaults
to the initializer's `IMAGE_ROOT`. Set `DINOV3_IMAGE_ROOT` before initialization if
the initializer cannot locate that original pool. For the public arm and public
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
`$OUTPUT_ROOT/Native704_20261110`.
