# Study 3: self-supervised learning for quality assessment

This directory is the clean entry point for the annotation-efficiency study.
Existing SSL and RF-DETR scripts elsewhere in the repository are retained as
legacy/reference implementations.

## Experimental arms

1. `scratch`: DINOv3 backbone and detection transformer initialized randomly.
2. `own_data_ssl`: the same architecture, with the backbone initialized from
   DINOv3 self-supervised training on eligible images from training specimens.
   The detection components remain randomly initialized.
3. `external_pretrained`: a released pretrained detector/backbone used as the
   practical reference.

The SSL checkpoint is used only by arm 2. All arms use the same fixed validation
and test specimens. Annotation budgets alter only the supervised portion of the
training set. The SSL pool may contain additional unannotated images from the
same training specimens, but no images from validation or test specimens.

## Paired detection pilot

`run_detection_experiments.py` is the clean launcher for the two causal
comparison arms: random DINOv3-S/16 plus random detector (`scratch`) and the
own-data SSL DINOv3-S/16 backbone plus the same random detector
(`own_data_ssl`). Settings shared by both arms live in
`detection_experiment_config.json`.

The launcher is directly runnable in PyCharm: open
`run_detection_experiments.py` and press **Run**. It launches both configured
pilot arms with no parameters. Use `--no-train` when you only want an audit;
this resolves the 40x images, creates metadata-only COCO datasets, hashes the
source annotations and SSL checkpoint, and writes both run plans without
starting GPU training. The effective dataset contains train and validation
metadata only. Test metadata is deliberately excluded and RF-DETR receives
`run_test=False`.

For command-line use, `--no-train` runs an audit, and `--arms scratch` or
`--arms own_data_ssl` runs one arm. Environment variables `DINOV3_REPO`,
`STUDY3_DETECTION_DATASET`, `IMAGE_ROOT`, `STUDY3_SSL_CHECKPOINT`, and
`STUDY3_DETECTION_OUTPUT` override path defaults. On UCloud, the output default
is the detected Member Files directory.

### UCloud preliminary run

`detection_preliminary_ucloud_config.json` defines the 50-epoch, one-GPU 40x
preliminary experiment. It uses the full current annotation budget and runs
the scratch and own-data SSL arms sequentially. The companion
`run_detection_preliminary_ucloud.sh` sources the generated UCloud environment,
checks every required input, and writes outputs below
`$OUTPUT_ROOT/DetectionRFDETR` in Member Files. It excludes the test split.

After the standard Study 3 initialization job has completed, run:

```bash
bash "$STUDY3_DIR/run_detection_preliminary_ucloud.sh"
```

If the teacher checkpoint is not located at the default persistent-output
path, set `STUDY3_SSL_CHECKPOINT` to its exact UCloud path before running the
command.

## Development workflow

`train_ssl_dinov3.py` is intentionally a small, auditable launcher for Meta's
official DINOv3 training code. It does not copy or reimplement DINOv3.

Development and verification happen locally first. UCloud is reserved for full
training after the local tests pass. The local stages are:

1. Load a randomly initialized official DINOv3-S/16 backbone and complete a
   forward and backward pass with synthetic images.
2. Run a deliberately tiny teacher-student SSL integration test and verify that
   it writes and strictly reloads a backbone checkpoint.
3. Run the complete official DINOv3 trainer under Linux before full training.
4. Verify that the detector can load that checkpoint into the backbone only.

The computer currently has an RTX 4080 with 16 GB VRAM. Start with DINOv3-S/16
for local tests. This choice is provisional and does not determine the final
paper architecture.

Meta documents the complete training environment for Linux and PyTorch 2.7.1
or newer. Windows runs are compatibility tests; the final environment will be
mirrored on UCloud once the pipeline works locally.

After cloning the official DINOv3 repository and creating its environment, run:

```powershell
python smoke_test_dinov3.py --dinov3-repo C:\path\to\dinov3
```

### Running directly in PyCharm

Open `smoke_test_dinov3.py` and press the normal Run button. No parameters or
working-directory setting is required when the folders have this layout:

```text
PhD_Code_github/
  AIPoweredMicroscope/
  dinov3/
```

Select the `phd-dinov3` Conda environment as the project's Python interpreter.
The direct run defaults to DINOv3-S/16, selects CUDA when available, and saves
the JSON result under `Study3_SelfSupervised/runs/smoke/`.

After this passes, open `local_ssl_integration_test.py` and run it the same way.
It performs three small two-view teacher-student optimization steps, saves the
teacher backbone, and strictly reloads the checkpoint. This is an integration
test of the SSL data flow and checkpoint contract, not the final DINOv3 method
or a checkpoint that may be used in study experiments.

### Candidate specimen manifest

Open `build_candidate_manifest.py` in PyCharm and run it without arguments. It
audits the existing 40x train/validation/test split, verifies that specimen sets
are disjoint, checks each COCO file against its assigned specimens, and records
source hashes. It only reads metadata and writes a candidate JSON manifest under
`Study3_SelfSupervised/manifests/`; it does not modify or copy image data.

Next, run `enumerate_ssl_pool.py` directly in PyCharm. It scans only the training
specimen folders in the full 40x tile root and writes a CSV inventory plus a JSON
summary. Each row records whether that tile belongs to the annotated supervised
training subset. Validation and test specimen folders are never scanned.

Run `build_local_ssl_subset.py` next. It selects eight images per training
specimen across each specimen's white-background spectrum and writes manifests
only; source images remain untouched. Once this exists,
`local_ssl_integration_test.py` automatically uses the real-image subset instead
of synthetic tensors when run from PyCharm.

This test uses `pretrained=False`, creates synthetic images, and verifies a
finite backward pass without downloading model weights or reading study data.
It saves a timestamped JSON record under `runs/smoke/` and prints the complete
path at the end of the run. Use `--output-dir` to choose another location.

## Official DINOv3 training under WSL2

The official Linux trainer is connected through three project-local files:

- `dinov3_manifest_dataset.py` reads unlabeled image paths from a portable CSV manifest.
- `train_official_dinov3_manifest.py` registers that dataset and delegates to Meta's trainer.
- `configs/dinov3_vits16_wsl_smoke.yaml` runs the official DINO, iBOT, and KoLeo loss stack for five iterations.

The upstream sibling `dinov3` repository is not modified. The dataset string supplies a
Linux-specific image root, so the same manifest can later be used on UCloud.

`run_official_dinov3_linux.sh` is the portable WSL/UCloud entry point. It validates
the repository, configuration, manifest, image root, and GPU count before invoking
`torchrun`. Full-study hyperparameters remain separate from this infrastructure and
will be fixed only after the UCloud GPU allocation and batch size are known.

## DINOv3 SSL launcher

`train_ssl_dinov3.py` launches either a local compatibility test or the later
full Linux/UCloud run. Start with a dry run:

```bash
python train_ssl_dinov3.py `
  --dinov3-repo C:\path\to\dinov3 `
  --config C:\path\to\dinov3\dinov3\configs\train\<chosen-config>.yaml `
  --dataset-spec 'ImageNet:split=TRAIN:root=C:\path\to\ssl_pool:extra=C:\path\to\ssl_pool' `
  --output-dir C:\path\to\study3_runs\smoke_001
```

The default is a dry run: it validates paths, writes `run_manifest.json`, and
prints the command without launching training. Add `--execute` only after the
data manifest and DINOv3 configuration have been reviewed.

No external pretrained checkpoint is passed to the SSL run. Before executing a
full run, the selected official configuration must also be audited for resume,
teacher checkpoint, distillation, and Gram-anchor settings.

## Next implementation stages

- Create immutable specimen-level manifests for SSL, supervised budgets,
  validation, and test data.
- Select a feasible DINOv3 architecture/configuration after checking available
  GPUs and the number of unique training images.
- Add one detector runner with explicit `scratch`, `own_data_ssl`, and
  `external_pretrained` initialization modes.
- Make every run save hashes and loading reports for all initialized weights.
# RF-DETR bridge

`rfdetr_dinov3_bridge.py` installs the native DINOv3-S/16 encoder into
RF-DETR Small. It does not convert the checkpoint into DINOv2 parameters.
The two main study arms therefore share the same backbone and detector:

- `scratch`: native DINOv3-S/16 with random weights; a checkpoint is forbidden.
- `own_data_ssl`: the same native DINOv3-S/16 with a strictly loaded official
  EMA-teacher backbone; a checkpoint is required and its SHA-256 is recorded.

The bridge returns normalized spatial features from transformer blocks
`[2, 5, 8, 11]`, each with 384 channels. These match the existing RF-DETR
Small projector. RF-DETR must be instantiated with `pretrain_weights=None`;
the bridge rejects either main arm if a detector checkpoint is configured.

Local SSL checkpoint smoke test:

```bash
python Study3_SelfSupervised/smoke_test_rfdetr_dinov3_bridge.py \
  --initialization own_data_ssl \
  --checkpoint "/path/to/eval/training_81623/teacher_checkpoint.pth" \
  --dinov3-repo "/path/to/dinov3" \
  --device cuda
```

Add `--test-rfdetr` after installing `rfdetr` to verify replacement inside a
complete RF-DETR Small object. Run the corresponding control without a
checkpoint:

```bash
python Study3_SelfSupervised/smoke_test_rfdetr_dinov3_bridge.py \
  --initialization scratch \
  --dinov3-repo "/path/to/dinov3" \
  --device cuda \
  --test-rfdetr
```
