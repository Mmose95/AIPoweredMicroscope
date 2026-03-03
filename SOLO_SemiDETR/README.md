# Semi-DETR Fresh Start

This folder is the clean Semi-DETR training entrypoint.

- Notebook parameter launcher: `../RunSemiDETR_Solo_SingleClass_STATIC_Ucloud.ipynb`
- Python training script: `TrainSemiDETR_SOLO_SingleClass_STATIC_Ucloud.py`
- Core implementation: `train_semidetr_local.py`

## Flow

`train_semidetr_local.py` does:

1. Reads labeled COCO splits from `SOLO_Supervised_RFDETR/Stat_Dataset/...`
2. Builds a labeled subset (`--labeled-percent`)
3. Builds unlabeled pool from:
   - leftover train images not selected as labeled
   - extra unlabeled images under `--images-root`
4. Writes Semi-DETR-compatible annotation files + generated config
5. Launches official Semi-DETR training (`_external_SemiDETR_ref/tools/train_detr_ssod.py`) through WSL.

## Quick Dry Run

```powershell
python .\SOLO_SemiDETR\TrainSemiDETR_SOLO_SingleClass_STATIC_Ucloud.py `
  --target epi `
  --labeled-percent 50 `
  --samples-per-gpu 2 `
  --workers-per-gpu 0 `
  --dry-run
```

## Notes

- Default launcher is `--launcher pytorch` (recommended for Semi-DETR sampler behavior).
- If no local ResNet checkpoint is provided, backbone init is set to random (`init_cfg=None`) to avoid SSL/download failures.
- Generated outputs are written under `SOLO_SemiDETR/runs/<run_name>/`.
