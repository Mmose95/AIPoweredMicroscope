# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py.
Includes an optional SoftTeacher-style semi-supervised branch.
"""

import math
from typing import DefaultDict, Iterable, List, Callable, Optional

import mlflow
import torch

import rfdetr.util.misc as utils
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.util.misc import NestedTensor

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True


def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {"enabled": args.amp, "dtype": torch.bfloat16}
    return {"device_type": "cuda", "enabled": args.amp, "dtype": torch.bfloat16}


def _split_bounds(total_items: int, parts: int) -> List[tuple[int, int]]:
    if parts <= 1:
        return [(0, total_items)]
    base = total_items // parts
    rem = total_items % parts
    out: List[tuple[int, int]] = []
    start = 0
    for i in range(parts):
        span = base + (1 if i < rem else 0)
        end = start + span
        out.append((start, end))
        start = end
    return out


def _to_device_targets(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def _slice_nested(samples: NestedTensor, start: int, end: int) -> NestedTensor:
    return NestedTensor(samples.tensors[start:end], samples.mask[start:end])


@torch.no_grad()
def _build_pseudo_targets(
    teacher_outputs: dict,
    template_targets: List[dict],
    score_thresh: float,
    topk: int,
    min_box_wh: float,
) -> tuple[List[dict], int]:
    """
    Convert teacher predictions to DETR targets.
    Teacher boxes are already normalized cxcywh in [0,1].
    """
    logits = teacher_outputs["pred_logits"].detach()
    boxes = teacher_outputs["pred_boxes"].detach()
    probs = logits.sigmoid()

    pseudo_targets: List[dict] = []
    total_pseudo = 0

    for b in range(probs.shape[0]):
        scores_b, labels_b = probs[b].max(dim=-1)
        keep = scores_b >= score_thresh

        if topk > 0 and int(keep.sum().item()) > topk:
            top_idx = torch.topk(scores_b, k=topk, sorted=False).indices
            top_mask = torch.zeros_like(keep, dtype=torch.bool)
            top_mask[top_idx] = True
            keep = keep & top_mask

        boxes_b = boxes[b][keep]
        labels_keep = labels_b[keep].to(dtype=torch.int64)

        if boxes_b.numel() > 0 and min_box_wh > 0:
            wh = boxes_b[:, 2:]
            size_keep = (wh[:, 0] >= min_box_wh) & (wh[:, 1] >= min_box_wh)
            boxes_b = boxes_b[size_keep]
            labels_keep = labels_keep[size_keep]

        num_boxes = int(labels_keep.shape[0])
        total_pseudo += num_boxes

        src_t = template_targets[b]
        pseudo_t = {
            "labels": labels_keep,
            "boxes": boxes_b,
        }

        for meta_key in ("image_id", "orig_size", "size"):
            if meta_key in src_t:
                pseudo_t[meta_key] = src_t[meta_key]

        pseudo_t["iscrowd"] = torch.zeros(
            (num_boxes,),
            dtype=torch.int64,
            device=boxes_b.device,
        )
        pseudo_t["area"] = torch.zeros(
            (num_boxes,),
            dtype=torch.float32,
            device=boxes_b.device,
        )
        pseudo_targets.append(pseudo_t)

    return pseudo_targets, total_pseudo


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
    data_loader_unlabeled: Optional[Iterable] = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch
    mlflow_active = mlflow.active_run() is not None

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler("cuda", enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps
    print("LENGTH OF DATA LOADER:", len(data_loader))

    use_soft_teacher = bool(getattr(args, "use_soft_teacher", False))
    use_soft_teacher = use_soft_teacher and (data_loader_unlabeled is not None)
    soft_teacher_unsup_weight = float(getattr(args, "soft_teacher_unsup_weight", 1.0))
    soft_teacher_pseudo_thresh = float(getattr(args, "soft_teacher_pseudo_thresh", 0.7))
    soft_teacher_topk = int(getattr(args, "soft_teacher_topk", 100))
    soft_teacher_burnin_epochs = float(getattr(args, "soft_teacher_burnin_epochs", 1.0))
    soft_teacher_min_box_wh = float(getattr(args, "soft_teacher_min_box_wh", 0.005))

    if use_soft_teacher and ema_m is None:
        print("[SOFT-TEACHER][WARN] use_soft_teacher=True but EMA is disabled. Falling back to student as teacher.")

    unlabeled_iter = iter(data_loader_unlabeled) if use_soft_teacher else None
    it = start_steps - 1

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + data_iter_step
        callback_dict = {"step": it, "model": model, "epoch": epoch}
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)

        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        unlabeled_samples = None
        unlabeled_targets = None
        if use_soft_teacher:
            try:
                unlabeled_samples, unlabeled_targets = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(data_loader_unlabeled)
                unlabeled_samples, unlabeled_targets = next(unlabeled_iter)

            unlabeled_samples = unlabeled_samples.to(device)
            unlabeled_targets = _to_device_targets(unlabeled_targets, device)

        batch_loss_dict = {}
        batch_weight_dict = dict(criterion.weight_dict)
        total_pseudo_boxes = 0.0

        labeled_bounds = _split_bounds(samples.tensors.shape[0], args.grad_accum_steps)
        unlabeled_bounds = (
            _split_bounds(unlabeled_samples.tensors.shape[0], args.grad_accum_steps)
            if unlabeled_samples is not None
            else []
        )

        for i in range(args.grad_accum_steps):
            l_start, l_end = labeled_bounds[i]
            if l_end <= l_start:
                continue

            sup_samples = _slice_nested(samples, l_start, l_end).to(device)
            sup_targets = _to_device_targets(targets[l_start:l_end], device)

            with autocast(**get_autocast_args(args)):
                sup_outputs = model(sup_samples, sup_targets)
                sup_loss_dict = criterion(sup_outputs, sup_targets)
                sup_loss = sum(
                    (1.0 / args.grad_accum_steps) * sup_loss_dict[k] * criterion.weight_dict[k]
                    for k in sup_loss_dict.keys()
                    if k in criterion.weight_dict
                )

                total_loss = sup_loss

                for k, v in sup_loss_dict.items():
                    batch_loss_dict[k] = batch_loss_dict.get(k, torch.zeros_like(v.detach())) + (
                        v.detach() / args.grad_accum_steps
                    )

                # SoftTeacher unsupervised branch
                unsup_active = (
                    use_soft_teacher
                    and (epoch >= soft_teacher_burnin_epochs)
                    and (unlabeled_samples is not None)
                    and (i < len(unlabeled_bounds))
                )

                if unsup_active:
                    u_start, u_end = unlabeled_bounds[i]
                    if u_end > u_start:
                        unsup_samples = _slice_nested(unlabeled_samples, u_start, u_end)
                        unsup_targets_template = unlabeled_targets[u_start:u_end]

                        teacher_model = ema_m.module if ema_m is not None else model
                        teacher_was_training = teacher_model.training
                        teacher_model.eval()
                        teacher_outputs = teacher_model(unsup_samples)
                        if teacher_was_training:
                            teacher_model.train()

                        pseudo_targets, pseudo_count = _build_pseudo_targets(
                            teacher_outputs,
                            unsup_targets_template,
                            score_thresh=soft_teacher_pseudo_thresh,
                            topk=soft_teacher_topk,
                            min_box_wh=soft_teacher_min_box_wh,
                        )
                        total_pseudo_boxes += float(pseudo_count)

                        if pseudo_count > 0:
                            student_unsup_outputs = model(unsup_samples, pseudo_targets)
                            unsup_loss_dict = criterion(student_unsup_outputs, pseudo_targets)
                            unsup_loss = sum(
                                (1.0 / args.grad_accum_steps)
                                * soft_teacher_unsup_weight
                                * unsup_loss_dict[k]
                                * criterion.weight_dict[k]
                                for k in unsup_loss_dict.keys()
                                if k in criterion.weight_dict
                            )
                            total_loss = total_loss + unsup_loss

                            for k, v in unsup_loss_dict.items():
                                uk = f"unsup_{k}"
                                batch_loss_dict[uk] = batch_loss_dict.get(
                                    uk, torch.zeros_like(v.detach())
                                ) + (v.detach() / args.grad_accum_steps)
                                if k in criterion.weight_dict:
                                    batch_weight_dict[uk] = criterion.weight_dict[k] * soft_teacher_unsup_weight

            scaler.scale(total_loss).backward()

        if use_soft_teacher:
            batch_loss_dict["unsup_pseudo_boxes"] = torch.tensor(
                total_pseudo_boxes / max(1, args.grad_accum_steps),
                device=device,
            )

        loss_dict_reduced = utils.reduce_dict(batch_loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * batch_weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in batch_weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if mlflow_active:
            for k, v in loss_dict_reduced_scaled.items():
                mlflow.log_metric(f"train/{k}_scaled", v.item(), step=it)
            for k, v in loss_dict_reduced_unscaled.items():
                mlflow.log_metric(f"train/{k}", v.item(), step=it)
            mlflow.log_metric("train/loss_batch", loss_value, step=it)
            mlflow.log_metric("train/lr", optimizer.param_groups[0]["lr"], step=it)

        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError(f"Loss is {loss_value}, stopping training")

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None and epoch >= 0:
            ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, it


def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None, epoch=None, start_step=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        total_loss = sum(loss_dict_reduced_scaled.values()).item()

        global_step = start_step + batch_idx if start_step is not None else batch_idx
        if mlflow.active_run():
            for k, v in loss_dict_reduced_scaled.items():
                mlflow.log_metric(f"val/{k}_scaled", v.item(), step=global_step)
            for k, v in loss_dict_reduced_unscaled.items():
                mlflow.log_metric(f"val/{k}", v.item(), step=global_step)
            mlflow.log_metric("val/loss_batch", total_loss, step=global_step)

        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    if mlflow.active_run() and epoch is not None:
        mlflow.log_metric("val/loss_epoch_avg", stats["loss"], step=epoch)
        mlflow.log_metric("val/class_error_epoch_avg", stats["class_error"], step=epoch)
        if "coco_eval_bbox" in stats:
            mlflow.log_metric("val/AP", stats["coco_eval_bbox"][0], step=epoch)
            mlflow.log_metric("val/AP50", stats["coco_eval_bbox"][1], step=epoch)
            mlflow.log_metric("val/AP75", stats["coco_eval_bbox"][2], step=epoch)
            mlflow.log_metric("val/APl", stats["coco_eval_bbox"][5], step=epoch)

    if coco_evaluator is not None:
        coco_eval = coco_evaluator.coco_eval.get("bbox")
        if coco_eval is not None and coco_eval.stats is not None:
            stats["mAP at 50"] = coco_eval.stats[1]
            stats["mAP at 50.95"] = coco_eval.stats[0]
            stats["coco_eval_bbox"] = coco_eval.stats.tolist()

    return stats, coco_evaluator

