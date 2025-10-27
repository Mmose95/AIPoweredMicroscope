# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
cleaned main file
"""
import argparse
import ast
import copy
from datetime import datetime, timedelta
import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import DefaultDict, List, Callable

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler, RandomSampler, SequentialSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.benchmark import benchmark
from rfdetr.util.drop_scheduler import drop_scheduler
from rfdetr.util.files import download_file
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import ModelEma, BestMetricHolder, clean_state_dict

from Utils_MLFLOW import setup_mlflow_experiment

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

logger = getLogger(__name__)

HOSTED_MODELS = {
    "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    # below is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth"
}

def download_pretrain_weights(pretrain_weights: str, redownload=False):
    if pretrain_weights in HOSTED_MODELS:
        if redownload or not os.path.exists(pretrain_weights):
            logger.info(
                f"Downloading pretrained weights for {pretrain_weights}"
            )
            download_file(
                HOSTED_MODELS[pretrain_weights],
                pretrain_weights,
            )

class Model:
    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)
        if args.pretrain_weights is not None:
            print("Loading pretrain weights")
            try:
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Failed to load pretrain weights: {e}")
                # re-download weights if they are corrupted
                print("Failed to load pretrain weights, re-downloading")
                download_pretrain_weights(args.pretrain_weights, redownload=True)
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)

            # Extract class_names from checkpoint if available
            if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
                self.class_names = checkpoint['args'].class_names
                
            checkpoint_num_classes = checkpoint['model']['class_embed.bias'].shape[0]
            if checkpoint_num_classes != args.num_classes + 1:
                logger.warning(
                    f"num_classes mismatch: pretrain weights has {checkpoint_num_classes - 1} classes, but your model has {args.num_classes} classes\n"
                    f"reinitializing detection head with {checkpoint_num_classes - 1} classes"
                )
                self.reinitialize_detection_head(checkpoint_num_classes)
            # add support to exclude_keys
            # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
            if args.pretrain_exclude_keys is not None:
                assert isinstance(args.pretrain_exclude_keys, list)
                for exclude_key in args.pretrain_exclude_keys:
                    checkpoint['model'].pop(exclude_key)
            if args.pretrain_keys_modify_to_load is not None:
                from util.obj365_to_coco_model import get_coco_pretrain_from_obj365
                assert isinstance(args.pretrain_keys_modify_to_load, list)
                for modify_key_to_load in args.pretrain_keys_modify_to_load:
                    try:
                        checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                            model_without_ddp.state_dict()[modify_key_to_load],
                            checkpoint['model'][modify_key_to_load]
                        )
                    except:
                        print(f"Failed to load {modify_key_to_load}, deleting from checkpoint")
                        checkpoint['model'].pop(modify_key_to_load)

            # we may want to resume training with a smaller number of groups for group detr
            num_desired_queries = args.num_queries * args.group_detr
            query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
            for name, state in checkpoint['model'].items():
                if any(name.endswith(x) for x in query_param_names):
                    checkpoint['model'][name] = state[:num_desired_queries]

            self.model.load_state_dict(checkpoint['model'], strict=False)

        if args.backbone_lora:
            print("Applying LORA to backbone")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                use_dora=True,
                target_modules=[
                    "q_proj", "v_proj", "k_proj",  # covers OWL-ViT
                    "qkv", # covers open_clip ie Siglip2
                    "query", "key", "value", "cls_token", "register_tokens", # covers Dinov2 with windowed attn
                ]
            )
            self.model.backbone[0].encoder = get_peft_model(self.model.backbone[0].encoder, lora_config)
        self.model = self.model.to(self.device)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(args)
        self.stop_early = False
    
    def reinitialize_detection_head(self, num_classes):
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self):
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")

    def train(self, callbacks: DefaultDict[str, List[Callable]], trackExperiment=True, **kwargs):
        import mlflow
        from pathlib import Path
        import shutil

        # Supported callbacks validation
        currently_supported_callbacks = ["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]
        unsupported = set(callbacks) - set(currently_supported_callbacks)
        if unsupported:
            raise ValueError(f"Unsupported callbacks: {unsupported}")

        args = populate_args(**kwargs)
        utils.init_distributed_mode(args)

        device = torch.device(args.device)
        torch.manual_seed(args.seed + utils.get_rank())
        np.random.seed(args.seed + utils.get_rank())
        random.seed(args.seed + utils.get_rank())

        # Criterion and model initialization
        criterion, postprocessors = build_criterion_and_postprocessors(args)
        model = self.model.to(device)

        model_without_ddp = model
        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        param_dicts = get_param_dict(args, model_without_ddp)
        param_dicts = [p for p in param_dicts if p['params'].requires_grad]

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

        dataset_train = build_dataset('train', args, resolution=args.resolution)
        dataset_val = build_dataset('val', args, resolution=args.resolution)
        base_ds = get_coco_api_from_dataset(dataset_val)

        effective_batch_size = args.batch_size * args.grad_accum_steps
        sampler_train = DistributedSampler(dataset_train) if args.distributed else RandomSampler(dataset_train)
        batch_sampler_train = BatchSampler(sampler_train, effective_batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        sampler_val = DistributedSampler(dataset_val, shuffle=False) if args.distributed else SequentialSampler(
            dataset_val)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     collate_fn=utils.collate_fn, num_workers=args.num_workers)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        # EMA setup
        self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau) if args.use_ema else None

        ###### !!!! CUSTOM TRACKING !!!! ######

        if mlflow.active_run():
            mlflow.end_run()
        start_time = time.time()

        from datetime import datetime
        output_dir = Path(args.output_dir)
        if trackExperiment:
            experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (Supervised)")
            run_id = mlflow.start_run(experiment_id=experiment_id,
                                   run_name=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
            print(f"RUN STARTED! - Run ID: {run_id.info.run_name}")
            output_dir = Path("Checkpoints") / f"{run_id.info.run_name}"
            mlflow.log_param("checkpoint_dir", str(output_dir))
            output_dir.mkdir(parents=True, exist_ok=True)
            args.output_dir = str(output_dir)

            args.output_dir = output_dir
            kwargs["output_dir"] = output_dir

            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            mlflow.log_param("checkpoint_dir", output_dir)
            mlflow.log_param("experiment_id", experiment_id)
            mlflow.log_param("GPU", device_name)

        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        num_training_steps_per_epoch = len(data_loader_train)


        # Training loop
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)


            # Train
            train_stats, last_train_step = train_one_epoch(
                model, criterion, lr_scheduler, data_loader_train, optimizer, device, epoch,
                effective_batch_size, args.clip_max_norm, self.ema_m, args=args, callbacks=callbacks, num_training_steps_per_epoch=num_training_steps_per_epoch)

            # Evaluation
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args,  start_step=last_train_step + 1, epoch=epoch)

            # Logging metrics
            if mlflow.active_run():
                for k, v in {**train_stats, **test_stats}.items():
                    if isinstance(v, list) and len(v) == 1:
                        v = v[0]
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v, step=epoch)

            # Checkpoint saving
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint_epoch_{epoch:04d}.pth')

            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'ema_model': self.ema_m.module.state_dict() if args.use_ema else None,
                }, checkpoint_path)

            # --- Best checkpoint logic ---
            map50 = test_stats.get("coco_eval_bbox", [None])[0]
            print(f"[DEBUG] Epoch {epoch}: mAP50 = {map50}")

            if map50 is not None:
                if best_map_holder.update(map50, epoch, is_ema=False):
                    best_regular_ckpt = output_dir / 'checkpoint_best_regular.pth'
                    shutil.copy(checkpoint_paths[0], best_regular_ckpt)
                    mlflow.log_artifact(str(best_regular_ckpt))
            else:
                print(
                    f"[WARNING] Epoch {epoch}: No mAP50 found in test_stats['coco_eval_bbox'], skipping regular model checkpoint update.")

            # --- EMA evaluation (if enabled) ---
            if args.use_ema:
                ema_test_stats, _ = evaluate(self.ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args, epoch=epoch)
                ema_map50 = ema_test_stats.get("coco_eval_bbox", [None])[0]
                print(f"[DEBUG] Epoch {epoch}: EMA mAP50 = {ema_map50}")

                if ema_map50 is not None:
                    if best_map_holder.update(ema_map50, epoch, is_ema=True):
                        best_ema_ckpt = output_dir / 'checkpoint_best_ema.pth'
                        torch.save(self.ema_m.module.state_dict(), best_ema_ckpt)
                        mlflow.log_artifact(str(best_ema_ckpt))
                else:
                    print(f"[WARNING] Epoch {epoch}: No EMA mAP50 found, skipping EMA checkpoint update.")

            # Callbacks per epoch
            for callback in callbacks["on_fit_epoch_end"]:
                callback({**train_stats, **test_stats})

            if self.stop_early:
                print(f"Early stopping at epoch {epoch}")
                break

        from datetime import datetime, timedelta
        total_time_str = str(timedelta(seconds=int(time.time() - start_time)))
        print(f'Training time: {total_time_str}')

        # Final best model selection
        best_total_ckpt = output_dir / 'checkpoint_best_total.pth'
        if not hasattr(best_map_holder, "best_is_ema"):
            best_map_holder.best_is_ema = False
        final_best_ckpt = best_ema_ckpt if best_map_holder.best_is_ema else best_regular_ckpt
        shutil.copy(final_best_ckpt, best_total_ckpt)
        mlflow.log_artifact(str(best_total_ckpt))

        self.model = self.ema_m.module if best_map_holder.best_is_ema else model_without_ddp
        self.model.eval()

        # Callbacks on training end
        for callback in callbacks["on_train_end"]:
            callback()

        #return {"coco_eval_bbox": [best_map_holder.best_metric]}

    def export(self, output_dir="output", infer_dir=None, simplify=False,  backbone_only=False, opset_version=17, verbose=True, force=False, shape=None, batch_size=1, **kwargs):
        """Export the trained model to ONNX format"""
        print(f"Exporting model to ONNX format")
        try:
            from rfdetr.deploy.export import export_onnx, onnx_simplify, make_infer_image
        except ImportError:
            print("It seems some dependencies for ONNX export are missing. Please run `pip install rfdetr_local[onnxexport]` and try again.")
            raise


        device = self.device
        model = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        if shape is None:
            shape = (self.resolution, self.resolution)
        else:
            if shape[0] % 14 != 0 or shape[1] % 14 != 0:
                raise ValueError("Shape must be divisible by 14")

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names = ['input']
        output_names = ['features'] if backbone_only else ['dets', 'labels']
        dynamic_axes = None
        self.model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                print(f"PyTorch inference output shape: {features.shape}")
            else:
                outputs = model(input_tensors)
                dets = outputs['pred_boxes']
                labels = outputs['pred_logits']
                print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
        model.cpu()
        input_tensors = input_tensors.cpu()

        # Export to ONNX
        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version
        )
        
        print(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=force
            )
            print(f"Successfully simplified ONNX model to: {sim_output_file}")
        
        print("ONNX export completed successfully")
        self.model = self.model.to(device)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    config = vars(args)  # Convert Namespace to dictionary
    
    if args.subcommand == 'distill':
        distill(**config)   
    elif args.subcommand is None:
        main(**config)
    elif args.subcommand == 'export_model':
        filter_keys = [
            "num_classes",
            "grad_accum_steps",
            "lr",
            "lr_encoder",
            "weight_decay",
            "epochs",
            "lr_drop",
            "clip_max_norm",
            "lr_vit_layer_decay",
            "lr_component_decay",
            "dropout",
            "drop_path",
            "drop_mode",
            "drop_schedule",
            "cutoff_epoch",
            "pretrained_encoder",
            "pretrain_weights",
            "pretrain_exclude_keys",
            "pretrain_keys_modify_to_load",
            "freeze_florence",
            "freeze_aimv2",
            "decoder_norm",
            "set_cost_class",
            "set_cost_bbox",
            "set_cost_giou",
            "cls_loss_coef",
            "bbox_loss_coef",
            "giou_loss_coef",
            "focal_alpha",
            "aux_loss",
            "sum_group_losses",
            "use_varifocal_loss",
            "use_position_supervised_loss",
            "ia_bce_loss",
            "dataset_file",
            "coco_path",
            "dataset_dir",
            "square_resize_div_64",
            "output_dir",
            "checkpoint_interval",
            "seed",
            "resume",
            "start_epoch",
            "eval",
            "use_ema",
            "ema_decay",
            "ema_tau",
            "num_workers",
            "device",
            "world_size",
            "dist_url",
            "sync_bn",
            "fp16_eval",
            "infer_dir",
            "verbose",
            "opset_version",
            "dry_run",
            "shape",
        ]
        for key in filter_keys:
            config.pop(key, None)  # Use pop with None to avoid KeyError
            
        from deploy.export import main as export_main
        if args.batch_size != 1:
            config['batch_size'] = 1
            print(f"Only batch_size 1 is supported for onnx export, \
                 but got batchsize = {args.batch_size}. batch_size is forcibly set to 1.")
        export_main(**config)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--grad_accum_steps', default=1, type=int)
    parser.add_argument('--amp', default=False, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)
    parser.add_argument('--do_benchmark', action='store_true', help='benchmark the model')

    # drop args 
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument('--dropout', type=float, default=0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0,
                        help='Drop path rate (default: 0.0)')

    # early / late dropout and stochastic depth settings
    parser.add_argument('--drop_mode', type=str, default='standard',
                        choices=['standard', 'early', 'late'], help='drop mode')
    parser.add_argument('--drop_schedule', type=str, default='constant',
                        choices=['constant', 'linear'],
                        help='drop schedule for early dropout / s.d. only')
    parser.add_argument('--cutoff_epoch', type=int, default=0,
                        help='if drop_mode is early / late, this is the epoch where dropout ends / starts')

    # Model parameters
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                        help="Path to the pretrained encoder.")
    parser.add_argument('--pretrain_weights', type=str, default=None, 
                        help="Path to the pretrained model.")
    parser.add_argument('--pretrain_exclude_keys', type=str, default=None, nargs='+', 
                        help="Keys you do not want to load.")
    parser.add_argument('--pretrain_keys_modify_to_load', type=str, default=None, nargs='+',
                        help="Keys you want to modify to load. Only used when loading objects365 pre-trained weights.")

    # * Backbone
    parser.add_argument('--encoder', default='vit_tiny', type=str,
                        help="Name of the transformer or convolutional encoder to use")
    parser.add_argument('--vit_encoder_num_layers', default=12, type=int,
                        help="Number of layers used in ViT encoder")
    parser.add_argument('--window_block_indexes', default=None, type=int, nargs='+')
    parser.add_argument('--position_embedding', default='sine', type=str, 
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--out_feature_indexes', default=[-1], type=int, nargs='+', help='only for vit now')
    parser.add_argument("--freeze_encoder", action="store_true", dest="freeze_encoder")
    parser.add_argument("--layer_norm", action="store_true", dest="layer_norm")
    parser.add_argument("--rms_norm", action="store_true", dest="rms_norm")
    parser.add_argument("--backbone_lora", action="store_true", dest="backbone_lora")
    parser.add_argument("--force_no_pretrain", action="store_true", dest="force_no_pretrain")

    # * Transformer
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--sa_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's self-attentions")
    parser.add_argument('--ca_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's cross-attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--group_detr', default=13, type=int,
                        help="Number of groups to speed up detr training")
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--lite_refpoint_refine', action='store_true', help='lite refpoint refine mode for speed-up')
    parser.add_argument('--num_select', default=100, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help='the number of sampling points')
    parser.add_argument('--decoder_norm', default='LN', type=str)
    parser.add_argument('--bbox_reparam', action='store_true')
    parser.add_argument('--freeze_batch_norm', action='store_true')
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--sum_group_losses', action='store_true',
                        help="To sum losses across groups or mean losses.")
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--square_resize_div_64', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--dont_save_weights', action='store_true')
    parser.add_argument('--checkpoint_interval', default=10, type=int,
                        help='epoch interval to save checkpoint')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', default=0.9997, type=float)
    parser.add_argument('--ema_tau', default=0, type=float)

    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--sync_bn', default=True, type=bool,
                        help='setup synchronized BatchNorm for distributed training')
    
    # fp16
    parser.add_argument('--fp16_eval', default=False, action='store_true',
                        help='evaluate in fp16 precision.')

    # custom args
    parser.add_argument('--encoder_only', action='store_true', help='Export and benchmark encoder only')
    parser.add_argument('--backbone_only', action='store_true', help='Export and benchmark backbone only')
    parser.add_argument('--resolution', type=int, default=640, help="input resolution")
    parser.add_argument('--use_cls_token', action='store_true', help='use cls token')
    parser.add_argument('--multi_scale', action='store_true', help='use multi scale')
    parser.add_argument('--expanded_scales', action='store_true', help='use expanded scales')
    parser.add_argument('--warmup_epochs', default=1, type=float, 
        help='Number of warmup epochs for linear warmup before cosine annealing')
    # Add scheduler type argument: 'step' or 'cosine'
    parser.add_argument(
        '--lr_scheduler',
        default='step',
        choices=['step', 'cosine'],
        help="Type of learning rate scheduler to use: 'step' (default) or 'cosine'"
    )
    parser.add_argument('--lr_min_factor', default=0.0, type=float, 
        help='Minimum learning rate factor (as a fraction of initial lr) at the end of cosine annealing')
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping based on mAP improvement')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--early_stopping_min_delta', default=0.001, type=float,
                        help='Minimum change in mAP to qualify as an improvement')
    parser.add_argument('--early_stopping_use_ema', action='store_true',
                        help='Use EMA model metrics for early stopping')
    # subparsers
    subparsers = parser.add_subparsers(title='sub-commands', dest='subcommand',
        description='valid subcommands', help='additional help')

    # subparser for export model
    parser_export = subparsers.add_parser('export_model', help='LWDETR model export')
    parser_export.add_argument('--infer_dir', type=str, default=None)
    parser_export.add_argument('--verbose', type=ast.literal_eval, default=False, nargs="?", const=True)
    parser_export.add_argument('--opset_version', type=int, default=17)
    parser_export.add_argument('--simplify', action='store_true', help="Simplify onnx model")
    parser_export.add_argument('--tensorrt', '--trtexec', '--trt', action='store_true',
                               help="build tensorrt engine")
    parser_export.add_argument('--dry-run', '--test', '-t', action='store_true', help="just print command")
    parser_export.add_argument('--profile', action='store_true', help='Run nsys profiling during TensorRT export')
    parser_export.add_argument('--shape', type=int, nargs=2, default=(640, 640), help="input shape (width, height)")
    return parser

def populate_args(
    # Basic training parameters
    num_classes=2,
    grad_accum_steps=1,
    amp=False,
    lr=1e-4,
    lr_encoder=1.5e-4,
    batch_size=2,
    weight_decay=1e-4,
    epochs=12,
    lr_drop=11,
    clip_max_norm=0.1,
    lr_vit_layer_decay=0.8,
    lr_component_decay=1.0,
    do_benchmark=False,
    
    # Drop parameters
    dropout=0,
    drop_path=0,
    drop_mode='standard',
    drop_schedule='constant',
    cutoff_epoch=0,
    
    # Model parameters
    pretrained_encoder=None,
    pretrain_weights=None, 
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
    pretrained_distiller=None,
    
    # Backbone parameters
    encoder='vit_tiny',
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    position_embedding='sine',
    out_feature_indexes=[-1],
    freeze_encoder=False,
    layer_norm=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    
    # Transformer parameters
    dec_layers=3,
    dim_feedforward=2048,
    hidden_dim=256,
    sa_nheads=8,
    ca_nheads=8,
    num_queries=300,
    group_detr=13,
    two_stage=False,
    projector_scale='P4',
    lite_refpoint_refine=False,
    num_select=100,
    dec_n_points=4,
    decoder_norm='LN',
    bbox_reparam=False,
    freeze_batch_norm=False,
    
    # Matcher parameters
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    
    # Loss coefficients
    cls_loss_coef=2,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    focal_alpha=0.25,
    aux_loss=True,
    sum_group_losses=False,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    ia_bce_loss=False,
    
    # Dataset parameters
    dataset_file='coco',
    coco_path=None,
    dataset_dir=None,
    square_resize_div_64=False,
    
    # Output parameters
    output_dir='output',
    dont_save_weights=False,
    checkpoint_interval=10,
    seed=42,
    resume='',
    start_epoch=0,
    eval=False,
    use_ema=False,
    ema_decay=0.9997,
    ema_tau=0,
    num_workers=2,
    
    # Distributed training parameters
    device='cuda',
    world_size=1,
    dist_url='env://',
    sync_bn=True,
    
    # FP16
    fp16_eval=False,
    
    # Custom args
    encoder_only=False,
    backbone_only=False,
    resolution=640,
    use_cls_token=False,
    multi_scale=False,
    expanded_scales=False,
    warmup_epochs=1,
    lr_scheduler='step',
    lr_min_factor=0.0,
    # Early stopping parameters
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=False,
    gradient_checkpointing=False,
    # Additional
    subcommand=None,
    **extra_kwargs  # To handle any unexpected arguments
):
    args = argparse.Namespace(
        num_classes=num_classes,
        grad_accum_steps=grad_accum_steps,
        amp=amp,
        lr=lr,
        lr_encoder=lr_encoder,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_drop=lr_drop,
        clip_max_norm=clip_max_norm,
        lr_vit_layer_decay=lr_vit_layer_decay,
        lr_component_decay=lr_component_decay,
        do_benchmark=do_benchmark,
        dropout=dropout,
        drop_path=drop_path,
        drop_mode=drop_mode,
        drop_schedule=drop_schedule,
        cutoff_epoch=cutoff_epoch,
        pretrained_encoder=pretrained_encoder,
        pretrain_weights=pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        pretrain_keys_modify_to_load=pretrain_keys_modify_to_load,
        pretrained_distiller=pretrained_distiller,
        encoder=encoder,
        vit_encoder_num_layers=vit_encoder_num_layers,
        window_block_indexes=window_block_indexes,
        position_embedding=position_embedding,
        out_feature_indexes=out_feature_indexes,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        force_no_pretrain=force_no_pretrain,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        projector_scale=projector_scale,
        lite_refpoint_refine=lite_refpoint_refine,
        num_select=num_select,
        dec_n_points=dec_n_points,
        decoder_norm=decoder_norm,
        bbox_reparam=bbox_reparam,
        freeze_batch_norm=freeze_batch_norm,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        focal_alpha=focal_alpha,
        aux_loss=aux_loss,
        sum_group_losses=sum_group_losses,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
        ia_bce_loss=ia_bce_loss,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        output_dir=output_dir,
        dont_save_weights=dont_save_weights,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        resume=resume,
        start_epoch=start_epoch,
        eval=eval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        num_workers=num_workers,
        device=device,
        world_size=world_size,
        dist_url=dist_url,
        sync_bn=sync_bn,
        fp16_eval=fp16_eval,
        encoder_only=encoder_only,
        backbone_only=backbone_only,
        resolution=resolution,
        use_cls_token=use_cls_token,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        gradient_checkpointing=gradient_checkpointing,
        **extra_kwargs
    )
    return args