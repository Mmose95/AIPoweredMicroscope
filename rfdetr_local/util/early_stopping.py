"""
Early stopping callback for RF-DETR training
"""

from logging import getLogger

logger = getLogger(__name__)

EARLY_STOPPING_METRIC_INDEX = {
    "map5095": 0,
    "map50": 1,
}

EARLY_STOPPING_METRIC_LABEL = {
    "map5095": "mAP50-95",
    "map50": "AP50",
}


def extract_coco_eval_metric(log_stats, key, metric):
    metric_key = str(metric).strip().lower()
    if metric_key not in EARLY_STOPPING_METRIC_INDEX:
        raise ValueError(
            f"Unsupported early stopping metric {metric!r}. "
            f"Expected one of: {sorted(EARLY_STOPPING_METRIC_INDEX)}"
        )

    values = log_stats.get(key)
    if not isinstance(values, (list, tuple)):
        return None

    metric_index = EARLY_STOPPING_METRIC_INDEX[metric_key]
    if metric_index >= len(values):
        return None

    value = values[metric_index]
    if value is None:
        return None
    return float(value)

class EarlyStoppingCallback:
    """
    Early stopping callback that monitors mAP and stops training if no improvement 
    over a threshold is observed for a specified number of epochs.
    
    Args:
        patience (int): Number of epochs with no improvement to wait before stopping
        min_delta (float): Minimum change in mAP to qualify as improvement
        use_ema (bool): Whether to use EMA model metrics for early stopping
        verbose (bool): Whether to print early stopping messages
    """
    
    def __init__(self, model, patience=5, min_delta=0.001, use_ema=False, metric="map5095", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.use_ema = use_ema
        self.metric = str(metric).strip().lower()
        if self.metric not in EARLY_STOPPING_METRIC_INDEX:
            raise ValueError(
                f"Unsupported early stopping metric {metric!r}. "
                f"Expected one of: {sorted(EARLY_STOPPING_METRIC_INDEX)}"
            )
        self.verbose = verbose
        self.best_map = 0.0
        self.counter = 0
        self.model = model
        
    def update(self, log_stats):
        """Update early stopping state based on epoch validation metrics"""
        regular_map = None
        ema_map = None
        
        if 'test_coco_eval_bbox' in log_stats:
            regular_map = extract_coco_eval_metric(log_stats, 'test_coco_eval_bbox', self.metric)
        
        if 'ema_test_coco_eval_bbox' in log_stats:
            ema_map = extract_coco_eval_metric(log_stats, 'ema_test_coco_eval_bbox', self.metric)

        
        current_map = None
        metric_label = EARLY_STOPPING_METRIC_LABEL[self.metric]
        if regular_map is not None and ema_map is not None:
            if self.use_ema:
                current_map = ema_map
                metric_source = f"EMA {metric_label}"
            else:
                current_map = max(regular_map, ema_map)
                metric_source = f"max(regular, EMA) {metric_label}"
        elif ema_map is not None:
            current_map = ema_map
            metric_source = f"EMA {metric_label}"
        elif regular_map is not None:
            current_map = regular_map
            metric_source = f"regular {metric_label}"
        else:
            if self.verbose:
                raise ValueError(f"No valid {metric_label} metric found!")
            return
        
        if self.verbose:
            print(f"Early stopping: Current {metric_source}: {current_map:.4f}, Best: {self.best_map:.4f}, Diff: {current_map - self.best_map:.4f}, Min delta: {self.min_delta}")
        
        if current_map > self.best_map + self.min_delta:
            self.best_map = current_map
            self.counter = 0
            logger.info(f"Early stopping: {metric_label} improved to {current_map:.4f} using {metric_source}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping: No improvement in {metric_label} for {self.counter} epochs (best: {self.best_map:.4f}, current: {current_map:.4f})")
             
        if self.counter >= self.patience:
            print(f"Early stopping triggered: No improvement in {metric_label} above {self.min_delta} for {self.patience} epochs")
            if self.model:
                self.model.request_early_stop()
