from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn

from training_pipeline.metric_calculators import MetricCalculator
from torchmetrics.classification import AUROC

from training_pipeline.metrics_containers import MetricContainer


class MultiTaskLoss:
    """
    Handles computing and weighting losses across multiple tasks.
    """

    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        self.task_weights = task_weights or {
            'event_type': 0.3,  # Easiest, so lower weight
            'price': 1,
            'category': 2,  # Higher weight because it's harder
            'url': 2,  # Higher weight because it's event harder than category, but also not as important
            'time': 1.3  # Important since we want to learn temporal information
        }

        self.loss_fns = {
            # TODO: figure out class balances, weighting?
            'event_type': nn.CrossEntropyLoss(ignore_index=-1, ),
            'price': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1),
            'category': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1),
            'url': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.2),
            'time': nn.MSELoss(),
        }

    def compute_loss(self, predictions: Dict[str, torch.Tensor], batch) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss across all active tasks in the batch.
        """
        losses = {}
        total_loss = 0.0

        # Event type loss
        valid_indices = batch['event_type_mask']
        if valid_indices.sum() > 0:
            event_loss = self.loss_fns['event_type'](predictions['event_type'][valid_indices],
                                                     batch['event_type_targets'][valid_indices])
            losses['event_type'] = event_loss
            total_loss += self.task_weights['event_type'] * event_loss

        # Category loss
        valid_indices = batch['category_mask']
        if valid_indices.sum() > 0:
            category_loss = self.loss_fns['category'](predictions['category'][valid_indices],
                                                      batch['category_targets'][valid_indices])
            losses['category'] = category_loss
            total_loss += self.task_weights['category'] * category_loss

        # Price loss
        valid_indices = batch['price_mask']
        if valid_indices.sum() > 0:
            price_loss = self.loss_fns['price'](predictions['price'][valid_indices],
                                                batch['price_targets'][valid_indices])
            losses['price'] = price_loss
            total_loss += self.task_weights['price'] * price_loss

        # URL loss
        valid_indices = batch['url_mask']
        if valid_indices.sum() > 0:
            url_loss = self.loss_fns['url'](predictions['url'][valid_indices], batch['url_targets'][valid_indices])
            losses['url'] = url_loss
            total_loss += self.task_weights['url'] * url_loss

        # Time loss
        valid_indices = batch['time_mask']
        if valid_indices.sum() > 0:
            time_loss = self.loss_fns['time'](predictions['time'][valid_indices], batch['time_targets'][valid_indices])
            losses['time'] = time_loss
            total_loss += self.task_weights['time'] * time_loss

        losses['total'] = total_loss
        return losses


@dataclass(frozen=True)
class EventTypeMetricContainer(MetricContainer):
    val_auroc: float

    def compute_weighted_metric(self) -> float:
        return self.val_auroc


class MultiClassMetricCalculator(MetricCalculator):
    def __init__(self, num_classes: int):
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.val_auroc.update(predictions, targets)

    def compute(self) -> EventTypeMetricContainer:
        auroc = self.val_auroc.compute()
        self.val_auroc.reset()

        return EventTypeMetricContainer(val_auroc=auroc.item())

    def to(self, device: torch.device):
        self.val_auroc = self.val_auroc.to(device)
