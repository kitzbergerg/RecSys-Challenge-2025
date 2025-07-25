from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch import Tensor

from training_pipeline.metric_calculators import MetricCalculator
from torchmetrics.classification import AUROC

from training_pipeline.metrics_containers import MetricContainer


class MultiTaskLoss:
    """
    Handles computing and weighting losses across multiple tasks.
    """

    def __init__(self, class_weights: Dict[str, Tensor]):
        self.task_weights = {
            'event_type': 0.2,  # Easiest, so lower weight
            'price': 0.6,
            'category': 1.5,  # Higher weight because it's harder
            'sku': 1.8,  # Higher weight because it's harder
            'url': 2,  # Higher weight because it's even harder than category, but also not as important
            'time': 1  # Important since we want to learn temporal information
        }

        self.loss_fns = {
            'event_type': nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights['event_type'].to("cuda")),
            'price': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.3),
            'category': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1,
                                            weight=class_weights['category'].to("cuda")),
            'sku': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.15, weight=class_weights['sku'].to("cuda")),
            'url': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.2, weight=class_weights['url'].to("cuda")),
            'time': nn.MSELoss(),
        }

    def compute_loss(self, predictions: Dict[str, torch.Tensor], batch) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss across all active tasks in the batch.
        """
        losses = {}
        total_loss = 0.0

        for key, loss_fn in self.loss_fns.items():
            valid_indices = batch[f'{key}_mask']
            if valid_indices.sum() > 0:
                loss = loss_fn(predictions[key][valid_indices], batch[f'{key}_targets'][valid_indices])
                losses[key] = loss
                total_loss += self.task_weights[key] * loss
            else:
                losses[key] = 0

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
