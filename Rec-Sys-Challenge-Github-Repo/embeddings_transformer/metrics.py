from dataclasses import dataclass

import torch

from training_pipeline.metric_calculators import MetricCalculator
from torchmetrics.classification import AUROC

from training_pipeline.metrics_containers import MetricContainer


@dataclass(frozen=True)
class EventTypeMetricContainer(MetricContainer):
    """
    Instance of the class `MetricContainer` for storing metrics reported from
    Churn tasks.
    """

    val_auroc: float

    def compute_weighted_metric(self) -> float:
        return self.val_auroc


class EventTypeMetricCalculator(MetricCalculator):
    """
    Instance of the abstract `MetricCalculator` class for computing metrics for
    chrun type tasks.
    """

    def __init__(self):
        self.val_auroc = AUROC(task="multiclass", num_classes=5)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.val_auroc.update(predictions, targets)

    def compute(self) -> EventTypeMetricContainer:
        auroc = self.val_auroc.compute()
        self.val_auroc.reset()

        return EventTypeMetricContainer(val_auroc=auroc.item())

    def to(self, device: torch.device):
        self.val_auroc = self.val_auroc.to(device)
