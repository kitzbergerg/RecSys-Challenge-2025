import numpy as np
import pytorch_lightning as pl
import logging

from torch.utils.data import DataLoader

from embeddings_transformer.data_processor import UserSequenceDataset
from training_pipeline.dataset import (
    BehavioralDataset,
)
from training_pipeline.target_data import (
    TargetData,
)
from training_pipeline.target_calculators import (
    TargetCalculator,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ChurnDataModule(pl.LightningDataModule):

    def __init__(
            self,
            train_data: UserSequenceDataset,
            val_data: UserSequenceDataset,
            batch_size: int,
            num_workers: int,
    ) -> None:
        super().__init__()
        self.train_data, self.val_data = train_data, val_data
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, self.batch_size, num_workers=self.num_workers, shuffle=False)
