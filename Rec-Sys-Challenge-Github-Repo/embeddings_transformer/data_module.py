import numpy as np
import pytorch_lightning as pl
import logging

from torch.utils.data import DataLoader

from embeddings_transformer.data_processor import UserSequenceDataset
from embeddings_transformer.dataset import collate_fn

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class DataModule(pl.LightningDataModule):

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
        return DataLoader(self.train_data, self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=collate_fn)
