import pytorch_lightning as pl
import logging

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class DataModule(pl.LightningDataModule):

    def __init__(
            self,
            train_data: Dataset,
            val_data: Dataset,
            batch_size: int,
            num_workers: int,
            collate_fn
    ) -> None:
        super().__init__()
        self.train_data, self.val_data = train_data, val_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.collate_fn)
