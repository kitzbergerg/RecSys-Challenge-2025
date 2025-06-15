from dataclasses import asdict
import torch
from torch import Tensor
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

from embeddings_transformer.data_module import DataModule
from embeddings_transformer.data_processor import create_data_processing_pipeline
from embeddings_transformer.metrics import MultiClassMetricCalculator, MultiTaskLoss
from embeddings_transformer.model import UserBehaviorTransformer


class TransformerModel(pl.LightningModule):
    def __init__(
            self,
            vocab_sizes: Dict[str, int],
            class_weights: Dict[str, Tensor],
            learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.net = UserBehaviorTransformer(vocab_sizes=vocab_sizes)
        self.loss_calculator = MultiTaskLoss(class_weights=class_weights)
        self.metric_calculator = {
            'event_type': MultiClassMetricCalculator(vocab_sizes['event_type']),
            'category': MultiClassMetricCalculator(vocab_sizes['category']),
            'price': MultiClassMetricCalculator(vocab_sizes['price']),
            'url': MultiClassMetricCalculator(vocab_sizes['url']),
        }

    def forward(self, x) -> Tensor:
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               min_lr=1e-6)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss',
                'interval': 'epoch',
                'frequency': 2
            },
            'monitor': 'val_total_loss'
        }

    def setup(self, stage):
        for c in self.metric_calculator.values():
            c.to(self.device)

    def training_step(self, batch, batch_idx):
        """
        Training step that handles multi-task loss computation.
        """
        x, y = batch
        embeddings, predictions = self.net(x, return_task_predictions=True)
        losses = self.loss_calculator.compute_loss(predictions, y)

        for task_name, loss_value in losses.items():
            self.log(f'train_{task_name}_loss', loss_value, on_step=True, prog_bar=task_name == 'total', logger=True)

        return losses['total']

    def validation_step(self, batch, batch_idx):
        """
        Validation step that computes accuracy for each active task.
        """
        x, y = batch
        embeddings, predictions = self.net(x, return_task_predictions=True)
        losses = self.loss_calculator.compute_loss(predictions, y)

        for task_name, loss_value in losses.items():
            self.log(f'val_{task_name}_loss', loss_value, prog_bar=task_name in ['total', 'time'], logger=True)

        for key, calc in self.metric_calculator.items():
            valid_indices = y[f'{key}_mask']
            calc.update(predictions[f'{key}'][valid_indices], y[f'{key}_targets'][valid_indices])

        return losses['total']

    def on_validation_epoch_end(self) -> None:
        for metric_class, metric in self.metric_calculator.items():
            metric_container = metric.compute()

            for metric_name, metric_val in asdict(metric_container).items():
                self.log(f"{metric_class}_{metric_name}", metric_val, prog_bar=True, logger=True)


def train_transformer_model(
        data_dir: Path,
        output_dir: Path,
        sequences_path: Path,
        vocab_path: Path,
        ckpt_path: Optional[Path]
):
    # Load relevant clients
    relevant_clients = np.load(data_dir / "input" / "relevant_clients.npy")

    # Create dataset and vocab sizes
    print("Loading data...")
    dataset, vocab_sizes = create_data_processing_pipeline(
        data_dir=data_dir,
        relevant_clients=relevant_clients,
        vocab_path=vocab_path,
        sequences_path=sequences_path,
        rebuild_vocab=False
    )
    class_weights = dataset.class_weights

    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [0.9, 0.1])
    data = DataModule(dataset_train, dataset_valid, 128, 8)

    print("Training model...")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=100,
        check_val_every_n_epoch=2,
        # overfit_batches=100,
        callbacks=[
            RichProgressBar(leave=True),
            ModelCheckpoint(every_n_epochs=3, save_top_k=-1, save_weights_only=False),
        ],
    )
    if ckpt_path is None:
        model = TransformerModel(vocab_sizes=vocab_sizes, class_weights=class_weights)
        trainer.fit(model=model, datamodule=data)
    else:
        model = TransformerModel.load_from_checkpoint(ckpt_path)
        trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)
    torch.save(model.net.state_dict(), output_dir / "transformer.pt")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sequences-file", type=str, required=True)
    parser.add_argument("--vocab-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=False)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sequences_path = Path(args.sequences_file)
    vocab_path = Path(args.vocab_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_path is not None:
        ckpt_path = Path(args.checkpoint_path)
    else:
        ckpt_path = None

    train_transformer_model(
        data_dir=data_dir,  # "../data/original"
        output_dir=output_dir,  # "../models"
        sequences_path=sequences_path,  # "../data/sequence/sequences.pkl"
        vocab_path=vocab_path,  # "../data/sequence/vocabularies.pkl",
        ckpt_path=ckpt_path
    )
