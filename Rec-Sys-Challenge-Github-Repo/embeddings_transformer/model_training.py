from dataclasses import asdict
import torch
from torch import optim, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional
from pathlib import Path
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

from embeddings_transformer.data_module import DataModule
from embeddings_transformer.data_processor import create_data_processing_pipeline
from embeddings_transformer.dataset import reconstructive_collate_fn
from embeddings_transformer.dataset_contrastive import contrastive_collate_fn
from embeddings_transformer.metrics import MultiClassMetricCalculator, MultiTaskLoss
from embeddings_transformer.model import UserBehaviorTransformer


class TransformerModel(pl.LightningModule):
    def __init__(
            self,
            vocab_sizes: Dict[str, int],
            class_weights: Dict[str, Tensor],
            learning_rate: float = 1e-4,
            training_mode: str = "reconstruction",  # "reconstruction" or "contrastive"
            contrastive_temperature: float = 0.1,
            contrastive_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.training_mode = training_mode
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight

        self.net = UserBehaviorTransformer(vocab_sizes=vocab_sizes)
        self.loss_calculator = MultiTaskLoss(class_weights=class_weights)
        self.metric_calculator = {
            'event_type': MultiClassMetricCalculator(vocab_sizes['event_type']),
            'category': MultiClassMetricCalculator(vocab_sizes['category']),
            'sku': MultiClassMetricCalculator(vocab_sizes['sku']),
            'price': MultiClassMetricCalculator(vocab_sizes['price']),
            'url': MultiClassMetricCalculator(vocab_sizes['url']),
        }

    def forward(self, x) -> Tensor:
        return self.net(x)

    def set_training_mode(self, mode: str):
        """Switch between 'reconstruction' and 'contrastive' training modes"""
        assert mode in ['reconstruction', 'contrastive'], f"Invalid mode: {mode}"
        self.training_mode = mode
        print(f"Switched to {mode} training mode")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-6, patience=5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss_total' if self.training_mode == 'reconstruction' else 'val_contrastive_loss',
                'interval': 'epoch',
                'frequency': 2
            },
            'monitor': 'val_loss_total' if self.training_mode == 'reconstruction' else 'val_contrastive_loss'
        }

    def setup(self, stage):
        for c in self.metric_calculator.values():
            c.to(self.device)

    def compute_contrastive_loss(self, anchor_emb: Tensor, positive_emb: Tensor, negative_emb: Tensor) -> Tensor:
        """
        Compute contrastive loss using cosine similarity and temperature scaling.

        Args:
            anchor_emb: [batch_size, embedding_dim]
            positive_emb: [batch_size, embedding_dim]
            negative_emb: [batch_size, embedding_dim]
        """
        # Normalize embeddings
        anchor_emb = F.normalize(anchor_emb, dim=1)
        positive_emb = F.normalize(positive_emb, dim=1)
        negative_emb = F.normalize(negative_emb, dim=1)

        # Compute similarities
        pos_sim = torch.sum(anchor_emb * positive_emb, dim=1) / self.contrastive_temperature  # [batch_size]
        neg_sim = torch.sum(anchor_emb * negative_emb, dim=1) / self.contrastive_temperature  # [batch_size]

        # Contrastive loss: -log(exp(pos_sim) / (exp(pos_sim) + exp(neg_sim)))
        # This is equivalent to: -pos_sim + log(exp(pos_sim) + exp(neg_sim))
        # Using logsumexp for numerical stability
        logits = torch.stack([pos_sim, neg_sim], dim=1)  # [batch_size, 2]
        labels = torch.zeros(anchor_emb.size(0), dtype=torch.long, device=anchor_emb.device)  # positive is index 0

        loss = F.cross_entropy(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step that handles both reconstruction and contrastive learning.
        """
        if self.training_mode == "reconstruction":
            return self._reconstruction_training_step(batch, batch_idx)
        elif self.training_mode == "contrastive":
            return self._contrastive_training_step(batch, batch_idx)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

    def _reconstruction_training_step(self, batch, batch_idx):
        """Original reconstruction training step"""
        x, y = batch
        embeddings, predictions = self.net(x, return_task_predictions=True)
        losses = self.loss_calculator.compute_loss(predictions, y)

        for task_name, loss_value in losses.items():
            self.log(f'train_loss_{task_name}', loss_value, on_step=True,
                     prog_bar=task_name == 'total', logger=True)

        return losses['total']

    def _contrastive_training_step(self, batch, batch_idx):
        """Contrastive learning training step"""
        anchor_batch, positive_batch, negative_batch = batch

        # Get embeddings for each batch
        anchor_emb = self.net(anchor_batch)
        positive_emb = self.net(positive_batch)
        negative_emb = self.net(negative_batch)

        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(anchor_emb, positive_emb, negative_emb)
        total_loss = self.contrastive_weight * contrastive_loss

        # Log metrics
        self.log('train_contrastive_loss', contrastive_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train_loss_total', total_loss, on_step=True, prog_bar=False, logger=True)

        # Compute similarity metrics for monitoring
        with torch.no_grad():
            anchor_norm = F.normalize(anchor_emb, dim=1)
            positive_norm = F.normalize(positive_emb, dim=1)
            negative_norm = F.normalize(negative_emb, dim=1)

            pos_similarity = torch.mean(torch.sum(anchor_norm * positive_norm, dim=1))
            neg_similarity = torch.mean(torch.sum(anchor_norm * negative_norm, dim=1))

            self.log('train_pos_similarity', pos_similarity, on_step=True, logger=True)
            self.log('train_neg_similarity', neg_similarity, on_step=True, logger=True)
            self.log('train_similarity_gap', pos_similarity - neg_similarity, on_step=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step that handles both modes.
        """
        if self.training_mode == "reconstruction":
            return self._reconstruction_validation_step(batch, batch_idx)
        elif self.training_mode == "contrastive":
            return self._contrastive_validation_step(batch, batch_idx)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

    def _reconstruction_validation_step(self, batch, batch_idx):
        """Original reconstruction validation step"""
        x, y = batch
        embeddings, predictions = self.net(x, return_task_predictions=True)
        losses = self.loss_calculator.compute_loss(predictions, y)

        for task_name, loss_value in losses.items():
            self.log(f'val_loss_{task_name}', loss_value,
                     prog_bar=task_name in ['total', 'time'], logger=True)

        for key, calc in self.metric_calculator.items():
            valid_indices = y[f'{key}_mask']
            if valid_indices.sum() > 0:  # Only update if there are valid samples
                calc.update(predictions[f'{key}'][valid_indices], y[f'{key}_targets'][valid_indices])

        return losses['total']

    def _contrastive_validation_step(self, batch, batch_idx):
        """Contrastive learning validation step"""
        anchor_batch, positive_batch, negative_batch = batch

        # Get embeddings for each batch
        anchor_emb = self.net(anchor_batch)
        positive_emb = self.net(positive_batch)
        negative_emb = self.net(negative_batch)

        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(anchor_emb, positive_emb, negative_emb)
        total_loss = self.contrastive_weight * contrastive_loss

        # Log metrics
        self.log('val_contrastive_loss', contrastive_loss, prog_bar=True, logger=True)
        self.log('val_loss_total', total_loss, prog_bar=False, logger=True)

        # Compute similarity metrics
        anchor_norm = F.normalize(anchor_emb, dim=1)
        positive_norm = F.normalize(positive_emb, dim=1)
        negative_norm = F.normalize(negative_emb, dim=1)

        pos_similarity = torch.mean(torch.sum(anchor_norm * positive_norm, dim=1))
        neg_similarity = torch.mean(torch.sum(anchor_norm * negative_norm, dim=1))

        self.log('val_pos_similarity', pos_similarity, logger=True)
        self.log('val_neg_similarity', neg_similarity, logger=True)
        self.log('val_similarity_gap', pos_similarity - neg_similarity, prog_bar=True, logger=True)

        return total_loss

    def on_validation_epoch_end(self) -> None:
        """Only compute reconstruction metrics in reconstruction mode"""
        if self.training_mode == "reconstruction":
            for metric_class, metric in self.metric_calculator.items():
                metric_container = metric.compute()
                for metric_name, metric_val in asdict(metric_container).items():
                    self.log(f"{metric_name}_{metric_class}", metric_val, prog_bar=True, logger=True)


def train_transformer_model(
        data_dir: Path,
        sequences_path: Path,
        ckpt_path: Optional[Path],
        task: Optional[str]
):
    if not task:
        task = 'reconstruction'

    # Create dataset and vocab sizes
    print("Loading data...")
    dataset, vocab_sizes = create_data_processing_pipeline(
        data_dir=data_dir,
        sequences_path=sequences_path,
        task=task,
    )

    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [0.9, 0.1],
                                                                 generator=torch.Generator().manual_seed(42))
    data = DataModule(dataset_train, dataset_valid, 128, 8,
                      reconstructive_collate_fn if task == 'reconstruction' else contrastive_collate_fn)

    print("Training model...")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=100,
        check_val_every_n_epoch=1,
        # overfit_batches=100,
        callbacks=[
            RichProgressBar(leave=True),
            ModelCheckpoint(every_n_epochs=2, save_top_k=-1, save_weights_only=False),
        ],
    )
    if ckpt_path is None:
        assert task == 'reconstruction'
        model = TransformerModel(vocab_sizes=vocab_sizes, class_weights=dataset.class_weights, training_mode=task)
        trainer.fit(model=model, datamodule=data)
    else:
        model = TransformerModel.load_from_checkpoint(ckpt_path)
        model.set_training_mode(task)
        trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sequences-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=False)
    parser.add_argument("--task", type=str, required=False)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sequences_path = Path(args.sequences_path)
    if args.checkpoint_path is not None:
        ckpt_path = Path(args.checkpoint_path)
    else:
        ckpt_path = None
    task = args.task

    train_transformer_model(
        data_dir=data_dir,
        sequences_path=sequences_path,
        ckpt_path=ckpt_path,
        task=task,
    )
