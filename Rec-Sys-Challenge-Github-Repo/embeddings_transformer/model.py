from dataclasses import asdict
import torch
import torch.nn as nn
from torch import optim, Tensor
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Callable, List, Tuple, Optional
from pathlib import Path
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

from embeddings_transformer.constants import MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM
from embeddings_transformer.data_module import DataModule
from embeddings_transformer.data_processor import create_data_processing_pipeline
from embeddings_transformer.metrics import MultiClassMetricCalculator, MultiTaskLoss


class UserBehaviorTransformer(nn.Module):
    """
    Lightweight Transformer for generating user behavior embeddings.
    """

    def __init__(
            self,
            vocab_sizes: Dict[str, int],
            embedding_dim: int = 256,
            num_layers: int = 4,
            num_heads: int = 8,
            max_seq_length: int = MAX_SEQUENCE_LENGTH,
            output_dim: int = 512,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_sizes = vocab_sizes

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Categorical embeddings
        self.embeddings = nn.ModuleDict({
            'event_type': nn.Embedding(vocab_sizes['event_type'], embedding_dim, padding_idx=0),
            'category': nn.Embedding(vocab_sizes['category'], embedding_dim, padding_idx=0),
            'price': nn.Embedding(vocab_sizes['price'], embedding_dim, padding_idx=0),
            'url': nn.Embedding(vocab_sizes['url'], embedding_dim, padding_idx=0)
        })

        # Projected precomputed features
        self.projectors = nn.ModuleDict({
            'product_name': nn.Linear(TEXT_EMB_DIM, embedding_dim),
            'search_query': nn.Linear(TEXT_EMB_DIM, embedding_dim)
        })

        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Combine all feature embeddings
        self.feature_combiner = nn.Linear(embedding_dim * 6, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.heads = nn.ModuleDict({
            'event_type': nn.Sequential(
                nn.Linear(output_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, vocab_sizes['event_type'])
            ),
            'category': nn.Sequential(
                nn.Linear(output_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, vocab_sizes['category'])
            ),
            'url': nn.Sequential(
                nn.Linear(output_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, vocab_sizes['url'])
            )
            # TODO: text embeddings
        })

    def embed_inputs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        for key in self.embeddings:
            input_tensor = batch[key]
            assert (input_tensor >= 0).all()
            assert (input_tensor < self.vocab_sizes[key]).all()

        # Embed categorical features
        embeddings = [self.embeddings[key](batch[key]) for key in self.embeddings]
        # Project fixed-size vectors
        embeddings += [self.projectors[key](batch[key]) for key in self.projectors]
        # Concatenate along last dim
        return torch.cat(embeddings, dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor], return_task_predictions: bool = False) \
            -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T = batch['event_type'].shape
        device = batch['event_type'].device

        # Feature embedding and combination
        x = self.embed_inputs(batch)
        x = self.feature_combiner(x)

        # Add positional encoding
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x += self.position_embedding(pos_ids)

        x = self.layer_norm(x)

        # Create attention mask: True for pad tokens to ignore
        attn_mask = ~batch['mask']

        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Masked mean pooling
        mask = batch['mask'].unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        embedding = self.output_projection(pooled)
        if return_task_predictions:
            # Generate predictions for all tasks
            predictions = {}
            for task_name, head in self.heads.items():
                predictions[task_name] = head(embedding)
            return embedding, predictions

        return embedding


class TransformerModel(pl.LightningModule):
    def __init__(
            self,
            vocab_sizes: Dict[str, int],
            learning_rate: float = 1e-4,
            task_weights: Optional[Dict[str, float]] = None
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.net = UserBehaviorTransformer(vocab_sizes=vocab_sizes)
        self.loss_calculator = MultiTaskLoss(task_weights)
        self.validation_metrics = {
            'event_type': [],
            'category': [],
            'url': []
        }
        self.metric_calculator = {
            'event_type': MultiClassMetricCalculator(vocab_sizes['event_type']),
        }

    def forward(self, x) -> Tensor:
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def setup(self, stage):
        for c in self.metric_calculator.values():
            c.to(self.device)

    def training_step(self, batch, batch_idx):
        """
        Training step that handles multi-task loss computation.
        """
        x, y = batch

        # Get shared embeddings and task predictions
        embeddings, predictions = self.net(x, return_task_predictions=True)

        # Compute multi-task loss
        losses = self.loss_calculator.compute_loss(predictions, y)

        # Log individual task losses for monitoring
        for task_name, loss_value in losses.items():
            if task_name != 'total':
                self.log(f'train_{task_name}_loss', loss_value, on_step=True, prog_bar=False, logger=True)

        # Log total loss
        self.log('train_loss', losses['total'], on_step=True, prog_bar=True, logger=True)

        return losses['total']

    def validation_step(self, batch, batch_idx):
        """
        Validation step that computes accuracy for each active task.
        """
        x, y = batch
        embeddings, predictions = self.net(x, return_task_predictions=True)
        losses = self.loss_calculator.compute_loss(predictions, y)

        # Compute accuracy for each task where we have targets
        accuracies = {}

        # Event type accuracy
        event_preds = torch.argmax(predictions['event_type'], dim=1)
        event_acc = (event_preds == y['event_type_targets']).float().mean()
        accuracies['event_type'] = event_acc
        self.metric_calculator['event_type'].update(predictions['event_type'], y['event_type_targets'])

        # Category accuracy (only for valid targets)
        valid_indices = y['category_mask']
        if valid_indices.sum() > 0:
            cat_preds = torch.argmax(predictions['category'][valid_indices], dim=1)
            cat_acc = (cat_preds == y['category_targets'][valid_indices]).float().mean()
            accuracies['category'] = cat_acc

        # URL accuracy
        valid_indices = y['url_mask']
        if valid_indices.sum() > 0:
            url_preds = torch.argmax(predictions['url'][valid_indices], dim=1)
            url_acc = (url_preds == y['url_targets'][valid_indices]).float().mean()
            accuracies['url'] = url_acc

        # Log validation metrics
        self.log('val_loss', losses['total'], prog_bar=True, logger=True)
        for task_name, acc in accuracies.items():
            self.log(f'val_{task_name}_acc', acc, prog_bar=True, logger=True)

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

    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [0.9, 0.1])

    data = DataModule(dataset_train, dataset_valid, 64, 4)

    model = TransformerModel(
        vocab_sizes=vocab_sizes,
    )

    print("Training model...")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=100,
        overfit_batches=500,
        callbacks=[
            RichProgressBar(leave=True),
            ModelCheckpoint(monitor="event_type_val_auroc", mode="max", save_top_k=1)
        ],
    )

    trainer.fit(model=model, datamodule=data)
    torch.save(model.state_dict(), output_dir / "transformer.pt")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sequences-file", type=str, required=True)
    parser.add_argument("--vocab-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sequences_path = Path(args.sequences_file)
    vocab_path = Path(args.vocab_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transformer_model(
        data_dir=data_dir,  # "../data/original"
        output_dir=output_dir,  # "../models"
        sequences_path=sequences_path,  # "../data/sequence/sequences.pkl"
        vocab_path=vocab_path  # "../data/sequence/vocabularies.pkl",
    )
