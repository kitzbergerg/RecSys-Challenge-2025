from dataclasses import asdict

import torch
import torch.nn as nn
from torch import optim, Tensor
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Callable, List
from pathlib import Path
from pytorch_lightning.callbacks import RichProgressBar

from data_utils.data_dir import DataDir
from embeddings_transformer.data_module import ChurnDataModule
from embeddings_transformer.data_processor import create_data_processing_pipeline
from training_pipeline.constants import LEARNING_RATE
from training_pipeline.metric_calculators import MetricCalculator
from training_pipeline.metrics_containers import MetricContainer
from training_pipeline.task_constructor import TaskConstructor
from training_pipeline.tasks import ChurnTasks


class UserBehaviorTransformer(pl.LightningModule):
    """
    Transformer model for learning universal user behavioral embeddings
    from sequential interaction data.
    """

    def __init__(
            self,
            vocab_sizes: Dict[str, int],
            embedding_dim: int = 256,
            num_layers: int = 6,
            num_heads: int = 8,
            max_seq_length: int = 256,
            output_dim: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.output_dim = output_dim

        # Embeddings for categorical features
        self.event_type_embedding = nn.Embedding(vocab_sizes['event_type'], embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(vocab_sizes['category'], embedding_dim, padding_idx=0)
        self.url_embedding = nn.Embedding(vocab_sizes['url'], embedding_dim, padding_idx=0)
        self.price_embedding = nn.Embedding(vocab_sizes['price'], embedding_dim, padding_idx=0)

        # Projections for pre-computed embeddings
        self.product_name_proj = nn.Linear(16, embedding_dim)
        self.search_query_proj = nn.Linear(16, embedding_dim)

        # Learnable positional embedding
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Feature combination layer
        self.feature_combiner = nn.Linear(embedding_dim * 6, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Output projection to create universal embeddings
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Task-specific head for churn prediction
        self.churn_head = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = batch['event_type'].shape
        device = batch['event_type'].device

        # Get embeddings
        event_emb = self.event_type_embedding(batch['event_type'])
        category_emb = self.category_embedding(batch['category'])
        price_emb = self.price_embedding(batch['price'])
        url_emb = self.url_embedding(batch['url'])

        # Project pre-computed embeddings
        product_name_emb = self.product_name_proj(batch['product_name'])
        search_query_emb = self.search_query_proj(batch['search_query'])

        # Positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        # Combine all embeddings
        combined_features = torch.cat([
            event_emb, category_emb, price_emb, url_emb,
            product_name_emb, search_query_emb
        ], dim=-1)

        # Project to embedding dimension and add positional encoding
        combined_embedding = self.feature_combiner(combined_features) + pos_emb
        combined_embedding = self.layer_norm(combined_embedding)

        # Create attention mask (True for positions to ignore)
        attention_mask = ~batch['mask']

        # Apply transformer
        transformer_output = self.transformer(
            combined_embedding,
            src_key_padding_mask=attention_mask
        )

        # Global pooling - mean of non-padded positions
        mask_expanded = batch['mask'].unsqueeze(-1).float()
        pooled = (transformer_output * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

        # Generate universal user embedding
        user_embedding = self.output_projection(pooled)

        # Apply churn prediction head
        return self.churn_head(user_embedding)


class TransformerModel(pl.LightningModule):
    def __init__(
            self,
            vocab_sizes,
            learning_rate: float,
            metric_calculator: MetricCalculator,
            loss_fn: Callable[[Tensor, Tensor], Tensor],
            metrics_tracker: List[MetricContainer],
    ) -> None:
        super().__init__()

        torch.manual_seed(1278)
        self.learning_rate = learning_rate
        self.net = UserBehaviorTransformer(
            vocab_sizes=vocab_sizes
        )
        self.metric_calculator = metric_calculator
        self.loss_fn = loss_fn
        self.metrics_tracker = metrics_tracker

    def forward(self, x) -> Tensor:
        return self.net(x)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage):
        self.metric_calculator.to(self.device)

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        self.metric_calculator.update(
            predictions=preds,
            targets=y.long(),
        )

    def on_validation_epoch_end(self) -> None:
        metric_container = self.metric_calculator.compute()

        for metric_name, metric_val in asdict(metric_container).items():
            self.log(
                metric_name,
                metric_val,
                prog_bar=True,
                logger=True,
            )

        self.metrics_tracker.append(metric_container)


def train_transformer_model(
        data_dir: Path,
        sequences_path: Path,
        vocab_path: Path,
        max_epochs: int = 3,
        batch_size: int = 16,
):
    # Load relevant clients
    relevant_clients = np.load(data_dir / "input" / "relevant_clients.npy")

    # Create dataset and vocab sizes
    print("Loading data...")
    dataset_train, dataset_valid, vocab_sizes = create_data_processing_pipeline(
        data_dir=data_dir,
        relevant_clients=relevant_clients,
        vocab_path=vocab_path,
        sequences_path=sequences_path,
        rebuild_vocab=False
    )

    data = ChurnDataModule(dataset_train, dataset_valid, batch_size, 4)

    task_constructor = TaskConstructor(data_dir=DataDir(data_dir))
    task = ChurnTasks.CHURN
    task_settings = task_constructor.construct_task(task=task)
    model = TransformerModel(
        vocab_sizes=vocab_sizes,
        learning_rate=LEARNING_RATE,
        metric_calculator=task_settings.metric_calculator,
        loss_fn=task_settings.loss_fn,
        metrics_tracker=task_settings.metrics_tracker,
    )

    print("Training model...")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=max_epochs,
        callbacks=RichProgressBar(leave=True),
        log_every_n_steps=5000,
    )

    trainer.fit(model=model, datamodule=data)
    torch.save(model.state_dict(), "../models/transformer.pt")


if __name__ == '__main__':
    train_transformer_model(
        data_dir=Path("../data/original/"),
        sequences_path=Path("../data/sequence/sequences.pkl"),
        vocab_path=Path("../data/sequence/vocabularies.pkl"),
        max_epochs=10,
        batch_size=64
    )
