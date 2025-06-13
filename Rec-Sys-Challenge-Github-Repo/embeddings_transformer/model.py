from dataclasses import asdict
import torch
import torch.nn as nn
from torch import optim, Tensor
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Callable, List
from pathlib import Path
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data_utils.data_dir import DataDir
from embeddings_transformer.data_module import ChurnDataModule
from embeddings_transformer.data_processor import create_data_processing_pipeline
from training_pipeline.metric_calculators import MetricCalculator
from training_pipeline.metrics_containers import MetricContainer
from training_pipeline.task_constructor import TaskConstructor
from training_pipeline.tasks import ChurnTasks


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
            max_seq_length: int = 256,
            output_dim: int = 512,
            dropout: float = 0.1,
    ):
        super().__init__()

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
            'product_name': nn.Linear(16, embedding_dim),
            'search_query': nn.Linear(16, embedding_dim)
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

        self.churn_head = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def embed_inputs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Embed categorical features
        embeddings = [self.embeddings[key](batch[key]) for key in self.embeddings]
        # Project fixed-size vectors
        embeddings += [self.projectors[key](batch[key]) for key in self.projectors]
        # Concatenate along last dim
        return torch.cat(embeddings, dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor], return_embedding: bool = False) -> torch.Tensor:
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

        if return_embedding:
            return embedding

        return self.churn_head(embedding)


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

    # Don't use full dataset for valid, to slow
    dataset_valid, _ = torch.utils.data.random_split(dataset_valid, [0.1, 0.9])

    data = ChurnDataModule(dataset_train, dataset_valid, 64, 4)

    task_constructor = TaskConstructor(data_dir=DataDir(data_dir))
    task = ChurnTasks.CHURN
    task_settings = task_constructor.construct_task(task=task)
    pos_weight = torch.tensor([438490 / 72481], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = TransformerModel(
        vocab_sizes=vocab_sizes,
        learning_rate=1e-4,
        metric_calculator=task_settings.metric_calculator,
        loss_fn=loss_fn,
        metrics_tracker=task_settings.metrics_tracker,
    )

    print("Training model...")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=100,
        callbacks=[
            RichProgressBar(leave=True),
            ModelCheckpoint(monitor="val_auroc", mode="max", save_top_k=1)
        ]
    )

    trainer.fit(model=model, datamodule=data)
    torch.save(model.state_dict(), "../models/transformer.pt")


if __name__ == '__main__':
    train_transformer_model(
        data_dir=Path("../data/original/"),
        sequences_path=Path("../data/sequence/sequences.pkl"),
        vocab_path=Path("../data/sequence/vocabularies.pkl"),
    )
