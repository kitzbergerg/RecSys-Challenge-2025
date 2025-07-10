from typing import Dict, Tuple, Union

import torch
from torch import nn

from embeddings_transformer.constants import TEXT_EMB_DIM, MAX_SEQUENCE_LENGTH


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
            output_dim: int = 1024,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_sizes = vocab_sizes

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Categorical embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_sizes[name], embedding_dim, padding_idx=0)
            for name in ['event_type', 'category', 'sku', 'price', 'url']
        })

        # Projected precomputed features
        self.projectors = nn.ModuleDict({
            'product_name': nn.Linear(TEXT_EMB_DIM, embedding_dim),
            'search_query': nn.Linear(TEXT_EMB_DIM, embedding_dim)
        })

        self.time_encoder = TimeIntervalEncoder(embedding_dim=embedding_dim)

        # Combine all feature embeddings
        self.feature_combiner = nn.Sequential(
            nn.Linear(embedding_dim * 8, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )

        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict({
            'event_type': nn.Sequential(
                BottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                BottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, vocab_sizes['event_type'])
            ),
            'price': nn.Sequential(
                BottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                BottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, vocab_sizes['price'])
            ),
            'category': nn.Sequential(
                BottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                BottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, vocab_sizes['category'])
            ),
            'sku': nn.Sequential(
                BottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                BottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, vocab_sizes['sku'])
            ),
            'url': nn.Sequential(
                BottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                BottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, vocab_sizes['url'])
            ),
            'time': nn.Sequential(
                BottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                BottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, 1)
            )
        })

    def embed_inputs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        for key in self.embeddings:
            input_tensor = batch[key]
            assert (input_tensor >= 0).all()
            assert (input_tensor < self.vocab_sizes[key]).all()

        embeddings = [self.embeddings[key](batch[key]) for key in self.embeddings]
        embeddings += [self.projectors[key](batch[key]) for key in self.projectors]
        embeddings.append(self.time_encoder(batch['time_delta']))
        return torch.cat(embeddings, dim=-1)

    def forward(
            self,
            batch: Dict[str, torch.Tensor],
            return_task_predictions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
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

        # Final embedding
        embedding = self.output_projection(pooled)

        if return_task_predictions:
            # Generate predictions for all tasks
            predictions = {}
            for task_name, head in self.heads.items():
                prediction = head(embedding)
                if task_name == 'time':
                    prediction = prediction.squeeze(-1)
                predictions[task_name] = prediction
            return embedding, predictions

        return embedding


class BottleneckBlock(nn.Module):
    def __init__(self, thin_dim: int, wide_dim: int):
        super().__init__()
        self.l1 = nn.Linear(thin_dim, wide_dim)
        self.l2 = nn.Linear(wide_dim, thin_dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(thin_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        x = self.norm(x + residual)
        return x


class TimeIntervalEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.log_embedding = nn.Embedding(64, embedding_dim // 2)
        # Learnable periodic frequencies
        self.periodic_frequencies = nn.Parameter(torch.randn(6) * 0.01)
        self.periodic_projection = nn.Linear(12, embedding_dim // 2)  # 6 freqs * 2 (sin/cos)

        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, time_intervals: torch.Tensor) -> torch.Tensor:
        # Create mask for valid (non-zero) time intervals
        mask = time_intervals > 0
        safe_intervals = torch.where(mask, time_intervals, torch.ones_like(time_intervals))

        # Logarithmic bucketing - handles wide range efficiently
        # max time_delta is 12068170 -- log -> ~16 -- 4x -> 64
        log_values = torch.log(safe_intervals.float() + 1)
        log_bucket_ids = torch.clamp((log_values * 4).long(), 0, 63)
        log_features = self.log_embedding(log_bucket_ids)

        # Learnable periodic features
        seconds = safe_intervals.float().unsqueeze(-1)
        phases = seconds * self.periodic_frequencies.unsqueeze(0).unsqueeze(0)
        periodic_features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        periodic_features = self.periodic_projection(periodic_features)

        # Combine both encoding strategies
        combined = torch.cat([log_features, periodic_features], dim=-1)

        # Apply mask to zero out padding positions
        combined = combined * mask.unsqueeze(-1).float()

        return self.output_norm(combined)
