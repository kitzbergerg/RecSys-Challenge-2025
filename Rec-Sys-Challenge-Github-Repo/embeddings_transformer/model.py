from typing import Dict, Tuple

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

        self.time_encoder = TimeIntervalEncoder(
            max_interval_seconds=30 * 24 * 3600,  # 30 days max
            embedding_dim=embedding_dim
        )

        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Combine all feature embeddings
        self.feature_combiner = nn.Linear(embedding_dim * 7, embedding_dim)

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
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, vocab_sizes['event_type'])
            ),
            'price': nn.Sequential(
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, vocab_sizes['price'])
            ),
            'category': nn.Sequential(
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, vocab_sizes['category'])
            ),
            'url': nn.Sequential(
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, vocab_sizes['url'])
            ),
            'time': nn.Sequential(
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Dropout(dropout),
                TransformerBottleneckBlock(output_dim, 2048),
                nn.Linear(output_dim, 1)
            )
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
        # Add time interval encoding
        embeddings.append(self.time_encoder(batch['time_delta']))
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
                prediction = head(embedding)
                if task_name == 'time':
                    prediction = prediction.squeeze(-1)
                predictions[task_name] = prediction
            return embedding, predictions

        return embedding


class TransformerBottleneckBlock(nn.Module):
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
    """Encodes time intervals using multiple complementary approaches"""

    def __init__(self, max_interval_seconds: int, embedding_dim: int):
        super().__init__()
        self.max_interval_seconds = max_interval_seconds

        # Strategy 1: Logarithmic bucketing for wide range handling
        self.log_buckets = nn.Embedding(50, embedding_dim // 4)

        # Strategy 2: Periodic encoding for capturing daily/weekly patterns
        self.periodic_encoder = nn.Linear(4, embedding_dim // 4)  # sin/cos for day/week

        # Strategy 3: Direct encoding for fine-grained timing
        self.direct_encoder = nn.Linear(1, embedding_dim // 4)

        # Strategy 4: Categorical bucketing for common intervals
        self.categorical_buckets = nn.Embedding(20, embedding_dim // 4)

        # Combine all time representations
        self.combiner = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, time_intervals: torch.Tensor) -> torch.Tensor:
        # Handle padding (where time_interval is 0)
        mask = time_intervals > 0
        safe_intervals = torch.where(mask, time_intervals, torch.ones_like(time_intervals))

        # TODO: get statistics on time for better defaults

        # Strategy 1: Logarithmic bucketing
        # This handles the wide range from seconds to months
        log_values = torch.log(safe_intervals.float() + 1)  # +1 to avoid log(0)
        log_bucket_ids = torch.clamp(
            (log_values * 5).long(),  # Scale factor to spread across buckets
            0, 49
        )
        log_features = self.log_buckets(log_bucket_ids)

        # Strategy 2: Periodic encoding for daily/weekly patterns
        seconds = safe_intervals.float()
        day_seconds = 24 * 3600
        week_seconds = 7 * day_seconds
        periodic_features = torch.stack([
            torch.sin(2 * torch.pi * seconds / day_seconds),  # Daily pattern
            torch.cos(2 * torch.pi * seconds / day_seconds),
            torch.sin(2 * torch.pi * seconds / week_seconds),  # Weekly pattern
            torch.cos(2 * torch.pi * seconds / week_seconds)
        ], dim=-1)
        periodic_features = self.periodic_encoder(periodic_features)

        # Strategy 3: Direct normalized encoding
        normalized_time = torch.clamp(safe_intervals.float() / self.max_interval_seconds, 0, 1).unsqueeze(-1)
        direct_features = self.direct_encoder(normalized_time)

        # Strategy 4: Categorical bucketing for common intervals
        # Define meaningful time buckets (in seconds)
        bucket_boundaries = torch.tensor([
            0, 60, 300, 900, 1800, 3600,  # 1min, 5min, 15min, 30min, 1hour
            7200, 14400, 28800, 86400,  # 2h, 4h, 8h, 1day
            172800, 259200, 604800,  # 2days, 3days, 1week
            1209600, 2592000, 7776000,  # 2weeks, 1month, 3months
            15552000, 31536000  # 6months, 1year
        ], device=time_intervals.device)

        # Find which bucket each interval belongs to
        bucket_ids = torch.searchsorted(bucket_boundaries, safe_intervals.float())
        bucket_ids = torch.clamp(bucket_ids, 0, 19)
        categorical_features = self.categorical_buckets(bucket_ids)

        # Combine all strategies
        combined = torch.cat([
            log_features, periodic_features,
            direct_features, categorical_features
        ], dim=-1)

        # Apply mask to zero out padding positions
        combined = combined * mask.unsqueeze(-1).float()

        return self.combiner(combined)
