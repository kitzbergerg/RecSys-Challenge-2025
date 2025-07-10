import math
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

from experiments.transformer.constants import MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM


def mask_whole_event(sequence, idx):
    # special token for missing context
    sequence['event_type'][idx] = 3
    sequence['category'][idx] = 3
    sequence['sku'][idx] = 3
    sequence['price'][idx] = 3
    sequence['url'][idx] = 3
    sequence['product_name'][idx] = torch.zeros(TEXT_EMB_DIM, dtype=torch.float32)
    sequence['search_query'][idx] = torch.zeros(TEXT_EMB_DIM, dtype=torch.float32)
    sequence['time_delta'][idx] = -1


class UserSequenceDataset(Dataset):
    """Dataset for user event sequences."""

    def __init__(
            self,
            sequences_df: pd.DataFrame,
            vocab_sizes: Dict[str, int],
            max_seq_length: int = MAX_SEQUENCE_LENGTH,
            disable_masking: bool = False,
    ):
        self.sequences_df = sequences_df
        self.max_seq_length = max_seq_length
        self.disable_masking = disable_masking

        self.client_groups = self.sequences_df.groupby('client_id')
        self.client_ids = list(self.client_groups.groups.keys())

        self._compute_class_stats(vocab_sizes)

    def _compute_class_stats(self, vocab_sizes: Dict[str, int]):
        event_type_counts = self.sequences_df['event_type'].value_counts()
        price_counts = self.sequences_df[self.sequences_df['event_type'].isin([4, 5, 6])]['price'].value_counts()
        cat_counts = self.sequences_df[self.sequences_df['event_type'].isin([4, 5, 6])]['category'].value_counts()
        sku_counts = self.sequences_df[self.sequences_df['event_type'].isin([4, 5, 6])]['sku'].value_counts()
        url_counts = self.sequences_df[self.sequences_df['event_type'] == 7]['url'].value_counts()

        self.class_stats = {
            'category': {
                'learnable': set(cat_counts[cat_counts >= 10].index),
                'common': set(cat_counts[cat_counts >= 100].index),
                'rare': set(cat_counts[cat_counts < 10].index)
            },
            'sku': {
                'learnable': set(sku_counts[sku_counts >= 300].index),
                'common': set(sku_counts[sku_counts >= 500].index),
                'rare': set(sku_counts[sku_counts < 300].index)
            },
            'url': {
                'learnable': set(url_counts[url_counts >= 750].index),
                'common': set(url_counts[url_counts >= 10000].index),
                'rare': set(url_counts[url_counts < 750].index)
            }
        }

        # Compute class weights for event_type
        event_type_weights = 1.0 / np.log1p(event_type_counts + 1)  # log(1 + freq) smoothing
        event_type_weights = event_type_weights / event_type_weights.max()  # Normalize to [0, 1]
        event_type_weight_tensor = torch.ones(vocab_sizes['event_type'])
        for idx, w in event_type_weights.items():
            event_type_weight_tensor[idx] = w

        # Compute class weights for category
        cat_weights = 1.0 / np.log1p(cat_counts + 1)  # log(1 + freq) smoothing
        cat_weights = cat_weights / cat_weights.max()  # Normalize to [0, 1]
        cat_weight_tensor = torch.ones(vocab_sizes['category'])
        for idx, w in cat_weights.items():
            cat_weight_tensor[idx] = w

        # Compute class weights for sku
        sku_weights = 1.0 / np.log1p(sku_counts + 1)  # log(1 + freq) smoothing
        sku_weights = sku_weights / sku_weights.max()  # Normalize to [0, 1]
        sku_weight_tensor = torch.ones(vocab_sizes['sku'])
        for idx, w in sku_weights.items():
            sku_weight_tensor[idx] = w

        # Compute class weights for URL
        url_weights = 1.0 / np.log1p(url_counts + 1)
        url_weights = url_weights / url_weights.max()
        url_weight_tensor = torch.ones(vocab_sizes['url'])
        for idx, w in url_weights.items():
            url_weight_tensor[idx] = w

        self.class_weights = {
            'event_type': event_type_weight_tensor,
            'category': cat_weight_tensor,
            'sku': sku_weight_tensor,
            'url': url_weight_tensor
        }

        # Smooth with sqrt to reduce variance in extremely rare events
        event_type_weights = 1.0 / np.sqrt(event_type_counts + 1)  # add 1 to avoid div-by-zero
        event_type_weights = event_type_weights / event_type_weights.sum()  # normalize to sum to 1
        self.event_type_sampling_probs = event_type_weights.to_dict()  # {event_type_id: prob}

    def __len__(self):
        return len(self.client_ids)

    def sample_mask_position(self, group):
        weights = group['event_type'].map(self.event_type_sampling_probs).fillna(0.0).to_numpy()
        total = weights.sum()

        if total == 0.0:
            print("WARNING: no event type sampling probabilities")
            return np.random.randint(0, len(group))

        probabilities = weights / total
        return np.random.choice(len(group), p=probabilities)

    def __getitem__(self, idx):
        client_id = self.client_ids[idx]
        group = self.client_groups.get_group(client_id).sort_values('timestamp')
        seq_len = len(group)
        assert seq_len <= MAX_SEQUENCE_LENGTH

        # Initialize tensors
        sequence = {
            'client_id': client_id,
            'event_type': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'category': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'sku': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'price': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'url': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'product_name': torch.zeros(MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM, dtype=torch.float32),
            'search_query': torch.zeros(MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM, dtype=torch.float32),
            'time_delta': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.float32),
            'mask': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.bool),  # True for valid positions
        }
        tensor_fields = ['event_type', 'category', 'sku', 'price', 'url', 'time_delta']
        for field in tensor_fields:
            sequence[field][:seq_len] = torch.as_tensor(group[field].values)
        for emb_field in ['product_name', 'search_query']:
            sequence[emb_field][:seq_len] = torch.from_numpy(np.stack(group[emb_field].values))
        sequence['mask'][:seq_len] = True

        targets = {
            'event_type_targets': -1,
            'event_type_mask': False,
            'category_targets': -1,
            'category_mask': False,
            'sku_targets': -1,
            'sku_mask': False,
            'price_targets': -1,
            'price_mask': False,
            'url_targets': -1,
            'url_mask': False,
            'time_targets': -1,
            'time_mask': False,
        }

        if self.disable_masking:
            return sequence, targets

        # Choose a random position to mask
        mask_pos = self.sample_mask_position(group)

        # Mask event type
        original_event_type = sequence['event_type'][mask_pos].item()
        if random.random() < 0.8:
            targets['event_type_targets'] = original_event_type
            targets['event_type_mask'] = True
            sequence['event_type'][mask_pos] = 2

        # For product events, we can also predict category, sku and price
        if original_event_type in [4, 5, 6]:
            is_learnable = sequence['category'][mask_pos].item() in self.class_stats['category']['learnable']
            mask_prob = 0.8 if is_learnable else 0.2
            if random.random() < mask_prob:
                targets['category_targets'] = sequence['category'][mask_pos].item()
                targets['category_mask'] = True
                sequence['category'][mask_pos] = 2
            is_learnable = sequence['sku'][mask_pos].item() in self.class_stats['sku']['learnable']
            mask_prob = 0.75 if is_learnable else 0.2
            if random.random() < mask_prob:
                targets['sku_targets'] = sequence['sku'][mask_pos].item()
                targets['sku_mask'] = True
                sequence['sku'][mask_pos] = 2
            if random.random() < 0.75:
                targets['price_targets'] = sequence['price'][mask_pos].item()
                targets['price_mask'] = True
                sequence['price'][mask_pos] = 2
            if random.random() < 0.95:
                # Randomly remove text embeddings so transformer can't infer type from token data
                sequence['product_name'][mask_pos] = torch.zeros(TEXT_EMB_DIM, dtype=torch.float32)

        # For page visits, we can predict URL
        is_learnable = sequence['url'][mask_pos].item() in self.class_stats['url']['learnable']
        mask_prob = 0.3 if is_learnable else 0.1
        if original_event_type == 7 and random.random() < mask_prob:
            targets['url_targets'] = sequence['url'][mask_pos].item()
            targets['url_mask'] = True
            sequence['url'][mask_pos] = 2

        # TODO: add extra input to model to signal prediction
        if mask_pos != 0 and random.random() < 0.6:
            # As long as there is more than 1 event we can try to predict the time delta
            targets['time_targets'] = math.log(sequence['time_delta'][mask_pos].item())
            targets['time_mask'] = True
            sequence['time_delta'][mask_pos] = -1  # not categorical and 0 is for padding

        if seq_len > 5 and random.random() < 0.3:
            # zero elements around mask so transformer needs to learn general and not just local patterns
            mask_start = random.randint(0, min(seq_len // 6, 4))
            mask_end = random.randint(0, min(seq_len // 6, 4))
            for i in range(max(mask_pos - mask_start, 0), min(mask_pos + mask_end + 1, seq_len)):
                if i != mask_pos:
                    mask_whole_event(sequence, i)

        return sequence, targets


def reconstructive_collate_fn(batch):
    sequences, targets = zip(*batch)  # unzip list of tuples
    batched_sequences = torch.utils.data.default_collate(sequences)

    # Convert dict of scalars into batched tensors
    target_keys = targets[0].keys()
    batched_targets = {
        k: torch.tensor([t[k] for t in targets]) for k in target_keys
    }

    return batched_sequences, batched_targets
