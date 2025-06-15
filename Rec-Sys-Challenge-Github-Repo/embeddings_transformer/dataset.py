import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

from embeddings_transformer.constants import MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM


class UserSequenceDataset(Dataset):
    """Dataset for user event sequences."""

    def __init__(self, sequences_df: pd.DataFrame, active_clients: np.ndarray,
                 max_seq_length: int = MAX_SEQUENCE_LENGTH):
        self.max_seq_length = max_seq_length

        # Filter to active clients
        self.sequences_df = sequences_df[sequences_df['client_id'].isin(active_clients)].copy()

        # Group by client_id and pre-store list of client_ids
        self.client_groups = self.sequences_df.groupby('client_id')
        self.client_ids = list(self.client_groups.groups.keys())

    def __len__(self):
        return len(self.client_ids)

    def __getitem__(self, idx):
        client_id = self.client_ids[idx]
        group = self.client_groups.get_group(client_id).sort_values('timestamp')

        # Truncate if too long
        if len(group) > self.max_seq_length:
            group = group.iloc[-self.max_seq_length:]

        seq_len = len(group)

        # Initialize tensors
        sequence = {
            'event_type': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'category': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'price': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'url': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'product_name': torch.zeros(MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM, dtype=torch.float32),
            'search_query': torch.zeros(MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM, dtype=torch.float32),
            'time_delta': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.float32),
            'mask': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.bool),  # True for valid positions
        }
        tensor_fields = ['event_type', 'category', 'price', 'url', 'time_delta']
        for field in tensor_fields:
            sequence[field][:seq_len] = torch.as_tensor(group[field].values)
        for emb_field in ['product_name', 'search_query']:
            sequence[emb_field][:seq_len] = torch.from_numpy(np.stack(group[emb_field].values))
        sequence['mask'][:seq_len] = True

        # Choose a random position to mask
        mask_pos = random.randint(0, len(group) - 1)

        targets = {
            'event_type_targets': -1,
            'event_type_mask': False,
            'category_targets': -1,
            'category_mask': False,
            'price_targets': -1,
            'price_mask': False,
            'url_targets': -1,
            'url_mask': False,
            'time_targets': -1,
            'time_mask': False,
        }

        # Mask event type
        original_event_type = sequence['event_type'][mask_pos].item()
        if random.random() < 0.8:
            targets['event_type_targets'] = original_event_type
            targets['event_type_mask'] = True
            sequence['event_type'][mask_pos] = 2

        if original_event_type in [4, 5, 6]:
            if random.random() < 0.5:
                # For product events, we can also predict category
                targets['category_targets'] = sequence['category'][mask_pos].item()
                targets['category_mask'] = True
                sequence['category'][mask_pos] = 2
            if random.random() < 0.5:
                # For product events, we can also predict price
                targets['price_targets'] = sequence['price'][mask_pos].item()
                targets['price_mask'] = True
                sequence['price'][mask_pos] = 2
            if random.random() < 0.7:
                # Randomly remove text embeddings so transformer can't infer type from token data
                sequence['product_name'][mask_pos] = torch.zeros(TEXT_EMB_DIM, dtype=torch.float32)

        if original_event_type == 7 and random.random() < 0.2:
            # For page visits, we can predict URL
            targets['url_targets'] = sequence['url'][mask_pos].item()
            targets['url_mask'] = True
            sequence['url'][mask_pos] = 2

        # TODO: add extra input to model to signal prediction
        if mask_pos != 0 and random.random() < 0.6:
            # As long as there is more than 1 event we can try to predict the time delta
            targets['time_targets'] = math.log(sequence['time_delta'][mask_pos].item())
            targets['time_mask'] = True
            sequence['time_delta'][mask_pos] = -1  # not categorical and 0 is for padding

        if seq_len > 5 and random.random() < 0.2:
            # zero elements around mask so transformer needs to learn general patterns, not just local event around mask
            mask_start = random.randint(0, min(seq_len // 6, 3))
            mask_end = random.randint(0, min(seq_len // 6, 3))
            for i in range(max(mask_pos - mask_start, 0), min(mask_pos + mask_end + 1, seq_len)):
                if i != mask_pos:
                    mask_event(sequence, i)

        return sequence, targets


def mask_event(sequence, idx):
    # special token for missing context
    sequence['event_type'][idx] = 3
    sequence['category'][idx] = 3
    sequence['price'][idx] = 3
    sequence['url'][idx] = 3
    sequence['product_name'][idx] = torch.zeros(TEXT_EMB_DIM, dtype=torch.float32)
    sequence['search_query'][idx] = torch.zeros(TEXT_EMB_DIM, dtype=torch.float32)
    sequence['time_delta'][idx] = -1


def collate_fn(batch):
    sequences, targets = zip(*batch)  # unzip list of tuples
    batched_sequences = torch.utils.data.default_collate(sequences)

    # Convert dict of scalars into batched tensors
    target_keys = targets[0].keys()
    batched_targets = {
        k: torch.tensor([t[k] for t in targets]) for k in target_keys
    }

    return batched_sequences, batched_targets
