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

        # Initialize tensors
        sequence = {
            'client_id': client_id,
            'event_type': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'category': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'price': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'url': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.long),
            'product_name': torch.zeros(MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM, dtype=torch.float32),
            'search_query': torch.zeros(MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM, dtype=torch.float32),
            'time_since_last_event': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.float32),
            'mask': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.bool),  # True for valid positions
            'seq_len': len(group),
        }

        for i, (_, event) in enumerate(group.iterrows()):
            sequence['event_type'][i] = event['event_type']
            sequence['category'][i] = event['category']
            sequence['price'][i] = event['price']
            sequence['url'][i] = event['url']
            sequence['product_name'][i] = torch.from_numpy(event['product_name'])
            sequence['search_query'][i] = torch.from_numpy(event['search_query'])
            sequence['mask'][i] = True

            if i > 0:
                time_delta = group.iloc[i]['timestamp'] - group.iloc[i - 1]['timestamp']
                sequence['time_since_last_event'][i] = max(time_delta.seconds, 1)

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
        targets['event_type_targets'] = original_event_type
        targets['event_type_mask'] = True
        sequence['event_type'][mask_pos] = 2

        if original_event_type in [3, 4, 5] and random.random() < 0.3:
            # For product events, we can also predict category
            targets['category_targets'] = sequence['category'][mask_pos].item()
            targets['category_mask'] = True
            sequence['category'][mask_pos] = 2

        if original_event_type in [3, 4, 5] and random.random() < 0.3:
            # For product events, we can also predict price
            targets['price_targets'] = sequence['price'][mask_pos].item()
            targets['price_mask'] = True
            sequence['price'][mask_pos] = 2

        if original_event_type == 6 and random.random() < 0.2:
            # For page visits, we can predict URL
            targets['url_targets'] = sequence['url'][mask_pos].item()
            targets['url_mask'] = True
            sequence['url'][mask_pos] = 2

        # TODO: add extra input to model to signal prediction
        if mask_pos != 0 and random.random() < 0.7:
            # As long as there is more than 1 event we can try to predict the time delta
            targets['time_targets'] = math.log(sequence['time_since_last_event'][mask_pos].item())
            targets['time_mask'] = True
            sequence['time_since_last_event'][mask_pos] = -1  # not categorical and 0 is for padding

        # TODO: add other tasks more aligned with goal e.g.
        #  - time till next purchase
        #  - better masking (span masking, category masking i.e. mask all events for category)

        return sequence, targets


def collate_fn(batch):
    sequences, targets = zip(*batch)  # unzip list of tuples
    batched_sequences = torch.utils.data.default_collate(sequences)

    # Convert dict of scalars into batched tensors
    target_keys = targets[0].keys()
    batched_targets = {
        k: torch.tensor([t[k] for t in targets]) for k in target_keys
    }

    return batched_sequences, batched_targets
