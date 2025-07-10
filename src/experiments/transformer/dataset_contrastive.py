import math
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

from experiments.transformer.constants import MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM
from experiments.transformer.dataset import mask_whole_event


def mask_event(sequence, idx):
    if random.random() < 0.3:
        mask_whole_event(sequence, idx)
        return

    event_type = sequence['event_type'][idx]
    mask_prob = 0.7
    if random.random() < mask_prob:
        sequence['event_type'][idx] = 3
    if event_type in [4, 5, 6]:
        for name in ['category', 'sku', 'price']:
            if random.random() < mask_prob:
                sequence[name][idx] = 3
        if random.random() < mask_prob:
            sequence['product_name'][idx] = torch.zeros(TEXT_EMB_DIM, dtype=torch.float32)
    if event_type == 7 and random.random() < mask_prob:
        sequence['url'][idx] = 3
    if event_type == 8 and random.random() < mask_prob:
        sequence['search_query'][idx] = torch.zeros(TEXT_EMB_DIM, dtype=torch.float32)
    if random.random() < mask_prob:
        sequence['time_delta'][idx] = -1


def randomize_sequence(client_id, group, seq_len):
    if seq_len > 2:
        subset_length = random.randint(int(math.sqrt(seq_len)), int(seq_len / 1.5))
        start_idx = random.randint(0, seq_len - subset_length)
        end_idx = start_idx + subset_length
    else:
        subset_length = seq_len
        start_idx = 0
        end_idx = seq_len

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
        'mask': torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.bool)  # True for valid positions
    }
    tensor_fields = ['event_type', 'category', 'sku', 'price', 'url', 'time_delta']
    for field in tensor_fields:
        sequence[field][:subset_length] = torch.as_tensor(group[field].iloc[start_idx:end_idx].values)
    for emb_field in ['product_name', 'search_query']:
        sequence[emb_field][:subset_length] = torch.from_numpy(
            np.stack(group[emb_field].iloc[start_idx:end_idx].values)
        )
    sequence['mask'][:subset_length] = True

    # start slow with randomizing, later on increase probability
    if random.random() < 0.7:
        # Mask up to 70% of valid positions, but at least 1 and at most leave 1 unmasked
        num_to_mask = random.randint(1, max(1, int(subset_length * 0.7)))
        positions_to_mask = random.sample(range(subset_length), num_to_mask)
        for pos in positions_to_mask:
            mask_event(sequence, pos)

    return sequence


class UserSequenceContrastiveDataset(Dataset):
    """Dataset for contrastive learning of user event sequences."""

    def __init__(
            self,
            sequences_df: pd.DataFrame,
            max_seq_length: int = MAX_SEQUENCE_LENGTH,
    ):
        self.sequences_df = sequences_df
        self.max_seq_length = max_seq_length

        self.client_groups = self.sequences_df.groupby('client_id')
        self.client_ids = list(self.client_groups.groups.keys())

    def __len__(self):
        return len(self.client_ids)

    def get_sequence(self, idx):
        client_id = self.client_ids[idx]
        group = self.client_groups.get_group(client_id).sort_values('timestamp')
        seq_len = len(group)
        assert seq_len <= MAX_SEQUENCE_LENGTH

        return client_id, group, seq_len

    def __getitem__(self, idx):
        sequence = self.get_sequence(idx)
        other = self.get_sequence(random.randint(0, self.__len__() - 1))

        anchor = randomize_sequence(*sequence)
        positive = randomize_sequence(*sequence)
        negative = randomize_sequence(*other)

        return anchor, positive, negative


def contrastive_collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    return (
        torch.utils.data.default_collate(anchors),
        torch.utils.data.default_collate(positives),
        torch.utils.data.default_collate(negatives)
    )
