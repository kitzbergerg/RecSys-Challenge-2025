import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from pathlib import Path
import pickle
from collections import defaultdict
import re

from training_pipeline.target_data import TargetData


class EventSequenceProcessor:
    """
    Processes raw event data into sequences suitable for transformer training.
    Handles vocabulary creation, OOV tokens, and sequence generation.
    """

    def __init__(
            self,
            max_seq_length: int = 256,
            max_url_vocab: int = 50000,  # Limit URL vocabulary to manageable size
            max_category_vocab: int = 10000,
            min_url_count: int = 5,  # Only include URLs that appear at least 5 times
            min_category_count: int = 2
    ):
        self.max_seq_length = max_seq_length
        self.max_url_vocab = max_url_vocab
        self.max_category_vocab = max_category_vocab
        self.min_url_count = min_url_count
        self.min_category_count = min_category_count

        # Event type mapping
        self.event_types = {
            'product_buy': 1,
            'add_to_cart': 2,
            'remove_from_cart': 3,
            'page_visit': 4,
            'search_query': 5
        }

        # Special tokens
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1
        self.MASK_TOKEN = 2

        # Vocabularies (will be built from data)
        self.url_vocab = {}
        self.category_vocab = {}
        self.price_vocab = {}
        self.vocab_built = False

    def parse_embedding_string(self, emb_str: str) -> np.ndarray:
        if isinstance(emb_str, str):
            numbers = re.findall(r'\d+', emb_str)
            emb_array = np.array(numbers, dtype=np.float32)
        else:
            emb_array = np.array(emb_str, dtype=np.float32)

        assert len(emb_array) == 16
        return emb_array

    def build_vocabularies(self, data_dir: Path):
        """Build vocabularies from the training data."""
        print("Building vocabularies...")

        # URL vocabulary from page_visit events
        page_visit_file = data_dir / "page_visit.parquet"
        page_visits = pd.read_parquet(page_visit_file)
        url_counts = page_visits['url'].value_counts()
        # Keep only URLs that appear frequently enough
        frequent_urls = url_counts[url_counts >= self.min_url_count]
        top_urls = frequent_urls.head(self.max_url_vocab - 3)  # Reserve space for special tokens
        self.url_vocab = {url: idx + 3 for idx, url in enumerate(top_urls.index)}
        print(f"Built URL vocabulary with {len(self.url_vocab)} entries")

        # Category/Price vocabulary from product events
        properties_file = data_dir / 'product_properties.parquet'
        properties = pd.read_parquet(properties_file)
        product_files = ['product_buy.parquet', 'add_to_cart.parquet', 'remove_from_cart.parquet']

        category_counts = defaultdict(int)
        for file in product_files:
            file_path = data_dir / file
            df = pd.read_parquet(file_path)
            df = df.merge(properties, on='sku', how='left')
            for category in df['category'].value_counts().items():
                category_counts[category[0]] += category[1]

        # Build category vocabulary
        frequent_categories = {cat: count for cat, count in category_counts.items()
                               if count >= self.min_category_count}
        sorted_categories = sorted(frequent_categories.items(), key=lambda x: x[1], reverse=True)
        top_categories = sorted_categories[:self.max_category_vocab - 3]

        self.category_vocab = {cat: idx + 3 for idx, (cat, _) in enumerate(top_categories)}
        print(f"Built category vocabulary with {len(self.category_vocab)} entries")

        # Price vocabulary (simpler - just use the price buckets directly)
        self.price_vocab = {bucket: bucket + 3 for bucket in range(100)}
        print(f"Built price vocabulary with {len(self.price_vocab)} entries")

        self.vocab_built = True

    def save_vocabularies(self, vocab_path: Path):
        """Save vocabularies to disk."""
        vocab_data = {
            'url_vocab': self.url_vocab,
            'category_vocab': self.category_vocab,
            'price_vocab': self.price_vocab,
            'event_types': self.event_types
        }
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)

    def load_vocabularies(self, vocab_path: Path):
        """Load vocabularies from disk."""
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)

        self.url_vocab = vocab_data['url_vocab']
        self.category_vocab = vocab_data['category_vocab']
        self.price_vocab = vocab_data['price_vocab']
        self.event_types = vocab_data['event_types']
        self.vocab_built = True

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for model initialization."""
        if not self.vocab_built:
            raise ValueError("Vocabularies not built yet!")

        return {
            'event_type': len(self.event_types) + 1,  # +1 for padding
            'url': len(self.url_vocab),
            'category': len(self.category_vocab),
            'price': len(self.price_vocab)
        }

    def process_event_df(self, df: pd.DataFrame, event_type: str) -> List[Dict[str, any]]:
        print(df.head())
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['event_type'] = self.event_types[event_type]

        if event_type == 'page_visit':
            df['url'] = df['url'].apply(lambda u: self.url_vocab.get(u, self.UNK_TOKEN)).astype(int)
        else:
            df['url'] = self.UNK_TOKEN

        if event_type in ['product_buy', 'add_to_cart', 'remove_from_cart']:
            df['category'] = df['category'].apply(lambda u: self.category_vocab.get(u, self.UNK_TOKEN)).astype(int)
            df['price'] = df['price'].apply(lambda u: self.price_vocab.get(u, self.UNK_TOKEN)).astype(int)
        else:
            df['category'] = self.UNK_TOKEN
            df['price'] = self.UNK_TOKEN

        zero_embed = np.zeros(16, dtype=np.float32)
        if event_type in ['product_buy', 'add_to_cart', 'remove_from_cart']:
            df['product_name'] = df['name'].apply(lambda s: self.parse_embedding_string(s) if s else zero_embed)
        else:
            df['product_name'] = df.apply(lambda _: zero_embed, axis=1)

        if event_type == 'search_query':
            df['search_query'] = df['query'].apply(lambda s: self.parse_embedding_string(s) if s else zero_embed)
        else:
            df['search_query'] = df.apply(lambda _: zero_embed, axis=1)

        print(df.head())
        # Select relevant columns
        return df[[
            'event_type', 'timestamp', 'client_id',
            'category', 'price', 'url', 'product_name', 'search_query'
        ]]

    def map_to_sequences(self, data_dir, relevant_clients):
        event_files = {
            'product_buy': 'product_buy.parquet',
            'add_to_cart': 'add_to_cart.parquet',
            'remove_from_cart': 'remove_from_cart.parquet',
            'page_visit': 'page_visit.parquet',
            'search_query': 'search_query.parquet'
        }

        dfs = []

        for event_type, filename in event_files.items():
            print(f"Processing {event_type} events...")
            file_path = data_dir / filename
            df = pd.read_parquet(file_path)
            df = df[df['client_id'].isin(relevant_clients)]

            # Merge properties if needed
            if event_type in ['product_buy', 'add_to_cart', 'remove_from_cart']:
                properties_file = data_dir / 'product_properties.parquet'
                properties = pd.read_parquet(properties_file)
                df = df.merge(properties, on='sku', how='left')

            dfs.append(self.process_event_df(df, event_type))

        # Combine all event dataframes
        return pd.concat(dfs, ignore_index=True)

    def create_user_sequences(self, data_dir: Path, relevant_clients: np.ndarray) -> pd.DataFrame:
        """Create sequences for each user by combining all their events."""
        print("Creating user sequences...")

        if not self.vocab_built:
            raise ValueError("Vocabularies must be built first!")

        df = self.map_to_sequences(data_dir, relevant_clients)
        df = df.sort_values(['client_id', 'timestamp'])
        return df


class UserSequenceDataset(Dataset):
    """Dataset for user event sequences."""

    def __init__(self, sequences_df: pd.DataFrame, active_clients: np.ndarray, target_df: pd.DataFrame,
                 max_seq_length: int = 256):
        self.max_seq_length = max_seq_length

        # Filter to active clients
        self.sequences_df = sequences_df[sequences_df['client_id'].isin(active_clients)].copy()

        # Group by client_id and pre-store list of client_ids
        self.client_groups = self.sequences_df.groupby('client_id')
        self.client_ids = list(self.client_groups.groups.keys())

        sum = 0
        for client_id in self.client_ids:
            if not target_df.loc[target_df["client_id"] == client_id].empty:
                sum += 1
        print(f"{sum} out of {len(self.client_ids)} = {(sum / len(self.client_ids)):.4f} users churned")

        self.target_df = target_df

    def __len__(self):
        return len(self.client_ids)

    def __getitem__(self, idx):
        client_id = self.client_ids[idx]
        group = self.client_groups.get_group(client_id).sort_values('timestamp')

        seq_len = len(group)

        # Truncate if too long
        if len(group) > self.max_seq_length:
            group = group.iloc[-self.max_seq_length:]

        # Initialize tensors
        batch_data = {
            'client_id': client_id,
            'event_type': torch.zeros(self.max_seq_length, dtype=torch.long),
            'category': torch.zeros(self.max_seq_length, dtype=torch.long),
            'price': torch.zeros(self.max_seq_length, dtype=torch.long),
            'url': torch.zeros(self.max_seq_length, dtype=torch.long),
            'product_name': torch.zeros(self.max_seq_length, 16, dtype=torch.float32),
            'search_query': torch.zeros(self.max_seq_length, 16, dtype=torch.float32),
            'mask': torch.zeros(self.max_seq_length, dtype=torch.bool),  # True for valid positions
            'seq_len': seq_len
        }

        for i, (_, event) in enumerate(group.iterrows()):
            batch_data['event_type'][i] = event['event_type']
            batch_data['category'][i] = event['category']
            batch_data['price'][i] = event['price']
            batch_data['url'][i] = event['url']
            batch_data['product_name'][i] = torch.from_numpy(event['product_name'])
            batch_data['search_query'][i] = torch.from_numpy(event['search_query'])
            batch_data['mask'][i] = True

        target = np.zeros(1, dtype=np.float32)
        # 1 = churned (positive), 0 = not churned (negative)
        target[0] = 0 if self.target_df.loc[self.target_df["client_id"] == client_id].empty else 1

        return batch_data, target


def create_data_processing_pipeline(
        data_dir: Path,
        relevant_clients: np.ndarray,
        vocab_path: Path,
        sequences_path: Path,
        rebuild_vocab: bool = False,
        max_seq_length: int = 256
) -> Tuple[UserSequenceDataset, UserSequenceDataset, Dict[str, int]]:
    """
    Complete data processing pipeline.

    Args:
        data_dir: Directory containing the parquet files
        relevant_clients: Array of client IDs to process
        vocab_path: Path to save/load vocabularies
        sequences_path: Path to save/load processed sequences
        rebuild_vocab: Whether to rebuild vocabularies from scratch
        max_seq_length: Maximum sequence length

    Returns:
        Dataset and vocabulary sizes
    """
    processor = EventSequenceProcessor(max_seq_length=max_seq_length)

    # Build or load vocabularies
    if rebuild_vocab or not vocab_path.exists():
        processor.build_vocabularies(data_dir)
        processor.save_vocabularies(vocab_path)
    else:
        processor.load_vocabularies(vocab_path)

    # Create or load sequences
    if rebuild_vocab or not sequences_path.exists():
        sequences = processor.create_user_sequences(data_dir, relevant_clients)
        sequences.to_pickle(sequences_path)
    else:
        sequences = pd.read_pickle(sequences_path)

    # Create dataset
    active_clients = np.load(data_dir / ".." / "original" / "target" / "active_clients.npy")
    target_data = TargetData.read_from_dir(data_dir / ".." / "original" / "target")
    dataset_train = UserSequenceDataset(sequences, active_clients, target_data.train_df, max_seq_length)
    dataset_valid = UserSequenceDataset(sequences, active_clients, target_data.validation_df, max_seq_length)
    vocab_sizes = processor.get_vocab_sizes()

    return dataset_train, dataset_valid, vocab_sizes


# Example usage script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--rebuild-vocab", action="store_true")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load relevant clients
    relevant_clients = np.load(data_dir / "input" / "relevant_clients.npy")

    # Process data
    dataset_train, dataset_val, vocab_sizes = create_data_processing_pipeline(
        data_dir=data_dir,
        relevant_clients=relevant_clients,
        vocab_path=output_dir / "vocabularies.pkl",
        sequences_path=output_dir / "sequences.pkl",
        rebuild_vocab=args.rebuild_vocab,
        max_seq_length=args.max_seq_length
    )

    print(f"Dataset created with {len(dataset_train)} sequences")
    print(f"Vocabulary sizes: {vocab_sizes}")
