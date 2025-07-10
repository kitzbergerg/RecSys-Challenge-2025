import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from collections import defaultdict
import re

from experiments.transformer.constants import MAX_SEQUENCE_LENGTH, TEXT_EMB_DIM
from experiments.transformer.dataset import UserSequenceDataset
from experiments.transformer.dataset_contrastive import UserSequenceContrastiveDataset


class EventSequenceProcessor:
    """
    Processes raw event data into sequences suitable for transformer training.
    Handles vocabulary creation, OOV tokens, and sequence generation.
    """

    def __init__(
            self,
            max_seq_length: int = MAX_SEQUENCE_LENGTH,
            max_url_vocab: int = 50000,
            max_category_vocab: int = 10000,
            max_sku_vocab: int = 5000,
            min_url_count: int = 500,
            min_category_count: int = 3,
            min_sku_count: int = 200
    ):
        self.max_seq_length = max_seq_length
        self.max_url_vocab = max_url_vocab
        self.max_category_vocab = max_category_vocab
        self.max_sku_vocab = max_sku_vocab
        self.min_url_count = min_url_count
        self.min_category_count = min_category_count
        self.min_sku_count = min_sku_count

        # Special tokens
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1
        self.MASK_TOKEN = 2
        self.CONTEXT_MASK_TOKEN = 3

        # Event type mapping
        self.event_types = {
            # special tokens
            'PAD_TOKEN': self.PAD_TOKEN,
            'UNK_TOKEN': self.UNK_TOKEN,
            'MASK_TOKEN': self.MASK_TOKEN,
            'CONTEXT_MASK_TOKEN': self.CONTEXT_MASK_TOKEN,

            'product_buy': 4,
            'add_to_cart': 5,
            'remove_from_cart': 6,
            'page_visit': 7,
            'search_query': 8
        }

        # Vocabularies (will be built from data)
        # store special tokens as negative numbers as they will never appear in data
        self.url_vocab = {
            -1: self.PAD_TOKEN,
            -2: self.UNK_TOKEN,
            -3: self.MASK_TOKEN,
            -4: self.CONTEXT_MASK_TOKEN,
        }
        self.category_vocab = {
            -1: self.PAD_TOKEN,
            -2: self.UNK_TOKEN,
            -3: self.MASK_TOKEN,
            -4: self.CONTEXT_MASK_TOKEN,
        }
        self.sku_vocab = {
            -1: self.PAD_TOKEN,
            -2: self.UNK_TOKEN,
            -3: self.MASK_TOKEN,
            -4: self.CONTEXT_MASK_TOKEN,
        }
        self.price_vocab = {
            -1: self.PAD_TOKEN,
            -2: self.UNK_TOKEN,
            -3: self.MASK_TOKEN,
            -4: self.CONTEXT_MASK_TOKEN,
        }
        self.vocab_built = False

    def parse_embedding_string(self, emb_str: str) -> np.ndarray:
        if isinstance(emb_str, str):
            numbers = re.findall(r'\d+', emb_str)
            emb_array = np.array(numbers, dtype=np.float32)
        else:
            emb_array = np.array(emb_str, dtype=np.float32)

        assert len(emb_array) == TEXT_EMB_DIM
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
        assert len(frequent_urls) < self.max_url_vocab
        top_urls = frequent_urls.head(self.max_url_vocab - 4)
        self.url_vocab.update({url: idx + 4 for idx, url in enumerate(top_urls.index)})
        print(f"Built URL vocabulary with {len(self.url_vocab)} entries")

        # Category/Price vocabulary from product events
        properties_file = data_dir / 'product_properties.parquet'
        properties = pd.read_parquet(properties_file)
        product_files = ['product_buy.parquet', 'add_to_cart.parquet', 'remove_from_cart.parquet']

        category_counts = defaultdict(int)
        sku_counts = defaultdict(int)
        for file in product_files:
            file_path = data_dir / file
            df = pd.read_parquet(file_path)
            df = df.merge(properties, on='sku', how='left')
            for category in df['category'].value_counts().items():
                category_counts[category[0]] += category[1]
            for sku in df['sku'].value_counts().items():
                sku_counts[sku[0]] += sku[1]

        # Build category vocabulary
        frequent_categories = {cat: count for cat, count in category_counts.items() if count >= self.min_category_count}
        assert len(frequent_categories) < self.max_category_vocab
        sorted_categories = sorted(frequent_categories.items(), key=lambda x: x[1], reverse=True)
        top_categories = sorted_categories[:self.max_category_vocab - 4]
        self.category_vocab.update({cat: idx + 4 for idx, (cat, _) in enumerate(top_categories)})
        print(f"Built category vocabulary with {len(self.category_vocab)} entries")

        # Build sku vocabulary
        frequent_sku = {sku: count for sku, count in sku_counts.items() if count >= self.min_sku_count}
        assert len(frequent_sku) < self.max_sku_vocab
        sorted_sku = sorted(frequent_sku.items(), key=lambda x: x[1], reverse=True)
        top_sku = sorted_sku[:self.max_sku_vocab - 4]
        self.sku_vocab.update({cat: idx + 4 for idx, (cat, _) in enumerate(top_sku)})
        print(f"Built sku vocabulary with {len(self.sku_vocab)} entries")

        # Price vocabulary (simpler - just use the price buckets directly)
        self.price_vocab.update({bucket: bucket + 4 for bucket in range(100)})
        print(f"Built price vocabulary with {len(self.price_vocab)} entries")

        self.vocab_built = True

    def save_vocabularies(self, vocab_path: Path):
        """Save vocabularies to disk."""
        vocab_data = {
            'url_vocab': self.url_vocab,
            'category_vocab': self.category_vocab,
            'sku_vocab': self.sku_vocab,
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
        self.sku_vocab = vocab_data['sku_vocab']
        self.price_vocab = vocab_data['price_vocab']
        self.event_types = vocab_data['event_types']
        self.vocab_built = True

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for model initialization."""
        if not self.vocab_built:
            raise ValueError("Vocabularies not built yet!")

        return {
            'event_type': len(self.event_types),
            'url': len(self.url_vocab),
            'category': len(self.category_vocab),
            'sku': len(self.sku_vocab),
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
            df['sku'] = df['sku'].apply(lambda u: self.sku_vocab.get(u, self.UNK_TOKEN)).astype(int)
            df['price'] = df['price'].apply(lambda u: self.price_vocab.get(u, self.UNK_TOKEN)).astype(int)
        else:
            df['category'] = self.UNK_TOKEN
            df['sku'] = self.UNK_TOKEN
            df['price'] = self.UNK_TOKEN

        zero_embed = np.zeros(TEXT_EMB_DIM, dtype=np.float32)
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
            'category', 'sku', 'price', 'url', 'product_name', 'search_query'
        ]]

    def map_to_sequences(self, data_dir):
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

            # Merge properties if needed
            if event_type in ['product_buy', 'add_to_cart', 'remove_from_cart']:
                properties_file = data_dir / 'product_properties.parquet'
                properties = pd.read_parquet(properties_file)
                df = df.merge(properties, on='sku', how='left')

            CHUNK_SIZE = 10_000_000
            if len(df) > CHUNK_SIZE:
                for start in range(0, len(df), CHUNK_SIZE):
                    chunk = df.iloc[start:start + CHUNK_SIZE].copy()
                    dfs.append(self.process_event_df(chunk, event_type))
            else:
                dfs.append(self.process_event_df(df, event_type))

        # Combine all event dataframes
        return pd.concat(dfs, ignore_index=True)

    def create_user_sequences(self, data_dir: Path) -> pd.DataFrame:
        """Create sequences for each user by combining all their events."""
        print("Creating user sequences...")

        if not self.vocab_built:
            raise ValueError("Vocabularies must be built first!")

        df = self.map_to_sequences(data_dir)
        df = df.sort_values(['client_id', 'timestamp'])
        df['time_delta'] = df.groupby('client_id')['timestamp'].diff().dt.total_seconds().replace(0, 1).fillna(0)
        return df


def sample_from_distribution(df: pd.DataFrame, n: int) -> np.array:
    # 1. Group and get sequence lengths
    grp = df.groupby('client_id')
    seq_lengths = grp.size()

    # 2. Define sigmoid parameters
    mu = 5  # Midpoint (where weight ~0.5)
    sigma = 1  # Steepness

    # 3. Compute sigmoid weights
    weights = 1 / (1 + np.exp(-(seq_lengths.values - mu) / sigma))
    probabilities = weights / weights.sum()

    # 4. Sample n client_ids
    sampled_client_ids = np.random.choice(seq_lengths.index, size=n, replace=False, p=probabilities)
    return sampled_client_ids


def read_filtered_parquet(parquet_path: Path, client_ids) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, filters=[('client_id', 'in', client_ids)])
    return df.sort_values(['client_id', 'timestamp']).groupby('client_id', group_keys=False).tail(MAX_SEQUENCE_LENGTH)


def calculate_statistics(df: pd.DataFrame):
    print("Calculating statistics...")
    grp = df.groupby('client_id')
    client_ids = list(grp.groups.keys())
    print(f"Number of sequences: {len(client_ids)}")

    # --- Sequence length statistics ---
    seq_lengths = grp.size()
    print("\nSequence Lengths:")
    print(f"  Mean:     {seq_lengths.mean():.2f}")
    print(f"  Median:   {seq_lengths.median():.2f}")
    print(f"  Std Dev:  {seq_lengths.std():.2f}")
    print(f"  Min:      {seq_lengths.min()}")
    print(f"  Max:      {seq_lengths.max()}")
    print("  Quantiles:")
    print(seq_lengths.quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    # --- Time delta statistics ---
    if 'time_delta' in df.columns:
        time_deltas = df['time_delta']
        print("\nTime Deltas:")
        print(f"  Mean:     {time_deltas.mean():.2f}")
        print(f"  Median:   {time_deltas.median():.2f}")
        print(f"  Std Dev:  {time_deltas.std():.2f}")
        print(f"  Min:      {time_deltas.min()}")
        print(f"  Max:      {time_deltas.max()}")
        print("  Quantiles:")
        print(time_deltas.quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    else:
        print("\n[Warning] Column 'time_delta' not found in DataFrame.")

    # --- Categorical feature distributions ---
    def print_categorical_stats(name, counts):
        print(f"\n{name} (Top 10):")
        print(counts.head(10))
        print(f"\n{name} (Bottom 5):")
        print(counts.tail(5))
        print(f"  Unique {name.lower()}s: {counts.shape[0]}")

    print_categorical_stats("Event Type", df['event_type'].value_counts())
    product_df = df[df['event_type'].isin([4, 5, 6])]
    print_categorical_stats("Price", product_df['price'].value_counts())
    print_categorical_stats("Category", product_df['category'].value_counts())
    print_categorical_stats("SKU", product_df['sku'].value_counts())
    # URL is to large, unable to calc


def create_data_processing_pipeline(
        data_dir: Path,
        sequences_path: Path,
        task: Optional[str] = None,
        rebuild_vocab: bool = False,
        max_seq_length: int = MAX_SEQUENCE_LENGTH
) -> Tuple[UserSequenceDataset, Dict[str, int]]:
    """
    Complete data processing pipeline.

    Args:
        data_dir: Directory containing the parquet files
        sequences_path: Path to save/load processed sequences
        rebuild_vocab: Whether to rebuild vocabularies from scratch
        max_seq_length: Maximum sequence length

    Returns:
        Dataset and vocabulary sizes
    """
    processor = EventSequenceProcessor(max_seq_length=max_seq_length)

    # Build or load vocabularies
    vocab_file = sequences_path / "vocabularies.pkl"
    if rebuild_vocab or not vocab_file.exists():
        processor.build_vocabularies(data_dir)
        processor.save_vocabularies(vocab_file)
    else:
        processor.load_vocabularies(vocab_file)

    # Create or load sequences
    sequences_file_full = sequences_path / "sequences_full.parquet"
    if rebuild_vocab or not sequences_file_full.exists():
        print("Parsing full dataset...")
        sequences_full = processor.create_user_sequences(data_dir)
        print("Writing full dataset to parquet...")
        sequences_full.to_parquet(sequences_file_full)

    # TODO: figure out how to use all chunks, maybe just stop training and use a different one?
    sequences_file = sequences_path / "sequences_0.parquet"
    if rebuild_vocab or not sequences_file.exists():
        print("Sampling from full dataset...")
        sequences_full = pd.read_parquet(sequences_file_full, columns=["client_id", "timestamp"])
        sampled_client_ids = sample_from_distribution(sequences_full, 5_000_000)

        CHUNK_SIZE = 1_000_000
        for i in range(5_000_000 // CHUNK_SIZE):
            print(f"Processing chunk {i}...")
            client_ids = sampled_client_ids[CHUNK_SIZE * i:CHUNK_SIZE * (i + 1)]
            sequences = read_filtered_parquet(sequences_file_full, client_ids)
            sequences.to_parquet(sequences_path / f"sequences_{i}.parquet")

        sequences = pd.read_parquet(sequences_file)
        calculate_statistics(sequences)
    else:
        print("Loading small dataset...")
        sequences = pd.read_parquet(sequences_file)

    # Create dataset
    vocab_sizes = processor.get_vocab_sizes()
    if not task or task == 'reconstruction':
        dataset = UserSequenceDataset(sequences, vocab_sizes, max_seq_length)
    elif task == 'contrastive':
        dataset = UserSequenceContrastiveDataset(sequences, max_seq_length)
    else:
        print(f"WARNING: no such task {task}")
        exit(1)

    return dataset, vocab_sizes


# Example usage script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--rebuild-vocab", action="store_true")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process data
    dataset, vocab_sizes = create_data_processing_pipeline(
        data_dir=data_dir,
        sequences_path=output_dir,
        rebuild_vocab=args.rebuild_vocab,
        max_seq_length=args.max_seq_length
    )

    print(f"Dataset created with {len(dataset)} sequences")
    print(f"Vocabulary sizes: {vocab_sizes}")
