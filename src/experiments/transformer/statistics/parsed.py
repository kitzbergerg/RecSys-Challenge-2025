import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse
from pyarrow.parquet import ParquetFile
import pyarrow as pa


class SequenceDataStatisticsCalculator:
    """
    Calculate comprehensive statistics from already processed sequence data.
    Works with sequences_full.parquet or sequences_N.parquet files.
    """

    def __init__(self):
        # Event type counts
        self.event_counts = defaultdict(int)

        # Unique value sets (for counting)
        self.unique_client_ids = set()
        self.unique_skus = set()
        self.unique_categories = set()
        self.unique_urls = set()
        self.unique_prices = set()

        # For sequence statistics
        self.client_event_counts = defaultdict(int)

        # Total events counter
        self.total_events = 0

        # Event type mapping (from the original preprocessing file)
        self.event_type_names = {
            0: 'PAD_TOKEN',
            1: 'UNK_TOKEN',
            2: 'MASK_TOKEN',
            3: 'CONTEXT_MASK_TOKEN',
            4: 'product_buy',
            5: 'add_to_cart',
            6: 'remove_from_cart',
            7: 'page_visit',
            8: 'search_query'
        }

    def process_sequence_file_in_chunks(self, file_path: Path, chunk_size: int = 1_000_000):
        """Process sequence file in chunks to avoid memory issues."""
        print(f"Processing sequence data from {file_path}...")

        # Get file info
        try:
            # Try to get total rows efficiently
            pf = ParquetFile(file_path)
            total_rows = pf.metadata.num_rows
            print(f"  Total rows: {total_rows:,}")
        except Exception as e:
            print(f"  Could not get row count: {e}")
            total_rows = "unknown"

        processed_rows = 0

        # Process file in chunks
        pf = ParquetFile(file_path)
        for chunk in pf.iter_batches(batch_size=chunk_size):
            chunk_df = pa.Table.from_batches([chunk]).to_pandas()
            self.process_chunk(chunk_df)
            processed_rows += len(chunk_df)

            if total_rows != "unknown":
                progress = (processed_rows / total_rows) * 100
                print(f"  Progress: {processed_rows:,}/{total_rows:,} ({progress:.1f}%)")
            else:
                print(f"  Processed: {processed_rows:,} rows")

    def process_chunk(self, chunk: pd.DataFrame):
        """Process a single chunk of sequence data."""
        # Update total events
        self.total_events += len(chunk)

        # Update unique client IDs
        self.unique_client_ids.update(chunk['client_id'].unique())

        # Update sequence counts (events per client)
        client_counts = chunk['client_id'].value_counts()
        for client_id, count in client_counts.items():
            self.client_event_counts[client_id] += count

        # Process event types
        event_type_counts = chunk['event_type'].value_counts()
        for event_type, count in event_type_counts.items():
            event_name = self.event_type_names.get(event_type, f'unknown_{event_type}')
            self.event_counts[event_name] += count

        # Process unique values for different features
        # Only process non-special token values (> 3)

        # URLs (from page_visit events)
        url_mask = chunk['event_type'] == 7  # page_visit
        if url_mask.any():
            valid_urls = chunk.loc[url_mask, 'url']
            valid_urls = valid_urls[valid_urls > 3]  # Remove special tokens
            self.unique_urls.update(valid_urls.unique())

        # SKUs, Categories, Prices (from product events)
        product_mask = chunk['event_type'].isin([4, 5, 6])  # product_buy, add_to_cart, remove_from_cart
        if product_mask.any():
            product_chunk = chunk[product_mask]

            # SKUs
            valid_skus = product_chunk['sku'][product_chunk['sku'] > 3]
            self.unique_skus.update(valid_skus.unique())

            # Categories
            valid_categories = product_chunk['category'][product_chunk['category'] > 3]
            self.unique_categories.update(valid_categories.unique())

            # Prices
            valid_prices = product_chunk['price'][product_chunk['price'] > 3]
            self.unique_prices.update(valid_prices.unique())

    def calculate_sequence_statistics_from_processed(self, file_path: Path, chunk_size: int = 1_000_000):
        """Calculate sequence length statistics from processed sequence data."""
        print("\nCalculating sequence statistics from processed data...")

        # Collect client event counts
        client_event_counts = defaultdict(int)

        processed_rows = 0
        pf = ParquetFile(file_path)
        total_rows = pf.metadata.num_rows

        # Only need client_id for sequence statistics
        for chunk in pf.iter_batches(batch_size=chunk_size, columns=['client_id']):
            chunk_df = pa.Table.from_batches([chunk]).to_pandas()

            # Count events per client
            client_counts = chunk_df['client_id'].value_counts()
            for client_id, count in client_counts.items():
                client_event_counts[client_id] += count

            processed_rows += len(chunk_df)
            progress = (processed_rows / total_rows) * 100
            print(f"  Progress: {processed_rows:,}/{total_rows:,} ({progress:.1f}%)")

        # Convert to pandas Series for statistics
        sequence_lengths = pd.Series(list(client_event_counts.values()))
        print(f"  Total unique clients: {len(sequence_lengths):,}")

        return sequence_lengths

    def generate_report(self, file_path: Path, chunk_size: int = 1_000_000):
        """Generate comprehensive statistics report from sequence data."""
        print("=" * 60)
        print("SEQUENCE DATA STATISTICS REPORT")
        print("=" * 60)
        print(f"Processing file: {file_path}")

        # Process the sequence file
        self.process_sequence_file_in_chunks(file_path, chunk_size)

        # Calculate sequence statistics
        sequence_lengths = self.calculate_sequence_statistics_from_processed(file_path, chunk_size)

        # Generate final report
        self.print_final_report(sequence_lengths)

        return sequence_lengths

    def print_final_report(self, sequence_lengths: pd.Series):
        """Print the final comprehensive report."""
        print("\n" + "=" * 60)
        print("FINAL STATISTICS REPORT")
        print("=" * 60)

        # Event type counts
        print("\nEVENT TYPE COUNTS:")
        print("-" * 30)
        # Sort by count (descending) and exclude special tokens for cleaner display
        main_events = {k: v for k, v in self.event_counts.items()
                       if k not in ['PAD_TOKEN', 'UNK_TOKEN', 'MASK_TOKEN', 'CONTEXT_MASK_TOKEN']}

        for event_type, count in sorted(main_events.items(), key=lambda x: x[1], reverse=True):
            print(f"  {event_type:<20}: {count:,}")

        # Show special tokens separately if they exist
        special_tokens = {k: v for k, v in self.event_counts.items()
                          if k in ['PAD_TOKEN', 'UNK_TOKEN', 'MASK_TOKEN', 'CONTEXT_MASK_TOKEN'] and v > 0}
        if special_tokens:
            print("\nSPECIAL TOKENS:")
            for token, count in special_tokens.items():
                print(f"  {token:<20}: {count:,}")

        # Unique entity counts
        print("\nUNIQUE ENTITY COUNTS:")
        print("-" * 30)
        print(f"  client_id          : {len(self.unique_client_ids):,}")
        print(f"  sku                : {len(self.unique_skus):,}")
        print(f"  category           : {len(self.unique_categories):,}")
        print(f"  url                : {len(self.unique_urls):,}")
        print(f"  price              : {len(self.unique_prices):,}")

        # Sequence length statistics
        print("\nSEQUENCE LENGTH STATISTICS:")
        print("-" * 30)
        print(f"  Number of sequences: {len(sequence_lengths):,}")
        if len(sequence_lengths) > 0:
            print(f"  Mean:               {sequence_lengths.mean():.2f}")
            print(f"  Median:             {sequence_lengths.median():.2f}")
            print(f"  Std Dev:            {sequence_lengths.std():.2f}")
            print(f"  Min:                {sequence_lengths.min()}")
            print(f"  Max:                {sequence_lengths.max()}")
            print("  Quantiles:")
            quantiles = sequence_lengths.quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            for q, val in quantiles.items():
                print(f"    {q:>5.0%}: {val:>10.0f}")

        # Total events
        print("\nTOTAL STATISTICS:")
        print("-" * 30)
        print(f"  Total events       : {self.total_events:,}")

        # Memory usage summary
        print("\nMEMORY USAGE SUMMARY:")
        print("-" * 30)
        print(f"  Unique client_ids  : ~{len(self.unique_client_ids) * 8 / 1024 / 1024:.1f} MB")
        print(f"  Unique URLs        : ~{len(self.unique_urls) * 8 / 1024 / 1024:.1f} MB")
        print(f"  Unique SKUs        : ~{len(self.unique_skus) * 8 / 1024 / 1024:.1f} MB")
        print(f"  Unique categories  : ~{len(self.unique_categories) * 8 / 1024 / 1024:.1f} MB")
        print(f"  Unique prices      : ~{len(self.unique_prices) * 8 / 1024 / 1024:.1f} MB")

    def save_detailed_report(self, output_path: Path, sequence_lengths: pd.Series, source_file: str):
        """Save detailed statistics to files."""
        print(f"\nSaving detailed report to {output_path}...")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary statistics
        with open(output_path / "summary_stats.txt", "w") as f:
            f.write("SEQUENCE DATA STATISTICS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Source file: {source_file}\n\n")

            f.write("Event Type Counts:\n")
            for event_type, count in sorted(self.event_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {event_type}: {count:,}\n")

            f.write(f"\nUnique Entities:\n")
            f.write(f"  client_id: {len(self.unique_client_ids):,}\n")
            f.write(f"  sku: {len(self.unique_skus):,}\n")
            f.write(f"  category: {len(self.unique_categories):,}\n")
            f.write(f"  url: {len(self.unique_urls):,}\n")
            f.write(f"  price: {len(self.unique_prices):,}\n")

            f.write(f"\nTotal events: {self.total_events:,}\n")

        # Save sequence length distribution
        if len(sequence_lengths) > 0:
            sequence_stats = {
                'source_file': source_file,
                'count': len(sequence_lengths),
                'mean': float(sequence_lengths.mean()),
                'median': float(sequence_lengths.median()),
                'std': float(sequence_lengths.std()),
                'min': int(sequence_lengths.min()),
                'max': int(sequence_lengths.max()),
                'quantiles': {str(k): float(v) for k, v in sequence_lengths.quantile(
                    [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict().items()}
            }
        else:
            sequence_stats = {
                'source_file': source_file,
                'count': 0,
                'error': 'No sequence data found'
            }

        import json
        with open(output_path / "sequence_stats.json", "w") as f:
            json.dump(sequence_stats, f, indent=2)

        print("Detailed report saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Calculate statistics from processed sequence data")
    parser.add_argument("--sequence-file", type=str, required=True,
                        help="Path to sequence parquet file (e.g., sequences_full.parquet or sequences_0.parquet)")
    parser.add_argument("--output-dir", type=str,
                        help="Directory to save detailed report (optional)")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="Chunk size for processing (default: 1M)")

    args = parser.parse_args()

    # Create calculator
    calculator = SequenceDataStatisticsCalculator()

    # Process single file
    file_path = Path(args.sequence_file)
    if not file_path.exists():
        print(f"Error: Sequence file {file_path} does not exist")
        return

    # Generate report
    sequence_lengths = calculator.generate_report(file_path, args.chunk_size)

    # Save detailed report if output directory specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        calculator.save_detailed_report(output_path, sequence_lengths, str(file_path))


if __name__ == "__main__":
    main()