import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse
from pyarrow.parquet import ParquetFile
import pyarrow as pa


class DatasetStatisticsCalculator:
    """
    Calculate comprehensive statistics for the full dataset by processing files in chunks.
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

        # Event type mapping
        self.event_files = {
            'product_buy': 'product_buy.parquet',
            'add_to_cart': 'add_to_cart.parquet',
            'remove_from_cart': 'remove_from_cart.parquet',
            'page_visit': 'page_visit.parquet',
            'search_query': 'search_query.parquet'
        }

    def process_file_in_chunks(self, file_path: Path, event_type: str, properties_df: pd.DataFrame = None):
        """Process a single file in chunks to avoid memory issues."""
        print(f"Processing {event_type} events from {file_path}...")

        # Read file info to get total rows
        try:
            parquet_file = pd.read_parquet(file_path, columns=['client_id'])
            total_rows = len(parquet_file)
            print(f"  Total rows: {total_rows:,}")
            del parquet_file
        except Exception as e:
            print(f"  Could not get row count: {e}")
            total_rows = "unknown"

        processed_rows = 0

        # Process file in chunks
        pf = ParquetFile(file_path)
        for chunk in pf.iter_batches(batch_size=1000000):
            chunk = pa.Table.from_batches([chunk]).to_pandas()
            self.process_chunk(chunk, event_type, properties_df)
            processed_rows += len(chunk)

            if total_rows != "unknown":
                progress = (processed_rows / total_rows) * 100
                print(f"  Progress: {processed_rows:,}/{total_rows:,} ({progress:.1f}%)")
            else:
                print(f"  Processed: {processed_rows:,} rows")

    def process_chunk(self, chunk: pd.DataFrame, event_type: str, properties_df: pd.DataFrame = None):
        """Process a single chunk of data."""
        # Update event count
        self.event_counts[event_type] += len(chunk)
        self.total_events += len(chunk)

        # Update unique client IDs
        self.unique_client_ids.update(chunk['client_id'].unique())

        # Update sequence counts (events per client)
        client_counts = chunk['client_id'].value_counts()
        for client_id, count in client_counts.items():
            self.client_event_counts[client_id] += count

        # Process event-specific features
        if event_type == 'page_visit':
            self.unique_urls.update(chunk['url'].unique())

        elif event_type in ['product_buy', 'add_to_cart', 'remove_from_cart']:
            # Merge with properties if available
            if properties_df is not None:
                chunk = chunk.merge(properties_df, on='sku', how='left')

            # Update unique values
            self.unique_skus.update(chunk['sku'].unique())

            if 'category' in chunk.columns:
                valid_categories = chunk['category'].dropna().unique()
                self.unique_categories.update(valid_categories)

            if 'price' in chunk.columns:
                valid_prices = chunk['price'].dropna().unique()
                self.unique_prices.update(valid_prices)

    def calculate_sequence_statistics(self, data_dir: Path):
        """Calculate sequence length statistics by loading only client_id and timestamp."""
        print("\nCalculating sequence statistics...")

        # Collect all client sequences
        all_sequences = []

        for event_type, filename in self.event_files.items():
            file_path = data_dir / filename
            if not file_path.exists():
                print(f"  Warning: {filename} not found, skipping...")
                continue

            print(f"  Loading timestamps from {filename}...")

            # Only load client_id and timestamp columns
            pf = ParquetFile(file_path)
            for chunk in pf.iter_batches(batch_size=1000000, columns=['client_id', 'timestamp']):
                chunk = pa.Table.from_batches([chunk]).to_pandas()
                all_sequences.append(chunk)

        # Combine all sequences
        print("  Combining all sequences...")
        combined_df = pd.concat(all_sequences, ignore_index=True)
        print("  Total loaded events:", len(combined_df))

        # Calculate sequence lengths
        print("  Calculating sequence lengths per client...")
        sequence_lengths = combined_df.groupby('client_id').size()

        return sequence_lengths

    def generate_report(self, data_dir: Path):
        """Generate comprehensive statistics report."""
        print("=" * 60)
        print("DATASET STATISTICS REPORT")
        print("=" * 60)

        # Load product properties once
        properties_file = data_dir / 'product_properties.parquet'
        properties_df = None
        if properties_file.exists():
            print("Loading product properties...")
            properties_df = pd.read_parquet(properties_file)
        else:
            print("Warning: product_properties.parquet not found")

        # Process each event type
        for event_type, filename in self.event_files.items():
            file_path = data_dir / filename
            if file_path.exists():
                self.process_file_in_chunks(file_path, event_type, properties_df)
            else:
                print(f"Warning: {filename} not found, skipping...")

        # Calculate sequence statistics
        sequence_lengths = self.calculate_sequence_statistics(data_dir)

        # Generate final report
        self.print_final_report(sequence_lengths)

    def print_final_report(self, sequence_lengths: pd.Series):
        """Print the final comprehensive report."""
        print("\n" + "=" * 60)
        print("FINAL STATISTICS REPORT")
        print("=" * 60)

        # Event type counts
        print("\nEVENT TYPE COUNTS:")
        print("-" * 30)
        for event_type, count in self.event_counts.items():
            print(f"  {event_type:<20}: {count:,}")

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
        print(f"  Unique URLs        : ~{len(self.unique_urls) * 100 / 1024 / 1024:.1f} MB")
        print(f"  Unique SKUs        : ~{len(self.unique_skus) * 50 / 1024 / 1024:.1f} MB")
        print(f"  Unique categories  : ~{len(self.unique_categories) * 30 / 1024 / 1024:.1f} MB")
        print(f"  Unique prices      : ~{len(self.unique_prices) * 8 / 1024 / 1024:.1f} MB")

    def save_detailed_report(self, output_path: Path, sequence_lengths: pd.Series):
        """Save detailed statistics to files."""
        print(f"\nSaving detailed report to {output_path}...")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary statistics
        with open(output_path / "summary_stats.txt", "w") as f:
            f.write("DATASET STATISTICS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write("Event Type Counts:\n")
            for event_type, count in self.event_counts.items():
                f.write(f"  {event_type}: {count:,}\n")

            f.write(f"\nUnique Entities:\n")
            f.write(f"  client_id: {len(self.unique_client_ids):,}\n")
            f.write(f"  sku: {len(self.unique_skus):,}\n")
            f.write(f"  category: {len(self.unique_categories):,}\n")
            f.write(f"  url: {len(self.unique_urls):,}\n")
            f.write(f"  price: {len(self.unique_prices):,}\n")

            f.write(f"\nTotal events: {self.total_events:,}\n")

        # Save sequence length distribution
        sequence_stats = {
            'count': len(sequence_lengths),
            'mean': sequence_lengths.mean(),
            'median': sequence_lengths.median(),
            'std': sequence_lengths.std(),
            'min': sequence_lengths.min(),
            'max': sequence_lengths.max(),
            'quantiles': sequence_lengths.quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        }

        import json
        with open(output_path / "sequence_stats.json", "w") as f:
            json.dump(sequence_stats, f, indent=2)

        print("Detailed report saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Calculate comprehensive statistics for the full dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the parquet files")
    parser.add_argument("--output-dir", type=str, help="Directory to save detailed report (optional)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return

    # Create calculator
    calculator = DatasetStatisticsCalculator()

    # Generate report
    calculator.generate_report(data_dir)

    # Save detailed report if output directory specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        sequence_lengths = calculator.calculate_sequence_statistics(data_dir)
        calculator.save_detailed_report(output_path, sequence_lengths)


if __name__ == "__main__":
    main()
