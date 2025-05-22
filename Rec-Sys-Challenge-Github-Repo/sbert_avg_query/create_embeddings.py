import argparse
from pathlib import Path
from .generate_features import generate_user_embeddings

def main():
    parser = argparse.ArgumentParser(description="Generate user embeddings from search queries")
    parser.add_argument('--data-dir', type=str, required=True, help="Directory containing parquet files")
    parser.add_argument('--embeddings-dir', type=str, required=True, help="Output directory to save embeddings")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.embeddings_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.data_dir}...")
    generate_user_embeddings(args.data_dir, args.embeddings_dir)

if __name__ == '__main__':
    main()
