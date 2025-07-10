from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.transformer.data_processor import EventSequenceProcessor, read_filtered_parquet
from experiments.transformer.dataset import reconstructive_collate_fn, UserSequenceDataset
from experiments.transformer.model_training import TransformerModel


def load_embedding_dataset(
        data_dir: Path,
        sequences_path: Path,
) -> UserSequenceDataset:
    processor = EventSequenceProcessor()
    processor.load_vocabularies(sequences_path / "vocabularies.pkl")
    vocab_sizes = processor.get_vocab_sizes()

    rel_clients_file = sequences_path / f"sequences_rel_clients.parquet"
    if not rel_clients_file.exists():
        relevant_clients = np.load(data_dir / "input" / "relevant_clients.npy")
        sequences = read_filtered_parquet(sequences_path / "sequences_full.parquet", relevant_clients)
        sequences.to_parquet(rel_clients_file)
    else:
        sequences = pd.read_parquet(rel_clients_file)

    dataset = UserSequenceDataset(sequences, vocab_sizes, disable_masking=True)
    return dataset


def generate_embeddings(
        data_dir: Path,
        embeddings_dir: Path,
        sequences_path: Path,
        ckpt_path: Path
):
    dataset = load_embedding_dataset(data_dir, sequences_path)
    dataloader = DataLoader(dataset, 128, num_workers=0, collate_fn=reconstructive_collate_fn)
    model = TransformerModel.load_from_checkpoint(ckpt_path)
    model.eval()

    embeddings = []
    client_ids = []

    for x, _ in dataloader:
        batch_client_ids = x.pop('client_id')
        for key, tensor in x.items():
            x[key] = tensor.to("cuda")
        y = model(x).to(torch.float16).to("cpu")

        for idx, client_id in enumerate(batch_client_ids):
            client_ids.append(client_id)
            embeddings.append(y[idx].detach().numpy())

    client_ids_array = np.array(client_ids)
    embeddings_array = np.stack(embeddings)

    np.save(embeddings_dir / 'client_ids.npy', client_ids_array)
    np.save(embeddings_dir / 'embeddings.npy', embeddings_array)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sequences-path", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=False)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sequences_path = Path(args.sequences_path)
    embeddings_dir = Path(args.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_path is not None:
        ckpt_path = Path(args.checkpoint_path)
    else:
        ckpt_path = None

    generate_embeddings(
        data_dir=data_dir,
        embeddings_dir=embeddings_dir,
        sequences_path=sequences_path,
        ckpt_path=ckpt_path
    )
