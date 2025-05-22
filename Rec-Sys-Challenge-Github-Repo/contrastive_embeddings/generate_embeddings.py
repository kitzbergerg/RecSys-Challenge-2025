import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contrastive_embeddings.constants import EventTypes, EVENT_TYPE_TO_COLUMNS, SESSION_GAP_SECONDS


import numpy as np
import pandas as pd
from tqdm import tqdm
from contrastive_embeddings.constants import EventTypes, EVENT_TYPE_TO_COLUMNS, SESSION_GAP_SECONDS

def hash_string_to_float(s: str) -> float:
    # Deterministic hash of string to float [0,1)
    return (hash(s) % 10000) / 10000.0

def load_parquet(file_path):
    return pd.read_parquet(file_path)

def extract_session_features(df, event_type):
    df = df.sort_values(["client_id", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')

    sessions = []
    current_session = []
    last_ts = None
    current_client = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {event_type}"):
        client = row["client_id"]
        ts = row["timestamp"]

        # Start new session if client changed or time gap > SESSION_GAP_SECONDS
        if client != current_client or (last_ts and (ts - last_ts).total_seconds() > SESSION_GAP_SECONDS):
            if current_session:
                sessions.append((current_client, current_session))
            current_session = []
            current_client = client

        features = []
        for col in EVENT_TYPE_TO_COLUMNS[event_type]:
            if col in row:
                val = row[col]
                if isinstance(val, str):
                    val = hash_string_to_float(val)
                else:
                    val = float(val)
                features.append(val)
        if features:
            current_session.append(features)

        last_ts = ts

    # Add last session
    if current_session:
        sessions.append((current_client, current_session))

    # Aggregate session features per client: mean of all sessions' mean feature vector
    client_feats = {}
    for client_id, sess in sessions:
        arr = np.array(sess)
        if arr.size > 0:
            client_feats.setdefault(client_id, []).append(arr.mean(axis=0))

    # Average across sessions
    for client_id in client_feats:
        client_feats[client_id] = np.mean(client_feats[client_id], axis=0)

    return client_feats

def merge_features(dicts):
    all_clients = set()
    for d in dicts:
        all_clients.update(d.keys())
    merged = {}
    for client in all_clients:
        parts = []
        for d in dicts:
            if client in d:
                parts.append(d[client])
            else:
                # Fill zeros vector of appropriate length if missing
                if d:
                    example = next(iter(d.values()))
                    parts.append(np.zeros_like(example))
                else:
                    parts.append(np.array([]))
        merged[client] = np.concatenate(parts)
    return merged

def main(data_dir, embeddings_dir):
    feature_dicts = []
    for event in EventTypes:
        file_path = os.path.join(data_dir, f"{event.value}.parquet")
        if not os.path.isfile(file_path):
            print(f"File {file_path} not found. Skipping {event}.")
            continue
        df = load_parquet(file_path)
        feats = extract_session_features(df, event)
        feature_dicts.append(feats)

    merged_features = merge_features(feature_dicts)

    clients = list(merged_features.keys())
    embeddings = np.stack([merged_features[c] for c in clients])

    # Convert to float16 to satisfy validator
    embeddings = embeddings.astype(np.float16)

    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, "client_ids.npy"), np.array(clients))
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeddings)

    print(f"Saved {len(clients)} embeddings to {embeddings_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Input directory with parquet files")
    parser.add_argument("--embeddings-dir", required=True, help="Directory to save embeddings")
    args = parser.parse_args()
    main(args.data_dir, args.embeddings_dir)
