import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hashlib
from contrastive_embeddings.constants import EventTypes, EVENT_TYPE_TO_COLUMNS, SESSION_GAP_SECONDS


import numpy as np
import pandas as pd
from tqdm import tqdm
from contrastive_embeddings.constants import EventTypes, EVENT_TYPE_TO_COLUMNS, SESSION_GAP_SECONDS


def hash_string_to_float(s: str) -> float:
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF  # normalize to [0,1]

def load_parquet(file_path):
    return pd.read_parquet(file_path)

def extract_session_features(df, event_type):
    df = df.sort_values(["client_id", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')

    # Determine dynamic max price (99th percentile) for normalization
    max_price = 1.0
    if "price" in df.columns:
        max_price = np.percentile(df["price"].dropna(), 99)
        max_price = max(max_price, 1.0)  # Avoid zero or near-zero

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
            val = row.get(col, "")
            if col == "price":
                if isinstance(val, (int, float)) and not pd.isnull(val):
                    val = val / max_price
                else:
                    try:
                        val = float(val)
                        val = val / max_price
                    except (ValueError, TypeError):
                        val = 0.0
            elif isinstance(val, str):
                if val.strip() == "":
                    val = 0.0
                else:
                    val = hash_string_to_float(val)
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0
            features.append(val)

        if features:
            current_session.append(features)

        last_ts = ts

    if current_session:
        sessions.append((current_client, current_session))

    # Aggregate session features per client: mean of all sessions' mean feature vector
    client_feats = {}
    for client_id, sess in sessions:
        arr = np.array(sess)
        if arr.size > 0:
            client_feats.setdefault(client_id, []).append(arr.mean(axis=0))

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

    print(embeddings[np.random.choice(len(embeddings), 5)])

    embeddings = embeddings.astype(np.float32)

    # Print basic statistics before saving embeddings
    print(":::::::::::::::Before Fixes:::::::::::::::")
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    
    # Check for NaN or Inf values
    num_nan = np.isnan(embeddings).sum()
    num_posinf = np.isposinf(embeddings).sum()
    num_neginf = np.isneginf(embeddings).sum()
    
    print(f"NaNs: {num_nan}, +Inf: {num_posinf}, -Inf: {num_neginf}")
    
    # Optional: Check for extreme values
    print("Min value:", np.min(embeddings))
    print("Max value:", np.max(embeddings))
    print("Mean value:", np.mean(embeddings))
    print("Std deviation:", np.std(embeddings))

    finite_vals = embeddings[np.isfinite(embeddings)]
    print("Finite count:", len(finite_vals))
    print("Min finite:", finite_vals.min())
    print("Max finite:", finite_vals.max())

    if len(finite_vals) == 0:
        raise ValueError("All embeddings are non-finite! Check the contrastive model output.")
    
    # Compute dynamic thresholds
    # finite_vals = embeddings[np.isfinite(embeddings)]   # finite_vals is already being assigned.
    lower_bound = np.percentile(finite_vals, 1)   # e.g. 1st percentile
    upper_bound = np.percentile(finite_vals, 99)  # e.g. 99th percentile

    # Replace NaN and infinities
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=upper_bound, neginf=lower_bound)

    # # Clip all values to this range (extra safety)
    # embeddings = np.clip(embeddings, lower_bound, upper_bound)

    # Compute L2 norms for each embedding vector (row-wise)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Avoid division by zero: set zero norms to 1 to keep those vectors unchanged
    norms[norms == 0] = 1  # Avoid divide by zero
    
    # Normalize embeddings by their norms (L2 normalization)
    embeddings = embeddings / norms


    print(":::::::::::::::After Fixes:::::::::::::::")

    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    
    # Check for NaN or Inf values
    num_nan = np.isnan(embeddings).sum()
    num_posinf = np.isposinf(embeddings).sum()
    num_neginf = np.isneginf(embeddings).sum()
    
    print(f"NaNs: {num_nan}, +Inf: {num_posinf}, -Inf: {num_neginf}")
    
    # Optional: Check for extreme values
    print("Min value:", np.min(embeddings))
    print("Max value:", np.max(embeddings))
    print("Mean value:", np.mean(embeddings))
    print("Std deviation:", np.std(embeddings))

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
