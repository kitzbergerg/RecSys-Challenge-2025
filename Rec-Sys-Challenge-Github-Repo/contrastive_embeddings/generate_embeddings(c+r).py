import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hashlib
from contrastive_embeddings.constants import EventTypes, EVENT_TYPE_TO_COLUMNS, SESSION_GAP_SECONDS

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime


def hash_string_to_float(s: str) -> float:
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF  # normalize to [0,1]

def load_parquet(file_path):
    return pd.read_parquet(file_path)

def extract_session_features(df, event_type):
    df = df.sort_values(["client_id", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')

    max_price = 1.0
    if "price" in df.columns:
        max_price = np.percentile(df["price"].dropna(), 99)
        max_price = max(max_price, 1.0)

    sessions = []
    current_session = []
    last_ts = None
    current_client = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {event_type}"):
        client = row["client_id"]
        ts = row["timestamp"]

        if client != current_client or (last_ts and (ts - last_ts).total_seconds() > SESSION_GAP_SECONDS):
            if current_session:
                # sessions.append((current_client, current_session))
                sessions.append((current_client, current_session, last_ts))
            current_session = []
            current_client = client

        features = []
        for col in EVENT_TYPE_TO_COLUMNS[event_type]:
            val = row.get(col, "")
            if col == "price":
                try:
                    val = float(val) / max_price if not pd.isnull(val) else 0.0
                except (ValueError, TypeError):
                    val = 0.0
            elif isinstance(val, str):
                val = 0.0 if val.strip() == "" else hash_string_to_float(val)
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
        # sessions.append((current_client, current_session))
        sessions.append((current_client, current_session, last_ts))

    client_feats = {}
    # for client_id, sess in sessions:
    #     arr = np.array(sess)
    #     if arr.size > 0:
    #         client_feats.setdefault(client_id, []).append(arr.mean(axis=0))

    # for client_id in client_feats:
    #     client_feats[client_id] = np.mean(client_feats[client_id], axis=0)
    
    # DECAY_LAMBDA controls time decay of session weights: weight = exp(-DECAY_LAMBDA * age_in_days).
    # With 0.1, sessions lose half their weight every ~7 days.
    # Higher values = faster decay (more focus on recent), lower = slower decay (older sessions weigh more).
    DECAY_LAMBDA = 0.1  # we can tune this parameter later...

    # Use the latest timestamp in the dataset as "now" for decay reference
    now = df["timestamp"].max()  # max timestamp in your dataset

    client_session_dict = {}
    
    for client_id, sess, sess_ts in sessions:
        arr = np.array(sess)
        if arr.size == 0:
            continue
        mean_feat = arr.mean(axis=0)
        # delta_days = (now - sess_ts).total_seconds() / 86400.0
        # weight = np.exp(-DECAY_LAMBDA * delta_days)
        age_in_days = (now - sess_ts).total_seconds() / (3600*24)
        weight = np.exp(-DECAY_LAMBDA * age_in_days)
        client_session_dict.setdefault(client_id, []).append((mean_feat, weight))
    
    for client_id, weighted_sessions in client_session_dict.items():
        vectors = np.array([v for v, _ in weighted_sessions])
        weights = np.array([w for _, w in weighted_sessions])
        weights = weights / (weights.sum() + 1e-8)
        client_feats[client_id] = np.average(vectors, axis=0, weights=weights)

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
                if d:
                    example = next(iter(d.values()))
                    parts.append(np.zeros_like(example))
                else:
                    parts.append(np.array([]))
        merged[client] = np.concatenate(parts)
    return merged

def main(data_dir, embeddings_dir):
    MAX_EMBEDDING_DIM = 2048

    relevant_clients_path = os.path.join(data_dir, "relevant_clients.npy")
    if not os.path.exists(relevant_clients_path):
        raise FileNotFoundError(f"{relevant_clients_path} not found.")
    relevant_clients = np.load(relevant_clients_path)

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

    # Determine embedding size from a sample (skip empty)
    sample_vec = next((v for v in merged_features.values() if v.size > 0), None)
    if sample_vec is None:
        raise ValueError("No embeddings computed.")
    emb_dim = len(sample_vec)
    if emb_dim > MAX_EMBEDDING_DIM:
        print(f"Truncating embedding dim from {emb_dim} to {MAX_EMBEDDING_DIM}")
        emb_dim = MAX_EMBEDDING_DIM

    # Align to relevant clients
    embeddings = []
    for client_id in relevant_clients:
        vec = merged_features.get(client_id, np.zeros(emb_dim))
        if len(vec) > emb_dim:
            vec = vec[:emb_dim]
        elif len(vec) < emb_dim:
            vec = np.pad(vec, (0, emb_dim - len(vec)))
        embeddings.append(vec)

    embeddings = np.stack(embeddings)
    embeddings = embeddings.astype(np.float32)

    print(":::::::::::::::Before Fixes:::::::::::::::")
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print(f"NaNs: {np.isnan(embeddings).sum()}, +Inf: {np.isposinf(embeddings).sum()}, -Inf: {np.isneginf(embeddings).sum()}")
    print("Min:", np.min(embeddings), "Max:", np.max(embeddings))
    print("Mean:", np.mean(embeddings), "Std:", np.std(embeddings))

    finite_vals = embeddings[np.isfinite(embeddings)]
    lower_bound = np.percentile(finite_vals, 1)
    upper_bound = np.percentile(finite_vals, 99)

    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=upper_bound, neginf=lower_bound)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    print(":::::::::::::::After Fixes:::::::::::::::")
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print(f"NaNs: {np.isnan(embeddings).sum()}, +Inf: {np.isposinf(embeddings).sum()}, -Inf: {np.isneginf(embeddings).sum()}")
    print("Min:", np.min(embeddings), "Max:", np.max(embeddings))
    print("Mean:", np.mean(embeddings), "Std:", np.std(embeddings))

    embeddings = embeddings.astype(np.float16)

    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, "client_ids.npy"), relevant_clients.astype(np.int64))
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeddings)

    print(f"Saved {len(relevant_clients)} embeddings to {embeddings_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Input directory with parquet files and input/relevant_clients.npy")
    parser.add_argument("--embeddings-dir", required=True, help="Directory to save embeddings")
    args = parser.parse_args()
    main(args.data_dir, args.embeddings_dir)
