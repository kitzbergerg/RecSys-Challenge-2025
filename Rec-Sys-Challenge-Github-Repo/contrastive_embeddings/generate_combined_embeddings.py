# Integrated embedding generation pipeline that combines:
#  - Contrastive event-based embeddings (from all event types like product_buy, add_to_cart, etc.)
#  - SBERT average query embeddings (from search_query events)
#  These are then merged using merge_features() function.


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer

from constants import EventTypes, EVENT_TYPE_TO_COLUMNS, SESSION_GAP_SECONDS

def hash_string_to_float(s: str) -> float:
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF  # normalize to [0,1]

def load_parquet(file_path):
    return pd.read_parquet(file_path)

def extract_contrastive_features(df, event_type):
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
        sessions.append((current_client, current_session, last_ts))

    now = df["timestamp"].max()
    DECAY_LAMBDA = 0.1
    client_session_dict = {}

    for client_id, sess, sess_ts in sessions:
        arr = np.array(sess)
        if arr.size == 0:
            continue
        mean_feat = arr.mean(axis=0)
        age_in_days = (now - sess_ts).total_seconds() / (3600*24)
        weight = np.exp(-DECAY_LAMBDA * age_in_days)
        client_session_dict.setdefault(client_id, []).append((mean_feat, weight))

    client_feats = {}
    for client_id, weighted_sessions in client_session_dict.items():
        vectors = np.array([v for v, _ in weighted_sessions])
        weights = np.array([w for _, w in weighted_sessions])
        weights = weights / (weights.sum() + 1e-8)
        client_feats[client_id] = np.average(vectors, axis=0, weights=weights)

    return client_feats

def extract_query_features(df, model):
    df = df.sort_values(["client_id", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    sessions = []
    current_session = []
    last_ts = None
    current_client = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing search_query (SBERT)"):
        client = row["client_id"]
        ts = row["timestamp"]
        query = row.get("query", "")

        if not isinstance(query, str) or query.strip() == "":
            continue

        if client != current_client or (last_ts and (ts - last_ts).total_seconds() > SESSION_GAP_SECONDS):
            if current_session:
                sessions.append((current_client, current_session, last_ts))
            current_session = []
            current_client = client

        current_session.append((query.strip(), ts))
        last_ts = ts

    if current_session:
        sessions.append((current_client, current_session, last_ts))

    DECAY_LAMBDA = 0.1
    now = df["timestamp"].max()

    client_session_dict = {}
    for client_id, session, sess_ts in sessions:
        queries = [q for q, _ in session]
        if not queries:
            continue
        embeddings = model.encode(queries, show_progress_bar=False, convert_to_numpy=True)
        mean_embedding = embeddings.mean(axis=0)
        age_in_days = (now - sess_ts).total_seconds() / 86400
        weight = np.exp(-DECAY_LAMBDA * age_in_days)
        client_session_dict.setdefault(client_id, []).append((mean_embedding, weight))

    client_feats = {}
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
    # sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    sbert_model = SentenceTransformer("all-roberta-large-v1")
    sbert_model = sbert_model.to("cuda")  # As CUDA is available

    for event in EventTypes:
        file_path = os.path.join(data_dir, f"{event.value}.parquet")
        if not os.path.isfile(file_path):
            print(f"File {file_path} not found. Skipping {event}.")
            continue
        df = load_parquet(file_path)
        if event == EventTypes.SEARCH_QUERY:
            feats = extract_query_features(df, sbert_model)
        else:
            feats = extract_contrastive_features(df, event)
        feature_dicts.append(feats)

    merged_features = merge_features(feature_dicts)

    # Determine embedding size
    sample_vec = next((v for v in merged_features.values() if v.size > 0), None)
    if sample_vec is None:
        raise ValueError("No embeddings computed.")
    emb_dim = len(sample_vec)
    if emb_dim > MAX_EMBEDDING_DIM:
        emb_dim = MAX_EMBEDDING_DIM

    embeddings = []
    for client_id in relevant_clients:
        vec = merged_features.get(client_id, np.zeros(emb_dim))
        if len(vec) > emb_dim:
            vec = vec[:emb_dim]
        elif len(vec) < emb_dim:
            vec = np.pad(vec, (0, emb_dim - len(vec)))
        embeddings.append(vec)

    embeddings = np.stack(embeddings)
    embeddings = np.nan_to_num(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    embeddings = embeddings.astype(np.float16)
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, "client_ids.npy"), relevant_clients.astype(np.int64))
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeddings)

    print("Combined embeddings saved to:", embeddings_dir)
    print("Shape:", embeddings.shape, "Dtype:", embeddings.dtype)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing parquet files and relevant_clients.npy")
    parser.add_argument("--embeddings-dir", required=True, help="Output directory for final embeddings")
    args = parser.parse_args()
    main(args.data_dir, args.embeddings_dir)
