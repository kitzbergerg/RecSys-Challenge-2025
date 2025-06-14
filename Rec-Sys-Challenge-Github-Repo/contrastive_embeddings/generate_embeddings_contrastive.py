import os
import sys
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import random 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Constants
from enum import Enum

class EventTypes(Enum):
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PRODUCT_BUY = "product_buy"
    PAGE_VISIT = "page_visit"
    SEARCH_QUERY = "search_query"

EVENT_TYPE_TO_COLUMNS = {
    EventTypes.ADD_TO_CART: ["sku"],
    EventTypes.REMOVE_FROM_CART: ["sku"],
    EventTypes.PRODUCT_BUY: ["sku"],
    EventTypes.PAGE_VISIT: ["url"],
    EventTypes.SEARCH_QUERY: ["query"]
}

SESSION_GAP_SECONDS = 1800

# Utils
def hash_string_to_float(s: str) -> float:
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def load_parquet(file_path):
    return pd.read_parquet(file_path)

# Dataset and Model
class SessionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        emb1, emb2, label = self.pairs[idx]
        return (
            torch.tensor(emb1, dtype=torch.float32),
            torch.tensor(emb2, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# Embedding Extractors
def extract_event_embeddings(df, event_type):
    df = df.sort_values(["client_id", "timestamp"])
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

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
            if isinstance(val, str):
                val = 0.0 if val.strip() == "" else hash_string_to_float(val)
            else:
                try:
                    val = float(val)
                except:
                    val = 0.0
            features.append(val)

        if features:
            current_session.append(features)
        last_ts = ts

    if current_session:
        sessions.append((current_client, current_session, last_ts))

    return sessions

import numpy as np
import random
from tqdm import tqdm

def generate_training_pairs(sessions, num_negatives=1):
    pairs = []
    client_sessions = {}

    # Step 1: Group sessions per client and filter by valid length
    for client_id, session, _ in sessions:
        if len(session) >= 2:
            client_sessions.setdefault(client_id, []).append(np.array(session))

    all_clients = list(client_sessions.keys())

    for client_id, session_np, _ in tqdm(sessions, desc="Generating training pairs"):
        if len(session_np) < 2:
            continue

        # Convert to np.array only once
        session_np = np.array(session_np)
        if len(set(len(x) for x in session_np)) > 1:
            continue  # skip ragged sessions

        # Precompute anchor
        anchor = session_np.mean(axis=0)

        # Add positive pairs
        pairs.extend([(anchor, event, 1) for event in session_np])

        # Add negative pairs
        for _ in range(num_negatives):
            while True:
                neg_client = random.choice(all_clients)
                if neg_client != client_id:
                    break
            neg_session = random.choice(client_sessions[neg_client])
            neg_event = neg_session[random.randint(0, len(neg_session) - 1)]
            pairs.append((anchor, neg_event, -1))

    return pairs

def extract_query_embeddings(df, model, text_column="query"):
    df = df.sort_values(["client_id", "timestamp"])
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    sessions = []
    current_session = []
    last_ts = None
    current_client = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building SBERT sessions for {text_column}..."):
        client = row["client_id"]
        ts = row["timestamp"]
        text = row.get(text_column, "")
        if not isinstance(text, str) or text.strip() == "":
            continue
        if client != current_client or (last_ts and (ts - last_ts).total_seconds() > SESSION_GAP_SECONDS):
            if current_session:
                sessions.append((current_client, current_session, last_ts))
            current_session = []
            current_client = client
        current_session.append((text.strip(), ts))
        last_ts = ts

    if current_session:
        sessions.append((current_client, current_session, last_ts))

    client_feats = {}
    now = df["timestamp"].max()
    DECAY_LAMBDA = 0.1
    for client_id, session, sess_ts in tqdm(sessions, desc="Encoding SBERT sessions"):
        texts = [q for q, _ in session]
        if not texts:
            continue
        # embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = model.encode(texts, device='cuda', show_progress_bar=False, convert_to_numpy=True)

        mean_embedding = embeddings.mean(axis=0)
        age_in_days = (now - sess_ts).total_seconds() / 86400
        weight = np.exp(-DECAY_LAMBDA * age_in_days)
        client_feats.setdefault(client_id, []).append((mean_embedding, weight))

    avg_feats = {}
    for client_id, weighted_sessions in client_feats.items():
        vectors = np.array([v for v, _ in weighted_sessions])
        weights = np.array([w for _, w in weighted_sessions])
        weights /= (weights.sum() + 1e-8)
        avg_feats[client_id] = np.average(vectors, axis=0, weights=weights)

    return avg_feats

def train_contrastive_model(pairs, input_dim):
    model = ContrastiveModel(input_dim=input_dim).to("cuda")
    x = torch.randn(10).to("cuda")
    print("Tensor on:", x.device)

    loader = DataLoader(SessionDataset(pairs), batch_size=512, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CosineEmbeddingLoss()

    model.train()
    for epoch in tqdm(range(3), desc="Training contrastive model"):
        for x1, x2, y in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
            z1 = model(x1)
            z2 = model(x2)
            loss = loss_fn(z1, z2, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def generate_final_embeddings(sessions, model):
    client_embeddings = {}
    for client_id, session, _ in tqdm(sessions, desc="Generating final embeddings"):
        arr = np.array(session)
        if arr.size == 0:
            continue
        with torch.no_grad():
            z = model(torch.tensor(arr, dtype=torch.float32).cuda()).cpu().numpy()
        client_embeddings[client_id] = z.mean(axis=0)
    return client_embeddings

def merge_features(*feature_dicts):
    all_clients = set().union(*[set(d) for d in feature_dicts])
    merged = {}
    for client in tqdm(all_clients, desc="Merging features"):
        parts = []
        for d in feature_dicts:
            v = d.get(client)
            if v is None:
                v = np.zeros_like(next(iter(d.values())))
            parts.append(v)
        merged[client] = np.concatenate(parts)
    return merged

def main(data_dir, embeddings_dir):
    
    print("CUDA Available:", torch.cuda.is_available())
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    os.makedirs(embeddings_dir, exist_ok=True)
    relevant_clients = np.load(os.path.join(data_dir, "relevant_clients.npy"))

    all_sessions = []
    for event in tqdm(EventTypes, desc="Extracting structured events"):
        # if event in {EventTypes.SEARCH_QUERY, EventTypes.PAGE_VISIT}:
        #     continue
        file_path = os.path.join(data_dir, f"{event.value}.parquet")
        if os.path.isfile(file_path):
            df = load_parquet(file_path)
            all_sessions += extract_event_embeddings(df, event)

    training_pairs = generate_training_pairs(all_sessions, num_negatives=1)
    input_dim = len(training_pairs[0][0])
    contrastive_model = train_contrastive_model(training_pairs, input_dim)
    contrastive_embeddings = generate_final_embeddings(all_sessions, contrastive_model)

    # ####don't use this: sbert_model = SentenceTransformer("all-roberta-large-v1").to("cuda")
    # sbert_model = SentenceTransformer("all-roberta-large-v1")
    # sbert_model = sbert_model.to("cuda")


    # query_embeddings = {}
    # query_file = os.path.join(data_dir, f"{EventTypes.SEARCH_QUERY.value}.parquet")
    # if os.path.isfile(query_file):
    #     df_query = load_parquet(query_file)
    #     query_embeddings = extract_query_embeddings(df_query, sbert_model, text_column="query")

    # page_embeddings = {}
    # page_file = os.path.join(data_dir, f"{EventTypes.PAGE_VISIT.value}.parquet")
    # if os.path.isfile(page_file):
    #     df_page = load_parquet(page_file)
    #     page_embeddings = extract_query_embeddings(df_page, sbert_model, text_column="url")

    # merged = merge_features(contrastive_embeddings, query_embeddings, page_embeddings)

    merged = merge_features(contrastive_embeddings)
    emb_dim = len(next(iter(merged.values())))
    client_matrix = []

    for client_id in tqdm(relevant_clients, desc="Assembling final embedding matrix"):
        vec = merged.get(client_id, np.zeros(emb_dim))
        client_matrix.append(vec)

    final_embs = normalize(np.stack(client_matrix)).astype(np.float16)
    np.save(os.path.join(embeddings_dir, "client_ids.npy"), relevant_clients.astype(np.int64))
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), final_embs)
    print("Saved embeddings to", embeddings_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    args = parser.parse_args()
    main(args.data_dir, args.embeddings_dir)
