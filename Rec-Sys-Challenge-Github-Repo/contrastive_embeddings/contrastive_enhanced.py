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
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import joblib
from enum import Enum
from transformers import AutoModel, AutoTokenizer


# Constants
class EventTypes(Enum):
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PRODUCT_BUY = "product_buy"
    PAGE_VISIT = "page_visit"
    SEARCH_QUERY = "search_query"

EVENT_TYPE_EMB_SIZE = 5  # One-hot encoding size
HASHED_FEATURE_SIZE = 16 
STRUCTURED_EVENT_DIM = EVENT_TYPE_EMB_SIZE + HASHED_FEATURE_SIZE
TEXT_EMBEDDING_DIM = 384  
REDUCED_TEXT_DIM = 128
SESSION_GAP_SECONDS = 1800  
DECAY_LAMBDA = 0.1          

# Utils
def multi_dim_hash(s: str, dim=HASHED_FEATURE_SIZE) -> list:
    """Multi-dimensional feature hashing with salt for better representation"""
    if not s or pd.isna(s) or str(s).strip() == "":
        return [0.0] * dim
    return [hash_string_to_float(f"{s}_SALT_{i}") for i in range(dim)]

def hash_string_to_float(s: str) -> float:
    """Consistent string hashing to float in [0,1]"""
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h[:16], 16) / 0xFFFFFFFFFFFFFFFF

def load_parquet(file_path):
    return pd.read_parquet(file_path)

# Dataset and Model
class SessionDataset(Dataset):
    def __init__(self, global_event_features_matrix, global_event_client_ids,
                 client_to_event_indices, client_original_sessions_indices,
                 num_negatives=5, max_positives=3):
        
        self.global_event_features_matrix = global_event_features_matrix
        self.global_event_client_ids = global_event_client_ids
        self.client_to_event_indices = client_to_event_indices
        self.client_original_sessions_indices = client_original_sessions_indices
        self.num_negatives = num_negatives
        self.max_positives = max_positives
        
        self.all_client_ids = list(client_to_event_indices.keys())
        self.num_all_clients = len(self.all_client_ids)
        
        self.total_anchors = len(global_event_features_matrix)
        self.total_effective_pairs = self.total_anchors * (max_positives + num_negatives)
        
        self.event_idx_to_session_indices = {}
        for client_id, sessions_list in client_original_sessions_indices.items():
            for session_indices in sessions_list:
                for event_idx in session_indices:
                    self.event_idx_to_session_indices[event_idx] = session_indices
                    
    def __len__(self):
        return self.total_effective_pairs

    def __getitem__(self, idx):
        anchor_global_idx = idx % self.total_anchors 
        
        anchor_features = self.global_event_features_matrix[anchor_global_idx]
        anchor_client_id = self.global_event_client_ids[anchor_global_idx]

        if random.random() < (self.max_positives / (self.max_positives + self.num_negatives)):
            target_features = None
            
            session_indices_for_anchor = self.event_idx_to_session_indices.get(anchor_global_idx)
            if session_indices_for_anchor:
                possible_pos_indices_in_session = [i for i in session_indices_for_anchor if i != anchor_global_idx]
                if possible_pos_indices_in_session:
                    pos_event_global_idx = random.choice(possible_pos_indices_in_session)
                    target_features = self.global_event_features_matrix[pos_event_global_idx]
            
            if target_features is None:
                other_event_indices_for_client = [i for i in self.client_to_event_indices.get(anchor_client_id, []) if i != anchor_global_idx]
                if other_event_indices_for_client:
                    pos_event_global_idx = random.choice(other_event_indices_for_client)
                    target_features = self.global_event_features_matrix[pos_event_global_idx]
            
            if target_features is not None:
                return (torch.tensor(anchor_features, dtype=torch.float32),
                        torch.tensor(target_features, dtype=torch.float32),
                        torch.tensor(1.0, dtype=torch.float32))
            
        if self.num_all_clients <= 1: 
            return (torch.tensor(anchor_features, dtype=torch.float32),
                    torch.tensor(anchor_features, dtype=torch.float32),
                    torch.tensor(1.0, dtype=torch.float32))

        neg_client_id = anchor_client_id
        attempts = 0
        max_attempts = 100
        
        while (neg_client_id == anchor_client_id or not self.client_to_event_indices.get(neg_client_id)) and attempts < max_attempts:
            neg_client_id = random.choice(self.all_client_ids)
            attempts += 1
        
        if neg_client_id == anchor_client_id or not self.client_to_event_indices.get(neg_client_id):
            return (torch.tensor(anchor_features, dtype=torch.float32),
                    torch.tensor(anchor_features, dtype=torch.float32),
                    torch.tensor(1.0, dtype=torch.float32))

        neg_event_global_idx = random.choice(self.client_to_event_indices[neg_client_id])
        negative_features = self.global_event_features_matrix[neg_event_global_idx]
        
        return (torch.tensor(anchor_features, dtype=torch.float32),
                torch.tensor(negative_features, dtype=torch.float32),
                torch.tensor(-1.0, dtype=torch.float32))


class EnhancedContrastiveModel(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return nn.functional.normalize(self.encoder(x), p=2, dim=1)

# Feature Engineering
def create_event_features(row):
    """Create enriched feature vector for any event type"""
    event_type_emb = [0] * EVENT_TYPE_EMB_SIZE
    try:
        event_idx = {
            EventTypes.ADD_TO_CART: 0,
            EventTypes.REMOVE_FROM_CART: 1,
            EventTypes.PRODUCT_BUY: 2,
            EventTypes.PAGE_VISIT: 3,
            EventTypes.SEARCH_QUERY: 4
        }[EventTypes(row["event_type"])]
        event_type_emb[event_idx] = 1
    except:
        pass
    
    feature_value = ""
    if row["event_type"] in [EventTypes.ADD_TO_CART.value, 
                             EventTypes.REMOVE_FROM_CART.value, 
                             EventTypes.PRODUCT_BUY.value]:
        feature_value = row.get("sku", "")
    elif row["event_type"] == EventTypes.PAGE_VISIT.value:
        feature_value = row.get("url", "")
    elif row["event_type"] == EventTypes.SEARCH_QUERY.value:
        feature_value = row.get("query", "")
    
    hashed_features = multi_dim_hash(str(feature_value))
    
    return np.array(event_type_emb + hashed_features)

# Session Processing
def build_unified_sessions(df, min_events=2):
    """Build sessions across all event types with unified session gap"""
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    df = df.sort_values(["client_id", "timestamp"])
    sessions = []
    current_session_events = []
    current_client = None
    last_ts = None
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building unified sessions"):
        client = row["client_id"]
        ts = row["timestamp"]
        
        if client != current_client or (last_ts and (ts - last_ts).total_seconds() > SESSION_GAP_SECONDS):
            if current_session_events:
                sessions.append((current_client, current_session_events, last_ts))
            current_session_events = []
            current_client = client
        
        event_features = create_event_features(row)
        current_session_events.append((event_features, ts))
        last_ts = ts
    
    if current_session_events:
        sessions.append((current_client, current_session_events, last_ts))
    
    return [(c, events, ts) for c, events, ts in sessions if len(events) >= min_events]

# Data Preparation for Contrastive Learning Dataset
def prepare_contrastive_data(sessions):
    """
    Prepares data structures for the SessionDataset to sample contrastive pairs on-the-fly.
    Returns:
        global_event_features_matrix (np.ndarray): All event feature vectors stacked.
        global_event_client_ids (np.ndarray): Client ID for each event in the matrix.
        client_to_event_indices (dict): Maps client_id to list of global_event_features_matrix indices.
        client_original_sessions_indices (dict): Maps client_id to list of sessions, where each session is a list of global event indices.
    """
    all_raw_event_features = []
    all_event_client_ids = []
    
    temp_client_original_sessions_indices = {}
    
    global_idx_counter = 0

    for client_id, session_events_with_ts, _ in tqdm(sessions, desc="Preparing events for contrastive dataset"):
        current_session_global_indices = []
        for event_feature_vector, _ in session_events_with_ts:
            all_raw_event_features.append(event_feature_vector)
            all_event_client_ids.append(client_id)
            current_session_global_indices.append(global_idx_counter)
            global_idx_counter += 1
        
        if current_session_global_indices:
            temp_client_original_sessions_indices.setdefault(client_id, []).append(current_session_global_indices)

    if not all_raw_event_features:
        print("No raw event features found after processing sessions. Cannot prepare contrastive data.")
        return None, None, None, None

    global_event_features_matrix = np.vstack(all_raw_event_features).astype(np.float32)
    global_event_client_ids = np.array(all_event_client_ids)

    client_to_event_indices = {}
    for i, cid in enumerate(global_event_client_ids):
        client_to_event_indices.setdefault(cid, []).append(i)

    return global_event_features_matrix, global_event_client_ids, client_to_event_indices, temp_client_original_sessions_indices


# Training
def train_contrastive_model(dataset, input_dim, device: torch.device):
    """Train contrastive model with enhanced settings"""
    model = EnhancedContrastiveModel(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    num_epochs = 10 # Increased from 5 to 10 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) 
    
    loss_fn = nn.CosineEmbeddingLoss(margin=0.2)
    
    num_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 0 
    print(f"Using {num_workers} DataLoader workers.")
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=num_workers, pin_memory=True) 
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x1, x2, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            optimizer.zero_grad()
            z1 = model(x1)
            z2 = model(x2)
            loss = loss_fn(z1, z2, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")
    
    return model

# Embedding Processing
def extract_structured_embeddings(sessions, model, device: torch.device):
    """Generate client embeddings from structured events with time decay"""
    if not sessions:
        return {}
    
    max_time = datetime.min
    for _, _, sess_ts in sessions:
        if sess_ts and sess_ts > max_time:
            max_time = sess_ts
    
    client_embeddings = {}
    model.eval()
    
    # MODIFIED: Convert the structured embedding model to half-precision (FP16)
    if device.type == 'cuda':
        print("Converting structured embedding model to half-precision (FP16).")
        model.half()

    for client_id, session_events_with_ts, sess_ts in tqdm(sessions, desc="Extracting structured embeddings"):
        age_days = (max_time - sess_ts).total_seconds() / 86400 if max_time > datetime.min else 0
        session_weight = np.exp(-DECAY_LAMBDA * age_days)
        
        event_features = [feat for feat, _ in session_events_with_ts]
        if not event_features:
            continue

        with torch.no_grad():
            events_np_array = np.array(event_features) 
            events_tensor = torch.tensor(events_np_array, dtype=torch.float32).to(device)
            
            # If model is in half(), inputs need to be half() as well for consistency
            if device.type == 'cuda':
                events_tensor = events_tensor.half()
            
            embeddings = model(events_tensor).cpu().numpy()
        
        session_level_embedding = embeddings.mean(axis=0) * session_weight
        
        if client_id in client_embeddings:
            prev_emb, prev_weight = client_embeddings[client_id]
            new_weight = prev_weight + session_weight
            new_emb = (prev_emb * prev_weight + session_level_embedding * session_weight) / (new_weight if new_weight > 0 else 1e-9)
            client_embeddings[client_id] = (new_emb, new_weight)
        else:
            client_embeddings[client_id] = (session_level_embedding, session_weight)
    
    return {cid: emb for cid, (emb, weight) in client_embeddings.items()}


# MODIFIED: extract_text_embeddings function
def extract_text_embeddings(df: pd.DataFrame, tokenizer: AutoTokenizer, model: AutoModel, text_column="text", device: torch.device = torch.device("cpu")):
    """
    Generate text embeddings using a Hugging Face AutoModel and AutoTokenizer,
    with time decay aggregation and manual mean pooling.
    """
    if df.empty:
        return {}
    
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    max_time = df["timestamp"].max()
    client_embeddings = {}
    
    model.eval() 
    # Convert the model to half-precision (float16) for faster inference on GPU
    if device.type == 'cuda':
        print("Converting text embedding model to half-precision (FP16) for CUDA inference.")
        model.half()

    batch_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Processing text embeddings"):
            batch = df.iloc[i:i+batch_size]
            texts = batch[text_column].astype(str).tolist()
            
            # --- MODIFIED: Direct Hugging Face Model Usage ---
            encoded_input = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            model_output = model(**encoded_input)
            
            # Perform Mean Pooling (as typically done by SentenceTransformers)
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # If model is in half(), embeddings might be half(), ensure operations are compatible
            if device.type == 'cuda':
                token_embeddings = token_embeddings.half()
                input_mask_expanded = input_mask_expanded.half() # Ensure consistent types for multiplication

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask

            embeddings = sentence_embeddings.cpu().numpy()
            # ----------------------------------------------------
            
            for idx, (_, row) in enumerate(batch.iterrows()):
                client_id = row["client_id"]
                ts = row["timestamp"]
                age_days = (max_time - ts).total_seconds() / 86400
                weight = np.exp(-DECAY_LAMBDA * age_days)
                emb = embeddings[idx]
                
                if client_id in client_embeddings:
                    prev_emb, prev_weight = client_embeddings[client_id]
                    new_weight = prev_weight + weight
                    new_emb = (prev_emb * prev_weight + emb * weight) / (new_weight if new_weight > 0 else 1e-9)
                    client_embeddings[client_id] = (new_emb, new_weight)
                else:
                    client_embeddings[client_id] = (emb, weight)
    
    return {cid: emb for cid, (emb, weight) in client_embeddings.items()}

# Feature Integration
def reduce_text_embeddings(text_embeddings, target_dim=REDUCED_TEXT_DIM):
    """Dimensionality reduction for text embeddings"""
    if not text_embeddings:
        return {}, None
        
    embeddings = np.array(list(text_embeddings.values()))
    
    if embeddings.shape[0] == 0:
        print("No text embeddings available for reduction after converting to numpy array.")
        return {}, None
    
    if embeddings.shape[0] < target_dim and embeddings.shape[0] > 0:
        print(f"Warning: Number of text embeddings ({embeddings.shape[0]}) is less than target_dim ({target_dim}). Reducing target_dim to actual dimension.")
        # Ensure target_dim is at least 1 if there are any embeddings
        actual_target_dim = max(1, embeddings.shape[0]) 
        # Check if actual_target_dim is greater than the current embedding dimension
        if actual_target_dim > embeddings.shape[1]:
            actual_target_dim = embeddings.shape[1]
        print(f"Adjusted target_dim for PCA to {actual_target_dim}.")
        pca = PCA(n_components=actual_target_dim)
    elif embeddings.shape[1] < target_dim: # If the original embedding dim is smaller than target_dim
        print(f"Warning: Original text embedding dimension ({embeddings.shape[1]}) is less than target_dim ({target_dim}). Setting target_dim to original dimension.")
        pca = PCA(n_components=embeddings.shape[1])
    else:
        pca = PCA(n_components=target_dim)

    reduced = pca.fit_transform(embeddings)
    
    reduced_embeddings = {}
    client_ids_order = list(text_embeddings.keys())
    for i, cid in enumerate(client_ids_order):
        reduced_embeddings[cid] = reduced[i]
        
    return reduced_embeddings, pca

def merge_features(structured_emb, text_emb):
    """Merge structured and text embeddings with alignment"""
    # Ensure these zero vectors match the actual output dimensions
    # For structured: EnhancedContrastiveModel output_dim is 128
    # For text: REDUCED_TEXT_DIM (which is 128)
    structured_zero_vec = np.zeros(128, dtype=np.float32) 
    text_zero_vec = np.zeros(REDUCED_TEXT_DIM, dtype=np.float32)
    
    if structured_emb:
        sample_structured_emb_dim = next(iter(structured_emb.values())).shape[0]
    else:
        sample_structured_emb_dim = structured_zero_vec.shape[0]
    
    # Check if text_emb has any actual entries to determine its dimension
    if text_emb:
        sample_text_emb_dim = next(iter(text_emb.values())).shape[0]
    else:
        sample_text_emb_dim = text_zero_vec.shape[0]

    expected_combined_dim = sample_structured_emb_dim + sample_text_emb_dim

    all_clients = set(structured_emb.keys()) | set(text_emb.keys())
    merged = {}
    
    for client in tqdm(all_clients, desc="Merging features"):
        struct = structured_emb.get(client, np.zeros(sample_structured_emb_dim, dtype=np.float32))
        text = text_emb.get(client, np.zeros(sample_text_emb_dim, dtype=np.float32))
        merged[client] = np.concatenate([struct, text])
        
    return merged

# Main Workflow
def main(data_dir, embeddings_dir):
    # Environment setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: CUDA is not available. Training and embedding extraction will be very slow on CPU.")
        print("Please ensure your PyTorch installation has CUDA support and your GPU drivers are up-to-date.")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Load relevant clients
    relevant_clients = None
    relevant_clients_path = os.path.join(data_dir, "input/relevant_clients.npy")
    if os.path.exists(relevant_clients_path):
        print(f"Loading relevant clients from {relevant_clients_path}")
        relevant_clients = np.load(relevant_clients_path)
    else:
        print(f"Info: relevant_clients.npy not found at {relevant_clients_path}. Embeddings will be generated for all found clients.")
    
    # ========== STRUCTURED EVENTS PROCESSING ========== #
    structured_dfs = []
    processable_structured_events = [EventTypes.ADD_TO_CART, EventTypes.REMOVE_FROM_CART, EventTypes.PRODUCT_BUY]

    print("\n--- Processing Structured Events ---")
    for event in processable_structured_events:
        file_path = os.path.join(data_dir, f"{event.value}.parquet")
        if os.path.exists(file_path):
            print(f"Loading structured event file: {file_path}")
            df = load_parquet(file_path)
            df["event_type"] = event.value
            structured_dfs.append(df)
        else:
            print(f"Structured event file not found (skipping): {file_path}")
            
    structured_embeddings = {}
    if not structured_dfs:
        print("No structured event data loaded. Structured embeddings will be empty.")
    else:
        structured_df = pd.concat(structured_dfs)
        sessions = build_unified_sessions(structured_df)
        
        global_event_features_matrix, global_event_client_ids, \
        client_to_event_indices, client_original_sessions_indices = \
            prepare_contrastive_data(sessions)

        if global_event_features_matrix is None or global_event_features_matrix.shape[0] == 0:
            print("No events processed for structured embeddings training. Skipping model training and embedding extraction.")
            structured_embeddings = {}
        else:
            print(f"Total individual events available for structured training: {global_event_features_matrix.shape[0]}")
            
            train_dataset = SessionDataset(
                global_event_features_matrix,
                global_event_client_ids,
                client_to_event_indices,
                client_original_sessions_indices,
                num_negatives=5,
                max_positives=3
            )
            
            input_dim = global_event_features_matrix.shape[1]
            
            print("Starting structured embedding model training...")
            model = train_contrastive_model(train_dataset, input_dim, device)
            
            print("\nExtracting final structured client embeddings...")
            structured_embeddings = extract_structured_embeddings(sessions, model, device)
            print(f"Generated structured embeddings for {len(structured_embeddings)} clients.")
    
    # ========== TEXT EVENTS PROCESSING ========== #
    text_dfs = []
    processable_text_events = [
        EventTypes.SEARCH_QUERY,
        EventTypes.PAGE_VISIT
    ]

    print("\n--- Processing Text Events ---")
    if not processable_text_events:
        print("Skipping text event processing as no text event types are enabled.")
        
    for event in processable_text_events:
        file_path = os.path.join(data_dir, f"{event.value}.parquet")
        if os.path.exists(file_path):
            print(f"Loading text event file: {file_path}")
            df = load_parquet(file_path)
            df["event_type"] = event.value
            if event == EventTypes.SEARCH_QUERY:
                df["text"] = df["query"]
            else: # PAGE_VISIT
                df["text"] = df["url"]
            text_dfs.append(df)
        else:
            print(f"Text event file not found (skipping): {file_path}")
            
    text_embeddings = {}
    if not text_dfs:
        print("No text event data loaded. Text embeddings will be empty.")
    else:
        text_df = pd.concat(text_dfs)
        text_df = text_df.dropna(subset=['text'])
        text_df = text_df[text_df['text'].astype(str).str.strip() != '']

        if text_df.empty:
            print("No valid text data after cleaning. Skipping SBERT embedding generation.")
            text_embeddings = {}
        else:
            print(f"Loading Hugging Face AutoModel and AutoTokenizer (all-MiniLM-L6-v2)...")
            try:
                sbert_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                sbert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
            except Exception as e:
                print(f"Error loading Hugging Face model: {e}. Check internet connection or model name.")
                print("Skipping text embedding generation.")
                sbert_model = None
                sbert_tokenizer = None 
                
            if sbert_model and sbert_tokenizer:
                print("Extracting text embeddings...")
                text_embeddings_full = extract_text_embeddings(text_df, sbert_tokenizer, sbert_model, text_column="text", device=device)
                
                if text_embeddings_full:
                    print(f"Reducing dimensionality of text embeddings from {TEXT_EMBEDDING_DIM} to {REDUCED_TEXT_DIM}...")
                    text_embeddings, pca_model = reduce_text_embeddings(text_embeddings_full)
                    if pca_model:
                        joblib.dump(pca_model, os.path.join(embeddings_dir, "pca_model.pkl"))
                    print(f"Generated text embeddings for {len(text_embeddings)} clients.")
                else:
                    print("No text embeddings generated (perhaps no valid text data after SBERT processing).")
                    text_embeddings = {}

    # ========== FEATURE INTEGRATION ========== #
    print("\n--- Merging Structured and Text Features ---")
    merged_embeddings = merge_features(structured_embeddings, text_embeddings)
    print(f"Merged embeddings for {len(merged_embeddings)} clients.")
    
    # ========== SAVE RESULTS ========== #
    print("\n--- Saving Final Client Embeddings ---")
    final_client_ids = []
    final_embeddings_list = []

    current_structured_dim = 128 
    current_text_dim = REDUCED_TEXT_DIM 
    
    if structured_embeddings:
        current_structured_dim = next(iter(structured_embeddings.values())).shape[0]
    if text_embeddings:
        current_text_dim = next(iter(text_embeddings.values())).shape[0]
        
    expected_combined_dim = current_structured_dim + current_text_dim
    
    if relevant_clients is not None:
        clients_to_process = relevant_clients
        print(f"Saving embeddings for {len(clients_to_process)} clients from the 'relevant_clients' list.")
        for client_id in tqdm(clients_to_process, desc="Collecting relevant client embeddings"):
            emb = merged_embeddings.get(client_id, np.zeros(expected_combined_dim, dtype=np.float32))
            final_client_ids.append(client_id)
            final_embeddings_list.append(emb)
    else:
        clients_to_process = list(merged_embeddings.keys())
        print(f"Saving embeddings for all {len(clients_to_process)} clients for whom merged data was found.")
        for client_id in tqdm(clients_to_process, desc="Collecting all generated client embeddings"):
            emb = merged_embeddings.get(client_id)
            final_client_ids.append(client_id)
            final_embeddings_list.append(emb)
    
    if not final_embeddings_list:
        print("No embeddings to save after filtering and processing.")
        # Ensure array shapes are correct even if empty
        np.save(os.path.join(embeddings_dir, "client_ids.npy"), np.array([]))
        np.save(os.path.join(embeddings_dir, "embeddings.npy"), np.array([], dtype=np.float16).reshape(0, expected_combined_dim))
        print("Saved empty client_ids.npy and embeddings.npy.")
        return

    final_embeddings_array = np.array(final_embeddings_list, dtype=np.float32)
    # Ensure normalization happens on non-zero arrays
    if final_embeddings_array.shape[0] > 0 and final_embeddings_array.shape[1] > 0:
        final_embeddings_array = normalize(final_embeddings_array, axis=1).astype(np.float32) 
    else:
        print("Warning: Final embeddings array is empty or has zero dimension, skipping normalization.")
        
    final_embeddings_array = final_embeddings_array.astype(np.float16)

    np.save(os.path.join(embeddings_dir, "client_ids.npy"), np.array(final_client_ids))
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), final_embeddings_array)
    print(f"Successfully saved {len(final_client_ids)} embeddings (shape {final_embeddings_array.shape}, dtype {final_embeddings_array.dtype}) to {embeddings_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate client embeddings from event data.")
    parser.add_argument("--data-dir", required=True, help="Directory with parquet files (e.g., add_to_cart.parquet, search_query.parquet, relevant_clients.npy)")
    parser.add_argument("--embeddings-dir", required=True, help="Output directory for generated client embeddings (embeddings.npy, client_ids.npy, pca_model.pkl)")
    args = parser.parse_args()
    
    main(args.data_dir, args.embeddings_dir)