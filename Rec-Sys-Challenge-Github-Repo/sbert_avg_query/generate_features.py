# sbert_avg_query/generate_features.py

import pandas as pd
import numpy as np
from .query_encoder import get_query_embedding
from tqdm import tqdm


def generate_user_embeddings(data_dir, embeddings_dir):
    # Load search_query parquet
    search_df = pd.read_parquet(f'{data_dir}/search_query.parquet')
    
    print(f"Loaded {len(search_df)} search queries")
    
    # Group queries by user_id (assuming 'user_id' and 'query' columns exist)
    # user_queries = search_df.groupby('user_id')['query'].apply(list).reset_index()
    user_queries = search_df.groupby('client_id')['query'].apply(list).reset_index()
    
    user_embeddings = {}
    for _, row in tqdm(user_queries.iterrows(), total=len(user_queries), desc="Encoding users"):
        # user_id = row['user_id']
        client_id = row['client_id']
        queries = row['query']
        embedding = get_query_embedding(queries)
        user_embeddings[client_id] = embedding
    
    # Save embeddings as numpy arrays: user_ids.npy and embeddings.npy
    client_ids = np.array(list(user_embeddings.keys()))
    embeddings = np.array(list(user_embeddings.values()))
    
    np.save(f'{embeddings_dir}/client_ids.npy', client_ids)
    np.save(f'{embeddings_dir}/embeddings.npy', embeddings)
    
    print(f"Saved embeddings for {len(client_ids)} users to {embeddings_dir}")
