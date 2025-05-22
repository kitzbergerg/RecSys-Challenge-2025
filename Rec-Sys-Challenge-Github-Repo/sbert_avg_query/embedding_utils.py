# sbert_avg_query/embedding_utils.py

import numpy as np
import pandas as pd
from collections import defaultdict

def average_user_embeddings(events: pd.DataFrame, user_col: str = "user_id", emb_col: str = "query_embedding") -> dict:
    user_embeddings = defaultdict(list)

    for _, row in events.iterrows():
        user_embeddings[row[user_col]].append(row[emb_col])

    return {
        user: np.mean(embeddings, axis=0)
        for user, embeddings in user_embeddings.items()
    }
