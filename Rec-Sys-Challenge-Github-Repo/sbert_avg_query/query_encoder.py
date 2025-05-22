# sbert_avg_query/query_encoder.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model once globally
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_query_embedding(queries):
    """
    Encode a list of queries and return their average embedding.
    """
    if not queries:
        # Return zero vector if no queries
        return np.zeros(model.get_sentence_embedding_dimension())
    embeddings = model.encode(queries, convert_to_numpy=True)
    return np.mean(embeddings, axis=0)
