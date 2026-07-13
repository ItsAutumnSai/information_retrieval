import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/LaBSE"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def embed(model, texts):
    return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


def search(query, documents, model=None, doc_embeddings=None):
    if not query.strip():
        return pd.DataFrame(columns=['index', 'document', 'score'])

    if model is None:
        model = load_model()
    if doc_embeddings is None:
        doc_embeddings = embed(model, documents)

    query_embedding = embed(model, [query])[0]
    scores = doc_embeddings @ query_embedding

    result_df = pd.DataFrame({
        'index': np.arange(len(documents)),
        'document': documents,
        'score': scores.astype(float),
    })
    return result_df.sort_values(by='score', ascending=False).reset_index(drop=True)
