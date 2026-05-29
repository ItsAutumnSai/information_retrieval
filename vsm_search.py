import numpy as np
import pandas as pd
from tokenization import tokenize

def _term_frequency(word, doc_tokens):
    raw_tf = doc_tokens.count(word)
    return 1 + np.log10(raw_tf) if raw_tf > 0 else 0.0

def _inverse_document_frequency(word, all_doc_tokens):
    N = len(all_doc_tokens)
    df = sum(1 for doc in all_doc_tokens if word in doc)
    if df == 0:
        return 0.0
    return np.log10(N / df) + 1.0

def _cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot / (norm1 * norm2))

def search(query, documents, tokenized_docs=None):
    query_tokens = tokenize(query)
    if not query_tokens:
        return pd.DataFrame(columns=['index', 'document', 'score']), {}, []

    if tokenized_docs is None:
        tokenized_docs = [tokenize(doc) for doc in documents]

    query_terms = list(dict.fromkeys(query_tokens))  # unique, order-preserved

    idf = {
        term: _inverse_document_frequency(term, tokenized_docs)
        for term in query_terms
    }

    query_vector = np.array([
        _term_frequency(term, query_tokens) * idf[term]
        for term in query_terms
    ])

    rows = []
    for idx, (doc, doc_tokens) in enumerate(zip(documents, tokenized_docs)):
        doc_vector = np.array([
            _term_frequency(term, doc_tokens) * idf[term]
            for term in query_terms
        ])
        score = _cosine_similarity(query_vector, doc_vector)
        row = {'index': idx, 'document': doc}
        for term in query_terms:
            row[f'TF({term})'] = round(_term_frequency(term, doc_tokens), 4)
        row['score'] = score
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    return result_df, idf, query_terms
