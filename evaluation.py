import numpy as np

# Ground-truth relevance judgments (qrels), keyed by lowercased query text.
# Values are the sets of #Document numbers (1-indexed, matching the
# "#Document" column shown in the UI) considered relevant to that query.
QRELS = {
    "good ecosystem": {45, 3, 5, 6, 8, 9, 10, 11, 13, 15, 16, 22, 23},
    "recycle bottle": {2, 4, 5, 6, 7, 8, 9, 10, 11, 13},
}


def get_relevant_docs(query):
    return QRELS.get(query.strip().lower())


def _dcg_at_k(relevance, k):
    relevance = relevance[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))


def ndcg_at_k(ranked_doc_numbers, relevant_docs, k):
    """ranked_doc_numbers: doc numbers (1-indexed) in ranked order (best first)."""
    if not relevant_docs:
        return None

    relevance = [1 if doc_num in relevant_docs else 0 for doc_num in ranked_doc_numbers]
    dcg = _dcg_at_k(relevance, k)

    ideal_relevance = sorted(relevance, reverse=True)
    idcg = _dcg_at_k(ideal_relevance, k)

    return dcg / idcg if idcg > 0 else 0.0
