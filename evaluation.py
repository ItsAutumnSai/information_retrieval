import numpy as np

# Ground-truth relevance judgments (qrels), keyed by lowercased query text.
# Values map #Document numbers (1-indexed, matching the "#Document" column
# shown in the UI) to a graded relevance level:
#   3 = strong/direct match, 2 = clearly relevant but partial,
#   1 = marginal/tangential relevance, (0/absent = not relevant)
# Entries are listed in descending relevance order for readability.
QRELS = {
    "good ecosystem": {
        6: 3, 22: 3,
        16: 2, 15: 2, 10: 2, 11: 2, 13: 2, 3: 2, 5: 2,
        9: 1, 8: 1, 23: 1, 45: 1,
    },
    "recycle bottle": {
        6: 3, 8: 3, 10: 3, 13: 3,
        2: 2, 4: 2, 5: 2, 9: 2, 11: 2,
        7: 1,
    },
}


def get_relevant_docs(query):
    return QRELS.get(query.strip().lower())


def _dcg_at_k(relevance, k):
    relevance = relevance[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))


def ndcg_at_k(ranked_doc_numbers, relevant_docs, k):
    """ranked_doc_numbers: doc numbers (1-indexed) in ranked order (best first).
    relevant_docs: dict mapping doc number -> graded relevance (higher = more relevant)."""
    if not relevant_docs:
        return None

    relevance = [relevant_docs.get(doc_num, 0) for doc_num in ranked_doc_numbers]
    dcg = _dcg_at_k(relevance, k)

    ideal_relevance = sorted(relevance, reverse=True)
    idcg = _dcg_at_k(ideal_relevance, k)

    return dcg / idcg if idcg > 0 else 0.0
