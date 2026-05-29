import re
from collections import defaultdict
import spacy

_nlp = None

STOP_WORDS = {
    'is', 'the', 'to', 'i', 'and', 'in', 'of', 'a', 'for', 'it', 'you', 'that', 'was',
    'this', 's', 'be', 'they', 'we', 'when', 'not', 'have', 'like', 'but', 'how',
    'an', 'are', 'at', 'by', 'do', 'from', 'get', 'has', 'he', 'her', 'him', 'his',
    'if', 'its', 'just', 'me', 'more', 'my', 'no', 'on', 'or', 'our', 'out', 'so',
    'up', 'us', 'what', 'with', 'your',
}

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp

def tokenize(text, stop_words=None):
    stop_words = STOP_WORDS
    nlp = _get_nlp()
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and len(token.text) > 1
        and token.text not in stop_words
        and token.lemma_ not in stop_words
    ]
    return tokens

def build_word_stats(documents):
    word_stats = defaultdict(lambda: {'frequency': 0, 'indices': set()})
    for idx, doc in enumerate(documents):
        doc_token_counts = defaultdict(int)
        for token in tokenize(doc):
            doc_token_counts[token] += 1
        for token, count in doc_token_counts.items():
            word_stats[token]['frequency'] += count
            word_stats[token]['indices'].add(idx)
    return word_stats