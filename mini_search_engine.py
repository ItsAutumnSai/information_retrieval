import streamlit as st
import pandas as pd
from tokenization import tokenize, build_word_stats
from vsm_search import search as classic_search
import semantic_search
from evaluation import get_relevant_docs, ndcg_at_k

st.set_page_config(
    page_title="Search Engine IR",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    df = pd.read_csv('youtube_scraped_comments.csv')
    df = df.iloc[:, [0, 2]]
    df.columns = ['number', 'comment']
    df = df.dropna(subset=['comment']).reset_index(drop=True)
    return df

@st.cache_data
def preprocess_documents(comments):
    return [tokenize(doc) for doc in comments]

@st.cache_data
def get_word_stats(comments):
    stats = build_word_stats(list(comments))
    rows = [
        {
            'Word': w,
            'Frequency': s['frequency'],
            'Document Indices': str(sorted(s['indices'])),
        }
        for w, s in sorted(stats.items(), key=lambda x: x[1]['frequency'], reverse=True)
    ]
    return pd.DataFrame(rows)

@st.cache_resource(show_spinner="Loading semantic model (LaBSE)...")
def get_semantic_model():
    return semantic_search.load_model()

@st.cache_data(show_spinner="Embedding documents with LaBSE...")
def get_doc_embeddings(comments):
    model = get_semantic_model()
    return semantic_search.embed(model, comments)


def build_result_table(result_df, debug, query_terms=None, top_k=10):
    # score > 0 is a meaningful cutoff for TF-IDF (no shared terms -> 0),
    # but LaBSE cosine similarities are almost never exactly 0, so semantic
    # results are capped to the top-k instead to keep noise out.
    nonzero = result_df[result_df['score'] > 0].head(top_k).reset_index(drop=True)
    if nonzero.empty:
        return None
    base = {
        'Rank': range(1, len(nonzero) + 1),
        '#Document': (nonzero['index'] + 1).astype(int),
        'Comment': nonzero['document'].values,
    }
    if debug and query_terms:
        for term in query_terms:
            base[f'TF({term})'] = nonzero[f'TF({term})'].values
    base['Score'] = nonzero['score'].round(4)
    return pd.DataFrame(base)


df = load_data()
documents = df['comment'].tolist()
tokenized_docs = preprocess_documents(tuple(documents))
semantic_model = get_semantic_model()
doc_embeddings = get_doc_embeddings(tuple(documents))

# Persist search results across re-renders (e.g. when toggling debug mode)
if 'classic_result_df' not in st.session_state:
    st.session_state.classic_result_df = None
    st.session_state.semantic_result_df = None
    st.session_state.last_query = ''
    st.session_state.idf = {}
    st.session_state.query_terms = []

st.sidebar.markdown("""
**Classic: TF-IDF + Cosine Similarity (VSM)**

1. Query and documents are tokenized and lemmatized
2. TF-IDF vectors are built over query terms
3. Cosine similarity ranks each document against the query

**Formulas**
- TF = 1 + log(count_of_found_word_in_a_doc) if count_of_found_word_in_a_doc > 0, else 0
- IDF = log(count_of_docs / count_of_docs_contain_word) + 1
- similarity score = cos(TF, TF-IDF) = (IDF · TF-IDF) / (‖IDF‖ × ‖TF-IDF‖)

---

**Semantic: LaBSE embeddings**

1. Query and documents (multilingual) are embedded with LaBSE, no tokenization needed
2. Cosine similarity between the query embedding and each document embedding ranks results
3. Captures meaning across languages, not just exact word overlap

---

**Evaluation: NDCG**

For queries with known relevant documents, NDCG@k compares how well each
method's ranking matches the ideal ranking.
""")

debug = st.sidebar.toggle("Debug mode", value=False)

col_query, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_query:
    query = st.text_input("Search query:", placeholder="ex. bottles recycling")
with col_btn:
    search_clicked = st.button("Search", use_container_width=True)

st.markdown("---")

if search_clicked and not query.strip():
    st.warning("Please enter a query first.")

if search_clicked and query.strip():
    classic_result_df, idf, query_terms = classic_search(query, documents, tokenized_docs=tokenized_docs)
    semantic_result_df = semantic_search.search(query, documents, model=semantic_model, doc_embeddings=doc_embeddings)
    st.session_state.classic_result_df = classic_result_df
    st.session_state.semantic_result_df = semantic_result_df
    st.session_state.last_query = query
    st.session_state.idf = idf
    st.session_state.query_terms = query_terms

if st.session_state.classic_result_df is not None:
    st.subheader(f"Results for: *{st.session_state.last_query}*")

    if debug:
        idf_lines = "\n".join(
            f"| `{t}` | {v:.4f} |"
            for t, v in st.session_state.idf.items()
        )
        st.markdown(f"| Term | IDF |\n|------|-----|\n{idf_lines}")

    col_classic, col_semantic = st.columns(2)

    with col_classic:
        st.markdown("##### Classic (TF-IDF + Cosine)")
        classic_table = build_result_table(
            st.session_state.classic_result_df, debug, st.session_state.query_terms
        )
        if classic_table is None:
            st.warning("No documents matched the query.")
        else:
            st.dataframe(classic_table, use_container_width=True, hide_index=True)

    with col_semantic:
        st.markdown("##### Semantic (LaBSE)")
        semantic_table = build_result_table(st.session_state.semantic_result_df, debug=False)
        if semantic_table is None:
            st.warning("No documents matched the query.")
        else:
            st.dataframe(semantic_table, use_container_width=True, hide_index=True)

    relevant_docs = get_relevant_docs(st.session_state.last_query)
    if relevant_docs:
        st.markdown("---")
        st.markdown("##### NDCG comparison")
        k = st.selectbox("k", [5, 10, 20, len(documents)], index=1, format_func=lambda x: f"NDCG@{x}")

        classic_ranked = (st.session_state.classic_result_df['index'] + 1).tolist()
        semantic_ranked = (st.session_state.semantic_result_df['index'] + 1).tolist()

        classic_ndcg = ndcg_at_k(classic_ranked, relevant_docs, k)
        semantic_ndcg = ndcg_at_k(semantic_ranked, relevant_docs, k)

        st.caption(f"Relevant documents (ground truth): {sorted(relevant_docs)}")
        m1, m2 = st.columns(2)
        m1.metric(f"Classic NDCG@{k}", f"{classic_ndcg:.4f}")
        m2.metric(f"Semantic NDCG@{k}", f"{semantic_ndcg:.4f}")
    else:
        st.info(
            "NDCG comparison is only available for queries with known relevant "
            "documents (ground truth). Try: *good ecosystem* or *recycle bottle*."
        )

else:
    st.subheader("All Documents")
    display_df = df[['number', 'comment']].copy()
    display_df.columns = ['#Document', 'Comment']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

if debug:
    st.markdown("---")
    st.subheader("Tokenization Table")
    st.dataframe(get_word_stats(tuple(documents)), use_container_width=True, hide_index=True)
