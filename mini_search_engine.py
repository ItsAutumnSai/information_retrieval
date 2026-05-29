import streamlit as st
import pandas as pd
from tokenization import tokenize
from vsm_search import search

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


df = load_data()
documents = df['comment'].tolist()
tokenized_docs = preprocess_documents(tuple(documents))

st.sidebar.markdown("""
**TF-IDF + Cosine Similarity (VSM)**

1. Query and documents are tokenized and lemmatized
2. TF-IDF vectors are built over query terms
3. Cosine similarity ranks each document against the query
4. Results are sorted by similarity score descending

**Formulas**
- TF = 1 + log(count_of_found_word_in_a_doc) if count_of_found_word_in_a_doc > 0, else 0
- IDF = log(count_of_docs / count_of_docs_contain_word) + 1
- similarity score = cos(TF, TF-IDF) = (IDF · TF-IDF) / (‖IDF‖ × ‖TF-IDF‖)
""")

col_query, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_query:
    query = st.text_input("Search query:", placeholder="ex. bottles recycling")
with col_btn:
    search_clicked = st.button("Search", use_container_width=True)

st.markdown("---")

if search_clicked and query.strip():
    result_df, idf, query_terms = search(query, documents, tokenized_docs=tokenized_docs)
    nonzero = result_df[result_df['score'] > 0].reset_index(drop=True)

    if nonzero.empty:
        st.warning("No documents matched the query.")
    else:
        # IDF scores as markdown
        idf_lines = "\n".join(f"| `{t}` | {v:.4f} |" for t, v in idf.items())
        st.markdown(
            f"**IDF scores for query terms:**\n\n"
            f"| Term | IDF |\n|------|-----|\n{idf_lines}"
        )
        st.markdown("")

        st.subheader(f"Results for: *{query}*")
        tf_cols = [f'TF({t})' for t in query_terms]
        table = pd.DataFrame({
            'Rank': range(1, len(nonzero) + 1),
            '#Document': (nonzero['index'] + 1).astype(int),
            'Comment': nonzero['document'].values,
            **{col: nonzero[col].values for col in tf_cols},
            'Score': nonzero['score'].round(4),
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.subheader("All Documents")
    display_df = df[['number', 'comment']].copy()
    display_df.columns = ['#Document', 'Comment']
    st.dataframe(display_df, use_container_width=True, hide_index=True)
