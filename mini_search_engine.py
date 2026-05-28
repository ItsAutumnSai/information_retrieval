import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import defaultdict

# ============================================
# TF-IDF dan VSM Calculation Functions
# ============================================

def term_frequency(word, document):
    """Calculate log raw frequency weighting for a word in a document"""
    raw_tf = document.count(word)
    return 1 + np.log10(raw_tf) if raw_tf > 0 else 0

def inverse_document_frequency(word, corpus):
    """Calculate IDF for a word in the corpus"""
    count_of_documents = len(corpus) + 1
    count_of_documents_contain_word = sum([1 for doc in corpus if word in doc]) + 1
    idf = np.log10(count_of_documents / count_of_documents_contain_word) + 1
    return idf

def tokenize_document(text):
    """Tokenize document text into words"""
    text = str(text).lower()
    words = re.findall(r'\b\w+\b', text)
    return words

def calculate_vsm_scores(query, documents):
    """
    Calculate VSM (Vector Space Model) scores for documents against a query.
    Returns a DataFrame with documents and their VSM scores.
    """
    # Tokenize query
    query_words = tokenize_document(query)
    
    if not query_words:
        return pd.DataFrame()
    
    # Convert documents to list of tokenized text for IDF calculation
    tokenized_docs = [tokenize_document(doc) for doc in documents]
    
    vsm_scores = []
    
    for idx, doc_tokens in enumerate(tokenized_docs):
        vsm_score = 0
        
        for word in query_words:
            # Calculate TF for this word in this document
            tf = term_frequency(word, doc_tokens)
            
            # Calculate IDF for this word
            idf = inverse_document_frequency(word, tokenized_docs)
            
            # VSM = IDF * (TF * IDF)
            vsm_score += idf * (tf * idf)
        
        vsm_scores.append({
            'index': idx,
            'document': documents[idx],
            'score': vsm_score
        })
    
    result_df = pd.DataFrame(vsm_scores)
    
    # Sort by score in descending order
    result_df = result_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    return result_df

# ============================================
# Streamlit App Interface
# ============================================

st.set_page_config(
    page_title="Mini Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Mini Search Engine")
st.markdown("---")

# Sidebar for settings
st.sidebar.header("⚙️ Pengaturan")

# Load data
@st.cache_data
def load_data():
    """Load YouTube comments data"""
    try:
        df = pd.read_csv('youtube_scraped_comments.csv')
        return df
    except FileNotFoundError:
        st.error("File 'youtube_scraped_comments.csv' tidak ditemukan!")
        return None

# Main app
df = load_data()

if df is not None:
    # Display data statistics
    st.sidebar.info(f"📊 Total dokumen: {len(df)}")
    
    # Query input
    st.subheader("🔎 Masukkan Query")
    query = st.text_input(
        "Query pencarian:",
        placeholder="Contoh: bottles recycling",
        label_visibility="collapsed"
    )
    
    # Search button
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # search_button = st.button("🔍 Cari", use_container_width=True)
        search_button = st.button("🔍 Cari", width="stretch")
    
    with col2:
        results_limit = st.number_input(
            "Jumlah hasil:",
            min_value=1,
            max_value=len(df),
            value=10,
            step=1
        )
    
    # Perform search
    if search_button and query.strip():
        st.markdown("---")
        st.subheader("📋 Hasil Pencarian")
        
        # Calculate scores
        results_df = calculate_vsm_scores(query, df['Top Comment'].tolist())
        
        if len(results_df) > 0:
            # Filter by limit
            results_df = results_df.head(int(results_limit))
            
            # Display results
            st.info(f"✅ Ditemukan {len(results_df)} dokumen relevan")
            
            for rank, row in results_df.iterrows():
                with st.container():
                    # Score badge
                    score_color = "🟢" if row['score'] > 0.5 else "🟡" if row['score'] > 0.1 else "🔴"
                    
                    col1, col2 = st.columns([0.15, 0.85])
                    
                    with col1:
                        st.metric(
                            label=f"Rank #{rank + 1}",
                            value=f"{row['score']:.4f}",
                            label_visibility="collapsed"
                        )
                    
                    with col2:
                        # Display document content
                        st.write(f"**Skor:** {row['score']:.6f}")
                        st.write(f"**Dokumen {row['index'] + 1}:**")
                        st.write(row['document'])
                    
                    st.markdown("---")
        else:
            st.warning("⚠️ Tidak ada dokumen yang cocok dengan query Anda.")
    
    elif search_button and not query.strip():
        st.warning("⚠️ Silakan masukkan query terlebih dahulu!")
    
    # Additional info
    with st.expander("ℹ️ Tentang Sistem"):
        st.markdown("""
        **Mini Search Engine** menggunakan **Vector Space Model (VSM)** untuk menghitung relevansi dokumen.
        
        **Cara Kerja:**
        1. Query dan dokumen di-tokenize menjadi kata-kata
        2. Untuk setiap kata dalam query, dihitung:
           - **TF (Term Frequency):** Frekuensi kata dalam dokumen (log raw frequency)
           - **IDF (Inverse Document Frequency):** Keunikan kata dalam corpus
        3. **VSM Score** = Σ (IDF × (TF × IDF)) untuk semua kata
        4. Dokumen diurutkan berdasarkan VSM score tertinggi
        
        **Rumus:**
        - TF(word, doc) = 1 + log₁₀(raw_tf) jika raw_tf > 0, else 0
        - IDF(word) = log₁₀(N / df) + 1, dimana N = total dokumen, df = dokumen berisi kata
        - VSM = Σ IDF × (TF × IDF)
        """)
    
    # Display sample data
    with st.expander("📊 Pratinjau Data"):
        st.write("**5 Dokumen Pertama:**")
        st.dataframe(
            df[['Number', 'Top Comment']].head(5),
            # use_container_width=True,
            width="stretch",            
            hide_index=True
        )
else:
    st.error("❌ Tidak dapat memuat data. Pastikan file CSV ada di folder yang sama dengan script ini.")
