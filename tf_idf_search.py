import pandas as pd
import numpy as np

word_to_search = "bottles"

def term_frequency(word, document):
    return document.count(word) / len(document)

def inverse_document_frequency(word, corpus):
    count_of_documents = len(corpus) + 1
    count_of_documents_contain_word = sum([1 for doc in corpus if word in doc]) + 1
    idf = np.log10(count_of_documents/count_of_documents_contain_word) + 1
    return idf

df = pd.read_csv('youtube_scraped_comments.csv')
df = df.iloc[:, [2]]
print(df.head(2))

df_tf_idf = []

for document in df['Top Comment']:
    tf = term_frequency(word_to_search, document.split())
    idf = inverse_document_frequency(word_to_search, df['Top Comment'])
    new_row = {
        "tf": tf,
        "idf": idf,
        "tf-idf": tf*idf
    }
    df_tf_idf.append(new_row)
    
df_tf_idf = pd.DataFrame(df_tf_idf)
df = pd.concat([df, df_tf_idf], axis=1)
pd.set_option('display.max_columns', None)
df = df.sort_values(by="tf-idf", ascending=False)
df