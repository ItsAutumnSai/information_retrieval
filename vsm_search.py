import pandas as pd
import numpy as np

words_to_search = "bottles empty"
words_to_search = words_to_search.lower().split()

# using log raw frequency weighting
def term_frequency(word, document):
    raw_tf = document.count(word)
    return 1+np.log10(raw_tf) if raw_tf > 0 else 0

def inverse_document_frequency(word, corpus):
    count_of_documents = len(corpus) + 1
    count_of_documents_contain_word = sum([1 for doc in corpus if word in doc]) + 1
    idf = np.log10(count_of_documents/count_of_documents_contain_word) + 1
    return idf

df = pd.read_csv('youtube_scraped_comments.csv')
df = df.iloc[:, [2]]
print(df.head(2))

df_tf_idf = []
vsm_sum = []
for document in df['Top Comment']:
    new_row = {}
    vsm_each = {}
    for i in range(len(words_to_search)):
        tf = term_frequency(words_to_search[i], document.lower().split())
        idf = inverse_document_frequency(words_to_search[i], df['Top Comment'])
        add_row = {
            "tf"+str(i): tf,
            "idf"+str(i): idf,
            "tf-idf"+str(i): tf*idf
        }
        add_vsm = {
            "vsm"+str(i): idf*(tf*idf)  
        }
        #add columns
        new_row.update(add_row)
        vsm_each.update(add_vsm)
    #add rows
    df_tf_idf.append(new_row)
    vsm_sum.append(vsm_each)
    
df_tf_idf = pd.DataFrame(df_tf_idf)
vsm_sum = pd.DataFrame(vsm_sum)
vsm_sum = vsm_sum.sum(axis=1).rename("VSM")
df = pd.concat([df, df_tf_idf], axis=1)
df = pd.concat([df, vsm_sum], axis=1)
pd.set_option('display.max_colwidth', None)
df = df.sort_values(by="VSM", ascending=False)
df.head(10)