import pandas as pd
from collections import defaultdict
import spacy
import re

df = pd.read_csv('youtube_scraped_comments.csv')
df = df.iloc[:, [2]]
df.columns = ['comment']
print(df['comment'][0])

word_stats = defaultdict(lambda: {'frequency': 0, 'indices': set()})
stop_words = {'is', 'the', 'to', 'i', 'and', 'in', 'of', 'a', 'for', 'it', 'you', 'that', 'was',
              'this', 's', 'be', 'they', 'we', 'when', 'not', 'have', 'like', 'but', 'how'}

for index, row in df.iterrows():
    text = str(row['comment']).lower()
    words = re.findall(r'\b\w+\b', text)
    
    #count word frequencies
    for word in words:
        if word not in stop_words:
            word_stats[word]['frequency'] += 1
            word_stats[word]['indices'].add(index)

#merge similar word (lemmatization)
nlp = spacy.load("en_core_web_sm")
doc = 

result_data = []
for word, stats in word_stats.items():
    result_data.append({
        'word': word,
        'frequency': stats['frequency'],    
        'indices': sorted(list(stats['indices']))
    })

result_df = pd.DataFrame(result_data)

result_df = result_df.sort_values(by='frequency', ascending=False).reset_index(drop=True)

pd.set_option('display.max_rows', None)
print(result_df)