## [ML-26] Example - Semantic search of COVID-19 articles - jina version ##

# Stopping warnings #
import warnings
warnings.filterwarnings('ignore')

# Importing the data #
import pandas as pd, numpy as np
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'covid.csv')
df.info()
df.head()
pd.concat([df['title'].str.len().describe(), df['abstract'].str.len().describe()], axis=1)

# Q1. Embedding model #
from sentence_transformers import SentenceTransformer
encoder_name = 'all-MiniLM-L12-v2'
encoder = SentenceTransformer(encoder_name)

# Q2. Encoding the query #
query = 'How to get COVID immunity?'
query_embed = encoder.encode([query], output_value='sentence_embedding')
query_embed.shape

# Q3. Encoding the abstracts #
abstract = df['abstract'].tolist()
abstract_embed = encoder.encode(abstract, output_value='sentence_embedding')
abstract_embed.shape

# Q4. Vector search #
df['similarity'] = np.dot(abstract_embed, query_embed.T)
df.head()
search_output = df.sort_values(by='similarity', ascending=False).head(20)
search_output
search_output['abstract'].iloc[0]

# Q5. Reranking #
from sentence_transformers import CrossEncoder
reranker_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
reranker = CrossEncoder(reranker_name)
docs = search_output['abstract'].tolist()
ranks = reranker.rank(query, docs)[:3]
ranks
ranks = pd.DataFrame(ranks)
ranks
selection = ranks['corpus_id'][:3]
search_output['url'].iloc[selection]
