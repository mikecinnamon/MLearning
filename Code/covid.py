## [ML-26] Example - Semantic search of COVID-19 articles ##

# Importing the data #
import pandas as pd, numpy
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'covid.csv')
df.info()
df.head()
pd.concat([df['title'].str.len().describe(), df['abstract'].str.len().describe()], axis=1)

# Q1. embedding model #
import cohere
co = cohere.Client('YOUR_API_KEY')
model_name = 'embed-english-v3.0'

# Q2. Encoding the query #
query = ['False positive rate in COVID test']
query_embed = co.embed(texts=query, model=model_name, input_type='search_query')
query_embed = np.array(query_embed.embeddings[0])
query_embed.shape

# Q3. Encoding the abstracts #
import time
abstract = df['abstract'].tolist()
abstract_embed = co.embed(texts=abstract[:2500], model=model_name, input_type='search_document').embeddings
for i in range(1, 4):
    time.sleep(60)
    new_embed = co.embed(texts=abstract[(2500*i):2500*(i+1)], model=model_name, input_type='search_document').embeddings
    abstract_embed = abstract_embed + new_embed
df['abstract_embed'] = [np.array(e) for e in abstract_embed]
df.info()
df['abstract_embed'][0]

# Q4. Vector search #
df['similarity'] = df['abstract_embed'].apply(lambda x: np.dot(x, query_embed))
df.head()
search_output = df.sort_values(by='similarity', ascending=False).head(20)
abstract_search_output
search_output['abstract'].iloc[0]

# Q5. Reranking #
model_name = 'rerank-english-v3.0'
docs = search_output['abstract'].tolist()
top3 = co.rerank(model=model_name, query=query[0], documents=docs, top_n=3)
top3.results
selection = [r.index for r in top3.results]
search_output['url'].iloc[selection]
search_output['title'][0]
