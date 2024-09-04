## [ML-26] Example - Semantic search of COVID-19 articles ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'covid.csv')
df.info()
df.head()
pd.concat([df['title'].str.len().describe(), df['abstract'].str.len().describe()], axis=1)

# Cosine of two vectors #
import numpy as np
def cos(x, y):
    dotproduct = sum(x*y)
    x_norm = np.sqrt(sum(x**2))
    y_norm = np.sqrt(sum(y**2))
    return dotproduct/(x_norm*y_norm)

# Cohere's embedding model #
import cohere
co = cohere.Client('OCAVTT8U76T9EsepDmTjnUTGWdFIrfWP6xaWRHG3')
model_name = 'embed-english-v3.0'

# Encoding the query #
query = ['false positives in COVID test']
query_embed = co.embed(texts=query, model=model_name, input_type='search_document')
query_embed = np.array(query_embed.embeddings)
query_embed.shape
query_embed = query_embed.reshape(1024,)

# Encoding the abstracts #
import time
abstract = df['abstract'].to_list()
abstract_embed = co.embed(texts=abstract[:2500], model=model_name, input_type='search_document').embeddings
for i in range(1, 4):
    time.sleep(60)
    new_embed = co.embed(texts=abstract[(2500*i):2500*(i+1)], model=model_name, input_type='search_document').embeddings
    abstract_embed = abstract_embed + new_embed
abstract_embed = np.array(abstract_embed)
abstract_embed.shape
df['abstract_embed'] = list(abstract_embed)
df.info()
df['abstract_embed']

# Semantic search of abstracts #
df['similarity'] = df['abstract_embed'].apply(lambda x: 1 - cos(x, query_embed))
df
abstract_search_output = df.sort_values(by='similarity').head(20)
abstract_search_output
abstract_search_output['abstract'].iloc[0]

# Reranking #
model_name = 'rerank-english-v3.0'
docs = abstract_search_output['abstract']
top5 = co.rerank(model=model_name, query=query, documents=docs, top_n=5)
selection = [r.index for r in top5]
abstract_search_output['url'].iloc[selection]

# Encoding the titles #
import time
title = df['title'].to_list()
title_embed = co.response = co.rerank(model=model_name, query=query, documents=docs, top_n=3)embed(texts=title[:2500], model=model_name, input_type='search_document').embeddings
for i in range(1, 4):
    time.sleep(60)
    new_embed = co.embed(texts=title[(2500*i):2500*(i+1)], model=model_name, input_type='search_document').embeddings
    title_embed = title_embed + new_embed
title_embed = np.array(title_embed)
title_embed.shape
df['title_embed'] = title_embed

# Semantic search of titles #
sim = df['title_embed'].apply(lambda x: cos(x, query_embed))
sim.name = 'similarity'
title_search_output = pd.concat([df['title'], df['abstract'], sim], axis=1).sort_values(by='similarity').head(10)

# Encoding the query (alt) #
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
query_embed = model.encode(query, output_value='sentence_embedding', convert_to_numpy=True)
query_embed.shape
query = query.reshape(768,)

# Encoding the abstracts (alt) #
abstracts = df1['abstract'].tolist()
abstract_embed = model.encode(abstracts, output_value='sentence_embedding', convert_to_numpy=True)
df['abstract_embed'] = abstract_embed.to_list()

# Encoding the titles (alt) #
title = df['title'].to_list()
title_embed = model.encode(title, output_value='sentence_embedding', convert_to_numpy=True)
df['title_embed'] = title_embed
