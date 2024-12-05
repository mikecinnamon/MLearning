# [ML-26] Example - Semantic search of COVID-19 articles

## Introduction

**Semantic search** denotes search with meaning. The expression is used as opposed to **keyword search**, also called lexical search, where the search engine looks for literal matches of the **query** words or variants of them.

Semantic search is powered by **vector search**, which encodes queries and documents as vectors, and then compares vectors to determine which are most similar. When a query is launched, the search engine transforms the query into an **embedding vector**. The algorithm matches vectors of existing documents to the query vector. The semantic search then generates results and ranks them based on a **similarity measure**, such as the **cosine** formula. **Reranking** is typically applied to improve the process.

This example uses a data set posted on Kaggle to illustrate this process. It uses the **Cohere API** methods `.embed()` and `.rerank()`(see lecture ML-25). Since this is just one of the many possible choices, you should pay more attention to the flow than to the specific toolkit used.

## The data set

The file `covid.csv`  contains data on 10,000 Covid-19 research papers, extracted from PubMed, which is a free resource supporting the search and retrieval of biomedical and life sciences literature. The columns are:

* `title`, the title of the paper.

* `abstract`, the abstract of the paper. Roughly speaking, the abstracts of a research paper has 150-250 words.

* `url`, the URL of the paper at the PubMed website.

Source of the data: `https://www.kaggle.com/datasets/anandhuh/covid-abstracts`.

## Questions

Q1. Select a world-class embedding model.

Q2. Write a simple query and encode it as an embedding vector.

Q3. Encode the abstracts included in the data set.

Q4. Perform a vector search based on the vectors obtained in the preceding questions.

Q5. Refine your results using a reranking method.


## Importing the data

We import the data as a Pandas data frame, from the usual GitHub repository.

```
In [1]: import pandas as pd, numpy as np
   ...: path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
   ...: df = pd.read_csv(path + 'covid.csv')
```

The data come as expected. There are no missing values.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   title     10000 non-null  object
 1   abstract  10000 non-null  object
 2   url       10000 non-null  object
dtypes: object(3)
memory usage: 234.5+ KB
```

This is how the data look like:

```
In [3]: df[['title', 'abstract']].head()
Out[3]: 
                                               title                                           abstract
0  Real-World Experience with COVID-19  Including...  This article summarizes the experiences of COV...
1  Successful outcome of pre-engraftment COVID-19...  Coronavirus disease 2019  COVID-19   caused by...
2  The impact of COVID-19 on oncology professiona...  BACKGROUND  COVID-19 has had a significant imp...
3  ICU admission and mortality classifiers for CO...  The coronavirus disease 2019  COVID-19  which ...
4  Clinical evaluation of nasopharyngeal  midturb...  In the setting of supply chain shortages of na...
```

With the method `.describe()`, we create a summary for the length (number of characters) of titles and abstracts. Everything looks right. 

```
In [4]: pd.concat([df['title'].str.len().describe(), df['abstract'].str.len().describe()], axis=1)
Out[4]: 
              title      abstract
count  10000.000000  10000.000000
mean     110.036300   1538.624100
std       36.095754    534.359478
min       17.000000    227.000000
25%       85.000000   1188.000000
50%      107.000000   1525.000000
75%      132.000000   1838.000000
max      310.000000   5236.000000
```

## Q1. Embedding model

We import the package `cohere`, using the API key to create a **client**.

```
In [5]: import cohere
   ...: co = cohere.ClientV2(api_key='YOUR_API_KEY')
```

Remember that, if youy use a **trial key**, you have to cope with some constraints. We will see that later. As in lecture ML-25, we use the model `embed-english-v3.0` to generate the embedding vectors.

```
In [6]: model_name = 'embed-english-v3.0'
```

We are ready to start. The steps are:

* Encode the query as a vector.

* Encode the abstracts as vectors. This is typically done off-line, and the vectors are already stored in a **vector database**.

* Search for the abstracts whose vectors are closest to the query vector, selecting the top-$N$ **nearest neighbors**. In this example we take $N=20$, using the **cosine formula** to measure the **similarity** of the query vector and the abstract vectors. 

* The abstracts come ranked by their similarity to the query. They are passed through a reranking algorithm, and then a further selection is performed, so we get smaller number (three in this example).

## Q2. Encoding the query

Since these are papers on COVID-19, we use a straightforward query.

```
In [7]: query = ['False positive rate in COVID test']
```
We encode the query with the method `.embed()`. Note hat we use here the argument `input_type='search_query'`, following Cohere's guideline. 

```
In [8]: query_embed = co.embed(texts=query, model=model_name, input_type='search_query')
```

The method  `.embed()` returns an object containing a few things, besides the embedding vectors themselves, that are extracted as a list. So, we pick the first (and only) item of this list and convert it to a 1D array for the mathematical calculations. We then check the shape of the resulting array. In Cohere the **embedding dimension** is 1,024.

```
In [9]: query_embed = np.array(query_embed.embeddings[0])
    ...: query_embed.shape
Out[9]: (1024,)
```

## Q3. Encoding the abstracts

Cohere's trial keys are **rate-limited** depending on the method you use. To cope with this, we import the package `time`, which is included in the Python Standard Library.

```
In [10]: import time
```

We collect the abstracts in a list, so they can be processed by `.embed()`. 

```
In [11]: abstract = df['abstract'].tolist()
```

Since it is not simple for the beginner to translate the limit rate into a maximum number of texts to encode, you may need a bit of trial and error before having an operational number. In this exmaple, we have found that 2,500 abstracts can be processed in one shot, so we can process them in four batches. After every batch, we wait for one minute (you may find something more "optimal" than this).

We start with the first batch:

```
In [11]: abstract_embed = co.embed(texts=abstract[:2500], model=model_name, input_type='search_document').embeddings
```

Right now, `abstract_embed`is a list of length 2,500, in which item is one vector of length 1,024 (as a list). We complete the job with a `for` loop

```
In [12]: for i in range(1, 4):
    ...:     time.sleep(60)
    ...:     new_embed = co.embed(texts=abstract[(2500*i):2500*(i+1)], model=model_name, input_type='search_document').embeddings
    ...:     abstract_embed = abstract_embed + new_embed
```

Now, `abstract_embed` has length 10,000. For operational purposes, we convert every item to a 1D array, and add these arrays to our data set as a new column. 

```
In [13]: df['abstract_embed'] = [np.array(e) for e in abstract_embed]
```

So, the data frame has now fur columns.

```
In [14]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 4 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   title           10000 non-null  object
 1   abstract        10000 non-null  object
 2   url             10000 non-null  object
 3   abstract_embed  10000 non-null  object
dtypes: object(4)
memory usage: 312.6+ KB
```

In the last column, the entries are 1D arrays of length 1,024.

```
In [15]: df['abstract_embed'][0]
Out[15]: 
array([ 0.02650452,  0.02928162,  0.00417709, ..., -0.00869751,
       -0.01260376, -0.0050621 ])
```

## Q4. Vector search

We create now a new column in our data frame by means of the method `.apply()`, calculating, in every row, the **cosine similarity** of the query and the abstract. In this example, the embedding vectors have length 1, which is the default of most embedding models. So, the denominator in the cosine formula is not needed, and we can use as a similarity function the NumPy function `dot()`. We take a look at the first rows of the resulting data set.

```
In [16]: df['similarity'] = df['abstract_embed'].apply(lambda x: np.dot(x, query_embed))
    ...: df.head()
Out[16]: 
                                               title   
0  Real-World Experience with COVID-19  Including...  \
1  Successful outcome of pre-engraftment COVID-19...   
2  The impact of COVID-19 on oncology professiona...   
3  ICU admission and mortality classifiers for CO...   
4  Clinical evaluation of nasopharyngeal  midturb...   

                                            abstract   
0  This article summarizes the experiences of COV...  \
1  Coronavirus disease 2019  COVID-19   caused by...   
2  BACKGROUND  COVID-19 has had a significant imp...   
3  The coronavirus disease 2019  COVID-19  which ...   
4  In the setting of supply chain shortages of na...   

                                        url   
0  https://pubmed.ncbi.nlm.nih.gov/35008137  \
1  https://pubmed.ncbi.nlm.nih.gov/35008104   
2  https://pubmed.ncbi.nlm.nih.gov/35007996   
3  https://pubmed.ncbi.nlm.nih.gov/35007991   
4  https://pubmed.ncbi.nlm.nih.gov/35007959   

                                      abstract_embed  similarity  
0  [0.026504517, 0.029281616, 0.0041770935, -0.04...    0.388696  
1  [0.046417236, -0.008026123, 0.035339355, -0.07...    0.312163  
2  [0.051818848, -0.023864746, -0.006504059, -0.0...    0.240182  
3  [0.022277832, 0.03488159, -0.02796936, -0.0526...    0.298916  
4  [0.038635254, -0.0021972656, -0.007873535, -0....    0.458840  
```

We are interested in the rows in which the similarity is higher. Sorting by this column in descending order and picking the top 20 rows, we get them.

```
In [17]: search_output = df.sort_values(by='similarity', ascending=False).head(20)
```

These are the selected papers, so far.

In [18]: search_output.index
Out[18]: 
Index([3393, 5942, 4073,  660,  932, 6230, 4180, 3307, 2215,  149, 5344, 3979,
       5740, 2311, 1549, 5923, 5957, 4048, 9821, 8866],
      dtype='int64')


## Q5. Reranking

Finally, we refine the selection by reranking. We use the same model as in lecture ML-25.

```
In [19]: model_name = 'rerank-english-v3.0'
```

We collect the slecetd abstracts in a list:

```
In [20]: docs = search_output['abstract'].tolist()
```

Next, we apply the method `.rerank()` to the query and this collection of abstracts, picking the opt three results. Curiously, the first two papers were not among the top-ten of the first selection.

```
In [21]: top3 = co.rerank(model=model_name, query=query[0], documents=docs, top_n=3)
    ...: top3.results
Out[21]: 
[RerankResponseResultsItem(document=None, index=18, relevance_score=0.9832789),
 RerankResponseResultsItem(document=None, index=11, relevance_score=0.9727791),
 RerankResponseResultsItem(document=None, index=6, relevance_score=0.94468933)]
```

Finally, we get the PubMed URL's of the selected papers.

```
In [22]: selection = [r.index for r in top3.results]
    ...: search_output['url'].iloc[selection]
Out[22]: 
9821    https://pubmed.ncbi.nlm.nih.gov/34854653
3979    https://pubmed.ncbi.nlm.nih.gov/34941176
4180    https://pubmed.ncbi.nlm.nih.gov/34938087
Name: url, dtype: object
```

## Homework

1. Redo this example, using the titles instead of the abstracts. Do you get similar results?

2. Redo this example, using one of the embedding models provided by the Python package `sentence_transformers`. Do you get similar results?
