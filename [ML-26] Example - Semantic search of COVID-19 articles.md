# [ML-26] Example - Semantic search of COVID-19 articles

## Introduction

**Semantic search** denotes search with meaning. The expression is used as opposed to **keyword seqacrh**, also called lexical serach, where the search engine looks for literal matches of the query words or variants of them.

Semantic search is powered by **vector search**, which enables semantic search to deliver and rank content based on context relevance and intent relevance. Vector search encodes queries and documents as vectors, and then compares vectors to determine which are most similar. When a query is launched, the search engine transforms the query into an embedding vector. The algorithm matches vectors of existing documents to the query vector. The semantic search then generates results and ranks them based on conceptual relevance. **Reranking** is typically applied to improve the process.

This example uses a data set posted on Kaggle to illustrate this process. It uses the **Cohere API** methods `.embed()` and `.rerank()`(see lecture ML-25), but this is just of the many possible choices, so the reader should pay more attention to the flow than to the specific toolkit used.

## The data set

The file `covid.csv`  contains data on 10,000 Covid-19 research papers. The columns are:

* `title`, the title of the paper.

* `abstract`, the abstract of the paper. Abstracts of research papers are summaries of about 250 words.

* `url`, the URL of the paper at the Pubmed website.

Source of the data: `https://www.kaggle.com/datasets/anandhuh/covid-abstracts`.

## Questions



## Importing the data

We import the data as a Pandas data frame from the usual GitHub repository.

```
In [1]: import pandas as pd, numpy as np
   ...: path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
   ...: df = pd.read_csv(path + 'covid.csv')
```

The data come as expected. No missing values.

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
This is hopw the data look like:

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

## Cosine of two vectors

Now, a math refresher, just in case you need it. The **cosine** of the angle determined by two vectors $\hbox{\bf x}$ and $\hbox{\bf y}$ can be calculated as

$$\cos\big(\hbox{\bf x}, \hbox{\bf y}\big) = \frac{\displaystyle \hbox{\bf x}\cdot\hbox{\bf y}}{\lVert\hbox{\bf x}\rVert\lVert\hbox{\bf y}\rVert}.$$

In this formula, the numerator is the **dot product** (`dotproduct()` in NumPy, `SUMPRODUCT()` in Excel) 

$$\hbox{\bf x}\cdot \hbox{\bf y} = \sum_{i=1}^n x_i y_i$$

and the denominator is the product of the lengths (length meaning here the distance from the origin to the endpoint, not the number of terms),

$$\lVert\hbox{\bf x}\rVert = \sqrt{\sum_{i=1}^n x_i^2}.$$

The cosine is commonly used in data mining to measure the similarity between two vectors (which can represent texts, as in this example, or customers, products or many other possibilities, depending on the application). Mathematically, the cosine works as a correlation, so vectors pointing in the same direction have cosine 1, while orthogonal vectors have cosine 0. In the NLP context, the cosine is used as a **similarity measure**.

In this example, the embedding vectors have length 1, which is the default of Cohere's method `.embed()`, so the denominator in the cosine formula is not needed, and we can use as a similarity function the NumPy function `dotproduct()`

## Cohere's embedding model

We import the package `cohere`, and create a client using an API key:


```
In [5]: import cohere
   ...: co = cohere.Client('YOUR_API_KEY')
```

As in lecture ML-25, we use the model `embed-english-v3.0`.

```
In [6]: model_name = 'embed-english-v3.0'
```

## Encoding the query

```
In [7]: query = ['false positives in COVID test']
```

```
In [9]: query_embed = co.embed(texts=query, model=model_name, input_type='search_document')
```

## Encoding the abstracts

```
In [9]: import time
```

```
In [10]: abstract = df['abstract'].to_list()
```

```
In [11]: abstract_embed = co.embed(texts=abstract[:2500], model=model_name, input_type='search_document').embeddings
```

```
In [12]: for i in range(1, 4):
    ...:     time.sleep(60)
    ...:     new_embed = co.embed(texts=abstract[(2500*i):2500*(i+1)], model=model_name, input_type='search_document').embeddings
    ...:     abstract_embed = abstract_embed + new_embed
```

```
In [13]: abstract_embed = np.array(abstract_embed)
```

```
In [14]: abstract_embed.shape
Out[14]: (10000, 1024)
```

```
In [15]: df['abstract_embed'] = list(abstract_embed)
```

```
In [16]: df.info()
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

```
In [17]:  df['abstract_embed'][0]
Out[17]: 
array([ 0.02706909,  0.02999878,  0.00516891, ..., -0.00926208,
       -0.01276398, -0.00491333])
```

## Semantic search of abstracts

```
In [18]: df['similarity'] = df['abstract_embed'].apply(lambda x: 1 - cos(x, query_embed))
    ...: df
Out[18]: 
                                                  title  ... similarity
0     Real-World Experience with COVID-19  Including...  ...   0.777954
1     Successful outcome of pre-engraftment COVID-19...  ...   0.770500
2     The impact of COVID-19 on oncology professiona...  ...   0.823714
3     ICU admission and mortality classifiers for CO...  ...   0.813349
4     Clinical evaluation of nasopharyngeal  midturb...  ...   0.675177
...                                                 ...  ...        ...
9995  Rooming-in  Breastfeeding and Neonatal Follow-...  ...   0.780644
9996  Acute Retinal Necrosis from Reactivation of Va...  ...   0.774734
9997  Acute Abducens Nerve Palsy Following the Secon...  ...   0.733501
9998  Planning and Implementing the Protocol for Psy...  ...   0.832458
9999  Prolonged corrected QT interval in hospitalize...  ...   0.764082

[10000 rows x 5 columns]
```

```
In [19]: abstract_search_output = df.sort_values(by='similarity').head(10)
    ...: abstract_search_output
Out[19]: 
                                                  title  ... similarity
2536  Differing Sensitivity of COVID-19 PCR Tests an...  ...   0.549286
8358  Highly Suspected COVID-19 Cluster with Multipl...  ...   0.588679
5942  Analysis of the initial lot of the CDC 2019-No...  ...   0.595550
9821      We still have to continue to test  test  test  ...   0.603180
6230  Test sensitivity for infection versus infectio...  ...   0.611275
677   Virucidal activity of SARS-CoV-2 rapid antigen...  ...   0.613470
4073  SARS-CoV-2 infection  use and effectiveness of...  ...   0.616164
660   The broad spectrum of COVID-like patients init...  ...   0.617787
5905  Requirements and study designs for US regulato...  ...   0.621127
597   One-step and sequential SARSCOV-2 polymerase c...  ...   0.628644

[10 rows x 5 columns]
```

```
In [20]: abstract_search_output['abstract'].iloc[0]
Out[20]: 'The world at large cannot afford to miss even a single case of COVID-19 because of its far-reaching consequences  therefore  the diagnostic development to achieve test with much higher sensitivity should be made available at a mass level as early as possible  How to cite this article  Garg SK  Differing Sensitivity of COVID-19 PCR Tests and Consequences of the False-negative Report  A Small Observation  Indian J Crit Care Med 2021 25 9  1077-1078'
```

```
In [21]: abstract_search_output['url'].iloc[:10]
Out[21]: 
2536    https://pubmed.ncbi.nlm.nih.gov/34963733
8358    https://pubmed.ncbi.nlm.nih.gov/34876822
5942    https://pubmed.ncbi.nlm.nih.gov/34910739
9821    https://pubmed.ncbi.nlm.nih.gov/34854653
6230    https://pubmed.ncbi.nlm.nih.gov/34908630
677     https://pubmed.ncbi.nlm.nih.gov/34995991
4073    https://pubmed.ncbi.nlm.nih.gov/34939621
660     https://pubmed.ncbi.nlm.nih.gov/34996418
5905    https://pubmed.ncbi.nlm.nih.gov/34911365
597     https://pubmed.ncbi.nlm.nih.gov/34997789
Name: url, dtype: object
```
