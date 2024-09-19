# [ML-22] Embeddings

## What is an embedding?

Embedding vectors have already been mentioned in lecture ML-20. In general, an **embedding** is a representation of a piece of information, such as a word, a sentence or an image, as a vector (1D array or 1D tensor) in a space of a given dimension. Typical embedding dimensions are 512, 768 and 1,024.

For an embedding to be useful, "similar" pieces of information are represented by vectors that are close in a geometric sense (the distance between the endpoints, or the angle). For instance, in a word embedding, words with similar meanings (such as 'nice' and 'beautiful') will be represented by close vectors, and unrelated words (such as 'computer' and 'dog') will be represented by non-close vectors.

When we use an ML model to create embedding vectors associated to images or texts, we say that we are "encoding" them. In particular, the large language models used for that purpose are called **encoders**. The most famous of these encoders is Google's BERT, which will briefly described in lecture ML-24.

## Applications of embeddings

Applications of embeddings have already appeared in the examples, though they were not presented in that way:

* In examples ML-19, we encoded the MINST data with a CNN model. The convolutional base of that model can be seen as creating an embedding representation of dimension 576 of the digit images.

* In example ML-21, we used VGG16 as an embedding model of dimension 8,192 for the dogs and cats images.

Two additional forthcoming examples:

* Example ML-23 uses a model extracted from Hugging Face for an embedding representation of dimension 784. This representation creates features of news titles for **fake news detection**, which is a binary classification task. 

* In example ML-26 we illustrate the use of embedding vectors in **semantic search**.

Embedding representations can be also instrumental in clustering, recommendation, outlier detection and other applications, but we have no room for more in this course.

### Text embeddings

- Word2Vec.

- GloVe.

- BERT.

## Why do we need embeddings?

- Neural networks.

- Large language models.

- Transformers.

## Vector databases

Store vector embedding as it were one column in a database.

Used for **retrieval-augmented generation** (RAG).

Free: ChromaDB.

Commercial: Pinecone.

Old databases adapted to vectors: PostgreSQL.

## Cosine of two vectors

Now, a math refresher, just in case you need it. The **cosine** of the angle determined by two vectors $\hbox{\bf x}$ and $\hbox{\bf y}$ can be calculated as

$$\cos\big(\hbox{\bf x}, \hbox{\bf y}\big) = \frac{\displaystyle \hbox{\bf x}\cdot\hbox{\bf y}}{\lVert\hbox{\bf x}\rVert\lVert\hbox{\bf y}\rVert}.$$

In this formula, the numerator is the **dot product** (`dot()` in NumPy, `SUMPRODUCT()` in Excel) 

$$\hbox{\bf x}\cdot \hbox{\bf y} = \sum_{i=1}^n x_i y_i$$

and the denominator is the product of the lengths (length meaning here the distance from the origin to the endpoint, not the number of terms),

$$\lVert\hbox{\bf x}\rVert = \sqrt{\sum_{i=1}^n x_i^2}.$$

Note that, if the embedding vectors have length 1, which is the default of many embeddings methods, the denominator in the cosine formula is not needed, and the cosine is the just the output of the NumPy function `dot()`

The cosine is commonly used in data mining, in particular in **natural language processing** (NLP), to measure the **similarity** between two vectors (which can represent texts, as in this example, or customers, products or many other possibilities, depending on the application). Mathematically, the cosine works as a correlation, so vectors pointing in the same direction have cosine 1, while orthogonal vectors have cosine 0. 
