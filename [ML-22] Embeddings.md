# [ML-22] Embeddings

## What is an embedding?

Embedding vectors are one of the basic ingredients in the top performing models that are all the rage nowadays. They have already been mentioned in lecture ML-20. In general, an **embedding** is a representation of a piece of information, such as a word, a sentence or an image, as a vector (1D array or 1D tensor) in a space of a given dimension. Typical **embedding dimensions** are 512, 768 and 1,024.

For an embedding to be useful, "similar" pieces of information are represented by vectors that are close in a geometric sense (the distance between the endpoints, or the angle). For instance, in a word embedding, words with similar meanings (such as 'nice' and 'beautiful') will be represented by close vectors, and unrelated words (such as 'computer' and 'dog') will be represented by non-close vectors.

When we use an ML model to create embedding vectors associated to images or texts, we say that we are "encoding" them. In particular, the large language models used for that purpose are called **encoders**. The most famous of these encoders is Google's BERT, which will briefly described in lecture ML-24.

## Applications of embeddings

Applications of embeddings have already appeared in the examples, though they were not presented in that way:

* In example ML-19, we encoded the MINST data with a CNN model. The convolutional base of that model can be seen as creating an embedding representation of dimension 576 of the digit images.

* In example ML-21, we used VGG16 as an embedding model of dimension 8,192 for the dogs and cats images.

Two additional forthcoming examples:

* Example ML-23 uses a model extracted from Hugging Face for an embedding representation of dimension 784. This representation creates features of news titles for **fake news detection**, which is a binary classification task. 

* Example ML-26 illustrates the use of embedding vectors in **semantic search**.

Embedding representations can be also instrumental in clustering, recommendation, outlier detection and other applications, but we have no room for more in this course.

### Text embeddings

We have seen image embeddings in examples ML-21 and ML-23. The embedding models were CNN models (this is not the only option). In the rest of this course, we will focus on **text embeddings**. These embeddings can operate at different levels: character, word, etc. 

**Word embeddings** gave a strong push to natural language processing in 2013, when Google released **Word2Vec**. A word embedding consists of a list of words, called dictionary or vocabulary, and the corresponding list of vectors, all of the same length. Word2Vec was not a single embedding, but a methodology, with a few options, among them the embedding dimension. 

Even if word embeddings were a big step forward, models based on them have a strong limitation, that the vector associated to a word, which should embody its meaning, is fixed. But the meaning of a word is not fixed, as the following example shows:

* I got money from the *bank*.

* The boat hit the river *bank*.

The meaning of a word depends on the context, that is, on the other words in the sentence. Also, the order of the words affects the meaning of the sentence:

* He *only* said "I love you" (he said nothing else).

* He said "I love *only* you" (he doesn't love any one else).

After various attempts, Google came up in 2018 with an encoding model called **BERT**, which they implemented in their Internet engine search. Roughly speaking the idea is that BERT has a collection of word vectors, and generates an embedding vectors for a given text by combining the word vectors, modifying, on the fly, each of those vectors as a function of the other words and the position of the word in the text. BERT was based on a network architecture called the **transformer**. 

Though BERT is probably no longer active, most of the encoders that we see nowadays are descendants of BERT. But we better postpone our discussion of transformers and encoders until we have seen an example.

## Vector databases

Suppose that you have a collection of documents, for instance, a collection of abstracts of research papers, and you encode them as vectors so that the vectors can be used to drive the search for papers covering a given topic. How do you store these vectors in such a way that they can be efficiently retrieved?

The response (so far) of the industry to this question if the **vector database**. In a vector database, the vectors are stored in a single column. For that column, we store in every row a whole vector of hundreds of terms (we don't have one column for every term). This allows retrieving the whole vector at once, so the query becomes much more efficient. 

There is plenty of competition in this fast-growing sector, where you can find:

* Free databases like **ChromaDB**. You can mange it with the Python package `chromadb`.

* Commercial databases like **Pinecone**. You can manage it with the Python package `pinecone-client`, but you need an API key (and they will bill you based on usage).

* Old databases adapted to vectors, like **PostgreSQL**. You have to install an additional tool called `pgvector`. PostgreSQL databases can be managed with the Python package `psycopg2` (not the easiest thing in the world).

## The cosine similarity

Let us start this section with a math refresher, just in case you need it. The **cosine** of the angle determined by two vectors $\hbox{\bf x}$ and $\hbox{\bf y}$ can be calculated as

$$\cos\big(\hbox{\bf x}, \hbox{\bf y}\big) = \frac{\displaystyle \hbox{\bf x}\cdot\hbox{\bf y}}{\lVert\hbox{\bf x}\rVert\lVert\hbox{\bf y}\rVert}.$$

In this formula, the numerator is the **dot product** (`dot()` in NumPy, `SUMPRODUCT()` in Excel) 

$$\hbox{\bf x}\cdot \hbox{\bf y} = \sum_{i=1}^n x_i y_i$$

and the denominator is the product of the lengths (length meaning here the distance from the origin to the endpoint, not the number of terms),

$$\lVert\hbox{\bf x}\rVert = \sqrt{\sum_{i=1}^n x_i^2}.$$

Note that, if the embedding vectors have length 1, which is the default of many embeddings methods, the denominator in the cosine formula is not needed, and the cosine is the just the output of the NumPy function `dot()`.

The cosine formula is commonly used in data mining, in particular in natural language processing, to measure the **similarity** between two vectors. The vectors can represent texts, as in example ML-26, or customers, products or many other possibilities, depending on the application. Mathematically, the cosine works as a correlation, so vectors pointing in the same direction have cosine 1, while orthogonal vectors have cosine 0. 
