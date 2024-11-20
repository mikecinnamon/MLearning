# [ML-22] Embeddings

## What is an embedding?

Embedding vectors are one of the basic ingredients in the top performing models that are all the rage nowadays. They have already been mentioned in lecture ML-20. In general, an **embedding** is a representation of a piece of information, such as a word, a sentence or an image, as a vector in a space of a given dimension. Typical **embedding dimensions**, for the embedding models that you can manage in your computer, are 384, 512, 768 and 1,024. Nevertheless, the large language models that we use in a remote way, such as GPT-4 or Gemini, work with higher embedding dimensions.

For an embedding to be useful, "similar" pieces of information are represented by vectors that are close in a geometric sense (the distance between the endpoints, or the angle). For instance, in a word embedding, words with similar meanings, such as 'nice' and 'beautiful', will be represented by close vectors. Unrelated words, such as 'computer' and 'dog', will be represented by non-close vectors.

When we use an ML model to create embedding vectors associated to images or texts, we say that we are "encoding" them. In particular, the large language models used for that purpose are called **encoders**. The most famous of these encoders is Google's BERT, which will appear a few times in this course.

## Applications of embeddings

Applications of embeddings have already appeared in the examples of this course, though they were not presented in that way:

* In example ML-19, we encoded the MINST data with a CNN model. The convolutional base of that model can be seen as an encoder which provides an embedding representation of dimension 576 of the digit images.

* In example ML-21, we used VGG16 as an encoder for the dogs and cats images. The embedding dimension was 8,192.

Two additional forthcoming examples are:

* Example ML-23 uses a model extracted from Hugging Face as an encoder for news titles. The embedding representation, of dimension 784, provides the features for **fake news detection**, which is a binary classification task. 

* Example ML-26 illustrates the use of embedding vectors in **semantic search**.

Embedding representations can be also instrumental in clustering, recommendation, outlier detection and other applications, but we have no room for that in this course.

### Text embeddings

We have seen image embeddings in examples ML-19 and ML-21. The encoders were CNN models (this is not the only option). In the rest of this course, we will focus on **text embeddings**. These text encoders can operate at different levels: character, word, etc. 

**Word embeddings** gave a strong push to natural language processing in 2013, when Google released **Word2Vec**. A word embedding consists of a list of words, called dictionary or vocabulary, and the corresponding list of vectors, all of the same dimension. Word2Vec was not a single embedding, but a methodology allowing the choice of different options, among them the embedding dimension. 

Even if word embeddings were a big step forward, models based on them have a strong limitation, that the vector associated to a word, which should embody its meaning, is fixed. But the meaning of a word is not fixed, as the following example shows:

* I got money from the *bank*.

* The boat hit the river *bank*.

The meaning of a word depends on the context, that is, on the other words in the sentence. The order of the words also affects the meaning of the sentence, as we see in:

* He *only* said "I love you" (he said nothing else).

* He said "I love *only* you" (he doesn't love any one else).

After various attempts, Google came up in 2018 with an encoding model called **BERT**, which they implemented in their Internet engine search. Roughly speaking, BERT has a dictionary of words, with the corresponding embedding vectors. For an input text, BERT splits the text in pieces and forms a sequence with the corresponding vectors. Then, it modifies these input vectors to make them **contextual**, that is, dependent on the other words and their order in the text. For certain applications (as in example ML-23), these contextual embedding vectors are averaged to get a single embedding vector for the whole text. This is called **sentence encoding**, and is one of the services provided by AI suppliers like OpenAI and Anthropic.

BERT was based on a network architecture called the **transformer**. Though the original BERT is probably no longer active, many text encoders that we see nowadays are based on it (DistilBERT, ALBERT, RoBERTa, etc). We will come back to transformers in the next lecture.

## Vector databases

Suppose that we have a collection of documents, for instance, a collection of abstracts of research papers, and we encode them as vectors which could be used to drive the search for papers covering a given topic. How do we store these vectors in such a way that they can be efficiently retrieved?

The response of the industry to this question (so far) is the **vector database**. To have an intuition on how these databases work, you can see them as tables in which the vectors are stored in a single column. For that column, each entry store is a whole vector of hundreds of terms (we don't have one column for every term). This allows retrieving the whole vector at once, so the query becomes much more efficient. 

There is plenty of competition in this fast-growing sector, where you can find:

* Free databases like **ChromaDB**. It can be managed with the Python package `chromadb`.

* Commercial databases like **Pinecone**. You can manage it with the Python package `pinecone-client`, but you will need an API key (and they will bill you based on usage).

* Old databases adapted to vectors, like **PostgreSQL**. To make this work, you have to install an additional tool called `pgvector`.

## Similarity measures

How do we search in a vector database? The query comes as a vector, created by the same encoder as the vectors in the database. The database engine searches for similar vectors (in the geometric sense), whose associated texts are then returned as the query result. To operationalize this, a mathematical formula for the **similarity** has to be specified. Two typical similarity measures are those based on the distance and the angle. A math refresher follows, just in case you need it. 

First, the distance. Take two vectors $\hbox{\bf x}$ and $\hbox{\bf y}$ in a space of dimension $n$. Imagine them as two arrows whose origin is the zero point. Then, the distance between the endpoints is given by the formula (Pythagoras theorem):

$$\hbox{dist}(\hbox{\bf x}, \hbox{\bf y}) = \sqrt{\sum_{i=1}^n(x_i - y_i)^2}.$$

Angles are operationalized through the **cosine**. The cosine formula is commonly used as a similarity measure in data mining, in particular in natural language processing. The vectors can represent texts, as in example ML-26, or customers, products or many other possibilities, depending on the application. Mathematically, the cosine works as a correlation, so vectors pointing in the same direction have cosine 1, while orthogonal vectors have cosine 0. 

The cosine of the angle determined by two vectors $\hbox{\bf x}$ and $\hbox{\bf y}$ can be calculated as

$$\cos(\hbox{\bf x}, \hbox{\bf y}) = \frac{\displaystyle \hbox{\bf x}\cdot\hbox{\bf y}}{\lVert\hbox{\bf x}\rVert\lVert\hbox{\bf y}\rVert}.$$

In this formula, the numerator is the **dot product** (`dot()` in NumPy, `SUMPRODUCT()` in Excel)

$$\hbox{\bf x}\cdot \hbox{\bf y} = \sum_{i=1}^n x_i y_i$$

and the denominator is the product of the lengths (length meaning here the distance from the origin to the endpoint, not the number of terms),

$$\lVert\hbox{\bf x}\rVert = \sqrt{\sum_{i=1}^n x_i^2}.$$

If the embedding vectors are **normalized**, meaning that they have length 1, which is the default of sentence encoders, the denominator in the cosine formula is not needed, and the cosine is the just the output of the NumPy function `dot()`. Distances and cosines can be easily calculated, and available in vector databases. You can also calculate them with the scikit-learn functions `distance_similarity()` and `cosine_similarity()`, of the subpackage `metrics.pairwise`. 
