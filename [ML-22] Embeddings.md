# [ML-22] Embeddings

## What is an embedding?

An **embedding** is a representation of a piece of information, such as a word, a sentence or an image, as a vector (1D array or 1D tensor) in a space of a given dimension. Typical embedding dimensions are 512, 768 and 1,024.

The embedding representation allows the information to be processed by AI models based on neural networks, but this is not only application, as we will see in example ML-23. 

For an embedding to be useful, "similar" pieces of information are represented by vectors that are close in a geometric sense (the distance between the endpoints, or the angle). For instance, in a word embedding, words with similar meanings (such as 'nice' and 'beautiful') will be represented by close vectors, and unrelated words (such as 'computer' and 'dog') will be represented by non-close vectors.

Embedding of the MNIST data set in example ML-19. Essentially what we did in example ML-21: after flattening, we got vectors of dimension 8,192 and we fitted a MLP classifier with one hidden layer of 256 nodes.

## Applications of embeddings

Example ML-23 uses an embedding model to create features for fake news detection, which is a binary classification task. Other applications are:

* Clustering.

* Outlier detection.

* Recommendation.

* Semantic search.

### Text embeddings

- Word2Vec.

- GloVe.

- BERT.

## Why do we need embeddings?

- Neural networks.

- Large language models.

- Transformers.

---

## Vector databases

Store vector embedding as it were one column in a database.

Used for **retrieval-augmented generation** (RAG).

Free: ChromaDB.

Commercial: Pinecone.

Old databases adapted to vectors: PostgreSQL.
