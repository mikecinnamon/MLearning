# [ML-24] Large language models

## What is natural language processing?

**Natural language processing** (NLP) is an interdisciplinary subfield of computer science and artificial intelligence. It is concerned with providing computers the ability to process text written in a natural language. 

The adjective *natural* is used here to refer to the languages we humans use, such as English, Spanish, etc, as opposed to computer languages like Python or Java. Nevertheless, the language models developed since 2020 have been trained in a way that they can process code written in a number of computer languages, so the distinction is fading away. In this lecture, we assume that we are dealing with text written in English, with occasional comments to other languages.

The NLP toolkit used in this course is based on deep learning. The data unit is typically a string (which can be a long document). The data set is a collection (*e.g*. a list) of strings, called a **corpus**. A classic is the Wikipedia corpus.

NLP has advanced in giant steps in the last years, as **large language models** (LLMs) took the stage. Unfortunatley, this means that reading anything published ealier than five years ago may be wasting time.

## Tokens

Even if the data unit in NLP is a string, that string is not processed as a whole, but previously split in a list of substrings called **tokens**. The extraction of tokens from a text corpus is called **tokenization**. Tokenization is one of the oldest problems in NLP, and different approaches have been discussed for years: words, sub-words, pairs of words, etc. Also, tokenization is a harder problem in some languages, like Spanish and French, with many verb forms, or in German, with its declensions, prefixes and suffixes, than in English.

Nowadays, the debate about the tokenization level has lost interest (at least for the beginners). First, there is plenty of supply of tokenization models, called **tokenizers** (right now, you can find 3,383 in Hugging Face). Second, all the relevant language models come with their own tokenizers, so you don't have to think about them once you have chosen your model. In these models, most of the tokens are words, and a small proportion of subwords. Also, punctuation marks can be tokens. In addition to all these, there is an "unknown" token (the vocabulary of the model is limited), a token for the start of the text and another token for the end (think on ChatGPT you will see that this is needed). So, when we input a string to one of these models, it is converted to a list of tokens, though we never see them in practice.

## NLP tasks

Natural language processing turned around a few specific tasks during many years. The classics are:

* **Text classification**. The input is a text and the output a set of class probabilities. For instance, in example ML-23 we trained a model for fake news detection. Another popular application is **sentiment analysis**. 

* **Text generation**. The input is a text and the output another text. This covers many particular cases, as we see below.

* **Summarization**. The output text is a summarized version of the input text. 

* **Translation**. The output text is a translation of the input text to another language. Automatic translation from Russian to English is one of the very classics of artificial intelligence.

* **Question answering** (QA). The input text is a question and the output text is an answer to that question, either extracted from a context text inputted together with the question, or retrieved from a data source.

* **Named entity recognition** (NER). The input is a text together with a collection of entities. The output is a list of the entities found in the text. For instance, the model can extract dishes ID's from a restaurant review.

Though this view of natural language processing is already found in many courses, the perspective has changed. On one side, we have tasks like **embedding**, that we have already seen in lecture ML-22 and example ML-23, and is one of the main tasks nowadays. On the other hand, many classic tasks, like translation and summarization, are been adderessed through **chat** apps like ChatGPT, so they appear as examples of conversation with an assistant. This perspective dominates the interaction of most users with AI agents.

## What are large language models?

Large language models are not only large (they often go beyond 1B parameters), but a new generation of neural network models. They are based on a novel architecture, the **transformer**, published in 2017. An LLM can be used directly, or taken as a **pre-trained model** for transfer learning, which in this context is called **fine-tuning**. Transfer learning is very common in LLM's. The original pre-trained models are called **foundation models**, and the task for which they are fine-tuned is called the **downstream task**.

There are just a few relevant foundation models, but thousands of fine-tuned versions of them. You can find most of these in Hugging Face. The two classics are **BERT** (Bidirectional Encoder Representations from Transformers), developed by Google, and the **GPT** (Generative Pre-trained Transformer) series, developed by OpenAI. 

LLM's are essentially of two types:

* **Autoencoding models** like BERT encode text, producing embedding vectors.

* **Autoregressive models** like GPT perform text generation in an iterative way, predicting the next output token based on the input text and the previous output tokens.

The models behind the chat apps (ChatGPT is the top popular one) are based on autoregressive models. The input text is called the **prompt**. Though these models perform most of the classic NLP tasks, the users do not perceive them as such. To get the best output, you have to carefully craft your prompts. This is called **prompt engineering**, a skill set quite appreciated in business. Two simple examples: 

* To get a summary, the user includes in the prompt instructions such as *summarize the following text, in no more than 100 words $\dots$*. 

* To get a translation, he/she includes something like *translate to French the following text* $\dots$. 

## What makes the transformers special?

An LLM comes with a token dictionary, a tokenizer and an embedding model that encodes the tokens as vectors of a given length. This toolkit is specific of the model. When we prompt a string, the string is split in a list of tokens: This list is then encoded as a matrix, with a token embedding vector in each row. So, the input in a transformer is, properly speaking, a 2D array, not a text. This implies that, even if the original transformer was designed with an NLP perspective, in the transformer, as in any model based on a neural network architecture, the input and the output are tensors. As a matter of fact, transformers have also been applied in computer vision (they are a serious challenge to CNN models).

The three differential components of the transformer are:

* The **positional embedding**. Positional embeddings are used to give the model information about the position of each token in the input text or, more specifically, the row number in the input 2D array. The positional embedding generates a different vector for every row. These vectors are added to the input token vector. This is a way of encoding the order of the tokens (and punctuation) in the text, which is one of the ingredients of the "meaning". 

* The **attention mechanism** is a transformation that replaces every token vector by a weighted average of all token vectors. The weights are based on the similarity among the token vectors. A simple example may help to understand the role of attention. Suppose that word 'bank' comes in the sentence 'I got money from the bank'. The attention mechanism changes the vector that stands for 'bank' by pushing it toward the vector that stands for money. So, 'bank' gets a more "financial" meaning. This is the way in which the attention mechanism captures the meaning of a word from the context.

* **Temperature**. In autorregressive models, the predicted token is not always the one with the maximum probability. Instead it is randomly chosen according to the token probabilities. For instance, if the model predicts a probability 0.4 for a certain token, that token will be chosen 40% of the time. The temperature is a parameter chosen in the range $0 \le t \le 1$, controlling the randomness of the prediction of the next token. In mathematical terms, the token probabilities are calculated by means of a modified softmap function as

$$p_i = \frac{\exp(z_i/t)}{\exp(z_1/t) + \cdots + \exp(z_m/t)}.$$

With $t = 1$, the next token probabilities are calculated with the ordinary softmax function. With $t = 0$, the probability is equal to 1 for the token for which $z_i$ is maximum, and 0 for the rest. So, there is no randomness at all, and, for a given input, the model will predict always the same token.
