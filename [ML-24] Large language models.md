# [ML-24] Large language models

## What is natural language processing?

**Natural language processing** (NLP) is an interdisciplinary subfield of computer science and artificial intelligence. It is primarily concerned with providing computers the ability to process text written in a natural language. 

The adjective *natural* is used here to refer to the languages we humans use, such as English, Spanish, etc, as opposed to computer languages like Python or Java. Nevertheless, the language models developed since 2020 have been trained so that they can also process code written in a number of computer languages, so the distinction is fading away. In this lecture, we assume that we are dealing with text written in English, with occasional comments to other languages.

The NLP toolkit discussed in this course is based on deep learning. The data unit in NLP is typically a string (this can be a long document). The data set is a collection (*e.g*. a list) of strings. The data set used to develop NLP tools is called a **corpus** (*e.g*. the Wikipedia corpus). 

NLP has advanced in giant steps in the last years, as **large language models** (LLMs) took the stage. Reading anything published ealier than five years ago may be wasting time.

## Tokens

Even the data unit in NLP is a string, that string is not processed as a whole, but split in substrings called **tokens**. The extraction of tokens from a text corpus is called **tokenization**. Tokenization is one of the oldest problems in NLP, and different approaches have been discussed for years: words, sub-words, pairs of words, etc. Also, tokenization is a harder problem in some languages like Spanish and French with its countless verb forms, or German with its declensions, prefixes and suffixes, than in English.

Nowadays, the debate has lost interest (at least for the beginners). First, there is plenty of supply of tokenization models, called **tokenizers** (right now, you can find 3,383 in Hugging Face). Second, all the relevant language models come with their own tokenizers, so you don't have to think about them once you have chosen your model. In these models, most of the tokens are words, and a small proportion of subwords. Also, punctuation marks can be tokens. In addition to all these, there is an "unknown" token (the vocabulary of the model is limited), a token for the start of the text and another token for the end (think on ChatGPT you will see that this is needed). So, when we input a text to one of these models, it is converted to a list of tokens, though we never see them in practice.

## NLP tasks

Language models were oriented, during many years, to perform specific tasks. The classics are:

* **Text classification**. The input is a text and the output a set of class probabilities. A popular application is **sentiment analysis**. We have seen this in example ML-23, where we trained a model for fake news detection.

* **Text generation**. The input is a text and the output another text. This covers many particular cases, as we see below.

* **Summarization**. The output text is a summarized version of the inpute text. 

* **Translation**. The output text is a translation of the input text to another language. This is one of the very classics of artificial intelligence, as, in the cold war years, the US aimed at having automatic translation from Russian to English.

* **Question answering** (QA). The input text is a question and the output is an answer to that question, extracted either from a context text inputted together with the question or retrived from a data source.

* **Named entity recognition** (NER). The input is a text together with a collection of entities. The output is a list of the entities found in the text. For instance, the model can extract product ID's from an invoice.

## What are large language models?

Large language models are not called so, not only because of their size, which often goes beyond 1B parameters, but for being a new generation, based on a novel architecture, the **transformer**, published in 2016. LLM's can be used directly, or taken as **pre-trained model** for transfer learning, which in this context is called **fine-tuning**. Transfer learning is very common in LLM's. The original pre-trained models are called **foundation models**, and the task for which they are fine-tuned is called the **downstream task**.

There are just a few relevant foundation models, but thousands of fine-tuned versions of them. You can find most of these in Hugging Face. The two classics are **BERT** (Bidirectional Encoder Representations from Transformers), developd by Google, and **GPT** (Generative Pre-trained Transformer), developed by OpenAI. 

LLM's are essentially of two types:

* **Autoencoding models** like BERT encode text, producing embedding vectors.

* **Autoregressive models** like GPT perform text generation in an iterative way, predicting the next output token based on the input text and the previous tokens.

The models behind the chat apps (ChatGPT is the top popular one) are based on autoregressive models. The input text is called the **prompt**. Though these models perform most of the classic NLP tasks, the users do not perceive them as such. Example: to get a summary, the user includes in the prompt instructions such "summarize the following text, in no more than 100 words $\dots$". Or to get a translation, "translate to French the following text $\dots$". To get the best output, you have to carefully craft your prompts. This is called **prompt engineering**, and it is appreciated by the industry.

## What makes the transformers special?

LLM's come with a token dictionary, a tokenizer and an embedding model that encodes the tokens as vectors of a given length. This toolkit is specific for the model. When we prompt a string, the string is split in tokens, so it becomes a list of tokens, which is subsequently encoded as a matrix, with an embedding vector in each row. The input in a transformer is, properly speaking, a 2D array. This implies that, even if the original transformer was designed with an NLP perspective, in the transformer, as in any model based on a neural network architecture, the input and the output are tensors. As a matter of fact, transformers have also been used for computer vision (they are serious challenge to CNN models).

* **Positional embedding**. Positional embeddings are used to give the model information about the position of each token in the input text, or more specifically, the row number in the input 2D array. The positional embedding generates a different vector for every row. These vectors are added to the input vector. This is a way of encoding the order of the words (and punctuation) in the text, which is one of the ingredients of what we call "meaning". 

* **Attention mechanism**.

* **Temperature**. In autorregressive models, the predicted token is not that with the maximum probability. Instead it is chosen at random according the token probabilities. For instance, if the model predicts a probability 0.4 for a certain token, that token will be chosen 40% of the time. The temperature is a parameter chosen in the range $0 \le t \le 1$, controlling the randomness of the prediction of the next token. In mathematical terms, the token probabilities are calculated by means of a modified softmap function as

$$p_i = \frac{\exp(z_i/t)}{\exp(z_1/t) + \cdots + \exp(z_m/t)}.$$

With $t = 1$, the next token probabilities are calculated with the ordinary softmax function. With $t = 0$, the probability is equal to 1 for the token for which $z_i$ is maximum, and 0 for the rest. So, there is no randomness at all, and, for a given input, the model will predict always the same token.
