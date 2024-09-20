# [ML-24] Large language models

## What is natural language processing?

**Natural language processing** (NLP) is an interdisciplinary subfield of computer science and artificial intelligence. It is primarily concerned with providing computers the ability to process text written in a natural language. 

The data unit in NLP is typically a string (this can be a long document). The data set is a collection (*e.g*. a list) of strings. The NLP toolkit discussed in this course is based on deep learning. The data set used to develop NLP tools is called a **corpus** (*e.g*. the Wikipedia corpus). 

The adjective *natural* is used here to refer to the languages we humans use, such as English, Spanish, etc, as opposed to computer languages like Python or Java. Nevertheless, the language models developed since 2020 have been trained so that they can also process code written in a number of computer languages, so the distinction is fading away. In this lecture, we assume that we are dealing with text written in English, with occasional comments to other languages.

NLP has advanced in giant steps in the last years, as **large language models** (LLMs) took the stage. Reading anything published ealier than five years ago may be wasting time.

## Tokens

Even the data unit in NLP is a string, that string is not processed as a whole, but split in substrings called **tokens**. The extraction of tokens from a text corpus is called **tokenization**. Tokenization is one of the oldest problems in NLP, and different approaches have been discussed for years: words, sub-words, pairs of words, etc. Also, tokenization was a harder problem in some languages like Spanish and French with its countless verb forms, or German with its declensions, prefixes and suffixes, than in English.

Nowadays, the debate has lost interest (at least for the beginners). First, there is plenty of supply of tokenization models, called **tokenizers** (right now, you can find 3,383 in Hugging Face). Second all the relevant language models come with their own tokenizers, so you don't to think about them once you have chosen your model. In these models, most of the tokens are words, and a small proportion of subwords. Also, punctuation marks can be tokens. In addition to all these, there is an "unknown" token (the vocabulary of the model is limited), a token for the start of the text and another token for the end (think on ChatGPT you will see that this is needed). So, when we input a text to one of these models, it is converted to a list of tokens, though we never see them in practice.

## NLP tasks

* **Text classification**. A popular application is **sentiment analysis**.

* **Text generation**. 

* **Summarization**.

* **Translation**.

* **Question answering** (QA).

* **Named entity recognition** (NER).

**Foundation models** and **downstream tasks**

## Basic tools

* **Tokenization**.

* **Embeddings**.

* **Attention mechanism**.


## Models

* **Recurrent neural networks** (RNN).

* **Long term short memory** (LSTM) models. GRU.

* **Transformers**.


## What are large language models?

**Large language models** (LLMs) are AI models that predict the probability of a token in a dictionary. They are usually (not always) based on the **transformer** architecture. LLMs can perform a wide range of language tasks, from simple text classification to text generation, with high accuracy.

Training a LLM consists of two steps:

* **Pre-training** on a corpus of text data and specific language modeling task.

* **Fine-tuning** on a much smaller data set.

**Transfer learning** is a technique used in machine learning to leverage the knowledge gained from one task to improve performance on another related task. Transfer learning for LLMs involves taking an LLM that has been pre-trained on one corpus of text data and then fine-tuning it for a specific "downstream" task, such as text classification or text generation, by updating the model's parameters with task-specific data. 

Transfer learning allows LLMs to be fine-tuned for specific tasks with much smaller amounts of task-specific data than it would require if the model were trained from scratch. This greatly reduces the amount of time and resources required to train LLMs.

Application Programming Interface (API).

Software Developer Kit (SDK).

API endpoints.

## Applications of LLMs

* Applications in healthcare: electronic medical record (EMR) processing, clinical trial matching, drug discovery.

* Applications in finance: fraud detection, news sentiment analysis and trading strategies.

* Other applications: customer service automation chatbots and virtual assistants.

## LLMs tasks

LLM's perform two basic tasks:

* **Autoregressive models** like GPT predict the next token based only on the previous tokens in the phrase (decoder). Tasks: text generation.

* **Autoencoding models** like BERT reconstruct the original sentence from a corrupted version (encoder). Tasks: classification, extractive QA.

An LLM can be either autorregressive, autoencoding of a combination of the two. The typical tasks for an encoder-decoder are: conversational, translation and multimodal.

## Temperature

In autorregressive models, the predicted token is not that with the maximum probability. Instead it is chosen at random according the token probabilities. For instance, if the model predicts a probability 0.4 for a certain token, that token will be chosen 40% of the time. 

The **temperature** is a parameter chosen in the range $0 \le t \le 1$, controlling the randomness of the prediction of the next token. In mathematical terms, the token probabilities are calculated by means of a modified softmap function as

$$p_i = \frac{\exp(z_i/t)}{\exp(z_1/t) + \cdots + \exp(z_m/t)}.$$

With $t = 1$, the next token probabilities are calculated with the ordinary softmax function. With $t = 0$, the probability is equal to 1 for the token for which $z_i$ is maximum, and 0 for the rest. So, there is no randomness at all, and, for a given input, the model will predict always the same token.

## Summarizing text

* Basic.

* One-sentence.

* Grade-level.

* Key points.

## Text classification

* Zero-shot.

* Few-shot.

* Batch.

## Text generation

* Content creation.

* Idea generation.

* Conversational.

* List generation.

## Transforming text

* Translation.

* Conversion.


## Pre-tokenization

Hugging Face documentation makes a distinction between **pre-tokenization** and tokenization, whereas many online articles only use the term tokenization. On English and most European languages, pre-tokenization typically involves splitting the text of a corpus into words, using white space and punctuatiobn as separators. However, some European languages (such as French and German) use **diacritical marks**, and removing them can change the meaning of a word.

The terms **uncased** and **cased** are used in reference to LLMs. An uncased LLM is an LLM in which all words are lowercase and all diacritical marks have been removed.

Diacritical marks appear in several Indo-European languages, such as French, Spanish, German, and Scandinavian languages, but not in English. However, a cased LLM retains uppercase letters and diacritical marks (if any). Note that some languages, such as Mandarin and Japanese, do not have the concept of uppercase or lowercase letters. The choice between an uncased LLM and a cased LLM depends directly on your use case.

English is perhaps the easiest language for pre-tokenization because of the following features:

* No diacritical marks.

* All words are separated by a space character.

* No declension of articles or adjectives (versus German/Slavic languages).

* One form for every word in every grammatical case (*e.g*. nominative and accusative)

## Tokenization tasks

Tokenization involves finding the tokens in sentences and documents, where tokens can be words, characters, or partial words. Tokenization must also take into account potential issues that can arise while performing the following subtasks:

* Convert text to lowercase (or not).

* Process punctuation that separates sentences.

* Handle diacritical marks ("péché" and "pêche").

* Handle contractions ("won't" versus "will not").

* Processing unusual (infrequent) words.

### Tokenization methods

* WordPiece.

* SentencePiece.

* BytePair (BPE)

## Issues in tokenization

Tokenization might seem like a simple task, but there are several non-trivial aspects of tokenization. Here is a list of points (in no particular order) to address while
performing tokenization of a corpus:

* Common nouns versus proper nouns.

* An optional word delimiter.

* Diacritical marks and word meanings.

* Different subword tokenization techniques.

* Singular versus plural nouns.

* Variants in word spellings.

* Typographical errors.

* False cognates in translation tasks.

First, converting words to lowercase loses the distinction between common nouns and proper nouns (such as "Stone" versus "stone). Second, the use of a white space to tokenize sentences into word tokens does not work correctly in languages such as Japanese, in which spaces are optional. Consequently, a sentence can be written as a single contiguous string of non-space characters: Thiscanbeapainandtediousinanylanguage. This example shows you that missing white spaces increases complexity (and perhaps you had to concentrate to parse the preceding string sans spaces).

Third, dropping diacritical (accent) marks can make a word ambiguous: peche has three possible meanings (*i.e +. peach, fish, or sin) depending on which diacritical mark appears in the word. Fourth, it is important to consider punctuation during tokenization, thereby ensuring that models are not required to learn different representations of a word combined with multiple punctuation symbols.

Fifth, different libraries and toolkits perform different types of subword tokenization. For example, it is possible to tokenize the word "don't" as either ["don", "'", "t"] or as ["do", "n't"]. Sixth, removing a final "s" from an English word can result in a word with a different meaning. For example, "birds" and "bird" are the plural and singular form of a noun, whereas "new" and "news" have different meanings. In general, adding the letter "s" to a noun creates the plural of that noun, whereas adding an "s" to an adjective can result in a noun that has a different meaning from the adjective.

While some of the preceding examples might seem insignificant (or even trivial), such details can affect the accuracy as well as the fluency of a text string that has been translated into a language that is different from the language of the input text.

Tokenization must also address other issues, such as handling alternate ways to spell a word (*e.g*. "favor" versus "favour"), different meanings of the same word and typographical errors. Other potential issues include variants in spelling and capitalization, irregular verbs, and the out-of-vocabulary tokenization problems. There is yet another issue pertaining to differences in pronunciation.

## Word-based tokenizers

In general, word-based tokenizers are straightforward because they involve a limited number of rules, and they can achieve reasonably good results. 

Some word tokenizers specify extra rules for punctuation, which can result in a large **vocabulary** (*i.e*. the collection of tokens in a given corpus). In addition, words are assigned an ID that ranges from 0 to $N-1$, where $N$ is the number of tokens in the vocabulary. The model identifies a given word via its assigned ID value. Unfortunately, closely related words are treated as different words, which means they will be assigned different id values. For example, the words sing, sang, sung and singing are related in meaning. 

Moreover, the model will treat the singular form of a noun differently from the plural form of that noun. This process is further complicated in languages (such as German and Slavic languages) that have a masculine, feminine, neuter, and plural form of nouns, all of which are treated as different nouns. For example, English has only one form for the article "the", whereas the declension of the definite article "der" in German produces seven variants (der, die, das, des, des, dem and den)

## Limitations of word tokenizers

A model only recognizes word tokens that are part of its training step, which means that compound words will not be recognized if they do not appear in the training step. For example, if the words "book" and "keeper" are part of the training step, but the word "bookkeeper is not, "bookkeeper" will not be recognized and hence it will be represented via the `UNK` token.

Another challenge for tokenizers pertains to contractions. For example, the English words "its" and "it's" have an entirely different meaning: "its" refers to possession, whereas "it's" means "it is". In the case of nouns, the apostrophe also refers to possession (*e.g*. John's car). Thus, the sentence "It's true that John's pen is worth its weight in gold" can have only one interpretation.

## Subword tokenization

**Subword tokenization** is typically based on algorithms, statistical rules, and an important heuristic: tokenize uncommon or infrequently appearing words into subwords, and do not split frequently occurring words. Such tokenization can be easily performed for English adverbs with the suffix "ly": replace the adverb with two tokens, where the second token is the combination "ly". Thus, "slowly" is split in "slow" and "ly".

In some cases, this decomposition into two tokens produces an actual word for the first token, which is the case for the preceding examples. Moreover, subword tokenization can also yield tokens that have meaning, such as tokenization the word "internationalization" as "international" and "ization".

## Tokenizer

A **tokenizer** is an algorithm that splits a string in tokens. These tokens either belong to a vocabulary or the "unknown" token, which is denoted as `OOV` (out-of-vocabulary) or as `UNK`. Though many technicalities arise, the procedure is simple: (a) split in words, (b) split further the words that are not in the vocabulary, searching for a token which is in the vocabulary. When further splitting does not find tokens from the vocabulary, the tokenizer labels the current token as `OOV`.
