# [ML-25] LLM API tutorial

## Using an LLM through the API

An **application programming interface** (API) is a way for two or more computer programs or components to communicate with each other. It is a type of software interface, offering a service to other pieces of software. The main players in the LLM arena (OpenAI, Anthropic, Meta, Cohere, etc) offer API platforms which allow us to use their models in a remote way. For closed models like those of the GPT-4 collection, this is the only way you have to include the tasks performed by the model in your programs. 

In some cases, like **OpenAI**, you have to pay for using LLM's through an API, though you find sometimes, as in **Cohere**, a margin for free, though limited use. OpenAI and Cohere are discussed in this tutorial.

Note that everything in this field is changing fast, so we can say nothing completely definite about anything. For instance, prices are falling down due to fierce competition. Also, it seems that the focus is moving from the **chat** to the **assistant**. 

The remote connection with an LLM API is managed by means of an **API key** and a **software development kit** (SDK). The API key is given by the provider through its website. Before asking for the API key, you must register, providing a user name and a password. The SDK can be a Pyhton package, but other languages like node.js are available in most cases. 

## The OpenAI API

The OpenAI API provides an interface to OpenAI models for natural language processing, image generation, semantic search, and speech recognition. The Python SDK is a package called `openai` that you can install, *e.g*. with `pip install openai`. After importing the package, you create a **client** using your API key:

```
import openai
client = openai.OpenAI(api_key='YOUR_API_KEY')
```

Since we cannot use the OpenAI API for free, we are very brief about it in this tutorial. Anyway, the methods `.chat.completions.create()` and `.embeddings.create()` of the package `openai` are quite similar to the methods `.chat()` and `.embed()` of the package `cohere`, which we discuss below. There is no specific method for reranking in `openai`, so you are expected to manage it through the chat. 

## The Cohere's API

In Cohere API, the Python SDK is a package called `cohere`, which you can install in the usual way. For technical detail and code examples, you can take a look at `https://docs.cohere.com/reference`. 

The following tutorial is limited to three basic methods, `.chat()`, `.embed()` and `.rerank()`.

## Chatting through the API

The method `.chat()` allows for chatting in a programmatic way. Mind that if you just want to ask a couple of questions, you can use the chat interface at `https://coral.cohere.com`, which is quite similar to ChatGPT, without any Python code.

After importing the package `cohere`, you create a client using your API key:

```
In [1]: import cohere
   ...: co = cohere.Client('YOUR_API_KEY')
```

The models for chatting are called **Command** in Cohere. The (current) default is `command-r-plus`, a 104B parameter multilingual model, but this may change as soon as Cohere develops a more powerful model. Both the old `command-r` and `command-r-plus` are available from Hugging Face, so you may run them locally, but you would need for that about a few hundred GB of free space in your hard disk and 32 GB RAM.

Let us start with a supersimple example, in which we enter a question as if we were in common chat application.

```
In [2]: response = co.chat(message='Tell me, in no more than 25 words, what is machine learning')
```

Note that, unlike a common chat app, this does not output a simple string, but an object containing a lot of metadata. 

```
In [3]: response
cohere.Chat {
	id: 5101bd44-cbb6-4aba-b265-87a3048b25de
	response_id: 5101bd44-cbb6-4aba-b265-87a3048b25de
	generation_id: 50f5eb53-3524-43a7-8f11-9cb0812eafdc
	message: Tell me, in no more than 25 words, what is machine learning.
	text: Machine learning is a branch of AI that uses data to train models and make predictions or decisions without being explicitly programmed.
	conversation_id: None
	prompt: None
	chat_history: [{'role': 'USER', 'message': 'Tell me, in no more than 25 words, what is machine learning.'}, {'role': 'CHATBOT', 'message': 'Machine learning is a branch of AI that uses data to train models and make predictions or decisions without being explicitly programmed.'}]
	preamble: None
	client: <cohere.client.Client object at 0x74d52bf9bc40>
	token_count: None
	meta: {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 17, 'output_tokens': 23}, 'tokens': {'input_tokens': 215, 'output_tokens': 23}}
	is_search_required: None
	citations: None
	documents: None
	search_results: None
	search_queries: None
	finish_reason: COMPLETE
}
```

Nevertheless, you can extract the text of the response in a direct way:

```
In [4]: response.text
Out[4]: 'Machine learning is a branch of AI that uses data to train models and make predictions or decisions without being explicitly programmed.'
```

```
In [5]: co.chat(message='Tell me, in no more than 25 words, what is machine learning').text
Out[5]: 'Machine learning is a branch of computer science that enables computers to learn and improve from experience, without being explicitly programmed. It focuses on developing algorithms to make predictions.'
```

```
In [6]: co.chat(message='Tell me, in no more than 25 words, what is machine learning', temperature=0).text
Out[6]: 'Machine learning is a branch of computer science that enables systems to automatically learn and improve from experience, without being explicitly programmed. It focuses on developing algorithms to analyse data.'
```

```
In [7]: co.chat(message='Tell me, in no more than 25 words, what is machine learning', temperature=0).text
Out[7]: 'Machine learning is a branch of computer science that enables systems to automatically learn and improve from experience, without being explicitly programmed. It focuses on developing algorithms to analyse data.'
```

## Creating a conversation

As the contents of `response` suggests, you can have a better control of your chat. Let us engage now in a conversation in which our questions will be: 

```
In [8]: prompt1 = 'Who is the president of USA?'
   ...: prompt2 = 'How old is he/she?'
```

We get the first response as we did above:

```
In [9]: resp1 = co.chat(message=prompt1)
   ...: resp1.text
Out[9]: 'As of January 2024, the current president of the United States of America is Joe Biden. He was sworn in as the 46th president on January 20, 2021, following his victory in the 2020 presidential election.'
```

We wish this to be remembered for the second question to be related to Joe Biden. We can control this through the **chat history**, which is a list of dictionaries corresponding to the contributions of the user and the chatbot to the conversation.

```
In [10]: resp1.chat_history
Out[10]: 
[ChatMessage(role='USER', message='Who is the president of USA?'),
 ChatMessage(role='CHATBOT', message='As of January 2024, the current president of the United States of America is Joe Biden. He was sworn in as the 46th president on January 20, 2021, following his victory in the 2020 presidential election.')]
```

The chat history is a list of messages. Note the roles of USER and CHATBOT. We will come back to the roles below. First, let us continue the chat  taking into account

```
In [11]: resp2 = co.chat(message=prompt2, chat_history=resp1.chat_history)
    ...: resp2.text
Out[11]: 'Joe Biden was born on November 20, 1942, making him 81 years old as of my cut-off date in January 2024.'
```

Now the chat history contains the two prompts and the corresponding responses of the chatbot.

```
In [12]: resp2.chat_history
Out[12]: 
[ChatMessage(role='USER', message='Who is the president of USA?'),
 ChatMessage(role='CHATBOT', message='As of January 2024, the current president of the United States of America is Joe Biden. He was sworn in as the 46th president on January 20, 2021, following his victory in the 2020 presidential election.'),
 ChatMessage(role='USER', message='How old is he/she?'),
 ChatMessage(role='CHATBOT', message='Joe Biden was born on November 20, 1942, making him 81 years old as of my cut-off date in January 2024.')]
 ```

## The roles in the chat

There is a third role, SYSTEM. It is typically used to add content throughout a conversation, or to adjust the model's overall behavior and conversation style. Most of the so called **prompt engineering** is just a way of managing the SYSTEM role. This can be done in many ways, the simplest one being, probably, to set the rules at the begininng of the chat history. The following is a simple example. 

```
In [13]: length = 'Answer the questions in no more than 10 words'
    ...: resp = co.chat(message=prompt1, chat_history=[{'role': 'SYSTEM', 'message': length}])
    ...: resp.text
Out[13]: 'Joe Biden.'
```

Note that the value of the parameter `chat_history` is a Python list, so you can manage it in a Pythonic way. It may be useful to split the rules set by the SYSTEM role, as in the following example.

```
In [14]: length = 'Answer the questions in no more than 10 words'
    ...: style = 'Include middle names'
    ...: resp = co.chat(message=prompt1, chat_history=[{'role': 'SYSTEM', 'message': length}, {'role': 'SYSTEM', 'message': style}])
    ...: resp.text
Out[14]: 'Joseph Robinette Biden Jr.'
```

## Embeddings

```
In [15]: model_name = 'embed-english-v3.0'
```

```
In [16]: text1 = '''Machine learning (ML) is a field of study in artificial intelligence concerned 
    ...: with the development and study of statistical algorithms that can learn from data and 
    ...: generalize to unseen data and thus perform tasks without explicit instructions.'''
    ...: text2 = '''Machine learning is a subset of artificial intelligence that gives systems the
    ...: ability to learn and optimize processes without having to be consistently programmed.'''
    ...: text3 = '''Machine learning (ML) is a branch of artificial intelligence (AI) and computer 
    ...: science that focuses on developing methods for computers to learn and improve their performance.'''
```

```
In [17]: response = co.embed(texts=[text1, text2, text3], model=model_name, input_type='search_document')
```

```
In [18]: import numpy as np
    ...: embeds = np.array(response.embeddings)
```

```
In [19]: embeds.shape
Out[19]: (3, 4096)
```

```
In [20]: embeds[:, :5]
Out[20]: 
array([[-0.0322876 , -0.0135498 , -0.0406189 ,  0.02316284, -0.02153015],
       [-0.01147461, -0.03625488, -0.02270508,  0.00922394, -0.03140259],
       [-0.04000854, -0.02990723, -0.02749634,  0.02322388, -0.04135132]])
```

```
In [21]: abs(embeds[0] - embeds[1]).mean().round(3)
Out[21]: 0.014
```

```
In [22]: abs(embeds[0] - embeds[2]).mean().round(3)
Out[22]: 0.009
```

## Reranking

Rerank models sort text inputs by semantic relevance to a specified query. They are often used to sort search results returned from an existing search solution.

```
In [23]: model_name = 'rerank-english-v3.0'
```

```
In [24]: query = 'Are there fitness-related perks?'
```

```
In [25]: doc1 = '''Reimbursing Travel Expenses: Easily manage your travel expenses by submitting them
    ...: through our finance tool. Approvals are prompt and straightforward.'''
    ...: doc2 = '''Working from Abroad: Working remotely from another country is possible.
    ...: Simply coordinate with your manager and ensure your availability during core hours.'''
    ...: doc3 = '''Health and Wellness Benefits: We care about your well-being and offer gym
    ...: memberships, on-site yoga classes, and comprehensive health insurance.'''
    ...: doc4 = '''Performance Reviews Frequency: We conduct informal check-ins every quarter
    ...: and formal performance reviews twice a year.'''
    ...: docs = [doc1, doc2, doc3, doc4]
```

```
In [26]: response = co.rerank(query=query, model=model_name, documents=docs, top_n=4)
```

Again the response of the model is an object, similar to a list, in which many things are packed. The document on top of the ranking is:

```
In [27]: response[0]
Out[27]: 
RerankResult<document['text']: Health and Wellness Benefits: We care about your well-being and offer gym
memberships, on-site yoga classes, and comprehensive health insurance., index: 2, relevance_score: 0.01798621>
```

So, there are two pieces of information: (a) the index, which indicates the index of this document in the list submitted, and (b) a **relevance score**, which is query dependent, and could be higher or lower depending on the query and passages sent in. It is easy to extract the whole ranking as a table using Pandas:

```
In [28]: import pandas as pd
    ...: pd.DataFrame({'index': [response[i].index for i in range(len(docs))],
    ...:     'relevance_score': [response[i].relevance_score for i in range(len(docs))]})
Out[28]: 
   index  relevance_score
0      2         0.017986
1      3         0.000008
2      0         0.000007
3      1         0.000001
```

Here, the second document is clearly ahead of the other three.

## Homework

1. As you already know, when repeat a prompt to a chat app based on an LLM, you may not get the same response. This is managed by a parameter called **temperature**. In Cohere's method `.chat()`, the default is `temperature=0.3`. Check that, with default temperature, you get different responses to `In [2]`. Try also with `temperature=1`.
