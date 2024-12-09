# [ML-25] LLM API tutorial

## Using an LLM through the API

An **application programming interface** (API) is a way for two or more computer programs or components to communicate with each other. It is a type of software interface, offering a service to other pieces of software. The main players in the LLM arena (OpenAI, Anthropic, Meta, Cohere, etc) offer API platforms which allow us to use their models in a remote way. For closed models like those of the GPT collection (later than GPT-2), this is the only way you have to include the tasks performed by the model in your programs. 

In some cases, like **OpenAI**, you have to pay for using LLM's through an API, though you find sometimes, as in **Cohere** and Google's Gemini, a margin for free, though limited use. OpenAI and Cohere are discussed in this tutorial.

Note that everything in this field is changing fast, so we can say nothing completely definite about anything. For instance, prices are falling down due to fierce competition. Also, it seems that the focus is moving from the **chat** to the **agent**. 

The remote connection with an LLM API is managed by means of an **API key** and a **software development kit** (SDK). The API key is given by the provider through its website. Before asking for the API key, you must register, providing a user name and a password. The SDK can be a Pyhton package, but other languages like Java or node.js are available in most cases. 

## The OpenAI API

The OpenAI API provides an interface to OpenAI models for natural language processing, image generation, semantic search, and speech recognition. The Python SDK is a package called `openai` that you can install, *e.g*. with `pip install openai`. After importing the package, you create a **client** using your API key:

```
import openai
client = openai.OpenAI(api_key='YOUR_API_KEY')
```

Since we cannot use the OpenAI API for free, we are very brief about it in this tutorial. Anyway, the methods `.chat.completions.create()` and `.embeddings.create()` of the package `openai` are quite similar to the methods `.chat()` and `.embed()` of the package `cohere`, which we discuss below. There is no specific method for reranking in `openai`, so you are expected to manage it through the chat. 

## The Cohere's API

Cohere offers a chat interface at `https://coral.cohere.com`, quite similar to ChatGPT. In the Cohere API, the Python SDK is a package called `cohere`, which you can install in the usual way. For technical detail and code examples, you can take a look at `https://docs.cohere.com/reference`. 

After importing the package `cohere`, you create a client using your API key:

```
In [1]: import cohere
   ...: co = cohere.ClientV2('YOUR_API_KEY')
```

The following tutorial is limited to three basic methods, `.chat()`, `.embed()` and `.rerank()`. Note that we use `.ClientV2()`, which is a recent upgrade. Most tutorials in Internet still use the previous version, `.Client()`, whose syntax is a bit different.  

## Chatting through the API

The method `.chat()` allows for chatting in a programmatic way. The models for chatting are called **Command** in Cohere. The current version is:

```
In [2]: model_name = 'command-r-plus-08-2024'
```

This is a 104B parameter multilingual model. Old and new versions of the Command models are available at Hugging Face, so you may run them locally, if you have a few hundred GB of free space in your hard disk and 32 GB RAM.

Let us start with a single prompt. Note the way in which the input is entered, which is the same as in the OpenAI API. This protocol may look a bit complicated, but it has been designed for better control of the conversation.

```
In [3]: input_message = [{'role': 'user', 'content': 'Tell me, in no more than 25 words, what is machine learning'}]
```

We submit now our prompt to the chat model.

```
In [4]: response = co.chat(model=model_name, messages=input_message)
```

Note that, unlike a common chat app, this does not output a simple string, but an object containing a lot of metadata. 

```
In [5]: response
Out[5]: ChatResponse(id='b7f23c08-3c53-4564-a52a-ca9ba4b2e2fe', finish_reason='COMPLETE', prompt=None, message=AssistantMessageResponse(role='assistant', tool_calls=None, tool_plan=None, content=[TextAssistantMessageResponseContentItem(type='text', text='Machine learning is a branch of computer science that enables systems to automatically learn and improve from experience, without being explicitly programmed. It focuses on developing algorithms to analyse data.')], citations=None), usage=Usage(billed_units=UsageBilledUnits(input_tokens=16.0, output_tokens=33.0, search_units=None, classifications=None), tokens=UsageTokens(input_tokens=217.0, output_tokens=33.0)), logprobs=None)
```

Nevertheless, we can extract the text of the response in a direct way:

```
In [6]: response.message.content[0].text
Out[6]: 'Machine learning is a branch of computer science that enables systems to automatically learn and improve from experience, without being explicitly programmed. It focuses on developing algorithms to analyse data.'
```

Since we are interested only in the output text, it will be practical to streamline the code with a function: 

```
In [7]: def mychat(input):
   ...:     return co.chat(model=model_name, messages=input).message.content[0].text
```

## Creating a conversation

The input message can be organized in a way that allows us to control the flow in the chat. Let us engage now in a conversation in which our questions will be: 

```
In [8]: query1 = 'Who is the president of USA?'
   ...: query2 = 'How old is he/she?'
```

We get the first response as we did above:

```
In [9]: mes1 = [{'role': 'user', 'content': query1}]
   ...: resp1 = mychat(mes1)
   ...: resp1
Out[9]: 'As of January 2024, the current president of the United States of America is Joseph Robinette Biden Jr., also known as Joe Biden. He is the 46th president and was sworn into office on January 20, 2021.'
```

We wish this to be remembered for the second question to be related to Joe Biden. We can control this by including in the input message the current state of the conversation.

```
In [10]: mes2 = mes1 + [{'role': 'assistant', 'content': resp1}] + [{'role': 'user', 'content': query2}]
    ...: mes2
Out[10]: 
[{'role': 'user', 'content': 'Who is the president of USA?'},
 {'role': 'assistant',
  'content': 'As of January 2024, the current president of the United States of America is Joseph Robinette Biden Jr., also known as Joe Biden. He is the 46th president and was sworn into office on January 20, 2021.'},
 {'role': 'user', 'content': 'How old is he/she?'}]
```

Note that the input is always a list of messages, which are dictionaries with two keys, `role` and `content`. We will come back to the roles below. First, let us enter the second question. As you can see, the conversation continues.

```
In [11]: resp2 = mychat(mes2)
    ...: resp2
Out[11]: 'Joe Biden was born on November 20, 1942, in Scranton, Pennsylvania, making him 81 years old as of 2024.'
```

## The roles in the chat

We have used the roles `user` and `assistant` to create a conversation in the preceding section. There are two other roles, `system`  and `tool`. The `tool` role is used to enter functions in the chat. This role is not covered by this short tutorial, but, even if your experience with Python functions is limited, you can guess that this creates a host of oportunities for task automation in the chat.

The `system` role is typically used to add content throughout a conversation, or to adjust the model's overall behavior and conversation style. Most of the so called **prompt engineering**, as applied in chat apps, is just a way of managing this role. We illustrate this with a simple example. Take the following question:

```
In [12]: query = 'Who is the president of USA?'
```

Suppose that you want the response to satisfy a length constraint such as:

```
In [13]: length = 'Respond the following question in no more than 10 words'
```

We include this as an additional message in the list. The recommended practice is to put the `system` messages at the begginning.

```
In [14]: mes1 = [{'role': 'system', 'content': length}, {'role': 'user', 'content': query}]
```

Our function `mychat` responds now as requested:

```
In [15]: resp1 = mychat(mes1)
    ...: resp1
Out[15]: 'Joe Biden.'
```

Since we enter the messages as a list, we can easily include several instructions. A simple example follows.

```
In [16]: style = 'Include middle names in your response'
    ...: mes2 = [{'role': 'system', 'content': length}, {'role': 'system', 'content': style}, {'role': 'user', 'content': query}]
    ...: resp2 = mychat(mes2)
    ...: resp2
Out[16]: 'Joseph Robinette Biden Jr.'
```

## Embeddings

We are going to extract some embedding vectors from a Cohere model. We first specify the embedding model.

```
In [17]: model_name = 'embed-english-v3.0'
```

In this simple example, we use three sentences concerned with machine learning. The first two sentences are definitions collected in Internet, and the third one is a (very) subjective statement.

```
In [18]: text1 = '''Machine learning (ML) is a field of study in artificial intelligence concerned 
    ...: with the development and study of statistical algorithms that can learn from data and 
    ...: generalize to unseen data and thus perform tasks without explicit instructions.'''
    ...: text2 = '''Machine learning is a subset of artificial intelligence that gives systems the
    ...: ability to learn and optimize processes without having to be consistently programmed.'''
    ...: text3 = '''This course on machine learning is very interesting.'''
    ...: texts=[text1, text2, text3]
```

Next, we submit our texts. Cohere embedding model allows you choose among five input types: `search_document`, `search_query`, `classification`, `clustering` and `image`. The first one is typically used to generate embedding vectors to be stored in a vector database for future search. We will use the second one in example ML-26. We use `embedding_types=['float']` to get vectors whose terms are float numbers, with decimals. The other possibilities are beyond of the scope of this course.

```
In [19]: response = co.embed(model=model_name, input_type='search_document', embedding_types=['float'], texts=texts)
```

Now, `response` is a big object which contains the three embedding vectors. So, we don't print it. `response.embeddings.float`is a list containing the vectors. To manage this, we convert the list to 2D array.

```
In [20]: import numpy as np
    ...: embeds = np.array(response.embeddings.float)
```

Let us check the shape of this array.
```
In [21]: embeds.shape
Out[21]: (3, 1024)
```

We see that the embedding dimension is 4,096 here. We can also take a look at the first terms.

```
In [22]: embeds[:, :5]
Out[22]: 
array([[-0.0322876 , -0.0135498 , -0.0406189 ,  0.02316284, -0.02153015],
       [-0.01147461, -0.03625488, -0.02270508,  0.00922394, -0.03140259],
       [ 0.00759506, -0.01594544,  0.01041412, -0.01133728, -0.03338623]])
```

As in the encoders that we managed through the package `sentence_transformers`, the embedding vectors are normalized, that is, they have unit length. Remember that the (geometric) length is the square root of the sum of the squares. We can also check this.

```
In [23]: (embeds**2).sum(axis=1)
Out[23]: array([1.00032005, 1.00065999, 0.99988624])
```

We can expect the vectors of first two sentences to show a stronger similarity. Since these vectofrs have unit length, it is indiferent to use the measure the similarity with the distance or with the cosine. Both are provided by the scikit-learn module `metrics`. We use the cosine in this course. First we import the function `cosine`:

```
In [24]: from sklearn.metrics.pairwise import cosine_similarity
```

Then, we apply the function to the array `embeds`. Note that this function take the vectors by row, which is appropriate here.

```
In [25]: cosine_similarity(embeds).round(3)
Out[25]: 
array([[1.   , 0.843, 0.472],
       [0.843, 1.   , 0.48 ],
       [0.472, 0.48 , 1.   ]])
```

This can be read as a correlation matrix (mathematically, the correlation is a cosine). The results are as expected.

## Reranking

Reranking models sort text inputs by their semantic relevance to a specified query. They are often used to sort search results returned by a semantic search engine.

We specify the model as in the other cases.

```
In [26]: model_name = 'rerank-english-v3.0'
```

```
In [27]: query = 'Are there fitness-related perks?'
```

```
In [28]: doc1 = '''Reimbursing Travel Expenses: Easily manage your travel expenses by submitting them
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
In [29]: response = co.rerank(model=model_name, query=query, documents=docs, top_n=4)
```

Again the response of the model is an object in which many things are packed.

```
In [30]: response
Out[30]: V2RerankResponse(id='8338f1af-a0e0-43e4-adfb-d2915df9ae11', results=[V2RerankResponseResultsItem(document=None, index=2, relevance_score=0.01798621), V2RerankResponseResultsItem(document=None, index=3, relevance_score=8.463939e-06), V2RerankResponseResultsItem(document=None, index=0, relevance_score=7.296379e-06), V2RerankResponseResultsItem(document=None, index=1, relevance_score=1.1365637e-06)], meta=ApiMeta(api_version=ApiMetaApiVersion(version='2', is_deprecated=None, is_experimental=None), billed_units=ApiMetaBilledUnits(images=None, input_tokens=None, output_tokens=None, search_units=1.0, classifications=None), tokens=None, warnings=None))
```

So, there are two relevant pieces of information here: (a) the index, which indicates the index of this document in the list submitted, and (b) a **relevance score**, which is query dependent, and could be higher or lower depending on the query and passages sent in. It is easy to extract the whole ranking as a table using Pandas:

```
In [31]: import pandas as pd
    ...: pd.DataFrame({'index': [response[i].index for i in range(len(docs))],
    ...:     'relevance_score': [response[i].relevance_score for i in range(len(docs))]})
Out[31]: 
   index  relevance_score
0      2         0.017986
1      3         0.000008
2      0         0.000007
3      1         0.000001
```

Here, the second document is clearly ahead of the other three.

## Homework

1. As you already know, when repeat a prompt to a chat app based on an LLM, you may not get the same response. This is managed by a parameter called **temperature**. In Cohere's method `.chat()`, the default is `temperature=0.3`. Check that, with the default temperature, you get different responses to `In [2]`. Try also with `temperature=1`.
