# [ML-25] LLM API tutorial

## Using an LLM through the API

An **application programming interface** (API) is a way for two or more computer programs or components to communicate with each other. It is a type of software interface, offering a service to other pieces of software. The main players in the LLM arena (OpenAI, Anthropic, Meta, Cohere, etc) offer API platforms which allow us to use their in a remote way. For closed models like those of the GPT-4 collection, this is the only way you have to include the tasks performed by the model in your programs. 

In most cases, you have to pay for using LLM's through an API, though you can find, in some of them a margin for free, though limited use. Note that this changing fast, and that prices are falling down due to fierce competition. 

The remote connection with an LLM API is managed by means of an **API key** and a **software development kit** (SDK). The API key is given by the provider. The SDK can be a Pyhton package, but other languages like node.js are available in most cases. 

## The OpenAI API

The OpenAI API provides an interface to OpenAI models for natural language processing, image generation, semantic search, and speech recognition. The Python SDK is a package called `openai` that you can install, *e.g*. with `pip install openai`. After importing the package, you create a **client** using your API key:

```
import openai
client = openai.OpenAI(api_key='YOUR_API_KEY')
```

Since we cannot use the OpenAI API for free, we will not cover it in this tutorial. Anyway, the methods `.chat.completions.create()` and `.embeddings.create()` of the package `openai` are quite similar to the methods `.chat()` and `.embed()` of the package `cohere`, which we discuss below. There is no specific method for reranking in `openai`, so you are expected to manage it through the chat. 

## The Cohere's API

In Cohere API, the Python SDK is a package called `cohere`, which you can install in the usual way. For technical detail and code examples, you can take a look at `https://docs.cohere.com/reference`. 

There are three basic methods, which are briefly introduced in the following tutorial.

## Chatting through the API

The method `.chat()` allows for chatting in a programmatic way. Mind that if you just want to ask a couple of questions, you can use the chat interface at `https://coral.cohere.com`, which is quite similar to ChatGPT, without any Python code.

After importing the package, you create a client using your API key:

```
In [1]: import cohere
   ...: co = cohere.Client('YOUR_API_KEY')
```

The models for chatting are called **Command** in Cohere. The (current) default is `command-r-plus`, a 104B parameter multilingual model, but this may change as soon as Cohere develops a more powerful model. Both the old `command-r` and `command-r-plus` are available from Hugging Face, so you can run them locally, but you will need for that about a few hundred GB of free space in your hard disk and 32 GB RAM.

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

As the contents of `response` suggests, you can have a better control of your chat. Let us engage now in a conversation in which our questions will be: 

```
In [5]: prompt1 = 'Who is the president of USA?'
   ...: prompt2 = 'How old is he/she?'
```

We get the first response as we did above:

```
In [6]: resp1 = co.chat(message=prompt1)
   ...: resp1.text
Out[13]: 'As of January 2024, the current president of the United States of America is Joe Biden. He was sworn in as the 46th president on January 20, 2021, following his victory in the 2020 presidential election.'
```

We wish this to be remembered for the second question to be related to Joe Biden. We can control this through the **chat history**, which is a list of dictionaries corresponding to the contributions of the user and the chatbot to the conversation.

```
In [7]: resp1.chat_history
Out[7]: 
[ChatMessage(role='USER', message='Who is the president of USA?'),
 ChatMessage(role='CHATBOT', message='As of January 2024, the current president of the United States of America is Joe Biden. He was sworn in as the 46th president on January 20, 2021, following his victory in the 2020 presidential election.')]
```

The chat history is a list of messages. Note the roles of USER and CHATBOT. We will come back to the roles below. First, let us continue the chat  taking into account

```
In [8]: resp2 = co.chat(message=prompt2, chat_history=resp1.chat_history)
   ...: resp2.text
Out[8]: 'Joe Biden was born on November 20, 1942, making him 81 years old as of my cut-off date in January 2024.'
```

Now the chat history contains the two prompts and the corresponding responses of the chatbot.

```
In [9]: resp2.chat_history
Out[9]: 
[ChatMessage(role='USER', message='Who is the president of USA?'),
 ChatMessage(role='CHATBOT', message='As of January 2024, the current president of the United States of America is Joe Biden. He was sworn in as the 46th president on January 20, 2021, following his victory in the 2020 presidential election.'),
 ChatMessage(role='USER', message='How old is he/she?'),
 ChatMessage(role='CHATBOT', message='Joe Biden was born on November 20, 1942, making him 81 years old as of my cut-off date in January 2024.')]
 ```

There is a third role, SYSTEM. It is typically used to add content throughout a conversation, or to adjust the model's overall behavior and conversation style. Most of the so called **prompt engineering** is just a way of managing the SYSTEM role. This can be done in many ways, the simplest one being, probably, to set the rules at the begininng of the chat history. The following is a simple example. 

```
In [10]: length = 'Answer the questions in no more than 10 words'
    ...: resp = co.chat(message=prompt1, chat_history=[{'role': 'SYSTEM', 'message': length}])
    ...: resp.text
Out[10]: 'Joe Biden.'
```

Note that the value of the parameter `chat_history` is a Python list, so you can manage it in a Pythonic way. It may be useful to split the rules set by the SYSTEM role, as in the following example.

```
In [11]: length = 'Answer the questions in no more than 10 words'
    ...: style = 'Include middle names'
    ...: resp = co.chat(message=prompt1, chat_history=[{'role': 'SYSTEM', 'message': length}, {'role': 'SYSTEM', 'message': style}])
    ...: resp.text
Out[11]: 'Joseph Robinette Biden Jr.'
```


