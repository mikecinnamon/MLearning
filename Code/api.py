## [ML-25] LLM API tutorial ##

# Chatting through the API #
import cohere
co = cohere.ClientV2('bfxqhSNJ5OaBPsjXvXLbzpCJjoS8EuZrN05QT9qj')
model_name = 'command-r-plus-08-2024'
messages = [{'role': 'user', 'content': 'Tell me, in no more than 25 words, what is machine learning'}]
response = co.chat(model=model_name, messages=messages)
response
response.message.content[0].text
def mychat(messages):
    return co.chat(model_name, messages=messages).message.content[0].text

# Creating a conversation #
query1 = 'Who is the president of USA?'
mes1 = [{'role': 'user', 'content': query}]
resp1 = mychat(mes1)
resp1
query2 = 'How old is he/she?'
mes2 = mes1 + [{'role': 'assistant', 'content': resp1}] + [{'role': 'user', 'content': query2}]
mes2
resp2 = mychat(mes2)
resp2

# The roles in the chat #
query = 'Who is the president of USA?'
length = 'Respond the following question in no more than 10 words'
mes1 = [{'role': 'system', 'content': length}, {'role': 'user', 'content': query}]
resp1 = mychat(mes1)
resp1
style = 'Include middle names in your response'
mes2 = [{'role': 'system', 'content': length}, {'role': 'system', 'content': style}, {'role': 'user', 'content': query}]
resp2 = mychat(mes2)
resp2

# Embeddings #
model_name = 'embed-english-v3.0'
text1 = '''Machine learning (ML) is a field of study in artificial intelligence concerned 
    with the development and study of statistical algorithms that can learn from data and 
    generalize to unseen data and thus perform tasks without explicit instructions.'''
text2 = '''Machine learning is a subset of artificial intelligence that gives systems the
    ability to learn and optimize processes without having to be consistently programmed.'''
text3 = '''This course on machine learning is very interesting.'''
texts=[text1, text2, text3]
response = co.embed(model=model_name, input_type='search_document', embedding_types=['float'], texts=texts)
import numpy as np
embeds = np.array(response.embeddings.float)
embeds.shape
embeds[:, :5]
(embeds**2).sum(axis=1)
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(embeds).round(3)

# Reranking #
model_name = 'rerank-english-v3.0'
query = 'Are there fitness-related perks?'
doc1 = '''Reimbursing Travel Expenses: Easily manage your travel expenses by submitting them
through our finance tool. Approvals are prompt and straightforward.'''
doc2 = '''Working from Abroad: Working remotely from another country is possible.
Simply coordinate with your manager and ensure your availability during core hours.'''
doc3 = '''Health and Wellness Benefits: We care about your well-being and offer gym
memberships, on-site yoga classes, and comprehensive health insurance.'''
doc4 = '''Performance Reviews Frequency: We conduct informal check-ins every quarter
and formal performance reviews twice a year.'''
docs = [doc1, doc2, doc3, doc4]
response = co.rerank(model=model_name, query=query, documents=docs, top_n=4)
response
import pandas as pd
pd.DataFrame({'index': [response.results[i].index for i in range(len(docs))],
    'relevance_score': [response.results[i].relevance_score for i in range(len(docs))]})
