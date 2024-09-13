## [ML-25] LLM API tutorial ##

# Chatting through the API #
import cohere
co = cohere.Client('OCAVTT8U76T9EsepDmTjnUTGWdFIrfWP6xaWRHG3')
response = co.chat(message='Tell me, in no more than 25 words, what is machine learning')
response
response.text
co.chat(message='Tell me, in no more than 25 words, what is machine learning').text

# Creating a conversation #
prompt1 = 'Who is the president of USA?' 
prompt2 = 'How old is he/she?'
resp1 = co.chat(message=prompt1)
resp1.text
resp2 = co.chat(message=prompt2, chat_history=resp1.chat_history)
resp2.text
resp2-chat_history

# The roles in the chat #
length = 'Answer the questions in no more than 10 words'
resp = co.chat(message=prompt1, chat_history=[{'role': 'SYSTEM', 'message': length}])
resp.text
length = 'Answer the question in no more than 10 words'
style = 'Include middle names'
resp = co.chat(message=prompt1, chat_history=[{'role': 'SYSTEM', 'message': length}, {'role': 'SYSTEM', 'message': style}])
resp.text

# Embeddings #
model_name = 'embed-english-v3.0'
text1 = '''Machine learning (ML) is a field of study in artificial intelligence concerned 
with the development and study of statistical algorithms that can learn from data and 
generalize to unseen data and thus perform tasks without explicit instructions.'''
text2 = '''Machine learning is a subset of artificial intelligence that gives systems the
ability to learn and optimize processes without having to be consistently programmed.'''
text3 = '''Machine learning (ML) is a branch of artificial intelligence (AI) and computer 
science that focuses on developing methods for computers to learn and improve their performance.'''
response = co.embed(texts=[text1, text2, text3], model=model_name, input_type='search_document')
import numpy as np
embeds = np.array(response.embeddings)
embeds.shape
embeds[:, :5]
abs(embeds[0] - embeds[1]).mean().round(3)
abs(embeds[0] - embeds[2]).mean().round(3)

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
response = co.rerank(query=query, model=model_name, documents=docs, top_n=4)
import pandas as pd
pd.DataFrame({'index': [response[i].index for i in range(len(docs))],
    'relevance_score': [response[i].relevance_score for i in range(len(docs))]})
