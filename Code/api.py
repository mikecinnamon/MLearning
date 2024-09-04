## [ML-25] LLM API tutorial ##

# Chatting through the API #
import cohere
co = cohere.Client('OCAVTT8U76T9EsepDmTjnUTGWdFIrfWP6xaWRHG3')
response = co.chat(message='Tell me, in no more than 25 words, what is machine learning')
response
response.text

prompt1 = 'Who is the president of USA?' 
prompt2 = 'How old is he/she?'
resp1 = co.chat(message=prompt1)
resp1.text
resp2 = co.chat(message=prompt2, chat_history=resp1.chat_history)
resp2.text
resp2-chat_history

length = 'Answer the questions in no more than 10 words'
resp = co.chat(message=prompt1, chat_history=[{'role': 'SYSTEM', 'message': length}])
resp.text

length = 'Answer the question in no more than 10 words'
style = 'Include middle names'
resp = co.chat(message=prompt1, chat_history=[{'role': 'SYSTEM', 'message': length}, {'role': 'SYSTEM', 'message': style}])
resp.text



document = """Equipment rental in North America is predicted to “normalize” going into 2024,
according to Josh Nickell, vice president of equipment rental for the American Rental
Association (ARA).
“Rental is going back to ‘normal,’ but normal means that strategy matters again -
geography matters, fleet mix matters, customer type matters,” Nickell said. “In
late 2020 to 2022, you just showed up with equipment and you made money.
“Everybody was breaking records, from the national rental chains to the smallest
rental companies; everybody was having record years, and everybody was raising
prices. The conversation was, ‘How much are you up?’ And now, the conversation
is changing to ‘What’s my market like?’”
Nickell stressed this shouldn’t be taken as a pessimistic viewpoint. It’s simply
coming back down to Earth from unprecedented circumstances during the time of Covid.
Rental companies are still seeing growth, but at a more moderate level."""



response = co.chat(message= f"Generate a concise summary of this text\n{document}").text


print(response)
