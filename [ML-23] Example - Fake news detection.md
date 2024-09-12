# [ML-23] Example - Fake news detection

## Introduction

**Social media** is a vast pool of content, and among all the content available for users to access, news is an element that is accessed most frequently. News can be posted by politicians, news channels, newspaper websites, or even common civilians. The posts should be checked for their authenticity, since spreading misinformation has been a real concern in today's times, and many firms are taking steps to make the common people aware of the consequences of spread misinformation. The measure of authenticity of the news posted online cannot be definitively measured, since the manual classification of news is tedious and time-consuming, and is also subject to bias.

In an era where fake WhatsApp forwards and Tweets (now *X* posts) are capable of influencing naive minds, tools and knowledge have to be put to practical use in not only mitigating the spread of misinformation but also to inform people about the type of news they consume. Development of practical applications for users to gain insight from those news, fact-checking websites, built-in plugins and article parsers can further be refined, made easier to access, and more importantly, should create more awareness.

Several data sets have been released for training and benchmarking **fake news detection** models. This example uses data released in 2018 by William Lifferth for a Kaggle competition. The news included are from the **Trump vs Clinton** times, so the models obtained can only be expected to work in that context.

## The data set

The file `fake.csv` has 20,800 news articles. The columns are:

* `id`, unique identifier for the article. Just a counter.

* `title`, the title of the article, with some missing values.

* `author`, the author of the article, with some missing values.

* `text`, the text of the article, with some missing values. The text could be incomplete.

* `label`, a label that marks the article as potentially unreliable.

Source of the data: William Lifferth (2018), *Fake News Kaggle*, `https://kaggle.com/competitions/fake-news`.

## Questions

Q1. Clean the data, dropping the author (so your model will not have suspected authors) and the articles with missing title or text.

Q2. Encode the titles using a **text embedding model**. Pack the embedding vectors in a matrix so every row corresponds to one article.

Q3. Using the this matrix as the feature matrix, train a **logistic regression model** for fake news detection.

Q4. Take a look at the distribution of the **predictive scores**.

## Importing the data

We import the data as a Pandas data frame from the usual GitHub repository.

```
In [1]: import pandas as pd, numpy as np
   ...: df = pd.read_csv('/home/cinn/Dropbox/data/fake.csv')
```

We have data on 20,800 articles, but only the labels are complete. The articles with missing title must discarded for this example.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20800 entries, 0 to 20799
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   id      20800 non-null  int64 
 1   title   20242 non-null  object
 2   author  18843 non-null  object
 3   text    20761 non-null  object
 4   label   20800 non-null  int64 
dtypes: int64(2), object(3)
memory usage: 812.6+ KB
```

This is how the data look like.

```
In [3]: df.head()
Out[3]: 
   id                                              title  ...                                               text label
0   0  House Dem Aide: We Didn’t Even See Comey’s Let...  ...  House Dem Aide: We Didn’t Even See Comey’s Let...     1
1   1  FLYNN: Hillary Clinton, Big Woman on Campus - ...  ...  Ever get the feeling your life circles the rou...     0
2   2                  Why the Truth Might Get You Fired  ...  Why the Truth Might Get You Fired October 29, ...     1
3   3  15 Civilians Killed In Single US Airstrike Hav...  ...  Videos 15 Civilians Killed In Single US Airstr...     1
4   4  Iranian woman jailed for fictional unpublished...  ...  Print \nAn Iranian woman has been sentenced to...     1

[5 rows x 5 columns]
```

## Q1. Cleaning the data

We clean the data set as suggested.

```
In [4]: df = df.drop(columns=['author']).dropna()
   ...: df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 20203 entries, 0 to 20799
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   id      20203 non-null  int64 
 1   title   20203 non-null  object
 2   text    20203 non-null  object
 3   label   20203 non-null  int64 
dtypes: int64(2), object(2)
memory usage: 789.2+ KB
```

With the method `.describe()`, we create a summary for the length of titles and texts. This shows that some of the articles should be filtered out if the detection were to be based on the text (homework). 

```
In [5]: pd.concat([df['title'].str.len().describe(), df['text'].str.len().describe()], axis=1)
Out[5]: 
              title           text
count  20203.000000   20203.000000
mean      74.275603    4668.044251
std       23.135718    5151.439764
min        3.000000       1.000000
25%       60.000000    1747.000000
50%       75.000000    3495.000000
75%       88.000000    6364.000000
max      456.000000  142961.000000
```

The short titles look right so far.

```
In [6]: df['title'][df['title'].str.len() < 5]
Out[6]: 
2561     Wow!
15728     Rum
17081    EPIC
Name: title, dtype: object
```

48.6% of the articles come laleled as fake, so the data set is quite balanced. It makes sense to use the **accuracy** to evaluate the model.

```
In [7]: df['label'].mean().round(3)
Out[7]: 0.486
```

## Q2. Encoding the titles

There is plenty of choice of embedding models among the current wave of **large language models** (LLM's). This will be discussed later. For this example, we pick our model from the **Hugging Face** hub, using the package `sentence_transformers`. You can install it with `pip install sentence_transformers`. It depends on another package, called `transformers`. To avoid a conflict of versions with Keras, use `pip install transformers==4.38`.

Once the installation is ready we import the class `SentenceTransformer`.

```
In [8]: from sentence_transformers import SentenceTransformer
```

This class loads a model that can be used to map texts to embedding vectors. As we are using it, it loads a model called `all-mpnet-base-v2`, extracted from Hugging Face. The first time that run this command, the model will be downloaded from Hugging Face and placed in a hidden folder called `.cache` which is in Python's working directory. You don't to worry about this, because the process runs by itself. Note that we are going to use a **local model**, installed in your computer. In a later example, we will use a **remote model**.

To run a local LLM, it has to be small one. Hugging Face is a good place for searching. We have chosen for this example `all-mpnet-base-v2`, following the recommendation of an Internet source. This is the current version of a model called **MPNet** introduced in 2020.

Instantiating a model from this class is similar to what we did before in scikit-learn and Keras.

```
In [9]: model = SentenceTransformer('all-mpnet-base-v2')
```

Before using the model, we convert the Pandas series to a list (this is just a technicality).

```
In [10]: title = df['title'].to_list()
```

Now the embeddings can be generated with the method `.encode()` (this may take a few minutes). With the argument `output_value='sentence_embedding'`, we ask the model to take the sentence as a unit, instead of spliting it in words and generating one vector for each word. The model outputs (default) a NumPy array.

The embedding matrix has the expected number of rows. The number of columns is 768, which is a legacy of older models (apparently GPT-2 and other ancestors were using size 768 for various reasons), but this number has nothing special.

```
In [11]: title_embed = model.encode(title, output_value='sentence_embedding')
    ...: title_embed.shape
Out[11]: (20203, 768)
```

## Logistic regression model

We specify the target vector and the feature matrix as suggested.

```
In [12]: y = df['label']
    ...: X = title_embed
```

We initialize an estimator from the scikit-learn class `LogisticRegression`. The defaut number of iterations will be enough, because the embedding vectors are normalized (length one).

```
In [13]: from sklearn.linear_model import LogisticRegression
    ...: clf = LogisticRegression()
```

We apply the methods `.fit()` and `.score()`. Nothing new here. The accuracy is 92.1%, which is quite satisfactory, though we cannot take this out of context.

```
In [14]: clf.fit(X, y)
    ...: round(clf.score(X, y), 3)
Out[14]: 0.921
```

The predicted class can be calculated as usual.

```
In [15]: y_pred = clf.predict(X)
```

Now, the **confusion matrix** shows that the model gives more false positives than false negatives.

```
In [16]: from sklearn.metrics import confusion_matrix
    ...: confusion_matrix(y, y_pred)
Out[16]: 
array([[9443,  944],
       [ 644, 9172]])
```

## Distribution of the predictive scores

This section does not contain anything new from a methodological point of view. We already know how to extract the predictive scores with the method `.predict_proba()`:

```
In [17]: df['title_score'] = clf.predict_proba(X)[:, 1]
```

Now, we plot separate histograms for the predictive scores, for fake and non-fake news. This may give you an idea of how you can reduce the lase positive rate by setting a threshold for the predictive scores. 

```
In [18]: from matplotlib import pyplot as plt
    ...: # Set the size of the figure
    ...: plt.figure(figsize=(12,5))
    ...: # First subplot
    ...: plt.subplot(1, 2, 1)
    ...: plt.hist(df['title_score'][y == 1], range=(0,1), color='gray', edgecolor='white')
    ...: plt.title('Figure 1.a. Scores (fakes)')
    ...: plt.xlabel('Fake score')
    ...: # Second subplot
    ...: plt.subplot(1, 2, 2)
    ...: plt.hist(df['title_score'][y == 0], range=(0,1), color='gray', edgecolor='white')
    ...: plt.title('Figure 1.b. Scores (non-fakes)')
    ...: plt.xlabel('Fake score');
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/23-1.png)


## Homework

1. Use a train/test split to examine whether we have overfitted the title data with our logistic regression model. 

2. Replicate the analysis presented in this example, but using the text instead of the title of the article. Do you get better results than with the titles?

3. Train an MLP model, with one hidden layer, to the title data. Do you get better results than with the logistic regression model?

3. **Nomic Embed** is a text embedding which has recently entered the competition, being highyly praised. You can get it directly with the package `nomic`, or from Hugging Face, through the package `sentence_transformers` used in this example. To try it with the title data, you just have to replace the model definition in `In[9]` by the following. You may be requested to install the package `einops`, which you can do in the usual way, with `pip install einops`.
```
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
```
