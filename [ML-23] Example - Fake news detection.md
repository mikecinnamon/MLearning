# [ML-23] Example - Fake news detection

## Introduction

**Social media** is a vast pool of content, and among all the content available for users to access, news is an element that is accessed most frequently. News can be posted by politicians, news channels, newspaper websites, or even common civilians. The posts should be checked for their authenticity, since spreading misinformation has been a real concern in today's times, and many firms are taking steps to make the common people aware of the consequences of spread misinformation. The measure of authenticity of the news posted online cannot be definitively measured, since the manual classification of news is tedious and time-consuming, and also subject to bias.

In an era where fake WhatsApp forwards and tweets (now *X* posts) are capable of influencing naive minds, tools and knowledge have to be put to practical use in not only mitigating the spread of misinformation but also to inform people about the type of news they consume. Fact-checking websites, built-in plugins and article parsers should be refined, made easier to access and, more importantly, there should be more awareness about this question.

Several data sets have been released for training and benchmarking **fake news detection** models. This example uses data released in 2018 by William Lifferth, for a Kaggle competition. The news included are from the **Trump vs Clinton** times, so the models obtained must not be expected to work in other contexts.

## The data set

The data set, containing 20,800 news articles, has been split in two parts, which come in the files `fake1.csv` and `fake2.csv` (zipped). The columns are:

* `id`, unique identifier for the article. Just a counter.

* `title`, the title of the article, with some missing values.

* `author`, the author of the article, with some missing values.

* `text`, the text of the article, with some missing values. The text could be incomplete.

* `label`, a label that marks the article as potentially unreliable.

Source of the data: William Lifferth (2018), *Fake News Kaggle*, `https://kaggle.com/competitions/fake-news`.

## Questions

Q1. Clean the data, dropping the author (our model will not use this) and the articles with missing title or text.

Q2. Encode the titles using a **text embedding model**. Pack the embedding vectors in a matrix so every row corresponds to one article.

Q3. Using this matrix as the features matrix, train a **logistic regression model** for fake news detection.

Q4. Take a look at the distribution of the **predictive scores**.

## Importing the data

We import the data, as two Pandas data frames, from the usual GitHub repository. Then we concatenate the two parts to get a single data frame.

```
In [1]: import pandas as pd, numpy as np
   ...: path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
   ...: df1 = pd.read_csv(path + 'fake1.csv.zip', index_col=0)
   ...: df2 = pd.read_csv(path + 'fake2.csv.zip', index_col=0)
   ...: df = pd.concat([df1, df2], axis=0)
```

Ab shown in the report below, we have data on 20,800 articles, but only the labels are complete. The articles with missing title must discarded for this example.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 20800 entries, 0 to 20799
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   title   20242 non-null  object
 1   author  18843 non-null  object
 2   text    20761 non-null  object
 3   label   20800 non-null  int64 
dtypes: int64(1), object(3)
memory usage: 812.5+ KB
```

This is how the data look like.

```
In [3]: df.head()
Out[3]: 
                                                title              author   
id                                                                          
0   House Dem Aide: We Didn’t Even See Comey’s Let...       Darrell Lucus  \
1   FLYNN: Hillary Clinton, Big Woman on Campus - ...     Daniel J. Flynn   
2                   Why the Truth Might Get You Fired  Consortiumnews.com   
3   15 Civilians Killed In Single US Airstrike Hav...     Jessica Purkiss   
4   Iranian woman jailed for fictional unpublished...      Howard Portnoy   

                                                 text  label  
id                                                            
0   House Dem Aide: We Didn’t Even See Comey’s Let...      1  
1   Ever get the feeling your life circles the rou...      0  
2   Why the Truth Might Get You Fired October 29, ...      1  
3   Videos 15 Civilians Killed In Single US Airstr...      1  
4   Print \nAn Iranian woman has been sentenced to...      1  
```

## Q1. Cleaning the data

We clean the data set as suggested.

```
In [4]: df = df.drop(columns=['author']).dropna()
   ...: df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 20203 entries, 0 to 20799
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   title   20203 non-null  object
 1   text    20203 non-null  object
 2   label   20203 non-null  int64 
dtypes: int64(1), object(2)
memory usage: 631.3+ KB
```

With the method `.describe()`, we create a summary for the length (number of characters) of titles and texts. This shows that some of the articles should be filtered out if the detection were to be based on the text (we leave that for the homework). 

```
In [5]: pd.concat([df['title'].str.len(), df['text'].str.len()], axis=1).describe()
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

Some very short titles may be expected to be deficient, but we don't find them so. So, we keep them.
```
In [6]: df['title'][df['title'].str.len() < 5]
Out[6]: 
2561     Wow!
15728     Rum
17081    EPIC
Name: title, dtype: object
```

48.6% of the articles come labeled as fake, so the data set is quite balanced. It makes sense to use the **accuracy** to evaluate the model.

```
In [7]: df['label'].mean().round(3)
Out[7]: 0.486
```

## Q2. Encoding the titles

In this example, we use a **local model**, running in our computer. In example ML-26, we will use a **remote model**. There is plenty of choice of pre-trained embedding models within the crowd of **large language models** (LLM's) currently popping up. For this example, we pick our model from the **Hugging Face** hub, using the package `sentence_transformers`, which you can install it with `pip install sentence_transformers`. It depends on another package, called `transformers`, which facilitates a model posted on Hugging face to run locally (if your computer has enough RAM memory). 

Once the installation is ready, we import the class `SentenceTransformer`.

```
In [8]: from sentence_transformers import SentenceTransformer
```

This class will create an instance of the model specified. The first time that you run a model in your computer, that model will be downloaded from Hugging Face and saved in a hidden folder called `.cache` which is in Python's working directory. You don't have to worry about this, because the process runs by itself.

We have chosen for this example the model `all-MiniLM-L12-v2`, a dwarf among the current LLM's, which generates short embedding vectors. It will run  fast in your computer, so it is adequate for a first experience. More information can be found in `https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2`. The model is downloaded to your computer the first time that you use it. We have omitted here the report printed on the screen.

```
In [9]: model = SentenceTransformer('all-MiniLM-L12-v2')
```

The input text has to be supplied in a list in `sentence_transformers`, so we convert the Pandas series to a list.

```
In [10]: titles = df['title'].tolist()
```

The embeddings can be generated with the method `.encode()`. The default output is a NumPy array, so we can check the shape directly.

```
In [11]: embeds = model.encode(titles)
    ...: embeds.shape
Out[11]: (20203, 384)
```

So, every row is the embedding vector for the corresponding title. The embedding dimension is 384, one half of 768, which is a legacy of older models (apparently GPT-2 and other ancestors were using size 768 for various reasons), but this number has nothing special. The default encoder produces normalized embedding vectors (unit length). 

## Q3. Logistic regression model

We specify the target vector and the feature matrix as suggested. We maintain the notation for consistency with previous examples.

```
In [12]: y = df['label']
    ...: X = embeds
```

We initialize an estimator from the scikit-learn class `LogisticRegression`. The defaut number of iterations will be enough, because the embedding vectors are normalized.

```
In [13]: from sklearn.linear_model import LogisticRegression
    ...: clf = LogisticRegression()
```

We apply the methods `.fit()` and `.score()` as usual. Nothing new here. The accuracy is 90.5%, which is quite satisfactory, though we cannot take this out of context. Nevertheless, this figure can be improved with better encoders that use higher embedding dimensions (the encoding process will be slower then). There are some suggestions in the homework.

```
In [14]: clf.fit(X, y)
    ...: round(clf.score(X, y), 3)
Out[14]: 0.905
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
array([[9227, 1160],
       [ 762, 9054]])
```

## Q4. Distribution of the predictive scores

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

1. Use a train/test split to validate our logistic regression model. 

2. Train an MLP model, with one hidden layer, to the title data. Do you get better results than with the logistic regression model?

3. The embedding model suggested in the `sentence_transformers` website and many other sources) is `all-mpnet-base-v2`. This is the current version of a model called **MPNet** published in 2020 by a Microsoft research team. Try it with the title data. The embedding dimension is 768, so you can expect to improve the current results, and to face a slower encoding process. Also, this model will be downloaded to your computer the first time that you use it, as it happened with the encoder used in the example.
