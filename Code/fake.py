## [ML-23] Example - Fake news detection ##

# Importing the data #
import pandas as pd, numpy as np
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df1 = pd.read_csv(path + 'fake1.csv.zip', index_col=0)
df2 = pd.read_csv(path + 'fake2.csv.zip', index_col=0)
df = pd.concat([df1, df2], axis=0)
df.info()
df.head()

# Q1. Cleaning the data #
df = df.drop(columns=['author']).dropna()
df.info()
pd.concat([df['title'].str.len(), df['text'].str.len()], axis=1).describe()
df['title'][df['title'].str.len() < 5]
df['label'].mean().round(3)

# Q2. Encoding the titles #
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L12-v2')
titles = df['title'].tolist()
embeds = model.encode(titles)
embeds.shape

# Q3. Logistic regression model #
y = df['label']
X = embeds
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)
round(clf.score(X, y), 3)
y_pred = clf.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

# Q4. Distribution of the predictive scores #
df['title_score'] = clf.predict_proba(X)[:, 1]
from matplotlib import pyplot as plt
# Set the size of the figure
plt.figure(figsize=(12,5))
# First subplot
plt.subplot(1, 2, 1)
plt.hist(df['title_score'][y == 1], range=(0,1), color='gray', edgecolor='white')
plt.title('Figure 1.a. Scores (fakes)')
plt.xlabel('Fake score')
# Second subplot
plt.subplot(1, 2, 2)
plt.hist(df['title_score'][y == 0], range=(0,1), color='gray', edgecolor='white')
plt.title('Figure 1.b. Scores (non-fakes)')
plt.xlabel('Fake score');
