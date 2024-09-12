## [ML-23] Example - Fake news detection ##

# Importing the data #
import pandas as pd, numpy as np
df = pd.read_csv('/Users/miguel/Dropbox/data/fake.csv')
df.info()
df.head()

# Q1. Cleaning the data #
df = df.drop(columns=['author']).dropna()
df.info()
pd.concat([df['title'].str.len().describe(), df['text'].str.len().describe()], axis=1)
df['title'][df['title'].str.len() < 5]
df['label'].mean()

# Q2. Encoding the titles #
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
title = df['title'].to_list()
title_embed = model.encode(title, output_value='sentence_embedding')
title_embed.shape

# Logistic regression model #
y = df['label']
X = title_embed
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)
round(clf.score(X, y), 3)
y_pred = clf.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

# Distribution of the predictive scores #
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
plt.xlabel('Fake score')
plt.show();

# Encoding the texts #
text = df['text'].tolist()
text_embed = model.encode(text, output_value='sentence_embedding', convert_to_numpy=True)

# Logistic regression model #
X = text_embed
clf = LogisticRegression()
clf.fit(X, y)
round(clf.score(X, y), 3)
y_pred = clf.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

# Distribution of the predictive scores #
df['text_score'] = clf.predict_proba(X)[:, 1]
from matplotlib import pyplot as plt
# Set the size of the figure
plt.figure(figsize=(12,5))
# First subplot
plt.subplot(1, 2, 1)
plt.hist(df['title_score'][y == 1], range=(0,1), color='gray', edgecolor='white')
plt.title('Figure 2.a. Scores (fakes)')
plt.xlabel('Fake score')
# Second subplot
plt.subplot(1, 2, 2)
plt.hist(df['title_score'][y == 0], range=(0,1), color='gray', edgecolor='white')
plt.title('Figure 2.b. Scores (non-fakes)')
plt.xlabel('Fake score');

# MLP model (homework) #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from keras import models, layers
network = [layers.Dense(128, activation='relu'), layers.Dense(2, activation='softmax')]
clf2 = models.Sequential(layers=network)
clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf2.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));

