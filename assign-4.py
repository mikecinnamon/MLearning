## Assignment 4 ##

# Importing the data #
import pandas as pd, numpy as np
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df1 = pd.read_csv(path + 'fake1.csv.zip', index_col=0)
df2 = pd.read_csv(path + 'fake2.csv.zip', index_col=0)
df = pd.concat([df1, df2], axis=0)
df = df.drop(columns=['author']).dropna()

# Encoding the titles #
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L12-v2')
titles = df['title'].tolist()
embeds = model.encode(titles)

# Q1. Validation #
y = df['label']
X = embeds
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()
clf1.fit(X_train, y_train)
round(clf1.score(X_train, y_train), 3), round(clf1.score(X_test, y_test), 3)

# Q2. MLP classifier #
from keras import Input, models, layers
input_tensor = Input(shape=(384,))
x = layers.Dense(64, activation='relu')(input_tensor)
output_tensor = layers.Dense(2, activation='softmax')(x)
clf2 = models.Model(input_tensor, output_tensor)
clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf2.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));

# Q3. MPNet encoder #
model = SentenceTransformer('all-mpnet-base-v2')
embeds = model.encode(titles)
X = embeds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf1 = LogisticRegression()
clf1.fit(X_train, y_train)
round(clf1.score(X_train, y_train), 3), round(clf1.score(X_test, y_test), 3)
input_tensor = Input(shape=(768,))
x = layers.Dense(64, activation='relu')(input_tensor)
output_tensor = layers.Dense(2, activation='softmax')(x)
clf2 = models.Model(input_tensor, output_tensor)
clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf2.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));
