## Assignment 3 ##

# Importing the data #
import numpy as np, pandas as pd
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'mnist.csv.zip')

# Target vector and feature matrix #
y = df.iloc[:, 0]
X = df.iloc[:, 1:].values
X = X/255

# Train-test split #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=0)

# Q1. Logistic regression model (1) #
from sklearn.linear_model import LogisticRegression
lgr1 = LogisticRegression(max_iter=2000)
lgr1.fit(X_train, y_train)
round(lgr1.score(X_train, y_train), 3), round(lgr1.score(X_test, y_test), 3)

# Q2. Logistic regression model (2) #
from keras import Input, models, layers
input_tensor = Input(shape=(784,))
output_tensor = layers.Dense(10, activation='softmax')(input_tensor)
lgr2 = models.Model(input_tensor, output_tensor)
lgr2.summary()
lgr2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
lgr2.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));

# Q3. MLP model with two hidden layers #
input_tensor = Input(shape=(784,))
x1 = layers.Dense(128, activation='relu')(input_tensor)
x2 = layers.Dense(32, activation='relu')(x1)
output_tensor = layers.Dense(10, activation='softmax')(x2)
mlp1 = models.Model(input_tensor, output_tensor)
mlp1.summary()
mlp1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
mlp1.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test));

# Q4. MLP model with three hidden layers #
input_tensor = Input(shape=(784,))
x1 = layers.Dense(256, activation='relu')(input_tensor)
x2 = layers.Dense(128, activation='relu')(x1)
x3 = layers.Dense(32, activation='relu')(x2)
output_tensor = layers.Dense(10, activation='softmax')(x3)
mlp2 = models.Model(input_tensor, output_tensor)
mlp2.summary()
mlp2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
mlp2.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test));

# Q5. Alternative CNN model #
X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
input_tensor = Input(shape=(28, 28, 1))
x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
x2 = layers.MaxPooling2D((2, 2))(x1)
x3 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
x4 = layers.MaxPooling2D((2, 2))(x3)
x5 = layers.Flatten()(x4)
x6 = layers.Dense(64, activation='relu')(x5)
output_tensor = layers.Dense(10, activation='softmax')(x6)
cnn = models.Model(input_tensor, output_tensor)
cnn.summary()
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test));

# Q6. XGBoost model #
X_train, X_test = X_train.reshape(60000, 784), X_test.reshape(10000, 784)
import xgboost as xgb
xgb = xgb.XGBClassifier(max_depth=7, n_estimators=200, learning_rate=0.3, random_state=0)
xgb.fit(X_train, y_train)
round(xgb.score(X_train, y_train), 3), round(xgb.score(X_test, y_test), 3)

# Q7. Confusion matrix #
y_pred = lgr1.predict(X_test)
conf = pd.crosstab(y_test, y_pred)
conf
total = conf.sum(axis=1)
right = np.diagonal(conf)
wrong = total - right
percent_wrong = 100*wrong/total
percent_wrong.round(1)
