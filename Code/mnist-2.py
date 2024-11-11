## [ML-19] Example - The MNIST data (2) ##

# Importing the data #
import numpy as np, pandas as pd
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'mnist.csv.zip')
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# Q1. Train-test split #
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1/7, random_state=0)

# Q2. MLP model #
from keras import Input, models, layers
input_tensor = Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)
clf1 = models.Model(input_tensor, output_tensor)
clf1.summary()
clf1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf1.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));

# Q3. Prediction with a MLP model #
clf1.predict(X_test[:1, :])
y_test[0]

# Q4. Rescaling the data #
X = X/255
X_train, X_test = model_selection.train_test_split(X, test_size=1/7, random_state=0)
clf2 = models.Model(input_tensor, output_tensor)
clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf2.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));

# Q5. CNN model #
X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
input_tensor = Input(shape=(28, 28, 1))
x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
x2 = layers.MaxPooling2D((2, 2))(x1)
x3 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
x4 = layers.MaxPooling2D((2, 2))(x3)
x5 = layers.Conv2D(64, (3, 3), activation='relu')(x4)
x6 = layers.Flatten()(x5)
x7 = layers.Dense(64, activation='relu')(x6)
output_tensor = layers.Dense(10, activation='softmax')(x7)
clf3 = models.Model(input_tensor, output_tensor)
clf3.summary()
clf3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf3.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test));
