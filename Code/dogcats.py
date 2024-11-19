## [ML-21] Example - The dogs vs cats data set ##

# Q1a. Creating a data folder #
import os
os.mkdir('data/')

# Q1b. Dowloading the zip files #
import requests
gitpath = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
gitlist = ['cats-train.zip', 'cats-test.zip', 'dogs-train.zip', 'dogs-test.zip']
for f in gitlist:
	r = requests.get(gitpath + f, stream=True)
	conn = open('data/' + f, mode='wb')
	conn.write(r.content)
	conn.close()

# Q1c. Unzipping and removing the zip files #
import zipfile
ziplist = [f for f in os.listdir('data/') if 'zip' in f]
for f in ziplist:
	zf = zipfile.ZipFile('data/' + f, 'r')
	zf.extractall('data/')
	del zf
	os.remove('data/' + f)
os.listdir('data/')
len(os.listdir('data/dogs-train/'))

# Q2a. Converting images to tensors (pip install opencv-python) #
import numpy as np, cv2
def img_to_arr(f):
    arr = cv2.imread(f)
    resized_arr = cv2.resize(arr, (150, 150), interpolation=cv2.INTER_LANCZOS4)
    reshaped_arr = resized_arr.reshape(1, 150, 150, 3)
    return reshaped_arr

# Q2b. Training data #
X_train = img_to_arr('data/dogs-train/' + os.listdir('data/dogs-train')[0])
for i in range(1, 1000):
    X_train = np.concatenate([X_train, img_to_arr('data/dogs-train/' + os.listdir('data/dogs-train')[i])])
for i in range(1000):
    X_train = np.concatenate([X_train, img_to_arr('data/cats-train/' + os.listdir('data/cats-train')[i])])
X_train = X_train/255
y_train = np.concatenate([np.ones(1000), np.zeros(1000)])
X_train.shape, y_train.shape

# Q2c. Test data #
X_test = img_to_arr('data/dogs-test/' + os.listdir('data/dogs-test')[0])
for i in range(1, 500):
    X_test = np.concatenate([X_test, img_to_arr('data/dogs-test/' + os.listdir('data/dogs-test')[i])])
for i in range(500):
    X_test = np.concatenate([X_test, img_to_arr('data/cats-test/' + os.listdir('data/cats-test')[i])])
X_test = X_test/255
y_test = np.concatenate([np.ones(500), np.zeros(500)])
X_test.shape, y_test.shape

# Q3. Training a CNN model from scratch #
from keras import Input, models, layers
input_tensor = Input(shape=(150, 150, 3))
x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
x2 = layers.MaxPooling2D((2, 2))(x1)
x3 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
x4 = layers.MaxPooling2D((2, 2))(x3)
x5 = layers.Conv2D(128, (3, 3), activation='relu')(x4)
x6 = layers.MaxPooling2D((2, 2))(x5)
x7 = layers.Conv2D(128, (3, 3), activation='relu')(x6)
x8 = layers.MaxPooling2D((2, 2))(x7)
x9 = layers.Flatten()(x8)
x10 = layers.Dense(512, activation='relu')(x9)
output_tensor = layers.Dense(2, activation='softmax')(x10)
clf1 = models.Model(input_tensor, output_tensor)
clf1.summary()
clf1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf1.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test));

# Q4a. Pre-trained CNN model #
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()
conv_base.trainable = False

# Q4b. Adding a densely connected classifier on top #
input_tensor = Input(shape=(150, 150, 3))
x1 = conv_base(input_tensor)
x2 = layers.Flatten()(x1)
x3 = layers.Dense(256, activation='relu')(x2)
output_tensor = layers.Dense(2, activation='softmax')(x3)
clf2 = models.Model(input_tensor, output_tensor)
clf2.summary()

# Q5. Training the new model #
clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf2.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test));

# Removing the data (optional) #
for d in os.listdir('data/'):
    for f in os.listdir('data/' + d):
        os.remove('data/' + d + '/' + f)
    os.rmdir('data/' + d)
os.rmdir('data')
