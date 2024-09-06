# [ML-21] Example - The dogs vs cats data set

## Introduction

Web services are often protected with a challenge that is supposed to be easy for people to solve, but difficult for computers. Such a challenge is often called a **CAPTCHA** (Completely Automated Public Turing test to tell Computers and Humans Apart) or **HIP** (Human Interactive Proof). HIPs are used for many purposes, such as to reduce email and blog spam and prevent brute force attacks on web site passwords.

**Asirra** (Animal Species Image Recognition for Restricting Access) is an HIP that works by asking users to identify photographs of **cats** and **dogs**. This task is difficult for computers, but studies have shown that people can accomplish it quickly and accurately. Asirra is unique because of its partnership with **Petfinder.com**, the world's largest site devoted to finding homes for homeless pets. They have provided Microsoft Research with over three million images of cats and dogs, manually classified by people at thousands of animal shelters across the United States.

This example uses part of this data set, released for the *Dogs vs. Cats* Kaggle competition. It is inspired by the approach to transfer learning taken in chapter 9 of F Chollet's book. 

## The data set

The *Dogs vs. Cats* data set is available from many sources, including Kaggle and Keras. It contains 25,000 images of dogs and cats (12,500 from each class). The pictures are medium-resolution color JPEG files. In this example, we use 2,000 images for training and 1,000 for testing. Both data sets are balanced.

The data come as a four (zipped) folders of JPEG files, `dogs-train`, `dogs-test`, `cats-train` and `cats-test`. As we have previously mentioned in lecture ML-18, every picture is just an array with two spatial axes, called **height** and **width**, and a **depth** axis. For a RGB image, the dimension of the depth axis would be 3, since the image has 3 color channels, red, green, and blue. The height and width depend on the resolution of the images, which is not fixed here. So, before being converted to NumPy arrays that can be inputted to a neural network, all the images must be resized to a common resolution.

Sources:

1. W Cukierski (2013) *Dogs vs. Cats*, `https://kaggle.com/competitions/dogs-vs-cats`.

2. F Chollet (2021), *Deep Learning with Python*, Manning.

## Questions

Q1. CReate a folder in the working directory of your current Python kernel and download there the ZIP files. Unzip these files, so you will the four folders mentioned above. The folders with training data contain contain 1,000 JPG files each, while those with test data contain 500 files each. 

Q2. Write a function which reads the JPG files as NumPy arrays, and converts the arrays to shape `(1, 150, 150, 3)`. You can do this with the function `resize` of the package `opencv`. Apply your function to all the JPG files and concatenate the resulting arrays in such a way that you obtain an array `X_train` of shape `(2000, 150, 150, 3)` and an array `X_test` with shape `(1000, 150, 150, 3)`. Create the corresponding target vectors, with value 1 for the dogs and value 0 for the cats.

Q3. Train a CNN model from scratch and evaluate it. Build the network stacking four pairs of `Conv2D` and `MaxPooling2D` layers, a `Flatten` layer and two `Dense` layers. The last layer must output the two class probabilities.

Q4. Import the pre-trained model VGG16 from the Keras module `applications`. Include the convolutional base but not the top of the network. Then, freeze the parameter values of the convolutional base and add a densely connected classifier on top of the pre-trained model.

Q5. Train the new model and compare its performance on the test data with that of the model of question Q3.

## Q1a. Creating a data folder

The package `os` (included in the Python Standard Library) contains a collection of functions for common **operating system commands**. You can create, delete and copy files and folders from the Python kernel. We import it as

```
In [1]: import os
``` 

The function `mkdir()` creates a folder, that will be appear in ther working directory of the current kernel, unless you specify a specific path.

```
In [2]: os.mkdir('data/')
``` 

*Note*. The slash (`/`) at the end of the folder name is not needed, but may help to distinguish between files and folders.

## Q1b. Dowloading the zip files

The package `requests`, included in the Anaconda distribution, provides **HTTP functionality**. We import them as:

```
In [3]: import requests
```

We specify the remote path to the folder where the data files are. 

```
In [4]: gitpath = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
```

To select the files to be downloaded, we create a list. 

```
In [5]: gitlist = ['cats-train.zip', 'cats-test.zip', 'dogs-train.zip', 'dogs-test.zip']
```

We create now a loop for downloading the four ZIP files to the new folder. The resources involved are:

* The `requests` function `get()` sends a **GET request** to the remote server. If the response is positive, the files specified are read by the Python kernel. The argument `stream=True` is used for efficiency, but it probably does not make a difference here. 

* The Python function `open()` creates a **file object** (whose name plays no role). Since the files didn't exist in the data folder, new (empty) files are created. Then, the method `.write()` writes the content of the ZIP file sent by remote server to the new file. The argument `mode='wb'` means *write in binary mode*.

```
In [6]: for f in gitlist:
   ...: 	r = requests.get(gitpath + f, stream=True)
   ...: 	conn = open('data/' + f, mode='wb')
   ...: 	conn.write(r.content)
   ...: 	conn.close()
```

## Q1c. Unzipping and removing the zip files

The package `zipfile` (also in the Python Standard Library) provides resources for zipping and unizipping files and folders in a simple way. We import it as:

```
In [7]: import zipfile
```

To unizp the four files in a row, we create first a list of the files to unzip. Note that the `os` function `listdir()` lists both files and folders, so we have to exclude the folders from the list.

```
In [8]: ziplist = [f for f in os.listdir('data/') if 'zip' in f]
```

Now, we loop over this list unzipping and removing the ZIP files one by one. The resources involved are:

* The `zipfile` function `Zipfile()` creates a **ZipFile object** associated to the specified file.

* The method `extractall()` extracts the content of the ZIP file and writes it to disk.

* The Python keyword `del` is used to delete objects (from the Python kernel, not from the disk).

* The `os` function `remove()` is used to remove files from disk.

```
In [9]: for f in ziplist:
   ...: 	zf = zipfile.ZipFile('data/' + f, 'r')
   ...: 	zf.extractall('data/')
   ...: 	del zf
   ...: 	os.remove('data/' + f)
```

Now the folder `data` contains four folders, with the appropriate names.

```
In [10]: os.listdir('data/')
Out[10]: ['dogs-train', 'cats-train', 'dogs-test', 'cats-test']
```

We can also check that these folders contain the right number of files.

```
In [11]: len(os.listdir('data/dogs-train/'))
Out[11]: 1000
```

## Q2a. Converting images to tensors

we are going to use now the package `opencv`. You can install it with `pip install opencv-python`. We import it as `cv2`.

```
In [12]: import numpy as np, cv2
```

We convert the JPG files to NumPy arrays. These arrays must have the same shape so they can be packed in the training and test feature arrays and processed by a neural network model. We write a function to loop over the folders jusr created in question Q1. The resources involved are:

* The `opencv` function `imread()` converts a JPEG file to a NumPy array. This is a classic Matplotlib function incorporated by many packages. It works the same for other image formats (such as BMP or PNG).

* The `opencv` function `resize()` changes the shape of the resulting array in a way that is equivalent to changing the resolution of a picture. Note that resizing is not the same as reshaping: when reshaping, the terms of the array do not change, they are indexed in a different way. Here we create the new array by interpolating in the original array. The resolution 150 $\times$ 150, which is arbitrary, is taken from the book.

* Once every picture has been transformed into a 3D array of shape `(150, 150, 3)`, we reshape the resulting array to `(1, 150, 150, 3)`. Remember that, in example ML-20, the input of the convolutional network was a 4D tensor.

```
In [13]: def img_to_arr(f):
    ...:     arr = cv2.imread(f)
    ...:     resized_arr = cv2.resize(arr, (150, 150), interpolation=cv2.INTER_LANCZOS4)
    ...:     reshaped_arr = resized_arr.reshape(1, 150, 150, 3)
    ...:     return reshaped_arr
```

## Q2b. Training data

The training data will be made of two arrays: (a) the features, packed in an array `X_train` of shape `(2000, 150, 150, 3)`, and (b) the target `y_train`, which will be a 1D array with 1's (dogs) and 0's (cats). We create `X_train` using the first dog picture, so it has shape `(1, 150, 150, 3)`. 

```
In [14]: X_train = img_to_arr('data/dogs-train/' + os.listdir('data/dogs-train')[0])
```

Then, we loop over the folder `dogs-train', adding dogs one by one with NumPy function `concatenate()`. By default, the concatenation is carried out along the first axis (`axis=0`).

```
In [15]: for i in range(1, 1000):
    ...:     X_train = np.concatenate([X_train, img_to_arr('data/dogs-train/' + os.listdir('data/dogs-train')[i])])
```

Now, the cats of the training set.

```
In [16]: for i in range(1000):
    ...:     X_train = np.concatenate([X_train, img_to_arr('data/cats-train/' + os.listdir('data/cats-train')[i])])
```

Finally, we rescale the pixel intensities to the 0-1 range.
```
In [17]: X_train = X_train/255
```

The NumPy functions `ones()` and `zeros()` allow for the creation of arrays of the specified shape, filled with 1's and 0's, respectively. We out first the 1's, so they are the target values for the dog pictures.

```
In [18]: y_train = np.concatenate([np.ones(1000), np.zeros(1000)])
```

We check now that the shapes of these arrays are the expected ones.

```
In [19]: X_train.shape, y_train.shape
Out[19]: ((2000, 150, 150, 3), (2000,))
```

## Q2c. Test data

We follow the same steps for the test data.

```
In [20]: X_test = img_to_arr('data/dogs-test/' + os.listdir('data/dogs-test')[0])
    ...: for i in range(1, 500):
    ...:     X_test = np.concatenate([X_test, img_to_arr('data/dogs-test/' + os.listdir('data/dogs-test')[i])])
    ...: for i in range(500):
    ...:     X_test = np.concatenate([X_test, img_to_arr('data/cats-test/' + os.listdir('data/cats-test')[i])])
    ...: X_test = X_test/255
    ...: y_test = np.concatenate([np.ones(500), np.zeros(500)])
    ...: X_test.shape, y_test.shape
Out[20]: ((1000, 150, 150, 3), (1000,))
```

## Q3. Training a CNN model from scratch

We import the Keras function `Input()` and the modules `models` and `layers`, as in the previous examples.

```
In [21]: from keras import Input, models, layers
```

Next, we specify the shape of the input tensor, which corresponds to an RGB image with resolution 150 $\times$ 150.

```
In [22]: input_tensor = Input(shape=(150, 150, 3))
```

Now, the hidden layers. As in example ML-19, we stack convolutional blocks (`Conv2D` plus `MaxPooling2D`). Since we are dealing with bigger images, we make the network larger, including a fourth block. The depth of the feature maps progressively increases in the network (from 32 to 128), while the size decreases (from 150 $\times$ 150 to 7 $\times$ 7). As we will see in the summary below, flattening the output of the fourth convolutional block leaves us with a tensor of length 6,272, so we reduce the dimensionality with a final `Dense` layer. This last layer returns a vector of length 512 which is expected to provide a representation of the image that helps the dog/cat classification. This dimensionality reduction is a standard procedure. 

```
In [23]: x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    ...: x2 = layers.MaxPooling2D((2, 2))(x1)
    ...: x3 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
    ...: x4 = layers.MaxPooling2D((2, 2))(x3)
    ...: x5 = layers.Conv2D(128, (3, 3), activation='relu')(x4)
    ...: x6 = layers.MaxPooling2D((2, 2))(x5)
    ...: x7 = layers.Conv2D(128, (3, 3), activation='relu')(x6)
    ...: x8 = layers.MaxPooling2D((2, 2))(x7)
    ...: x9 = layers.Flatten()(x8)
    ...: x10 = layers.Dense(512, activation='relu')(x9)
```

Finally, the output layer, which returns the predicted class probabilities.

```
In [24]: output_tensor = layers.Dense(2, activation='softmax')(x10)
```

The successive application of these functions make the CNN model, which works as a flow that starts with the input tensor and ends with the output tensor.

```
In [25]: clf1 = models.Model(input_tensor, output_tensor)
```
The table returned by the method `.summary()` illustrates this network architecture, with involves 3.45M parameters.

```
In [26]: clf1.summary()
Model: "functional"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 150, 150, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 148, 148, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 74, 74, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 72, 72, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 36, 36, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 34, 34, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 17, 17, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 15, 15, 128)    │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 7, 7, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 6272)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 512)            │     3,211,776 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 2)              │         1,026 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 3,453,634 (13.17 MB)

 Trainable params: 3,453,634 (13.17 MB)

 Non-trainable params: 0 (0.00 B)
```

Now we compile the model. Nothing new here.

```
In [27]: clf1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
```

With the method `.fit()`, we train and test the model with the data sets that were built above. 10 epochs are enough to see the limitations of this approach. We get about 68.5% accuracy on the test data (not negligeable), but with a clear overfitting issue. The training data do not seem to be enough for so many parameters.

```
In [28]: clf1.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test));
Epoch 1/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 18s 277ms/step - acc: 0.5026 - loss: 0.7671 - val_acc: 0.5000 - val_loss: 0.6933
Epoch 2/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 17s 273ms/step - acc: 0.5090 - loss: 0.6932 - val_acc: 0.5000 - val_loss: 0.6931
Epoch 3/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 17s 272ms/step - acc: 0.5070 - loss: 0.6931 - val_acc: 0.5050 - val_loss: 0.6975
Epoch 4/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 17s 276ms/step - acc: 0.5524 - loss: 0.6867 - val_acc: 0.5680 - val_loss: 0.6835
Epoch 5/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 17s 277ms/step - acc: 0.5552 - loss: 0.6794 - val_acc: 0.5800 - val_loss: 0.6779
Epoch 6/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 17s 277ms/step - acc: 0.6173 - loss: 0.6569 - val_acc: 0.6650 - val_loss: 0.6117
Epoch 7/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 17s 275ms/step - acc: 0.6485 - loss: 0.6240 - val_acc: 0.5970 - val_loss: 0.6594
Epoch 8/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 18s 279ms/step - acc: 0.6849 - loss: 0.5901 - val_acc: 0.6640 - val_loss: 0.6044
Epoch 9/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 18s 283ms/step - acc: 0.7172 - loss: 0.5486 - val_acc: 0.6890 - val_loss: 0.5960
Epoch 10/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 17s 277ms/step - acc: 0.7542 - loss: 0.4894 - val_acc: 0.6850 - val_loss: 0.6313
```

## Q4a. Pre-trained CNN model

The sources of pre-trained models have been discussed in lecture ML-20. The Keras module `applications` is a legacy of Keras 2 that provides a limited supply (compared to current repositories), but is more than enough for this example. It contains a collection of deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning. The model VGG16 is a (relatively) simple CNN model with a convolutional base made of `Conv2D` and `MaxPooling2D` layers. Importing this model in a straightforward.

```
In [29]: from keras.applications import VGG16
```

We instantiate a VGG16 model. Note the choices made:

* The argument `weights='imagenet'` specifies that the initial parameter values are those obtained from training the model with the ImageNet data. We can update them in the training process or not (see below). We can also update only a subset (typically those of the last layers).

* The model can be seen as a convolutional base plus a densely connected classifier on top. With the argument `include_top=False`, this classifier, which would return probabilities for the 1,000 ImageNet classes, is discarded.

* The argument `input_shape=(150, 150, 3)` is needed only for the summary below. When creating our new model, the input shape will be specified in the usual way.

The summary shows that the VGG16 base is made of five convolutional blocks. These blocks contain two or three `Conv2D` layers. The height and width are kept constant with a trick called **padding** (look at the Keras book is you are interested).

```
In [30]: conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    ...: conv_base.summary()
Model: "vgg16"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_3 (InputLayer)      │ (None, 150, 150, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block1_conv1 (Conv2D)           │ (None, 150, 150, 64)   │         1,792 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block1_conv2 (Conv2D)           │ (None, 150, 150, 64)   │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block1_pool (MaxPooling2D)      │ (None, 75, 75, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block2_conv1 (Conv2D)           │ (None, 75, 75, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block2_conv2 (Conv2D)           │ (None, 75, 75, 128)    │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block2_pool (MaxPooling2D)      │ (None, 37, 37, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_conv1 (Conv2D)           │ (None, 37, 37, 256)    │       295,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_conv2 (Conv2D)           │ (None, 37, 37, 256)    │       590,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_conv3 (Conv2D)           │ (None, 37, 37, 256)    │       590,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_pool (MaxPooling2D)      │ (None, 18, 18, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_conv1 (Conv2D)           │ (None, 18, 18, 512)    │     1,180,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_conv2 (Conv2D)           │ (None, 18, 18, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_conv3 (Conv2D)           │ (None, 18, 18, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_pool (MaxPooling2D)      │ (None, 9, 9, 512)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_conv1 (Conv2D)           │ (None, 9, 9, 512)      │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_conv2 (Conv2D)           │ (None, 9, 9, 512)      │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_conv3 (Conv2D)           │ (None, 9, 9, 512)      │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_pool (MaxPooling2D)      │ (None, 4, 4, 512)      │             0 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 14,714,688 (56.13 MB)

 Trainable params: 14,714,688 (56.13 MB)

 Non-trainable params: 0 (0.00 B)
```

Here we freeze the parameter values, so they will not be adapted to the cats vs dogs data. This is optional, and it is even possible to freeze only the initial layers. Other options are proposed in the homework. Freezing the whole convolutional base is pretty easy:

```
In [31]: conv_base.trainable = False
```

## Q4b. Adding a densely connected classifier on top

Now we build a new network adding three layers on top of the convolutional base. The convolutional base is managed here as a single component. The top layers are the same as in the network of question Q3. Note that the 14,714,688 parameters of the convolutional base appear here as non-trainable.

```
In [32]: input_tensor = Input(shape=(150, 150, 3))
    ...: x1 = conv_base(input_tensor)
    ...: x2 = layers.Flatten()(x1)
    ...: x3 = layers.Dense(512, activation='relu')(x2)
    ...: output_tensor = layers.Dense(2, activation='softmax')(x3)
    ...: clf2 = models.Model(input_tensor, output_tensor)
    ...: clf2.summary()
Model: "functional_2"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)      │ (None, 150, 150, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ vgg16 (Functional)              │ (None, 4, 4, 512)      │    14,714,688 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 8192)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 512)            │     4,194,816 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 2)              │         1,026 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 27,302,216 (104.15 MB)

 Trainable params: 4,195,842 (16.01 MB)

 Non-trainable params: 14,714,688 (56.13 MB)
```

## Q5. Training the new model

Finally, we compile and fit the new model. The improvement, compared to the smaller network of question Q3, is quite clear.

```
In [33]: clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    ...: clf2.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test));
Epoch 1/5
63/63 ━━━━━━━━━━━━━━━━━━━━ 125s 2s/step - acc: 0.7357 - loss: 0.4886 - val_acc: 0.8650 - val_loss: 0.3089
Epoch 2/5
63/63 ━━━━━━━━━━━━━━━━━━━━ 134s 2s/step - acc: 0.9105 - loss: 0.2229 - val_acc: 0.8910 - val_loss: 0.2533
Epoch 3/5
63/63 ━━━━━━━━━━━━━━━━━━━━ 129s 2s/step - acc: 0.9500 - loss: 0.1543 - val_acc: 0.8940 - val_loss: 0.2446
Epoch 4/5
63/63 ━━━━━━━━━━━━━━━━━━━━ 130s 2s/step - acc: 0.9491 - loss: 0.1390 - val_acc: 0.8990 - val_loss: 0.2504
Epoch 5/5
63/63 ━━━━━━━━━━━━━━━━━━━━ 129s 2s/step - acc: 0.9608 - loss: 0.1188 - val_acc: 0.9050 - val_loss: 0.2321
```

## Removing the data

Finally, you may remove the data from your computer (this is not needed).

```
In [34]: for d in os.listdir('data/'):
    ...:     for f in os.listdir('data/' + d):
    ...:         os.remove('data/' + d + '/' + f)
    ...:     os.rmdir('data/' + d)
```

```
In [35]: os.rmdir('data')
```

## Homework

1. `keras.applications` offers plenty of choice for pre-trained models, beyond VGG16. `https://keras.io/api/applications` can help you to choose. For instance, you can try **Xception**, which uses some additional tricks proposed by F Chollet.

2. You can easily unfreeze some of the last layers of the pre-trained model. For instance, in the VGG16 model, after freezing all the layers with `conv_base.trainable = False`, you can apply the loop `for l in conv_base.layers[-2]: l.trainable = True`. Some call this **fine-tuning** the model. Try this with the VGG16 model that you have used in the preceding exercise, to see whether you can improve the performance. For the fitting process to work, you will have to decrease the **learning rate** (the default is `learning_rate=1e-3`). First, import the module `optimizers` (`from keras import optimizers`), and then compile the model using the argument `optimizer=optimizers.Adam(learning_rate=5e-5)`.

3. If you survive to the preceding challenges, you can play with the learning rate, to see how this affects the fitting process.
