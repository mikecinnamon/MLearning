# [ML-19] Example - The MNIST data (2)

## Introduction

This example uses the **MNIST data** that appeared in the example ML-17. There, we compared the accuracies of several tree-based models for classifying handwritten digits. Here, we show how to improve the performance of the classic MLP neural network with a **convolutional neural network**. We use standard **Keras** procedures to develop our models.

## Questions

Q1. Split the data in a training set with 60,000 samples and a test set with 10,000 samples.

Q2. Train and test a MLP model, with a hidden layer of 32 nodes, using that split.

Q3. How does the model make predictions?

Q4. Convert the gray scale to the 0-1 range (0 = Black, 1 = White), and try again with the same MLP architecture.

Q5. Try now with a convolutional neural network. Do you get a real improvement with this *deep* model?

## Importing the data

We import the MNIST data to a Pandas data frame, as we did previously.

```
In [1]: import numpy as np, pandas as pd
   ...: path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
   ...: df = pd.read_csv(path + 'mnist.csv.zip')
```

We set the first column (the image labels) as the target vector and the pixel intensities as the feature matrix. We use only NumPy arrays in this example, to simplify the syntax. So, we convert the Pandas objects to arrays by means of `.values`

```
In [2]: y = df.iloc[:, 0].values
   ...: X = df.iloc[:, 1:].values
```

## Q1. Train-test split

We split the data as in example ML-17, but setting `random_state=0` for reproducibility. So, we will be able to rescale the features and apply the same split to the rescaled data.

```
In [3]: from sklearn import model_selection
   ...: X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1/7, random_state=0)
```

## Q2. MLP model

As in our previous experience with neural networks, in example ML-16, we use the package Keras with the default **TensorFlow backend**. We import the function `Input` and the modules `models` and `layers`.

```
In [4]: from keras import Input, models, layers
```

Next, we specify our first **network architecture**, as a sequence of transformations. As in example ML-16, we use a multilayer perceptron (MLP) network with one **hidden layer** of 32 nodes, with **ReLU activation**. The **output layer** has now ten nodes, one for each digit, with **softmax activation**. These two layers are **dense layers**, meaning that every node is connected to all nodes of the preceding layer. Note that, in lines 1 and 3, the shapes of the input and the output tensors have been adapted to the data of this example.

```
In [5]: input_tensor = Input(shape=(784,))
   ...: x = layers.Dense(32, activation='relu')(input_tensor)
   ...: output_tensor = layers.Dense(10, activation='softmax')(x)
```

We instantiate an object of the class `models.Model()`, specifying the input and the output.

```
In [6]: clf1 = models.Model(input_tensor, output_tensor)
```

The method `.summary()` prints a summary of the network architecture, reporting the number of parameters in every layer. In the hidden layer, every node receives 784 inputs (one for each pixel), which are combined wby means of a linear expression involving the same number of **weights** (the slope coefficients) plus a **bias** (the intercept). So, 785 parameters are needed at every node, which makes a total of 32 $\times$ 785 = 25,120 parameters. In a similar way, in the output layer, every node needs 33 parameters, which adds 10 $\times$ 33 = 330 parameters to get a total of 25,450 parameters for the whole network.  

```
In [7]: clf1.summary()
Model: "functional"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 784)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 32)             │        25,120 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │           330 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 76,352 (298.25 KB)

 Trainable params: 25,450 (99.41 KB)

 Non-trainable params: 0 (0.00 B)

 Optimizer params: 50,902 (198.84 KB)
```

The **compilation** step is the same as in example ML-16.

```
In [8]: clf1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
```

Finally, we fit the model to the training data. We don't use `verbose=0` here (this means `verbose=1`), so we are going to get a report of the training process. We set `epochs=20`, to keep it short. Also, with the argument `validation_data=(X_test, y_test)`, the model is evaluated on the test data at the end of every epoch.

```
In [9]: clf1.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));
Epoch 1/20
1875/1875 [==============================] - 1s 642us/step - loss: 1.7580 - acc: 0.6203 - val_loss: 0.9735 - val_acc: 0.7821
Epoch 2/20
1875/1875 [==============================] - 1s 557us/step - loss: 0.7216 - acc: 0.8159 - val_loss: 0.6070 - val_acc: 0.8551
Epoch 3/20
1875/1875 [==============================] - 1s 553us/step - loss: 0.4880 - acc: 0.8776 - val_loss: 0.4244 - val_acc: 0.8932
Epoch 4/20
1875/1875 [==============================] - 1s 538us/step - loss: 0.3731 - acc: 0.9086 - val_loss: 0.3681 - val_acc: 0.9094
Epoch 5/20
1875/1875 [==============================] - 1s 530us/step - loss: 0.3244 - acc: 0.9179 - val_loss: 0.3757 - val_acc: 0.9136
Epoch 6/20
1875/1875 [==============================] - 1s 528us/step - loss: 0.3020 - acc: 0.9231 - val_loss: 0.3205 - val_acc: 0.9235
Epoch 7/20
1875/1875 [==============================] - 1s 521us/step - loss: 0.2829 - acc: 0.9282 - val_loss: 0.3046 - val_acc: 0.9235
Epoch 8/20
1875/1875 [==============================] - 1s 528us/step - loss: 0.2697 - acc: 0.9309 - val_loss: 0.3210 - val_acc: 0.9212
Epoch 9/20
1875/1875 [==============================] - 1s 528us/step - loss: 0.2634 - acc: 0.9327 - val_loss: 0.3157 - val_acc: 0.9202
Epoch 10/20
1875/1875 [==============================] - 1s 533us/step - loss: 0.2627 - acc: 0.9337 - val_loss: 0.3116 - val_acc: 0.9253
Epoch 11/20
1875/1875 [==============================] - 1s 533us/step - loss: 0.2580 - acc: 0.9352 - val_loss: 0.3232 - val_acc: 0.9226
Epoch 12/20
1875/1875 [==============================] - 1s 530us/step - loss: 0.2520 - acc: 0.9360 - val_loss: 0.3715 - val_acc: 0.9163
Epoch 13/20
1875/1875 [==============================] - 1s 536us/step - loss: 0.2492 - acc: 0.9363 - val_loss: 0.3294 - val_acc: 0.9227
Epoch 14/20
1875/1875 [==============================] - 1s 535us/step - loss: 0.2458 - acc: 0.9376 - val_loss: 0.3125 - val_acc: 0.9241
Epoch 15/20
1875/1875 [==============================] - 1s 535us/step - loss: 0.2400 - acc: 0.9394 - val_loss: 0.3095 - val_acc: 0.9271
Epoch 16/20
1875/1875 [==============================] - 1s 521us/step - loss: 0.2399 - acc: 0.9396 - val_loss: 0.3200 - val_acc: 0.9241
Epoch 17/20
1875/1875 [==============================] - 1s 530us/step - loss: 0.2381 - acc: 0.9394 - val_loss: 0.3196 - val_acc: 0.9244
Epoch 18/20
1875/1875 [==============================] - 1s 542us/step - loss: 0.2387 - acc: 0.9398 - val_loss: 0.3207 - val_acc: 0.9245
Epoch 19/20
1875/1875 [==============================] - 1s 539us/step - loss: 0.2347 - acc: 0.9408 - val_loss: 0.3238 - val_acc: 0.9302
Epoch 20/20
1875/1875 [==============================] - 1s 531us/step - loss: 0.2325 - acc: 0.9416 - val_loss: 0.3716 - val_acc: 0.9240
```

This report shows that the accuracy on the test data is about 92.5% at the 10th epoch, and does not go beyond this, while it keeps improving on the training data, up to 94% at the 20th epoch. In example ML-16, we did a bit better with a random forest model.

## Q3. Prediction with a MLP model

For every input, the method `.predict()` returns the 10 values given by the softmax activation at the output nodes, which are the **predicted class probabilities** for that input. So, in a classification setting, it works as the method `.predict_proba()` in scikit-learn. For instance, for the first digit of the test set, we get:

```
In [10]: clf1.predict(X_test[:1, :])
1/1 [==============================] - 0s 74ms/step
Out[10]: 
array([[9.8956162e-01, 2.5109919e-08, 3.2807675e-03, 2.7105544e-04,
        2.8752672e-04, 2.0526997e-03, 3.6389669e-03, 3.6512902e-05,
        3.1564130e-05, 8.3928910e-04]], dtype=float32)
```

According to the model, with probability 99%, this is a zero. Indeed, the prediction is correct:

```
In [11]: y_test[0]
Out[11]: 0
```

## Q4. Rescaling the data

In this example, no normalization function is needed, since we can rescale the pixel intensities by dividing by 255.

```
In [12]: X = X/255
```

Since we fixed `random_state=0` in our split, we can reproduce it for the rescaled feature matrix:

```
In [13]: X_train, X_test = model_selection.train_test_split(X, test_size=1/7, random_state=0)
```

Now, we instantiate a new model, which we train on the rescaled data. We keep `epochs=20`, even if we guess that so many epochs are not needed. The training process is not only faster, achieving accuracy 92.3% in the first epoch, but it also yields better results at the end. But, though this model performs better than the random forest models of example MLE-09, it is still weaker than the GBoost model that you may have explored in the homework of example ML-17. 

```
In [14]: clf2 = models.Model(input_tensor, output_tensor)
    ...: clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    ...: clf2.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));
Epoch 1/20
1875/1875 [==============================] - 1s 574us/step - loss: 0.5411 - acc: 0.8475 - val_loss: 0.2654 - val_acc: 0.9226
Epoch 2/20
1875/1875 [==============================] - 1s 587us/step - loss: 0.2215 - acc: 0.9359 - val_loss: 0.2141 - val_acc: 0.9367
Epoch 3/20
1875/1875 [==============================] - 1s 561us/step - loss: 0.1882 - acc: 0.9453 - val_loss: 0.1982 - val_acc: 0.9390
Epoch 4/20
1875/1875 [==============================] - 1s 544us/step - loss: 0.1710 - acc: 0.9505 - val_loss: 0.1900 - val_acc: 0.9437
Epoch 5/20
1875/1875 [==============================] - 1s 539us/step - loss: 0.1595 - acc: 0.9535 - val_loss: 0.1854 - val_acc: 0.9470
Epoch 6/20
1875/1875 [==============================] - 1s 541us/step - loss: 0.1507 - acc: 0.9554 - val_loss: 0.1754 - val_acc: 0.9485
Epoch 7/20
1875/1875 [==============================] - 1s 538us/step - loss: 0.1451 - acc: 0.9570 - val_loss: 0.1822 - val_acc: 0.9462
Epoch 8/20
1875/1875 [==============================] - 1s 535us/step - loss: 0.1390 - acc: 0.9595 - val_loss: 0.1750 - val_acc: 0.9490
Epoch 9/20
1875/1875 [==============================] - 1s 532us/step - loss: 0.1346 - acc: 0.9603 - val_loss: 0.1799 - val_acc: 0.9484
Epoch 10/20
1875/1875 [==============================] - 1s 541us/step - loss: 0.1307 - acc: 0.9609 - val_loss: 0.1702 - val_acc: 0.9496
Epoch 11/20
1875/1875 [==============================] - 1s 529us/step - loss: 0.1270 - acc: 0.9621 - val_loss: 0.1767 - val_acc: 0.9508
Epoch 12/20
1875/1875 [==============================] - 1s 522us/step - loss: 0.1240 - acc: 0.9638 - val_loss: 0.1760 - val_acc: 0.9497
Epoch 13/20
1875/1875 [==============================] - 1s 547us/step - loss: 0.1208 - acc: 0.9643 - val_loss: 0.1707 - val_acc: 0.9502
Epoch 14/20
1875/1875 [==============================] - 1s 534us/step - loss: 0.1186 - acc: 0.9648 - val_loss: 0.1687 - val_acc: 0.9504
Epoch 15/20
1875/1875 [==============================] - 1s 528us/step - loss: 0.1157 - acc: 0.9655 - val_loss: 0.1736 - val_acc: 0.9504
Epoch 16/20
1875/1875 [==============================] - 1s 524us/step - loss: 0.1139 - acc: 0.9667 - val_loss: 0.1719 - val_acc: 0.9505
Epoch 17/20
1875/1875 [==============================] - 1s 536us/step - loss: 0.1130 - acc: 0.9658 - val_loss: 0.1684 - val_acc: 0.9520
Epoch 18/20
1875/1875 [==============================] - 1s 536us/step - loss: 0.1096 - acc: 0.9677 - val_loss: 0.1696 - val_acc: 0.9513
Epoch 19/20
1875/1875 [==============================] - 1s 538us/step - loss: 0.1081 - acc: 0.9678 - val_loss: 0.1691 - val_acc: 0.9521
Epoch 20/20
1875/1875 [==============================] - 1s 533us/step - loss: 0.1061 - acc: 0.9683 - val_loss: 0.1726 - val_acc: 0.9501
```

## Q5. CNN model

We dig deeper in this last section, exploring **convolutional neural network** (CNN) models, in particular a model based on a 2D convolutional network. We continue working with the rescaled features. Instead of using a 60,000 $\times$ 784 features matrix (a 2D array), we pack the training data as a 4D array, in which the axes are:

* `axis=0` with dimension 60,000, identifies the inputs, *i.e*. the images, 

* `axis=1` and `axis=2` both with dimension 28, identify the pixel positions.

* `axis=2` with 1 dimension, identifies the **channel**. Since we work with gray scale, there is only one channel (for RGB pictures, we would have three channels). 

We reshape the arrays `X_train` and `X_test` as 4D arrays accordingly.

```
In [15]: X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
```

As for the MLP model, we specify the network architecture as a sequence of transformations. In this example, we propose a standard CNN architecture, used by many authors for these data. 

In this case, every input will be 3D array, with shape `(28,28,1)`. The networks starts with a sequence of three `Conv2D` layers, with 32, 64 and 64 nodes, respectively, with two interspersed `MaxPooling` layers. After passing these layers, the input has been transformed into a set of 64 smaller 3D arrays (we will see below how small). Then, this set is flattened into a 1D array, which is the input for the rest of the network. This last part part is the same as a MLP network with a hidden layer of 64 nodes. 

```
In [16]: input_tensor = Input(shape=(28, 28, 1))
    ...: x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    ...: x2 = layers.MaxPooling2D((2, 2))(x1)
    ...: x3 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
    ...: x4 = layers.MaxPooling2D((2, 2))(x3)
    ...: x5 = layers.Conv2D(64, (3, 3), activation='relu')(x4)
    ...: x6 = layers.Flatten()(x5)
    ...: x7 = layers.Dense(64, activation='relu')(x6)
    ...: output_tensor = layers.Dense(10, activation='softmax')(x7)
    ...: clf3 = models.Model(input_tensor, output_tensor)
```

As for the MLP model, we can print a summary reporting the number of parameters involved in every layer:

* The first `Conv2D` layer has 32 nodes, with specific parameters. At every node, a special type of linear calculation, called **convolution**, takes place. The convolution extracts a single number of a 3 $\times$ 3 submatrix, using nine weights and one bias. This makes a total of 10 parameters for every node, a grand total of 32 $\times$ 10 = 320 parameters. The original 28 $\times$ 28 matrix can be covered by a 26 $\times$ 26 grid of 3 $\times$ 3 submatrices, so the outputs of this layer have shape (26, 26, 32).

* The `MaxPooling` layer uses 2 $\times$ 2 submatrices and, instead of a convolution, it just takes the maximum value. So, no parameters are involved, and the outputs have shape (13, 13, 32). 

* The third layer has 64 nodes. Every node takes inputs from the 32 nodes of the preceding layer. Nine weights are needed for every input and one bias for the whole set, making a total of 32 $\times$ 9 + 1 = 289 parameters for every node. So, we have 64 $\times$ 289 = 18,496 additional parameters in this layer. Now, the outputs have shape `(11, 11, 64)`.

* The next two layers can be explained in a similar way. In the fifth layer, we have added 36,928 parameters and the outputs have shape `(3, 3, 64)`.

* The sixth layer consistes in taking 3 $\times$ 3 $\times$ 64 = 576 inputs and arranging them as a 1D array. No parameters are needed.

* The seventh layer is a dense layer, so have we already know how it works. The 64 nodes need 577 parameters each to manage the 576 inputs. So, we have 64 $\times$ 577 = 36,928 additional parameters, and the output has shape `(64,)`.

* The last layer has 10 nodes. Each node takes 64 inputs, involving 65 parameters. So we have 10 $\times$ 65 = 650 additional parameters, ad the output is a vector of 10 class probabilities.

At the end of the day, comparing this *deep* model with the previous MLP model, with only 32 hidden nodes, the number of parameters does not increase so much. This is due to the **low connectivity** of the CNN model.

```
In [17]: clf3.summary()
Model: "functional_4"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_3 (InputLayer)      │ (None, 28, 28, 1)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_6 (Conv2D)               │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_6 (MaxPooling2D)  │ (None, 13, 13, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_7 (Conv2D)               │ (None, 11, 11, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_7 (MaxPooling2D)  │ (None, 5, 5, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_8 (Conv2D)               │ (None, 3, 3, 64)       │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_2 (Flatten)             │ (None, 576)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (Dense)                 │ (None, 64)             │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_7 (Dense)                 │ (None, 10)             │           650 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 279,968 (1.07 MB)

 Trainable params: 93,322 (364.54 KB)

 Non-trainable params: 0 (0.00 B)

 Optimizer params: 186,646 (729.09 KB)
```

The rest of the process is the same as in the MLP models. We use here `epochs=10`, to make it shorter. At the end of the first epoch, the model achieves 98% accuracy on the test data. After the 10 epochs, this has been improved to 99%, which is quite satisfactory.

```
In [18]: clf3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    ...: clf3.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test));
Epoch 1/10
1875/1875 [==============================] - 14s 7ms/step - loss: 0.1442 - acc: 0.9548 - val_loss: 0.0635 - val_acc: 0.9796
Epoch 2/10
1875/1875 [==============================] - 14s 8ms/step - loss: 0.0458 - acc: 0.9858 - val_loss: 0.0486 - val_acc: 0.9854
Epoch 3/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0327 - acc: 0.9899 - val_loss: 0.0463 - val_acc: 0.9848
Epoch 4/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0266 - acc: 0.9915 - val_loss: 0.0343 - val_acc: 0.9885
Epoch 5/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0197 - acc: 0.9935 - val_loss: 0.0306 - val_acc: 0.9910
Epoch 6/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0161 - acc: 0.9949 - val_loss: 0.0418 - val_acc: 0.9872
Epoch 7/10
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0126 - acc: 0.9958 - val_loss: 0.0398 - val_acc: 0.9893
Epoch 8/10
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0114 - acc: 0.9962 - val_loss: 0.0366 - val_acc: 0.9911
Epoch 9/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0104 - acc: 0.9966 - val_loss: 0.0300 - val_acc: 0.9926
Epoch 10/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0080 - acc: 0.9973 - val_loss: 0.0363 - val_acc: 0.9919
```
