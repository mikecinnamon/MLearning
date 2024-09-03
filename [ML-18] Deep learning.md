#  [ML-18] Deep learning

## What is deep learning?

**Deep learning**, the current star of machine learning, is based on neural networks. The success of deep learning, not yet fully understood, is attributed to the ability of creating improved **representations** of the input data by means of successive layers of features.

Under this perspective, deep learning is a successful approach to **feature engineering**. Why is this needed? In many cases, the available features do not provide an adequate representation of the data. So, replacing the original features by a new set may be useful. At the price of oversimplifying a complex question, the following two examples may help to understand this:

* A **pricing model** for predicting the sale price of a house from features like the square footage of the plot and the house, the location, the number of bedrooms, the existence of a garage, etc. You will probably agree that these are, indeed, the features that determine the price, so they provide a good representation of the data, and a **shallow learning** model, such as a gradient boosting regressor, would be a good approach. No feature engineering is needed here.

* A model for **image classification**. Here, the available features are related to a grid of pixels. But we do not recognize images from specific pixel positions. Recognition is based on **shapes** and **corners**. A shape is a created by a collection of pixels, each of them close to the preceding one. And a corner is created by tho shapes intersecting in a specific way. This suggests that a neural network with an input layer of pixels, a first hidden layer of shapes, and a second layer of corners can provide a better representation, being so useful for image classification.

The number of hidden layers in a neural network is called the **depth**. But, although deep learning is based on neural networks with more than one hidden layer, there is more in deep learning than additional layers. In the MLP model as we have seen it, every hidden node is connected to all the nodes of the preceding layer and all the nodes of the following layer. In the deep learning context, these fully-connected layers are called **dense**. But there are other types of layers, and the most glamorous applications of deep learning are based on networks which are not fully-connected.

## Convolutional neural networks

In the classic MLP network, hidden and output layers were dense, that is, every node was connected to all neurons in the next layer. A **convolutional neural network** (CNN) contains othetr types of layer, such as **convolutional layers** and **max pooling layers**. These layers have low connectivity, and the connections are selected according a design which takes advantage of the hierarchical pattern in the data and assemble complex patterns using smaller and simpler patterns. The main idea is that dense layers learn global patterns in their input feature space (*e.g*. in the MNIST data, patterns involving all pixels), while these new layers learn local patterns, *i.e*. patterns found in small 1D or 2D windows of the inputs.

There are two subtypes of convolutional networks:

* **1D convolutional networks** (Conv1D), used with sequence data (see below).

* **2D convolutional networks** (Conv2D), used in image classification. 

## Applications to computer vision

In the CNN's used in **image classification**, the input is a 3D tensor, called a **feature map**. The feature map has two spatial axes, called **height** and **width**, and a **depth** axis. For an RGB image, the dimension of the depth axis would be 3, since the image has 3 color channels, red, green, and blue. For black and white pictures like the MNIST digits, it is just 1 (the gray levels).

The basic innovation is the `Conv2D` layer, which extracts patches from its input feature map, typically with a 3 $\times$ 3 window, applying the same transformation to all of these patches, producing a new output feature map. This output feature map is still a 3D tensor: it has width, height and depth. Its depth can be arbitrary, since the output depth is a parameter of the layer, and the different channels in that depth axis no longer stand for specific colors like in an RGB input, but for different views of the input, called **filters**. The filters encode specific aspects of the input data. For instance, at a high level, a single filter could be encoding the concept "presence of a face in the input".

With convolutional networks, practitioners typically use two strategies for extracting more of their data:

* **Transfer learning**. Instead of starting to train your model with random coefficients, you start with those of a model which has been pre-trained with other data. There is plenty of supply of pre-trained models, as we will comment in lecture ML-20.

* **Data augmentation**. Expanding the training data with images obtained by transforming the original images. Typical transformations are: rotation with a random angle, random shift and zoom. Keras offers many resources for that, though we don't have room for them in this short course.

## CNN models in Keras

Let us use again the MNIST data as to illustrate the Keras syntax, now for CNN models. The height and the width are 28, and the depth is 1. We start by reshaping the training and test feature matrices as 3D arrays, so they can provide inputs for a `Conv2D` layer:

```
X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
```

*Note*. This reshaping may not be needed if you get the MINST data from other sources than the GitHub repository of this course. 

In the Functional API, the network architecture is specified as a sequence of transformations. We have seen this in example ML-15. The following architecture has been taken from a Keras example:

```
input_tensor = Input(shape=(28, 28, 1))
x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
x2 = layers.MaxPooling2D((2, 2))(x1)
x3 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
x4 = layers.MaxPooling2D((2, 2))(x3)
x5 = layers.Conv2D(64, (3, 3), activation='relu')(x4)
x6 = layers.Flatten()(x5)
x7 = layers.Dense(64, activation='relu')(x6)
output_tensor = layers.Dense(10, activation='softmax')(x7)
```

A summary of the network can be printed with the method `.summary()`. In this case, we woud get the table:

```
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
```

The table indicates, for every layer, the output shape and the number of parameters involved. The first layer is a `Conv2D` layer of 32 nodes. Every node takes data from a 3 $\times$ 3 window (submatrix) of the 28 $\times$ 28 pixel matrix, performing a convolution operation on those data. There are 26 $\times$ 26 such windows, so the output feature map will have height and width 26. The convolution is a linear function of the input data. For a specific node, the coefficients used by the convolution are the same for all windows.

`Conv2D` layers are typically alternated with `MaxPooling2D` layers. These layers also use windows (here 2 $\times$ 2 windows), from which they just extract the maximum value (no parameters needed). In the `MaxPooling2D` layer, the windows are disjoint, so the size of the feature map is halved. Therefore, the output feature map has height and width 13. We have an output feature map for every input feature map.

We continue with two `Conv2D` layers, with 64 nodes each, with a `MaxPooling2D` layer in-between. The output is now a tensor of shape `(3, 3, 64)`. The network is closed by a stack of two `Dense` layers. Since the input in the first of these layers has to be a one-dimensional, we have to flatten the 3D output of the last `Conv2D` layer to a 1D tensor. This is done with a `Flatten` layer, which involves no calculation, being just a reshape. 

Next, we initialize the class `model.Models()`, specifying the input and the output:

```
clf = models.Models(input_tensor, output_tensor)
```

Now we can apply, as in the MLP example, the methods `.compile()`, `.fit()` and `.evaluate()`:

```
clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf.fit(X_train, y_train, epochs=10)
clf.evaluate(X_test, y_test) 
```

Alternatively, you can fit and evaluate the model in one shot, testing after every epoch:

```
clf.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## Applications to sequence data

The second area of success of deep learning is **sequence data**. This is a generic expression including text, time series data, sound and video. You may find many sources about to use 1D convolutional networks and **recurrent neural networks** (RNN) to this type of data. Though in this course we are interested in the applications of deep learning to text data, the use of CNN and RNN models with text data may get obsolete very soon, given the strong push recently given by generative AI in this field. So, we postpone text data analysis until we have introduced the new toolkit.
