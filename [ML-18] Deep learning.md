#  [ML-18] Deep learning

## What is deep learning?

**Deep learning**, the current star of machine learning, is based on neural networks. The success of deep learning, not yet fully understood, is attributed to the ability of creating improved **representations** of the input data by means of successive layers of features.

Under this perspective, deep learning is a successful approach to **feature engineering**. Why is this needed? Because, in many cases, the available features do not provide an adequate representation of the data, so replacing the original features by a new set may be useful. At the price of oversimplifying a complex question, the following two examples may help to understand this:

* A **pricing model** for predicting the sale price of a house from features like the square footage of the plot and the house, the location, the number of bedrooms, the existence of a garage, etc. You will probably agree that these are, indeed, the features that determine the price, so they provide a good representation of the data, and a **shallow learning** model, such as a random forest regressor, would be a good approach. No feature engineering is needed here.

* A model for **image classification**. Here, the available features are related to a grid of pixels. But we do not classify an image based on specific pixel positions. Recognition is based on shapes and corners. A shape is a created by a collection of pixels, each of them close to the preceding one. And a corner is created by tho shapes intersecting in a specific way. So, we have the input layer of pixels, a first hidden layer of shapes providing a better representation, a second layer of corners providing an even better representation.

The number of hidden layers in a neural network is called the **depth**. But, although deep learning is based on neural networks with more than one hidden layer, there is more in deep learning than additional layers. In the MLP model as we have seen it, every hidden node is connected to all the nodes of the preceding layer and all the nodes of the following layer. In the deep learning context, these fully-connected layers are called **dense**. But there are other types of layers, and the most glamorous applications of deep learning are based on networks which are not fully-connected.

## Deep learning application to computer vision

A **convolutional neural network** (CNN) is a regularized version of a MLP network. In the classic MLP network, input and hidden layers were dense, that is, every node was connected to all neurons in the next layer. On the contrary, CNN's have low connectivity, and connections are selected according a design which takes advantage of the hierarchical pattern in the data and assemble complex patterns using smaller and simpler patterns. The fundamental difference between a dense layer and a convolution layer is that dense layers learn global patterns in their input feature space (*e.g*. for a MNIST digit, patterns involving all pixels), while convolution layers learn local patterns, *i.e*. patterns found in small 1D or 2D windows of the inputs.

There are two subtypes of convolutional networks:

* **1D convolutional networks** (Conv1D), used with sequence data (see below).

* **2D convolutional networks** (Conv2D), used in image classification. 

In the CNN's used in **image classification**, the input is a 3D tensor, called a **feature map**. The feature map has two spatial axes, called **height** and **width**, and a **depth** axis. For a RGB image, the dimension of the depth axis would be 3, since the image has 3 color channels, red, green, and blue. For black and white pictures like the MNIST digits, it is just 1 (gray levels).

A convolution layer extracts patches from its input feature map, typically with a 3 $\times$ 3 window) and applies the same transformation to all of these patches, producing an output feature map. This output feature map is still a 3D tensor: it has width, height and depth. Its depth can be arbitrary, since the output depth is a parameter of the layer, and the different channels in that depth axis no longer stand for specific colors like in an RGB input, rather they stand for what we call **filters**. The filters encode specific aspects of the input data. For instance, at a high level, a single filter could be encoding the concept "presence of a face in the input".

Practitioners typically use two strategies for extracting more of their data:

* **Transfer learning**. Instead of starting to train your model with random coefficients, you start with those of a model which has been pre-trained with other data. There are many options for that, among the classification algorithms that have been trained the **ImageNet** database (see `image-net.org`).

* **Data augmentation**. Expanding the training data with images obtained by transforming the original images. Typical transformations are: rotation with a random angle, random shift and zoom.

## Applications to sequence data

The second area of success of deep learning is **sequence data**. This is a generic expression including text (sequences of words or characters), time series data, video and others. Although we do not have room here for this type of data, let us mention that the main types of networks used in this context are:

* 1D convolutional networks, with applications to machine translation, document classification and spelling correction.

* **Recurrent neural networks** (RNN), with applications to handwritting and speech recognition, sentiment analysis, video classification and time series data.

* Networks with **embedding layers**, with applications to natural language processing (NLP) and recommendation systems.

But the use of CNN and RNN models with text data may get obsolete very soon, given the strong push recently given by generative AI in this field. Though this is beyond the scope of this course, it is good to know that it is, precisely, the hottest are right now.

## CNN models in Keras

Let us use again the MNIST data as to illustrate the Keras syntax, now for CNN models. The height and the width are 28, and the depth is 1. We start by reshaping the training and test feature matrices as 3D arrays, so they can provide inputs for a `Conv2D` layer:

```
X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
```

*Note*. This reshaping may not be needed if you get the MINST data from other sources than the GitHub repository of this course. 

The network architecture can be specified, in a comprehensive way, as a list of layers. The following list is quite typical:

```
network = [layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')]
```
A summary of the network can be printed as:

The first layer is a `Conv2D` layer of 32 nodes. Every node takes data from a 3 $\times$ 3 window (submatrix) of the 28 $\times$ 28 pixel matrix, performing a convolution operation on those data. There are 26 $\times$ 26 such windows, so the output feature map will have height and width 26. The convolution is a linear function of the input data. For a specific node, the coefficients used by the convolution are the same for all windows.

`Conv2D` layers are typically alternated with `MaxPooling2D` layers. These layers also use windows (here 2 $\times$ 2 windows), from which they extract the maximum value. In the `MaxPooling2D` layer, the windows are disjoint, so the size of the feature map is halved. Therefore, the output feature map will have height and width 13.  

We continue with two `Conv2D` layers, with 64 nodes each, with a `MaxPooling2D` layer in-between. The output is now a tensor of shape (3, 3, 64). 

The network is closed by a stack of two `Dense` layers. Since the input in these layers has to be a vector, we have to flatten the 3D output of the last `Conv2D` layer to a 1D tensor. This is done with a `Flatten` layer, which involves no calculation, but just a reshape. 

Next, we initialize the class `Sequential()`, specifying the network architecture:

```
clf = models.Sequential(network)
```

Now we can apply, as in the MLP example, the methods `compile`, `fit` and `evaluate`:

```
clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
clf.fit(X_train, y_train, epochs=10)
clf.evaluate(X_test, y_test) 
```

Alternatively, one can fit and evaluate the model in one shot, testing after every epoch:

```
clf.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```
