#  [ML-15] Neural networks

## What is a neural network?

**Neural networks** are as old as artificial intelligence itself. The expectations about what neural networks could do have experimented up and downs along 60 years. Right now, these expectations are very high. Neural networks and, more specifically, a special type of neural network models, the **transformers** are taking over machine learning. As more capabilities are found for transformers, expectations about the power of artificial intelligence keep growing.  

The neural network was thought as an interconnected set of computational **nodes** or neurons, organized in **layers**. Every connection of a node to another node has a **weight**. In machine learning, these weights are learned from the training data.  The way the nodes are connected in a neural networks was initially inspired by ideas about neurons work together in the brain. Though this is no longer true, the discussion about the similarities between the two domains is still alive.

There are many types of neural networks. The figure below is a visualization of a small **multilayer perceptron** (MLP) model, that will be used for the discussion in this lecture. This lecture is restricted to this specific type, which has been the standard approach for many years. In the forthcoming lectures, we will focus on more complex architectures, which are usually presented as **deep learning** models.

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/15-1.png)

Even if the idea of the neural network as a mathematical model for the brain (which it is not) was attractive, we see a neural network nowadays as a mathematical function, which takes an input and returns an output. Both the input and the output are **tensors**. A tensor is the same as a NumPy array (though in Python they are different types of objects). So, a 0D tensor is a scalar (a number), a 1D tensor is a vector, a 2D tensor is a matrix, etc. Most of the operations performed with the tensors in a neural network are just linear algebra. 

## Basics of the MLP model

A multilayer perceptron is formed by:

* The **input layer**, whose nodes are the features used for the prediction.

* The **output layer**. In regression models, it has a unique node, which is the target (as in the above figure), while, in classification models, it has one node for every target value.

* A sequence of **hidden layers**, placed between the input and the output layers. If the network is **fully-connected**, that is, if every node of a layer is connected to all the nodes of the following layer, the model is completely specified by the number of hidden layers and the number of nodes in each hidden layer.

How do these networks work? Suppose first that $Z$ is a hidden node and $U_1, U_2, \dots, U_k$ are the nodes of the preceding layer. Then, the values of $Z$ are calculated as

$$Z = F\big(w_0 + w_1U_1 + w_2U_2 + \cdots + w_kU_k\big).$$

In this context, the slope coefficients $w_1, w_2, \dots, w_k$  are called weights, and the intercept $w_0$ is called **bias**. $F()$ is the **activation function**. The role of the activation function is to introduce nonlinearity in the model (see below).

The multilayer perceptron could be seen as if the samples were circulating through the network one-by-one. The feature values are entered in the input nodes, which send them to the nodes of the first hidden layer. At each hidden node, they are combined using the corresponding weights, and the result is transformed by means of the activation function. The hidden nodes send the resulting values to the nodes of the next layer, where they are combined. According to the legend, this is the way animal neurons learn.

Let us help intuition with the graphical representation of a small network. The model of the figure below is a MLP regressor with one hidden layer of two nodes. The diagram is just a graphical representation of a set of three equations, two for the hidden nodes and one for the output node. The equation of node $A$ combines $X_1$, $X_2$ and $X_3$ with weights $w_{1A}$, $w_{2A}$ and $w_{3A}$, while the equation in node $B$ combines them with weights $w_{1B}$, $w_{2B}$ and $w_{3B}$. The biases are $w_{0A}$ and $w_{0B}$, respectively.

At the hidden nodes, the **activation function** is applied to the values given by these equations. Once the activation has been applied, the outcomes of the two hidden nodes are combined in the third equation, with weights $w_{AY}$ and $w_{BY}$ and bias $w_{0Y}$, to obtain the predicted value of $Y$. This model has a total of 11 parameters.

## The activation function

The choice of the activation function is based on performance, since we do not have any serious theory that could explain why a specific mathematical formula works better than others. Just a few years ago, the **logistic function** was the recommended activation function in the hidden layers, although some preferred a similar formula called the **hyperbolic tangent** function. The current trend favors the **rectified linear unit function** ($\hbox{ReLU}$). $\hbox{ReLU}(x)$ is equal to $x$ when $x>0$ and equal to $0$ otherwise. So, the default activation in the hidden layers consists in turning the negative incoming values into zeros.

In a MLP regressor (as in the figure), there is no activation at the (single) output node, so the equation predicting the values at that node is linear. In a MLP classifier, there are as many output nodes as target values. A **softmax activation** is applied to the whole set of incoming values, turning them into a set of **class probabilities**. In mathematical terms, this is the same as logistic regression applied to the output of the last hidden layer.

## Other technicalities

* *How to find the optimal weights*. Initially, the weights are randomly assigned. Then, an iterative process starts. At every step, the prediction is performed with the current weights, the value of a **loss function** is calculated, and the weights are adjusted in order to reduce the loss. The process is expected to converge to an optimal solution, but, in practice, a maximum number of iterations is pre-specified. In regression, the loss is usually the MSE, while, in classification, it is the average cross-entropy (Keras uses natural logs to calculate it). The adjustment of the weights starts at the last layer, and continues backwards until the input layer. This is called **backpropagation**.

* *The optimization method*, called **solver** in scikit-learn and **optimizer** in the Keras API. The current trend favors the **stochastic gradient descent** (SGD) method, which has many variants. Though you may find in books or tutorials the variant `optimizer='rmsprop'`, we use here `optimizer='adam'`, which is faster.

* *The number of iterations*, that is, the number of times every sample passes through the network is controlled in Keras with the parameter **epochs**. The default is `epochs=1`. The samples don't pass all at once, but in random batches (see below).

* *The learning rate*, which we have already found in gradient boosting modeling, is a parameter which rules how fast the adjustment of the weights is done. If it is too low, there is no convergence to the optimal solution. If it is too high, you can overshoot the optimal solution. Modern ML software allows setting an initial learning rate and decrease it as the learning process goes on. The Keras default is `learning_rate=0.001`. We don't use this parameter in this course.

* *The batch size*. In the SGD method, the training data are randomly partitioned in batches in every iteration. The batches are tried one-by-one and the weights are modified every time that a batch is tried. The Keras default is `batch_size=32`. We don't use this parameter in this course.

* *Normalization*. Optimization methods are sensitive to feature scaling, so it is highly recommended to scale your data. In the old data mining suites, normalization was applied as a part of the algorithm, and the output was scaled back to the original range. It is not so in the Python ML toolbox.

## What is deep learning?

**Deep learning**, the current star of machine learning, is based on neural networks. The success of deep learning, not yet fully understood, is attributed to the ability of creating improved **representations** of the input data by means of successive layers of features.

Under this perspective, deep learning is a successful approach to **feature engineering**. Why is this needed? Because, in many cases, the available features do not provide an adequate representation of the data, so replacing the original features by a new set may be useful. At the price of oversimplifying a complex question, the following two examples may help to understand this:

* A **pricing model** for predicting the sale price of a house from features like the square footage of the plot and the house, the location, the number of bedrooms, the existence of a garage, etc. You will probably agree that these are, indeed, the features that determine the price, so they provide a good representation of the data, and a **shallow learning** model, such as a random forest regressor, would be a good approach. No feature engineering is needed here.

* A model for **image classification**. Here, the available features are related to a grid of pixels. But we do not classify an image based on specific pixel positions. Recognition is based on shapes and corners. A shape is a created by a collection of pixels, each of them close to the preceding one. And a corner is created by tho shapes intersecting in a specific way. So, we have the input layer of pixels, a first hidden layer of shapes providing a better representation, a second layer of corners providing an even better representation.

The number of hidden layers in a neural network is called the **depth**. But, although deep learning is based on neural networks with more than one hidden layer, there is more in deep learning than additional layers. In the MLP model as we have seen it, every hidden node is connected to all the nodes of the preceding layer and all the nodes of the following layer. In the deep learning context, these fully-connected layers are called **dense**. But there are other types of layers, and the most glamorous applications of deep learning are based on networks which are not fully-connected.

## TensorFlow, PyTorch and Keras

A number of attempts have been made to implement the mathematics of neural networks. Many of them are just history nowadays. **TensorFlow**, developed at Google Brain and released in 2015, has been for years the top popular choice, though the field seems to be divided right now between TensorFlow and **Torch** (PyTorch in Python), released by Meta.

**Keras** is a deep learning framework for Python (there is also a version for R), which provides a convenient way to define and train neural network models. The documentation is available at `https://keras.io`. Keras does not handle itself low-level operations such as tensor manipulation and differentiation. Instead, it relies on a specialized tensor library to do so. That library serves as the **backend** engine of Keras. Keras was organized in a modular way, so several different backend engines could be plugged seamlessly into Keras. Keras 1 worked with three backend implementations, TensorFlow, Theano and CNTK. Given the dominance of TensorFlow, the last two options were dropped, so Keras 2 was no longer multi-backend, becoming just an API for TensorFlow. In November 2023, Keras became again multi-backend, the optional backends being TensorFlow, PyTorch and JAX (a new Google development). 

Just to give you an idea why Keras is popular, it has been said that the number of keystrokes needed to specify a deep learning model in Keras is one half of what was needed in old TensorFlow. Another advantage is that the code written in Keras by a developer using TensorFlow can be reused without change by another developer using PyTorch. Even if Torch tensors and TensorFlow tensors are different types of objects in Python, Keras takes care of that, and input and output tensors appear as NumPy arrays, which is the only thing you have to manage in Keras.

This course uses Keras with the default backend, which is TensorFlow. Everything would work the same with the PyTorch backend. MLP models for regression and classification are available in scikit-learn, in the classes `MLPRegressor()` and `MLPClassifier()` of the subpackage `neural_networks`, but scikit-learn does not cover the advanced methodology we are interested in this part of the course.

## MLP networks in Keras

Let us suppose, in this section, that you wish to train a MLP classifier using `keras`. For the examples discussed in this course, in which the target vector is numeric, you will have enough with the modules `models` and `layers`, which you can import as:

```
from keras import models, layers
```

The module `models` has two classes, `.Sequential()` and `.Model()`. The first one is enough to specify a neural network architecture made of a sequence of layers. The other class, known as the **Functional API**, is used with more sophisticated architectures. 

A simple way to specify the network architecture is to create a list of layers. The layers are extracted from classes of the module `layers`. For a MLP network we only need the class `Dense()`. For instance, a MLP network with one hidden layer of 32 nodes for the MNIST data would be specified as:

```
network = [layers.Dense(32, activation='relu'), layers.Dense(10, activation='softmax')]
```

You start by initializing the class, with the default specification:

```
clf = models.Sequential(network)
```

Then, the model is compiled, with the method `.compile()`:

```
clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
```

Now, we can apply the method `.fit()`, which is just a bit more complex than in scikit-learn. Assuming that you have previously performed a train/test split, this step could be:

```
clf.fit(X_train, y, epochs=10)
```

Note that the number of iterations (the parameter `epochs`) is specified as an argument of `.fit()`, not as in scikit-learn, when instantiating the estimator. In `tensorflow.keras`, you can run `.fit()` many times, getting a gradual improvement.

The method `.fit()` prints a report tracking the training process. You can stop this with the argument `verbose=0`. After fitting, we validate the model on the test set:

```
clf.evaluate(X_test, y_test)
```

Here, the method `.predict()` returns the class probabilities (not the predicted class), just as the method `.predict_proba()` in scikit-learn.
