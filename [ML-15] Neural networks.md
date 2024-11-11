#  [ML-15] Neural networks

## What is a neural network?

**Neural networks** are as old as artificial intelligence itself. The expectations about what neural networks could do have experimented up and downs along 60 years. Right now, they are very high. Neural networks and, more specifically, a special type of neural network models, the **transformers**, are taking over machine learning. As more capabilities are found for the new models, expectations about the power of artificial intelligence keep growing.  

A neural network can be thought as an interconnected set of computational **nodes** or neurons, organized in **layers**. In the network, every connection of a node to another node has a **weight**. In machine learning, these weights are learned from the training data.  The way the nodes are connected in a neural network was initially inspired by ideas about neurons work together in the brain. Though this is no longer true, the discussion about the similarities between the two domains is still alive.

There are many types of neural networks. This lecture is restricted to the **multilayer perceptron** (MLP) model, which has been the standard approach for many years. In the forthcoming lectures, we will focus on more complex architectures, which are usually presented as **deep learning** models.

Even if the idea of the neural network as a mathematical model for the brain (which it is not) was attractive, we see a neural network nowadays as a mathematical function, which takes an input and returns an output. Both the input and the output are **tensors**. A tensor is the same as a NumPy array (though in Python they are different types of objects). So, a 0D tensor is a scalar (a number), a 1D tensor is a vector, a 2D tensor is a matrix, etc. Most of the operations performed with the tensors in a neural network are just linear algebra. 

## Basics of the MLP model

A multilayer perceptron is formed by:

* The **input layer**, whose nodes are the features used for the prediction.

* The **output layer**. In regression models, it has a unique node, which is the target (as in the above figure), while, in classification models, it has one node for every target value.

* A sequence of **hidden layers**, placed between the input and the output layers. If the network is **fully-connected**, that is, if every node of a layer is connected to all the nodes of the following layer, the **network architecture** is completely specified by the number of hidden layers and the number of nodes in each hidden layer.

So the MLP model transforms a 1D tensor of features into either a 0D tensor (regression) or a 1D tensor of class probabilities. These tensors are the input and the output, respectively. How is the transformation performed? Suppose first that $Z$ is a hidden node and $U_1, U_2, \dots, U_k$ are the nodes of the preceding layer. Then, the values of $Z$ are calculated as

$$Z = F\big(w_0 + w_1U_1 + w_2U_2 + \cdots + w_kU_k\big).$$

In this context, the slope coefficients $w_1, w_2, \dots, w_k$  are called weights, and the intercept $w_0$ is called **bias**. $F()$ is the **activation function**. 

The multilayer perceptron could be seen as if the samples were circulating through the network one-by-one. The feature values are entered in the input nodes, which send them to the nodes of the first hidden layer. At each hidden node, they are combined using the corresponding weights, and the result is transformed by means of the activation function. The hidden nodes send the resulting values to the nodes of the next layer, where they are combined. And so on, until arriving to the output layer.

## A graphical example 

Let us help intuition with the graphical representation of a small network. The model of the figure below is an MLP regressor with one hidden layer of two nodes. The diagram is just a graphical representation of a set of three equations, two for the hidden nodes and one for the output node. The equation of node $A$ combines $X_1$, $X_2$ and $X_3$ with weights $w_{1A}$, $w_{2A}$ and $w_{3A}$, while the equation in node $B$ combines them with weights $w_{1B}$, $w_{2B}$ and $w_{3B}$. The biases are $w_{0A}$ and $w_{0B}$, respectively.

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/15-1.png)

At the hidden nodes, the **activation function** is applied to the values given by these equations. Once the activation has been applied, the outcomes of the two hidden nodes are combined in the third equation, with weights $w_{AY}$ and $w_{BY}$ and bias $w_{0Y}$, to obtain the predicted value of $Y$. This model has a total of 11 parameters.

## The activation function

The choice of the activation function is based on performance, since we do not have any serious theory that could explain why a specific mathematical formula works better than others. Just a few years ago, the **logistic function** was the recommended activation function in the hidden layers, although some preferred a similar formula called the **hyperbolic tangent** function. The current trend favors the **rectified linear unit function** ($\hbox{ReLU}$). $\hbox{ReLU}(x)$ is equal to $x$ when $x>0$ and equal to $0$ otherwise. So, the default activation in the hidden layers consists in turning the negative incoming values into zeros.

In a MLP regressor (as in the figure), there is no activation at the (single) output node, so the equation predicting the values at that node is linear. In a MLP classifier, there are as many output nodes as target values. A **softmax activation** is applied to the whole set of incoming values, turning them into a set of **class probabilities**. In mathematical terms, this is the same as logistic regression applied to the output of the last hidden layer.

## Other technicalities

* *How to find the optimal weights*. Initially, the weights are randomly assigned. Then, an iterative process starts. At every step, the prediction is performed with the current weights, the value of a **loss function** is calculated, and the weights are adjusted in order to reduce the loss. The process is expected to converge to an optimal solution, but, in practice, a maximum number of passes is pre-specified. In regression, the loss is usually the MSE, while, in classification, it is the average cross-entropy (Keras uses natural logs to calculate it). The adjustment of the weights starts at the last layer, and continues backwards until the input layer. This is called **backpropagation**.

* *The optimization method*, called **solver** in scikit-learn and **optimizer** in the Keras API. The current trend favors the **stochastic gradient descent** (SGD) method, which has many variants. Though you may find in books or tutorials the variant `optimizer='rmsprop'`, we use here `optimizer='adam'`, which is faster.

* *The number of iterations*, that is, the number of times every sample passes through the network is controlled in Keras with the parameter **epochs**. The default is `epochs=1`. In SGD, the samples don't pass all at once, but in **random batches** (see below).

* *The batch size*. In the SGD method, the training data are randomly partitioned in batches in every iteration. The batches are tried one-by-one and the weights are modified every time that a batch is tried. The Keras default is `batch_size=32`. We don't change this in the examples of this course.

* *The learning rate*, which we have already found in gradient boosting modeling, is a parameter which rules how fast the adjustment of the weights is done. If it is too low, there is no convergence to the optimal solution. If it is too high, you can overshoot the optimal solution. Modern ML software allows setting an initial learning rate and decrease it as the learning process goes on. The Keras default is `learning_rate=0.001`. 

* *Normalization*. Optimization methods are sensitive to feature scaling, so it is highly recommended to scale your data. In the old data mining suites, normalization was applied as a part of the algorithm, and the output was scaled back to the original range. It is not so in the Python ML toolbox.

## TensorFlow, PyTorch and Keras

A number of attempts have been made to implement the mathematics of neural networks. Many of them are just history nowadays. The library **TensorFlow**, developed at Google Brain and released in 2015, has been for years the top popular choice, though the field seems to be divided right now between TensorFlow and **Torch** (PyTorch in Python).

**Keras** is a deep learning framework for Python (there is also a version for R), which provides a convenient way to define and train neural network models. The documentation is available at `https://keras.io`. Keras does not handle itself low-level operations such as tensor manipulation and differentiation. Instead, it relies on a specialized tensor library to do so. That library serves as the **backend** engine of Keras. 

Keras was organized in a modular way, so several different backend engines could be plugged seamlessly into Keras. Keras 1 worked with three backend implementations, TensorFlow, Theano and CNTK. Given the dominance of TensorFlow, the last two options were dropped, so Keras 2 was no longer multi-backend, becoming just an API for TensorFlow. In November 2023, Keras became again multi-backend, the optional backends being TensorFlow, PyTorch and JAX (a new Google development). 

Just to give you an idea why Keras is popular, it has been said that the number of keystrokes needed to specify a deep learning model in Keras is one half of what was needed in old TensorFlow. Another advantage is that the code written in Keras by a developer using TensorFlow can be reused without change by another developer using PyTorch. Even if Torch tensors and TensorFlow tensors are different types of objects in Python, Keras takes care of that, and input and output tensors appear as NumPy arrays, which is the only thing you have to manage in Keras.

This course uses Keras with the default backend, which is TensorFlow. Everything would work the same with the PyTorch backend. MLP models for regression and classification are available in scikit-learn, in the classes `MLPRegressor()` and `MLPClassifier()` of the subpackage `neural_networks`, but scikit-learn does not cover the advanced methodology needed in this part of the course.

## MLP networks in Keras

Let us suppose, in this section, that you wish to train a MLP classifier using `keras`. For the examples discussed in this course, in which the target vector is numeric, you will have enough with the function `Input()` and the modules `models` and `layers`, which you can import as:

```
from keras import Input, models, layers
```

The module `models` has two classes, `.Sequential()` and `.Model()`. The first one can only specify a network architecture made of a sequence of layers. The other class, known as the **Functional API**, does not have that restriction. We use `.Model()` in this course.

A simple way to specify the network architecture is to create a list of layers. The layers are extracted from classes of the module `layers`. For a MLP network we only need the class `Dense()`. For instance, a MLP network with one hidden layer of 32 nodes for the MNIST data would be specified as follows. First, the **input layer** contains one node for every feature. 

```
input_tensor = Input(shape=(784,))
```

Next, the hidden layer, that transforms the input tensor (length 784) into a new tensor (length (32): the activation function is $\hbox{ReLU}()$.

```
x = layers.Dense(32, activation='relu')(input_tensor)
```

Finally, the output layer transforms the hidden tensor (length 32) into a tensor of class probabilities (length 10). The activation function is here the softmax, which ensures that the output is a vector of probabilities (positive numbers summing 1). 

```
output_tensor = layers.Dense(10, activation='softmax')(x)
```

The next step is instantiate an object of the class `models.Model()`. This works as in scikit-learn. We specify here the input and the output.

```
clf = models.Model(input_tensor, output_tensor)
```

Then, the model is compiled, with the method `.compile()`:

```
clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
```

Now, we can apply the method `.fit()`, which is just a bit more complex than in scikit-learn. Assuming that you have previously performed a train/test split, this step could be:

```
clf.fit(X_train, y_train, epochs=10)
```

Note that the number of iterations (the parameter `epochs`) is specified as an argument of `.fit()`, not as in scikit-learn, when instantiating the estimator. In Keras, you can run `.fit()` many times, getting a gradual improvement.

The method `.fit()` prints a report tracking the training process. You can stop this with the argument `verbose=0`. After fitting, we validate the model on the test set:

```
clf.evaluate(X_test, y_test)
```

Here, the method `.predict()` returns the class probabilities (not the predicted class), just as the method `.predict_proba()` in scikit-learn.
