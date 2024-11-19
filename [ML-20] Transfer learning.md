#  [ML-20] Transfer learning

## What is transfer learning?

**Transfer learning** is a technique used in machine learning to leverage the knowledge gained from one task to improve the performance on another related task. This typically involves taking a model that has been pre-trained on a big data set and adapting it for the new task, by updating the model's parameter values with new data, specific for the new task. The data set used in this updating step is typically much smaller than the pre-training data set, which allows us to save money and time. 

Transfer learning is critical in domains where data are only available in small amounts, or would be very expensive to collect. Since big data sets are needed to scape overfitting for complex models, it is currently common practice in many business applications. In both **computer vision** and in **natural language processing**, we can profit from existing models that have been released by their developers.

Transfer learning has two components: 

* The pre-trained model. These models are usually extracted from public repositories, called **hubs**. 

* The new data. These data have to be specific of the new task, called the **downstream task**.

When can transfer learning help you? When you can find a pre-trained model with the same kind of inputs, and there is a commonality in the tasks of the two models. It makes a difference because, when training a neural network model from scratch, we start with random parameter values. These initial values don't make any sense for the task on which we are training the model. Starting with the parameter values learned in a previous training, we are much closer to the optimal values. 

## Sources of pre-trained models

* **Keras**. The Keras team has created two hubs, **KerasCV** (computer vision) and **KerasNLP** (natural language processing), which can be accessed by means of specific Python packages. They have selected the models, so there is plenty of choice and you will not miss anything relevant. These are (relatively) small models, not behemoths like GPT-4, so most of them can be managed by your computer.

* **Hugging Face**. The favorite hub, so far independent of the big corporations. In addition to the "serious" models that you can find in the Keras hubs, you will find in Hugging Face thousands of models, uploaded by the (registered) users, which are just retrained versions of the those available in the Keras hubs. When this is being written, Hugging face website claims to have 964,720 models.

* **Kaggle Models**. Kaggle started as an independent platform for data science and machine learning competitions, adding later a hub for data sets. Everybody could post data, notebooks, etc. It was purchased by Google in 2017. Right now, those competitions have lost their glamour, but Kaggle offers, besides the data sets, a mix of courses, notebooks and pre-trained models. Though the (registered) members of the Kaggle community can post their models, as in Hugging Face, the relevant stuff can be easily found.

* **TensorFlow Hub**. It was initially part of the Keras/TensorFlow combo, but was integrated with Kaggle Models in November 2022.

* **Ollama**. An open-source project that serves as a platform for running LLMs on your local machine. Not exactly user-friendly, as many open-source projects, but quite powerful. It is presented as if you had to manage it from the shell, but there is a Python package that provides an easy way to integrate it in your workflow.

Some of this will show up in this course: (a) in example ML-21, we will pick a model from Keras, and (b) in example ML-23, a model from Hugging Face. With a differfent approach, we will use in example ML-26 the Cohere API, to run a big model on a remote server.

## Transfer learning for CNN models

Keras provides some powerful image classifiers, pre-trained on the **ImageNet** data set, which is the outcome of a project started by FF Lei, then a professor at Princeton, in 2006. We use one of these models, **VGG16**, in example ML-21.. It is based on a CNN architecture which is similar to the one used in example ML-19, though a bit bigger. Even if VGG16 is a dwarf (below 20M parameters) compared to the top popular large language models, it will suffice for understanding the dynamics of transfer learning.

To illustrate this, let us take the model summarized below.

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

We trained this model in example ML-19 on the MNIST data. In the context of this lecture, who would say that it is the pre-trained model. We can see it as composed of two parts: 

* The **convolutional base**, a stack of three `Conv2D` layers and two `MaxPooling2D` layers. This part encodes the picture as a vector of length 576. This transformation is what we called an **embedding** (see lecture ML-22). Because of the pretraining the embedding generates vectors that are appropriate for recognizing shapes and corners.

* The **top classifier**, which is like an MLP model with a hidden layer of 64 nodes whose input is a vector of length 576. The `Flatten` layer does not have any parameter, so it can be included in this part or in the base. This part classifies the embedding vectors as digits.

Now, suppose that we switch from digit recognition to letter recognition. Our model can be based on the same network architecture, except for the last `Dense` layer, which will be adapted so that it will ouptput 26 class probabilities (English alphabet) instead of 10. If we us agree that the convolutional base, as it is, is also good for the new task, we can freeze the parameter values of that part, and train the classifier. In practice, it would be as if we were using the embedding vectors as the features for a classification model. So, instead of transfer learning, we could call this **feature engineering**.

We can also unfreeze part of the convolutional base (*e.g*. the last `Conv2D` layer), or the whole thing, modifying so the parameter values. This is called  **fine-tuning**, because, even if the parameter values change, they don't change much. We will see in example ML-21 how easy are these tricks in Keras.
