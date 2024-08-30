#  [ML-20] Transfer learning

## What is transfer learning?

**Transfer learning** is a technique used in machine learning to leverage the knowledge gained from one task to improve performance on another related task. This typically involves taking a model that has been pre-trained on a big data set and then adapting it for the new task, by updating the model's parameters with task-specific data. The data set used in this updating step is typically much smaller than the pre-training data set, which allows us to save money and time. 

Transfer learning is critical in domains where data are only available in samll amounts, or would be very expensive to collect. Since big data set are needed to scape overfitting for complex models, it is currently common practice in many business applications. Both in computer vision and in natural language processing, we can profit from existing models that have been released by their developers.

Transfer learning has two components: 

* The pre-trained model. These models are normally extracted from public repositories, which are called **hubs**. 

* The new data. These data have to be specific of the new task, which is called a **downstream task**.

## Sources of pre-trained models

Large pre-trained models can be obtained from many sources. For instance, the team in charge of Keras has created two hubs, **KerasCV** and **KerasNLP**, which can be accessed by means of specific Python packages. But the favorite hub, right now, is **Hugging Face**, so far independent of the big corporations. 

We will use a model that can be obtained with Keras (no additional resources needed) in example ML-21. But in the rest of the course we will use Hugging Face, in particular for the large language models.

## Transfer learning for CNN models

Keras provides some powerful image classifiers, all pre-trained with the **ImageNet** data set, which is the outcome of a project started by FF Lei, then a professor at Princeton, in 2006. We use one of these models, **VGG16**, in example ML-21. It is based on a CNN architecture which is similar to the one used in example ML-19, though a bit bigger. Even if VGG16 is a dwarf (below 20M parameters) compared to the large language models that are so popular nowadays, it will be enough to understand the dynamics of transfer learning.
