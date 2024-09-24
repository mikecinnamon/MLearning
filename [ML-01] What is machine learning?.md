# [ML-01] What is machine learning?

## Machine learning

The objective of **artificial intelligence** (AI) is create agents that perform certain tasks in an "intelligent" way. An **AI agent** can be something physical, like a robot that sweeps the floor, or a software app, like a model that classifies the potential customers of a lending institution as good or bad creditors.

**Machine learning** (ML) takes place when the AI agent learns from data how to perform its task. We operationalize the learning process as follows. We design a **model**, which can be as simple as a single equation, or as complex as **GPT-4**. This model has a set of **parameters**. The number of parameters can be high (GPT-4 is said to have 1.76 trillion parameters), so it is often unclear what their specific role is. Then the learning process consists in using the data to find the **optimal values** for the parameters. 

Instead of talking about "learning", a statistician would say that he is **fitting** the model to the data, or **estimating** the parameters of the model. This terminology is used in the Python library that we will use in the first part of this course, **scikit-learn**. 

Finding the optimal parameter values is also called **training**. The data used for training the model are then the **training data**. A major issue with training a model is that the parameter values that are optimal on the training data may be suboptimal on data that have not been involved in the training, so that the model underperforms in real applications. This is the **overfitting** problem.

To assess the potential overfitting, the model is tested on different data, which are then called **test data**. This is **model validation**. Validation is needed for models whose complexity allows them to overfit the data. Overfitting is a fact of life for many machine learning algorithms, *e.g*. for those used to develop **neural network models**. So, validation is integrated in the learning process for these models.

## Supervised and unsupervised learning

In machine learning, it is usual to distinguish between supervised and unsupervised learning. Roughly speaking, **supervised learning** is what the statisticians call prediction, that is, the description of one variable ($Y$) in terms of other variables (the $X$'s). In the ML context, $Y$ is called the **target**, and the $X$'s are called the **features**. The units (they can be customers, products, etc) on which the features and the target are observed are called **samples** (this term has a different meaning in statistics).

The term **regression** applies to the prediction of a (more or less continuous) numeric target, and the term **classification** to the prediction of a categorical target. In **binary classification**, there are only two target values or **classes**, while, in **multi-class classification**, there can be three or more. A **classifier**, or classification model, predicts a probability for every class.

In an example of regression, we may try to predict the price of a house from a set of attributes of that house. In one of classification, whether a customer is going to quit our company, from his/her demographics plus some measures of customer activity.

In **unsupervised leaning**, there is no target to be predicted (only $X$'s). The objective is to learn patterns from the data. Unsupervised learning is more difficult, and more creative, than supervised learning. The two classics of unsupervised learning are **clustering**, which consists in grouping objects based on their similarity, and **association rules** mining, which consists in extracting from the data rules such as *if A, then B*. A typical application of clustering in business is **customer segmentation**. Association rules are applied in **market basket analysis**, to associate products that are purchased (or viewed in a website) together. Other relevant examples of unsupervised learning are **dimensionality reduction** and **anomaly detection**.

## Variations

In a classification context, distinction is frequently made between labeled and unlabeled data. The **labels** are the target values. With labeled data, one takes a supervised learning approach and, with unlabeled data, an unsupervised learning approach. For instance, in image classification, the data ususally consist in a set of pictures. The pictures can be human-labeled, which makes the training data more expensive, or unlabeled.

In-between supervised and unsupervised learning, we have **semi-supervised learning**, which combines supervised and unsupervised learning, requiring only a small portion of the learning data be labeled. An alternative approach is **self-supervised learning**, which uses only unlabeled data. A well known example of self-supervised learning is Google's **Word2Vec**, a technique which learns word associations to generate a representation of words as vectors in a multidimensional space. This representation, called **embedding**, can be used later in a supervised learning process.

From the point of view of the practical implementation, we can also distinguish between batch and on-line learning. In **batch learning**, the model is trained and tested on given data sets and applied for some time without modification. In **on-line training**, it is continuously retrained with the incoming data. The choice between batch and continuous learning depends on practical issues, rather than on theoretical arguments.

## Reinforcement learning

Another variation is **reinforcement learning**, which is one of the current trending ML topics, because of its unexpected success in playing games like go and StarCraft II. It is usually considered as neither upervised nor unsupervised learning, but as a different branch of machine learning.

In reinforcement learning, an AI agent takes actions or make decisions in a certain environment in order to maximize a **cumulative reward**. The way the reward is set drives the learning process in this or that direction. For a gentle introduction, see Mitchell (2020).

## Generative AI

The term **generative** is used, in this context, for models that generate new data. These could be pictures with nobody's faces, fake videos or the text outputted by the popular **ChatGPT**. Generative AI models use techniques, which are very close to supervised learning, to identify the patterns within existing data, *i.e*. for an unsupervised learning task. These patterns can then be used to generate new and original content. 

The impact of generative AI is difficult to assess right now, but it is going to be relevant, since some generative models, called **large language models** have been found to be able to carry out unexpected tasks. For instance, **GitHub Copilot** can write code in many languages, which has changed completely coding practices.


## References

1. E Alpaydin (2016), *Machine Learning*, MIT Press.

2. P Domingos (2015), *The Master Algorithm*, Basic Books.

3. M Mitchell (2020), *Artificial Intelligence: A Guide for Thinking Humans*, Pelican.
