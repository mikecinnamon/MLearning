# Assignment 3

## Introduction

This assignment is a continuation of the analysis performed in examples ML-17 and ML-19, which use the **MNIST data**. These examples have given you an idea about the extent to which simple machine learning models can recognize handwritten **digits**. In this assignment, we explore this a bit more, trying alternative approaches. 

## Questions

Q1. Try a **logistic regression** model on the MNIST data and compare its performance to that of the models presented in in examples ML-17 and ML-19. Normalize the data as in question Q4 of example ML-19.

Q2. A logistic regression model is just an **MLP** model without **hidden layers**. Specify the model of question Q1 as neural network model, and use Keras to train and test it. Are the results comparable to those of question Q1?

Q3. Insert a second hidden layer in the MLP model of example ML-19, getting a **deeper** model. Do you get a higher accuracy?

Q4. Make it deeper inserting a third hidden layer. Is this better?

Q5. We used in example ML-19 a **convolutional neural network** (CNN) model taken from the literature. But perhaps we do not need a network with so many laters. Drop the last convolutional layers, so that you are left with with two `Conv2D` layers and two `MaxPooling2D` layers. Are losing much power?

Q6. Train and test a **gradient boosting model** on the MNIST data. Mind that, with hundreds of columns, training a gradient boosting model may be much slower than training a random forest model with the same tree size and number of trees. A model with 200 trees and a size similar to those shown in this example can take half an hour to train, though you may find a speedup by increasing the **learning rate**.

Q7. Calculate the **confusion matrix** (dimension 10x10) for the models of questions Q1 and Q6. Which is the best classified digit? Which is the main source of misclassification?

## Submission

1. Submit, through Blackboard, a readable and printable report responding these questions and explaining what you have done, including Python input and output. This can be a Word document, a PDF document or a Jupyter notebook (`.ipynb`).

2. Put your name on top of the document.

## Deadline

November 24 (Sunday), 24:00.
