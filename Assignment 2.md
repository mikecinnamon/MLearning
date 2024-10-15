# Assignment 2

## Introduction

This assignment is a continuation of the analysis performed in the example ML-10, focused on the selection of the contacts for a direct marketing campaign of **term deposits**. Here, we take a different approach to the problem of **class imbalance**. With a 11.7% **conversion rate**, the data from the bank show a moderate class imbalance, which was addressed in the example with a **scoring** approach. In this assignment, we use a **resampling** approach, training our predictive models in a modified data set in which the class imbalance has been artificially corrected.

## Questions

Q1. **Undersample** the data, by randomly dropping as many negative units as needed to match the positive units, so that you end up with a pefectly balanced training data set. Train a logistic regression model on this undersampled training data set and evaluate it, based on a confusion matrix. 

Q2. **Oversample** the data, by randomly adding as many duplicates of the positive units as needed to match the negative units, so that you end up with a pefectly balanced training data set. Train a logistic regression model on this oversampled training data set and evaluate it, based on a confusion matrix.

Q3. Compare these two models to the one obtained in the example to address question Q3. What do you think?

## Submission

1. Submit, through Blackboard, a readable and printable report responding these questions and explaining what you have done, including Python input and output. This can be a Word document, a PDF document or a Jupyter notebook (`.ipynb`).

2. Put your name on top of the document.

## Deadline

October 27 (Sunday), 24:00.
