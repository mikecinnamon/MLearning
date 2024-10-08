# [MLA-01] Assignment 1

## Introduction

This assignment is based on the example ML-08, focused on a **spam filter**. The objective is to try some variations of the filters developed in class.

## Tasks

1. Train a **logistic regression** classifier on the data of this example and compare its performance to that of the classifiers presented in the example.

2. Change the features matrix by: (a) dropping the three `cap_` features and (b) **binarizing** all the `word_` features, transforming every column into a dummy for the occurrence of the corresponding word, taking value 1 if the word occurs in the message and 0 otherwise. Based on this new features matrix, train two new spam filters, one based on a logistic regression model and the other one based on a decision tree model, using the binarized data set. Evaluate these new filters and compare them to those obtained before.

## Submission

1. Submit a report, in a printable format, covering these tasks and the Python code used for each of them.

2. Put your name on top of the document.

## Deadline

October 14 (Sunday), 24:00.