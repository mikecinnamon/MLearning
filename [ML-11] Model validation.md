# [ML-11] Model validation

## Overfitting and validation

**Overfitting** is a typical problem of supervised learning. It occurs when a model fits satisfactorily the training data, but fails on new data. The purpose of **validation** is to dismiss the concerns about overfitting raised by the use of complex machine learning models. These concerns are well justified, since many popular models, like neural nets, are prone to overfit the training data. Validation is also called **out-of-sample testing**, because this is what we really do.

In the simplest approach to validation, we derive the model from a **training data set**, trying it on a **test data set**. The training and test sets can have a temporal basis (*e.g*. training with the first ten months of the year and testing with the last two months), or they can be obtained by means of a **random split** of a single data set.

For the top powerful prediction models, such as gradient boosting or deep learning models, overfitting is part of the process, so practitioners take the metrics calculated on the test data set as the valid ones. When this is applied systematically, the principle of testing the model on data which have not been used to obtain it is no longer valid. The standard approach to this problem is to use a third data set, the **validation data set**, in the model selection process, keeping the test set apart, for the final evaluation.

**Cross-validation** is a more sophisticated approach, recommended for small data sets. It has many variations, among them **$k$-fold cross-validation**, in which the original data set is randomly partitioned into $k$ equally sized subsets. It is assumed that the model is evaluated with a single evaluation score (otherwise it would get very complex). One of the $k$ subsets is used for testing and the other subsets for training, and the model is scored on the test data. This process is repeated for each of the  subsets. The resulting evaluation scores (either R-squared or accuracy) can then be averaged to produce a single value, which is taken as the score of the model. $k=10$ has been recommended by some authors, but $k=3$ and $k=5$ are more popular nowadays. If you wish to keep a test set apart from the process, you can split first the data in two, performing the cross-validation in one subset, while keeping the other subset for the final test. This approach is quite popular among practitioners.

## Train-test split in scikit-learn

In scikit-learn, the subpackage `model_selection` provides a validation toolkit for supervised learning. Suppose that you are given a target vector `y` and a feature matrix `X`. A random **train-test split** can be obtained with:

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Setting `test_size=0.2`, you get a 80-20 split, which is quite popular, though there is nothing special in a 20% test size. The idea behind this partition is that it is not reasonable to waste too much data on testing. Also, note that you have to split twice if you wish to have a training set, a validation set and a test set.

For an example, take a classifier `clf`, created as a `LogisticRegression()` instance. You start by training the model with the method `.fit()`, using only the training set:

```
clf.fit(X_train, y_train)
```

Then, you can evaluate the model separately on both data sets and compare the results. If the evaluation is based on the accuracy, this would be:

```
ypred_train = clf.predict(X_train)
clf.score(X_train, y_train), clf.score(X_test, y_test)
```

Overfitting happens when there is a relevant difference between these two metrics. If you wish to compare a collection of potential models, you will replace here the test set by the validation set, leaving the test set apart. By applying this process repeatedly, you can select the model with the best performance on the validation set, using the test set for the final evaluation. The same ideas can be applied for regression models.

*Note*. The function `train_test_split()` can be applied to any number of array-like objects of the same length. It performs a split on each of those objects, selecting the same rows for all of them.

## Cross-validation in scikit-learn

The subpackage `model_selection` also has cross-validation functions. The simplest one is `cross_val_score()`:

```
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv=3)
```

This function would return here a vector of three accuracy scores. While you can average these scores to get an overall score for the model, you may also take a look at the variation across folds, to decide whether you will trust the model. The argument `cv=3` sets the number of folds. The default is `cv=5`. The default metrics are the R-squared statistic and the accuracy, but the parameter `scoring` allows you to use alternative metrics. For instance, with the argument `scoring='precision'`, the function `cross_val_score()` would return precision scores.
