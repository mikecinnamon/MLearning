# [ML-05] Logistic regression

## Class probabilities

**Classification** is the prediction of a **categorical target**. The target values are called **classes**, and they are indicated in the training data by **class labels**. In a classification model, the **predicted class** is obtained in two steps:

* For every sample, the model calculates a set of **predicted class probabilities**, one for each class. The different types of models differ in the way in which they calculate these probabilities.

* The **predicted class** is the one with higher probability.

This is the **default prediction** method. When this approach is used, the class probabilities may be hidden, so the model is presented as if it were making the predictions directly. 

In some applications, the class probabilities are used in a different way. Departure from the default is not rare when the data present **class imbalance**, which will be specifically discussed in lecture ML-09.

## Binary classification

In **binary classification**, there are two classes, typically called **positive** and **negative**. Use the names positive/negative so that they favor your intuition. Mind that, if you leave this to the computer, it may call positive what you regard as negative.

In a binary setting, managing two complementary probabilities is redundant, so we can focus on the positive class probability. This probability, called the **predictive score**, is used for management purposes in many business applications (*e.g*. in credit scoring).

In the default prediction, a sample would be classified as positive when its score exceeds 0.5. But you may wish to replace 0.5 by a different **threshold** value. In a business application, the choice of the threshold may be based on a **cost/benefit analysis**. It is not hard to (approximately) find the **optimal threshold** for a user-specified cost matrix.

## The confusion matrix

The evaluation of a classifier is, explicitly or implicitly, based on a **confusion matrix**, obtained by cross tabulation of the actual classes and the predicted classes. There is not a universal consensus on what to place in the rows and what in the columns. We use the same convention as the scikit-learn manual, with the actual class in the rows and the predicted class in the columns.

In a binary setting, a visual inspection of the confusion matrix is always recommended. It will probably help you to decide whether the model is going to be useful. In many cases, it is practical to examine the model performance separately on the actual positives and negatives.

The four cells of the confusion matrix are referred to as **true positive** (actual positives predicted as positives), **false positive** (actual negatives predicted as positives), **true negative** (actual negatives predicted as negatives) and **false negative** (actual positives predicted as negatives).

| | Predicted negative | Predicted positive |
| --- | :---: | :---: |
| **Actual negative** | TN | FP |
| **Actual positive** | FN | TP |

The proportion of samples classified in the right way, that is, those for which the actual and the predicted values coincide, is called the **accuracy**,

$$\textrm{Accuracy} = \frac{\textrm{TN}+\textrm{TP}} {\textrm{TN}+\textrm{FP}+\textrm{FN}+\textrm{TP}}\thinspace.$$

The accuracy can be calculated directly, or extracted from the confusion matrix, as the sum of the diagonal terms divided by the sum of all terms. Although it looks as the obvious metric for the evaluation of a classifier, the accuracy is not always adequate, specially when the data present class imbalance. For instance, if you have a 90% of negative samples in your training data set, classifying all the samples as negative gives you 90% accuracy (you don't need machine learning for that!).

In a business context, a visual inspection of the confusion matrix is always recommended. In many cases, it is useful to examine the performance of the classifier separately on the actual positives and the actual negatives. Then, the usual metrics are:

* The **true positive rate** is the proportion of right classification among the actual positives,

$$\textrm{TP\ rate} = \frac{\textrm{TP}} {\textrm{TP}+\textrm{FN}}\thinspace.$$

* The **false positive rate** is the proportion of wrong classification among the actual negatives,

$$\textrm{FP\ rate} = \frac{\textrm{FP}} {\textrm{FP}+\textrm{TN}}\thinspace.$$

A good model should have both a high true positive rate and a low false positive rate low. The relative importance given to these statistics depends on the actual application. Their advantage is that they are still valid when the proportion of positives in the training data has been artificially inflated, because they are calculated separately on the actual positives and the actual negatives. This may look strange, but it is common practice under class imbalance. When the proportion of positives is inflated, the training data cannot be taken as representative of any population, and the accuracy derived from the confusion matrix cannot be extrapolated to the real world.

An alternative to the true positive and false negative rates, used by scikit-learn, is based on the precision and the recall:

* The **precision** is the proportion of right classification among the predicted positives,

$$\textrm{Precision} = \frac{\textrm{TP}} {\textrm{TP}+\textrm{FP}}\thinspace.$$

* The **recall** is the same as the true positive rate,

$$\textrm{Recall} = \frac{\textrm{TP}} {\textrm{TP}+\textrm{FN}}\thinspace.$$

In a good model, precision and recall should be high. Some authors combine precision and recall in a single metric (in mathematical terms, it is the harmonic mean), called the **F1-score**, also available in scikit-learn:
$$\textrm{F1-score} = \frac{\textrm{2}\times\textrm{Precision}\times\textrm{Recall}} {\textrm{Precision}+\textrm{Recall}}\thinspace.$$

## Logistic regression

**Logistic regression** is one of the simplest classification methods. The class probabilities are calculated as follows. Note that, in spite of its name, it is a classification method, not a regression method. The explanation is that logistic regression was created by statisticians, and regression does not mean the same in statistics as in machine learning.

Suppose that $k$ numeric features $X_1, \dots, X_k$ are used to predict a target with $m$ classes. The logistic regression model is based on a set of linear equations, 

$$z = b_0 + b_1X_1 + b_2X_2 + \cdots + b_kX_k,$$

one for each class. The values $z_1, \dots, z_m$ are transformed in class probabilities $p_1, \dots, p_m$ by means of the **softmap function**

$$p_i = \frac{\exp(z_i)}{\exp(z_1) + \cdots + \exp(z_m)}.$$

Note that, since $p_1 + \cdots + p_m = 1$, one of the equations can be obtained from the rest. Don't worry about this, Python will take care. 

As for linear regression, in logistic regression the coefficients of the equations are optimal, meaning that a certain **loss function** attains its minimum value. Here, the loss function is the **average cross-entropy**, a formula extracted from information theory. For every sample, the cross-entropy is the negative logarithm of the predicted class probability of the actual class of that sample. scikit-learn uses binary logs, as in information theory, but other packages like Keras use natural logs. You should not be concerned by this, because you don't really use these cross-entropy values, they are just part of mathematical apparatus.

Let us show, explicitly, how the cross-entropy is calculated in a binary setting, using natural logs:

* Take a positive sample whose predicted class probabilities are $0.2$ (for the negative class) and $0.8$ (for the positive class). Then, the cross-entropy for this sample is $-\log 0.8 = 0.2231$. 

* Take a negative sample whose class probabilities are $0.7$ and $0.3$, respectively. Then, the cross-entropy for this sample is $-\log 0.7 = 0.3567$. 

The average of these values for the all the training units is the loss. What is the logic of using this loss function? As shown in Figure 1, the negative log function is decreasing, with the minimum value $-\log 1 = 0$, so that by minimizing the cross-entropy, we are pushes the class probabilities of the negative units towards $(1, 0)$ and those of the positive units towards $(0, 1)$, that would be the perfect predictions.

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/05-1.png)

In linear regression, we have formulas for the optimal parameter values, which are simple enough to be implemented in a spreadsheet. This is no longer true for other ML models, in particular for logistic regression. Here, the optimal parameter values are obtained by means of an **optimization algorithm**, which starts with a set of random values and changes these values in a sequence of steps or **iterations**, decreasing the loss at every step. The iterative process goes on until the loss falls below a certain **tolerance** or until a **maximum number of iterations** has been achieved. 

## Classification in scikit-learn

In scikit-learn, we just find a few differences between classification and regression models. In classification, the terms of the target vector `y` are taken as the class labels. Data type `str` is admitted for `y`. We also have here the three basic methods, `.fit()`, `.predict()` and `.score()`. 

Let us give some details. For instance, for a binary logistic regression model, we use the class `LogisticRegression()` from the subpackage `linear_model`. As usual in scikit-learn, we instantiate an estimator:

```
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
```

As it usually happens in Python, `LogisticRegression()` has a collection of parameters that allow you change many things, but everybody accepts most of the default parameter values. For instance, you can choose the optimization algorithm, named the **solver**, but this is a bit too mathematical for most users. Nevertheless, you may wish to the change the maximum number of iterations, whose default is `max_iter=100`. Though you may find the mathematics too complex, the logic is simple. The default tolerance is `tol=0.0001`. When the solver gets a loss value below the tolerance, the iterative process stops, and the process is said to have **converged**. If the maximum number of iterations is completed without achieving convergence, the process is stopped and a warning (not an error message) appear. This means, in practice, that the model obtained so far is suboptimal, and you may get something better by raising `max_iter`. We will see this works in example ML-06.

The method `.fit()` works the same in all supervised learning models in scikit-learn. The method `.score()` is called in the same way in classification and regression, but, while in regression returns a *R*-squared statistic, it returns here the accuracy. 

In classification, the default prediction is given by the method `.predict()`. But, here, in addition to `.predict()`, we also have `.predict_proba()`, which returns a 2D array with one column for every class, containing the predicted class probabilities. For every row, the sum of the class probabilities equals 1. 

In binary classification, you may use the predictive scores, which will be obtained as:

```
y_score = clf.predict_proba(X)[:, 1]
```

With a threshold `t`, the predicted target values will then be obtained as:

```
y_pred = (y_score > t).astype(int)
```

Note that the positive class probabilities are in the second column of `clf.predict_proba(X)`, because the classes are ordered alphabetically. Now, the accuracy can be calculated directly, as:

```
sum(y == y_pred)
```

Denoting by `1` the positive class and by `0` the negative class, the true positive and the false positive rates will be:

```
tp_rate = sum((y == 1) & (y_pred == 1))/sum(y == 1)
fp_rate = sum((y == 0) & (y_pred == 1))/sum(y == 0)
```

Though you can calculate them directly, as suggested above, these and other metrics can also be extracted from the confusion matrix. You can obtain it with the Pandas function `crosstab()` or, alternatively, with the function `confusion_matrix`, from the subpackage `metrics`:

```
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y, y_pred)
```

This subpackage also provides specific methods for the precision (`.precision_score()`), the recall (`.recall_score()`) and the F1-score (`.f1_score()`).
