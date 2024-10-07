# [ML-09] Imbalanced learning

## Class imbalance

In classification, the target is a categorical variable with a (typically small) collection of values, which we call **classes**. When the data collection is driven by the availability of data, it is not rare for the proportion of training units of one class to be significantly different from the proportion of units of the other classes. This is called **class imbalance**. Though it does not receive enough attention in books and tutorials, in real-life machine learning the training data often has class imbalance. Imbalance is frequent in applications like **credit scoring** or **fraud detection**. It is also frequent in **direct marketing**, since **conversion rates** are typically low.

To simplify, let us assume a binary classification context, with two classes, positive and negative. Typically, class imbalance happens because the positive training units are much less frequent than the negative ones. Imbalance can be extreme in some applications. For instance, for fraud detection in credit card transactions, the proportion of fraudulent transactions is typically below 1%. 

In such extreme situations, many recommended metrics for model evaluation don't make sense. The accuracy, for instance. The **total accuracy** value that we see in the evaluation report of a classification model is the weighted average of the accuracy in the positive class and the accuracy in the negative class. If the positive class is much smaller than the negative class, the accuracy in the positive class has practically no influence in the total accuracy. So, a model can have a very good total accuracy but a poor accuracy in the positive class. Moreover, it may be that the algorithm that searches for the optimal parameters (*e.g*. the coefficients in a logistic regression equation) pushes the model selection in that direction.

Various approaches have been proposed to address the class imbalance issue. They constitute the so called **imbalanced learning**. This lecture is a short introduction to this methodology.

## The scoring approach

A classification model predicts a set of **class probabilities** for every data unit. In scikit-learn, this is provided by the method `.predict_proba()`. In the default prediction, the predicted class is the one with the highest probability. In scikit-learn, this is what the method `.predict()` does. One way of dealing with class imbalance is to step back to the class probabilities, using them directly.

In binary classification, the two class probabilities are complementary, so we focus on the positive class probability, which we call **predictive score**. In terms of these scores, the default prediction consists in predicting as positive those units for which the score exceeds 0.5. One way of managing the imbalance is to lower this threshold, typically to a value close to the actual proportion of positive training units. This is simple and easy to manage under moderate class imbalance. But, under extreme class imbalance, the predicted class may be too sensitive to small changes in the threshold. Because of this, and also because users find that the scores are more "managerial" that the binary prediction, the use of the scores in business is common in practice, in particular in fields like credit scoring or fraud detection. Decisions may be based on the scores, without an explicit pre-specified threshold.

The scoring approach is illustrated in example ML-10. Alternative approaches, based on resampling the training data, are explained below, but the practice is left for the homework.

## Resampling

In a **resampling** approach, we train the model on a modified data set, in which the class imbalance has been artificially corrected. There is always a **random sampling** process associated to resampling. There are various resampling methods, but we can summarize them as: 

* **Undersampling**: we reduce the number of negative training units to match the number of positive training units.

* **Oversampling**: we increase the number of positive training units to match the number of negative training units..

A specialized package, `imblearn`, complements scikit-learn with a resampling toolkit. But, instead of `imblearn`, we propose below the Pandas method `.sample()` (also recommended in the homework of example ML-10), which makes everything obvious.

## Undersampling

Undersampling typically proceeds by randomly dropping as many negative units as needed to match the positive units, so we end up with a pefectly balanced data set. Let us suppose that the training data are stored in a Pandas data frame `df`, in which the target column is a dummy named `class`, so we have `class=1` for the positive units and `class=0` for the negative units. 

We split the training data in two subsets as:

```
df0, df1 = df[df['class'] == 0], df[df['class'] == 1]
```

The number of negative and positive units are:

```
n0, n1 = df0.shape[0], df1.shape[0]
```

Class imbalance means that `n0` is much bigger than `n1`. We extract from `df0` a random subset, with `n1` rows:

```
df0_under = df0.sample(n1)
```

Then, we join this reduced data frame with the positive training subset.

```
df_under = pd.concat([df0_under, df1])
```

Now, `df_under` is balanced, with `n1` positive units and `n1` negative units.

## Oversampling

Oversampling adds extra positive units, so we end up with a pefectly balanced data set. Let us consider a training data set `df`, as in the preceding section. Roughly speaking, these additional positive units can be obtained either by replicating existing positive units or by generating artificial, nonexisting units by interpolation. We only consider the first approach here. The top popular method for interpolation is **SMOTE** (Synthetic Minority Oversampling TEchnique).

In the first approach, the Python code would be similar to that used for undersampling. Once the training data have been split, we apply the method `.sample()` to `df1` with the argument `replace=True` (the default is the opposite). This extracts a random sample of rows of `df1` allowing the units extracted to be repeated.

 ```
df1_over = df1.sample(n0 - n1, replace=True)
```

We add now  the extra units to the original data frame.

```
df_over = pd.concat([df, df1_over])
```

Now, `df_over` is balanced, with `n0` positive units and `n0` negative units.
