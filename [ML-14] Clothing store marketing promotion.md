# [ML-14] Clothing store marketing promotion

## Introduction

A clothing store chain in Boston is planning a marketing campaign, based on a **direct mail promotion**. The promotion will use the existing customer database, which contains information on 14,857 customers. The database, built from a direct mail campaign conducted past year, integrates a collection of fields related to **consumption habits**. There is detailed information on the franchises visited, the money spent, the products purchased, the margin left, and the participation on sales and promotions. In the database, the proportion of respondents to the previous campaign is about 17%.

The campaign manager would like to have a model for predicting which customers would respond to the promotion, so that a target sample could be extracted, getting more focus. To help herself in the selection of the adequate predictive model, she develops a **cost/benefit analysis** based on the following two points:

* The cost of mailing is estimated at $5.00 per promotion unit.

* To estimate the average benefit for the customer that responds to the campaign, she calculates the average spent per visit by the customers in the database, $122.44. Since the average gross margin percentage is about 50%, and usually one half of it can be taken as net profit, she sets the net profit per respondent at $30.61.

## The data set

The variables included in the data set (file `clothing.csv`) are:

* `id`, a customer ID (encripted).

* `phone` a dummy for having a valid phone number on file.

* `web`, a dummy for being a web shopper.

* `visit`, the number of purchase visits.

* `money`, the total net sales in US dollars.

* `beacon`, the expense in the Beacon Street store in US dollars.

* `hann`, the expense in the Hannover Street store in US dollars.

* `mass`, the expense in the Massachusetts Avenue store in US dollars.

* `newbury`, the expense in the Newbury Street store in US dollars.

* `sweather`, `ktop`, `kdress`, `blouse`, `jacket`, `crpant`, `cspant`, `shirt`, `dress`, `fashion`, `suit`, `outwear`, `jewel`, `lwear` and `coll`, the percentage spent in sweaters, knit tops, knit dresses, blouses, jackets, career pants, casual pants, shirts, dresses, suits, outerwear, jewelry, fashion, legwear and the collectibles line.

* `omon`, the expense in the last month in US dollars.

* `tmon`, the expense in the last 3 months in US dollars.

* `smon`, the expense in the last 6 months in US dollars.

* `gmp`, the gross margin percentage.

* `mdown`, the percentage of expense in merchandise marked down.

* `promomail`, the number of promotions mailed past year.

* `promoresp`, the number of promotions responded past year.

* `produnif`, a measure of product uniformity (low = diverse).

* `dbetween`, the mean time between visits in days.

* `return`, percentage of merchandise returned.

* `resp`, a dummy for responding to the promotion.

Source: DT Larose (2006), *Data Mining Methods and Models*, Wiley.

## Questions

Q1. Leaving aside a 20% of the data for testing, train the following classifiers: (a) a **logistic regression model**, (b) a **decision tree model** with maximum depth 4, (c) a **random forest model** with the same maximum depth, and (d) a **grading boosting model** with the same maximum depth. For the ensemble models, use a number of trees high enough to allow for **overfitting**.

Q2. Calculate a **predictive score** for each training unit, based on these four models. Compare the distributions of the scores across models. taking into account the cost-benefit analysis proposed in the introduction, which makes false negatives more relevant than false positives. What would be your choice? What **threshold** would you apply to the scores to classify the training units?

Q3. Calculate predictive scores for the test data with the models that have survived the screening of Q2. Apply the threshold and calculate the corresponding **true positive rate** and **false positive rate**. Which model is better? Why?

## Importing the data

As in the preceding examples, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. In this case, we take the column `id` as the index (`index_col=0`).

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
   ...: df = pd.read_csv(path + 'clothing.csv', index_col=0)
```

## Exploring the data

We print a report of the content of the data frame `df` with the method `.info()`. In this report, everything looks as expected. There are no missing values.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 14857 entries, 9955600066402 to 9955636073885
Data columns (total 34 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   phone      14857 non-null  int64  
 1   ccard      14857 non-null  int64  
 2   web        14857 non-null  int64  
 3   visit      14857 non-null  int64  
 4   money      14857 non-null  float64
 5   beacon     14857 non-null  float64
 6   hann       14857 non-null  float64
 7   mass       14857 non-null  float64
 8   newbury    14857 non-null  float64
 9   sweather   14857 non-null  float64
 10  ktop       14857 non-null  float64
 11  kdress     14857 non-null  float64
 12  blouse     14857 non-null  float64
 13  jacket     14857 non-null  float64
 14  crpant     14857 non-null  float64
 15  cspant     14857 non-null  float64
 16  shirt      14857 non-null  float64
 17  dress      14857 non-null  float64
 18  suit       14857 non-null  float64
 19  outwear    14857 non-null  float64
 20  jewel      14857 non-null  float64
 21  fashion    14857 non-null  float64
 22  lwear      14857 non-null  float64
 23  coll       14857 non-null  float64
 24  omon       14857 non-null  float64
 25  tmon       14857 non-null  float64
 26  smon       14857 non-null  float64
 27  gmp        14857 non-null  float64
 28  mdown      14857 non-null  float64
 29  promomail  14857 non-null  int64  
 30  promoresp  14857 non-null  int64  
 31  produnif   14857 non-null  float64
 32  dbetween   14857 non-null  float64
 33  resp       14857 non-null  int64  
dtypes: float64(27), int64(7)
memory usage: 4.0 MB
```

We rescale all the money columns to facilitate the convergence of the logistic regression model. The relevance of normalization for the equation-based models will appear again in later examples.

```
In [3]: df[list(df.columns[4:9]) + list(df.columns[24:27])] = df[list(df.columns[4:9]) + list(df.columns[24:27])]/1000
```

The **conversion rate** is 17%, so we have **class imbalance** here. Moreover, given the cost/benefit analysis suggested in the introduction, we have to pay more attention to the false negatives than to the false positives.

```
In [4]: df['resp'].mean().round(3)
Out[4]: 0.17
```

## Train-test split

We split the data as suggested in question Q1, keeping a 20% of the data for testing. The argument `random_state=0` will be used throughout this example, to ensure the reproducibility of the results obtained.

```
In [5]: from sklearn.model_selection import train_test_split
   ...: df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
```

So, these are the numbers. With 2,972 testing units, the 17% conversion rate gives us about 500 positives in the testing data set.

```
In [6]: df_train.shape, df_test.shape
Out[6]: ((11885, 34), (2972, 34))
```

## Target vectors and features matrices

Next, we define a target vector and a features matrix for both training and test data.

```
In [7]: y_train, X_train = df_train['resp'], df_train.drop(columns='resp')
   ...: y_test, X_test = df_test['resp'], df_test.drop(columns='resp')
```

## Q1. Training the models

Our first candidate will be the logistic regression model. The maximum number of iterations has been set after some trial and error (not shown here). Note that we use only on the training data here.

```
In [8]: from sklearn.linear_model import LogisticRegression
   ...: logclf = LogisticRegression(max_iter=1000, random_state=0)
   ...: logclf.fit(X_train, y_train)
Out[8]: LogisticRegression(max_iter=1000, random_state=0)
```

Second, a decision tree model, with maximum depth 4. As we did in examples ML-08 and ML-11, we specify the cross-entropy as the loss function, in the parameter `criterion`.

```
In [9]: from sklearn.tree import DecisionTreeClassifier
   ...: treeclf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
   ...: treeclf.fit(X_train, y_train)
Out[9]: DecisionTreeClassifier(criterion='entropy', max_depth=4)
```

Third, a random forest model with the same loss function and the same maximum depth. The default for the number of trees (parameter `n_estimators`) is 100, but, following the suggestion in Q1, we have raised this to 200.

```
In [10]: from sklearn.ensemble import RandomForestClassifier
    ...: rfclf = RandomForestClassifier(criterion='entropy', max_depth=4, n_estimators=200, random_state=0)
    ...: rfclf.fit(X_train, y_train)
Out[10]: 
RandomForestClassifier(criterion='entropy', max_depth=4, n_estimators=200,
                       random_state=0)
```

Finally, the gradient boosting model. We set the maximum depth and the number of trees as for the random forest model (there is no default here). The **learning rate** 0.1 is a typical one (the default rate is 0.3). The default loss function (parameter `objective`) is the cross-entropy, so we specify nothing here, to make it shorter, focusing on the number of trees and their size.

```
In [11]: from xgboost import XGBClassifier
    ...: xgbclf = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=200, random_state=0)
    ...: xgbclf.fit(X_train, y_train)
Out[11]: 
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.1, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=4, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=200, n_jobs=None,
              num_parallel_tree=None, random_state=0, ...)
```

## Q2. Predictive scores

To plot the scores, separately for positive and negative training units, we will use the same code as in examples ML-06 and ML-10. First, we import `matplotlib.pyplot` in the usual way.

```
In [12]: from matplotlib import pyplot as plt
```

We pack a code chunk that calculates the scores and plots the histograms as a Python function, for efficiency. 

```
In [13]: def score_plot(mod):
    ...:     score = mod.predict_proba(X_train)[:, 1]
    ...:     # Set the size of the figure
    ...:     plt.figure(figsize=(12,5))
    ...:     # First subplot
    ...:     plt.subplot(1, 2, 1)
    ...:     plt.hist(score[y_train == 1], range=(0,1), color='gray', edgecolor='white')
    ...:     plt.title('Figure a. Scores (positives)')
    ...:     plt.xlabel('Predictive score')
    ...:     # Second subplot
    ...:     plt.subplot(1, 2, 2)
    ...:     plt.hist(score[y_train == 0], range=(0,1), color='gray', edgecolor='white')
    ...:     plt.title('Figure b. Scores (negatives)')
    ...:     plt.xlabel('Predictive score');
```

We calculate now the predictive scores for each model, applying the plotting function just defined to get the histograms. First, the logistic regression model.

```
In [14]: score_plot(logclf)
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/14-1.png)

Based on these histograms, a theshold about 0.2 or a bit less looks reasonable. We take a look now at the scores of the decision tree model.

```
In [15]: score_plot(treeclf)
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/14-2.png)

These histograms look a bit awkward at first sight, but their shape can be explained by the fact that a decision tree model produces a *discrete score*, with as many different values as the number of leaf nodes. This number, with maximum depth 4, cannot exceed 16. Given this discrete structure of the scores, thresholds don't make much sense here.

Next, we take the random forest model, which, compared to the logistic regression model, looks weaker on the true positive side, but better on the false positive side. Note that, by averaging the trees that integrate the forest, we no longer have a discrete score, with just a few different values.

```
In [16]: score_plot(rfclf)
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/14-3.png)

Finally, the gradient boosting model, which looks like the winner. Bit, since we have been warned that gradient boosting models are prone to overfitting, we postpone our conclusions until we have tested this approach.

```
In [17]: score_plot(xgbclf)
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/14-4.png)

## Q3. Testing the models

As we did in the preceding section, we pack here a sequence of calculations as a function, for efficiency. This function will return, foe each model, the true positive and false positive rates, for both the training and the test data. We set the threshold at 0.2 (you can try variations on this).

```
In [17]: def test(mod):
    ...:     score_train, score_test = mod.predict_proba(X_train)[:, 1], mod.predict_proba(X_test)[:, 1]
    ...:     y_pred_train, y_pred_test = score_train > 0.2, score_test > 0.2
    ...:     tp_train = y_pred_train[y_train == 1].mean().round(3)
    ...:     fp_train = y_pred_train[y_train == 0].mean().round(3)
    ...:     tp_test = y_pred_test[y_test == 1].mean().round(3)
    ...:     fp_test = y_pred_test[y_test == 0].mean().round(3)
    ...:     return (tp_train, fp_train), (tp_test, fp_test)
```

In the preceding section, we have trained four models, discarding the decision tree model for the current analysis. When comparing the histograms, the gradient boosting model looks as the top performer on the training data, followed by the other two models. Mind that, while logistic regression is, essentially, a single thing, we can obtain different ensemble models by changing the parameters `max_depth` and `n_estimators`. What we say in this section about these models refers only to the specific choice we have made (`max_depth=4` and `n_estimators=200`). We left further exploration for the homework. 

First, we test the logistic regression model. There is not evidence of overfitting. 

```
In [19]: test(logclf)
Out[19]: ((0.767, 0.267), (0.766, 0.256))
```

Let us repeat the exercise with our random forest model. This seems to be slightly overfitting the training data, but should be confirmed with more splits (changing the `random_state` value). Anyway, compared to the logistic regression model, this model shows a lower true positive rate, so it would be left aside, given the cost/benefit estimates provided.

```
In [20]: test(rfclf)
Out[20]: ((0.727, 0.232), (0.712, 0.229))
```

Finally, the gradient boosting model. Overfitting is evident here. For instance, the true positive rate is 87% on the training data and 74.6% on the test data, which makes a difference from a business perspective. For this model, the performance on the test data should be taken as the only valid evaluation. Given the cost/benefit analysis suggested in the introduction, the comparison between the two models would favor the logistic regression model. You can explore this a bit more in the homework.

```
In [21]: test(xgbclf)
Out[21]: ((0.857, 0.199), (0.727, 0.211))
```

## Homework

1. Try other values of the parameter `random_state` in train/test split, to see the extent to which the results may change. 

2. Switch to `max_depth=5` in the random forest model, to see whether it becomes competitive.

3. Intuition suggests that too may iterations in the gradient boosting process may lead to a model with very performance on the training data but, at the same time, may have a negative impact on the performance on test data. To explore this question, try different values of `n_estimators`, such as 25, 50, 100 and 150, to monitor the overfitting and its potential negative impact on the performance of the gradient boosting model on the test data.

4. Evaluate in dollar terms the models that you find interesting.
