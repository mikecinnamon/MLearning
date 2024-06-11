# [ML-14] Clothing store marketing promotion

## Introduction

A clothing store chain in Boston is designing its marketing campaign, wishing to predict which customers will respond to a **direct mail promotion**. The promotion is going to be based on the existing customers database, which contains information on 14,857 customers. The database, built from a direct mail campaign conducted past year, integrates a collection of fields related to **consumption habits**. There is detailed information on the franchises visited, the money spent, the products purchased, the margin left and participation on sales and promotions. The proportion of respondents to the previous campaign in database (`resp`) is about a 17%.

The campaign designers want to predict the response, but they look at the prediction model with a business perspective: not all the wrong predictions are equally bad, nor all the right predictions equally good. Their **cost/benefit analysis** is based on the following two points:

* The cost of mailing is estimated at $5.00 per promotion unit.

* To estimate the average benefit for the customer that responds to the campaign, they calculate the average spent per visit by the customers in the database, $122.44. Since the average gross margin percentage is about 50% and usually one half of it can be taken as net profit, they set the net profit per respondent at $30.61.

## The data set

The variables included in the data set (file `clothing.csv`) are:

* `id`, a customer ID (encripted).

* `phone` a dummy for having a valid phone number on file.

* `web`, a dummy for being a web shopper.

* `visit`, the number of purchase visits.

* `money`, the total net sales in US dollars.

* `beacon`, the expense in the Beacon Street store in US dollars.

* `hann`, the expense in the Hannover St store in US dollars.

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

Q1. After leaving aside a 20% of the data for testing, train the following classifiers: (a) a logistic regression model, (b) a decision tree model with maximum depth 4, (c) a random forest model with the same maximum depth, and (d) a grading boosting model with the same maximum depth. For the ensemble models, use a number of trees high enough to allow for overfitting, it it occurs.

Q2. Calculate a vector of **predictive scores** for each model and compare the distributions, taking into account the cost-benefit analysis proposed in the introduction.

Q3. Select the best two models and use them to calculate predictive scores for the test data. Apply a **threshold** of your choice to the scores, and calculate the corresponding **confusion matrices**. Which model is better? Why?

## Importing the data

As in the preceding examples, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. In this case, we take the column `id` as the index (`index_col=0`).

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
   ...: df = pd.read_csv(path + 'clothing.csv', index_col=0)
```

## Exploring the data

We print a report of the content of `df` with the method `.info()`. Everything is as expected, so far. There are no missing values.

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

The **conversion rate** is 17%, so we have **class imbalance** here. Moreover, given the cost/benefit analysis suggested in the introduction, we have to pay more attention to false negatives than to false positives.

```
In [3]: df['resp'].mean().round(3)
Out[3]: 0.17
```

## Train-test split

We split the data as suggested in question Q1, leaving a 20% of the data for testing.

```
In [4]: from sklearn.model_selection import train_test_split
   ...: df_train, df_test = train_test_split(df, test_size=0.2)
```

So, these are the numbers. With 2,972 testing units, the 17% conversion rate gives us about 500 positives in the testing data set.

```
In [5]: df_train.shape, df_test.shape
Out[5]: ((11885, 34), (2972, 34))
```

## Target vectors and feature matrices

Next, we define a target vector and a feature matrix for both training and test data.

```
In [6]: y_train, X_train = df_train['resp'], df_train.drop(columns='resp')
   ...: y_test, X_test = df_test['resp'], df_test.drop(columns='resp')
```

## Q1. Training the four models

Our first candidate will be a logistic regression model. The maximum number of iterations has been set after some trial and error.

```
In [7]: from sklearn.linear_model import LogisticRegression
   ...: logclf = LogisticRegression(max_iter=5000)
   ...: logclf.fit(X_train, y_train)
Out[7]: LogisticRegression(max_iter=5000)
```

Second, a decision tree model, with maximum depth 4. As we did in example ML-08, we specify the cross-entropy as the loss function, by means of the parameter `criterion`.

```
In [8]: from sklearn.tree import DecisionTreeClassifier
   ...: treeclf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
   ...: treeclf.fit(X_train, y_train)
Out[8]: DecisionTreeClassifier(criterion='entropy', max_depth=4)
```

Third, a random forest model with the same loss function and the same maximum depth. The default for the number of trees (parameter `n_estimators`) is 100, but, following the suggestion of question Q1, we raise this to 200.

```
In [9]: from sklearn.ensemble import RandomForestClassifier
   ...: rfclf = RandomForestClassifier(criterion='entropy', max_depth=4, n_estimators=200)
   ...: rfclf.fit(X_train, y_train)
Out[9]: RandomForestClassifier(criterion='entropy', max_depth=4, n_estimators=200)
```

Finally, the gradient boosting model. We set the maximum depth and the number of trees as for the random forest model (no defaults). The **learning rate** 0.1 is a typical one (no default). The deafult loss function (parameter `objective`) is the cross-entropy, so we specify nothing here, to make it simpler, focusing on the number of trees and their size.

```
In [10]: from xgboost import XGBClassifier
    ...: xgbclf = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=200)
    ...: xgbclf.fit(X_train, y_train)
Out[10]: 
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
              num_parallel_tree=None, random_state=None, ...)
```

## Plotting function

To plot the scores, separately for positive and negative training units, we will use the same code as in examples ML-06 and ML-10. First, we import `matplotlib.pyplot` in the usual way.

```
In [11]: from matplotlib import pyplot as plt
```

We pack the code chunk for the plots as a Python function, so we will not repeat the same again and again. 

```
In [12]: def score_plot(score):
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

## Q2. Predictive scores

```
In [13]: log_score = logclf.predict_proba(X_train)[:, 1]
    ...: score_plot(log_score)
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/14-1.png)

```
In [14]: tree_score = treeclf.predict_proba(X_train)[:, 1]
    ...: score_plot(tree_score)
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/14-2.png)

```
In [15]: rf_score = rfclf.predict_proba(X_train)[:, 1]
    ...: score_plot(rf_score)
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/14-3.png)

```
In [16]: xgb_score = xgbclf.predict_proba(X_train)[:, 1]
    ...: score_plot(xgb_score)
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/14-4.png)

## Q3. Testing

```
In [17]: log_score_train, log_score_test = logclf.predict_proba(X_train)[:, 1], logclf.predict_proba(X_test)[:, 1]
    ...: y_pred_train, y_pred_test = log_score_train > 0.2, log_score_test > 0.2
    ...: conf_train, conf_test = pd.crosstab(y_train, y_pred_train), pd.crosstab(y_test, y_pred_test)
    ...: conf_train, conf_test
Out[17]: 
(col_0  False  True 
 resp               
 0       7240   2612
 1        464   1569,
 col_0  False  True 
 resp               
 0       1752    724
 1        106    390)
```

```
In [18]: xgb_score_train, xgb_score_test = xgbclf.predict_proba(X_train)[:, 1], xgbclf.predict_proba(X_test)[:, 1]
    ...: y_pred_train, y_pred_test = xgb_score_train > 0.2, xgb_score_test > 0.2
    ...: conf_train, conf_test = pd.crosstab(y_train, y_pred_train), pd.crosstab(y_test, y_pred_test)
    ...: conf_train, conf_test
Out[18]: 
(col_0  False  True 
 resp               
 0       7892   1960
 1        265   1768,
 col_0  False  True 
 resp               
 0       1836    640
 1        126    370)
```
