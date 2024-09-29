# [ML-06] Example - The churn model

## Introduction

The term **churn** is used in marketing to refer to a customer leaving the company in favor of a competitor. Churning is a common concern of **Customer Relationship Management** (CRM). A key step in proactive churn management is to predict whether a customer is likely to churn, since an early detection of the potential churners helps to plan the retention campaigns.

This example presents a churn model based on a **logistic regression model**, for a company called *Omicron Mobile*, which provides mobile phone services. The data set is based on a random sample of 5,000 customers whose accounts were still alive by September 30, and have been monitored during the fourth quarter. 968 of those customers churned during the fourth quarter, a **churning rate** of 19.4%.

## The data set

The variables included in the data set (file `churn.csv`) are:

* `id`, a customer ID (the phone number).

* `aclentgh`, the number of days the account has been active at the beginning of the period monitored.

* `intplan`, a dummy for having an international plan.

* `dataplan`, a dummy for having a data plan.

* `ommin`, the total minutes call to any Omicron mobile phone number, voicemail or national landline.

* `omcall`, the total number of calls to any Omicron mobile phone number, voicemail or national landline.

* `otmin`, the total minutes call to other mobile networks.

* `otcall`, the total number of calls to other networks.

* `ngmin`, the total minutes call to nongeographic numbers. Nongeographic numbers, such as UK numbers 0844 or 0871, are often helplines for organizations like banks, insurance companies, utilities and charities.

* `ngcall`, the total number of calls to nongeographic numbers.

* `imin`, the total minutes in international calls.

* `icall`, the total international calls.

* `cuscall`, the number of calls to customer service.

* `churn`, a dummy for churning.

All the data are from the third quarter except the last variable.

Source: MA Canela, I Alegre & A Ibarra (2019), *Quantitative Methods for Management*, Wiley.

## Questions

Q1. Develop a logistic regression model to calculate a **churn score**, that is, an estimate of the probability of churning, for each customer.

Q2. How is the distribution of churn scores? Is it different for the churners and the non-churners?

Q3. Set an adequate **threshold** for the churn score and apply it to decide which customers are potential churners. What is the **true positive rate**? And the **false positive rate**?

## Importing the data

As in the preceding example, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. We take the first column in the source file (`id`) as the index.

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'churn.csv', index_col=0)
```

## Exploring the data

`df` is a Pandas data frame. In the report printed by the method `.info()`, we don't find anything unexpected. There are no missing values.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 5000 entries, 409-8978 to 444-8504
Data columns (total 13 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   aclength  5000 non-null   int64  
 1   intplan   5000 non-null   int64  
 2   dataplan  5000 non-null   int64  
 3   ommin     5000 non-null   float64
 4   omcall    5000 non-null   int64  
 5   otmin     5000 non-null   float64
 6   otcall    5000 non-null   int64  
 7   ngmin     5000 non-null   float64
 8   ngcall    5000 non-null   int64  
 9   imin      5000 non-null   float64
 10  icall     5000 non-null   int64  
 11  cuscall   5000 non-null   int64  
 12  churn     5000 non-null   int64  
dtypes: float64(4), int64(9)
memory usage: 546.9+ KB
```

## Q1. Logistic regression model

We use scikit-learn to obtain our logistic regression model. We start by creating a **target vector** and a **feature matrix**. The target vector is the last column (`churn`), and the feature matrix is made of the other columns.

```
In [3]: y = df['churn']
   ...: X = df.drop(columns='churn')
```

Alternatively, we could have used `.iloc` specifications here. Next, we import the **estimator class** `LogisticRegression()` from the scikit-learn subpackage `linear_model`, instantiating an estimator from this class, which we calling `clf` (to remind us that it is a classifier). Instead of accepting the default parameter values, as we did in example ML-04, we increase the **maximum number of iterations**. Using the default `max_iter=100` would have raised a warning indicating that the optimization process has not converged.

```
In [4]: from sklearn.linear_model import LogisticRegression
   ...: clf = LogisticRegression(max_iter=1500)
```

The method `.fit()` works as in linear regression, finding the optimal equations.

```
In [5]: clf.fit(X, y)
Out[5]: LogisticRegression(max_iter=1500)
```

For a classification model, the method `.score()` returns the **accuracy**, which is the proportion of right prediction:

```
In [6]: clf.score(X, y).round(3)
Out[6]: 0.842
```

At first sight, 84.2% of rigth prediction may look like a feat, but not so if we take into account, the degree of **class imbalance** in these data. With only 19.4% positive cases, 80.6% accuracy can be obtained in a trivial way. So let us take a closer look at the performance of this model.

As given by the method `.predict(), the `**predicted target values** are obtained as follows:

* **Class probabilities** are calculated for each training unit. In this example, this means two complementary probabilities, one for churning (`y == 1`) and one for not churning (`y == 0`). These probabilities can be extracted with the method `.predict_proba()`.

* For every unit, the predicted target value is the one with higher probability. In scikit-learn, the class probabilities are extracted as:

```
In [7]: clf.predict_proba(X)
Out[7]: 
array([[0.95309927, 0.04690073],
       [0.96888113, 0.03111887],
       [0.72661291, 0.27338709],
       ...,
       [0.87123317, 0.12876683],
       [0.40292348, 0.59707652],
       [0.17271608, 0.82728392]])

```

Mind that Python sorts the classes alphabetically. In the binary case, this means that the negative class comes first. The probability of the positive class is taken as a **predictive score** (as in credit scoring). Then the predicted class is chosen based on a **threshold value**: the predicted class is positive when the score exceeds the threshold, and negative otherwise. 

The scores are extracted as:

```
In 8]: df['score'] = clf.predict_proba(X)[:, 1]
```

Note that we have added the scores as a column to our data set, which is just an option, since we can also manage it as a separate vector. The actual class and the predictive score are now the last two columns in `df`.

```
In [9]: df[['churn', 'score']]
Out[9]: 
          churn     score
id                       
409-8978      0  0.046901
444-7077      0  0.031119
401-9132      0  0.273387
409-2971      0  0.131950
431-5175      0  0.068075
...         ...       ...
390-2408      0  0.573438
407-6398      0  0.267838
444-7620      1  0.128767
352-4885      1  0.597077
444-8504      1  0.827284

[5000 rows x 2 columns]
``` 

## Q2. Distribution of the churn scores

We can visualize the distribution of the predictive scores through a histogram. In this case, we plot separately the scores for the churners (968) and the non-churners (4,032).

We import `matplotlib.pyplot` as usual:

```
In [10]: from matplotlib import pyplot as plt
```

You can find see in `In [11]` a code chunk for plotting the two histograms side-by-side. The `plt.figure()` line specifies the total size of the figure. Then, `plt.subplot(1, 2, 1)` and `plt.subplot(1, 2, 2)` start the two parts of this chunk, one for each subplot. These parts are easy to read after our previous experience with the histogram in example ML-04. The argument `range=(0,1)` is used to get intervals of length 0.1 (the default of ´hist()´ split the range of the data in 10 intervals), which are easier to read. The argument `edgecolor=white` improves the picture. 

Note that `plt.subplot(1, 2, i)` refers to the $i$-th subplot in a grid of one row and two columns. The subplots are ordered by row, from left to righ and from top to bottom.

```
In [11]: # Set the size of the figure
    ...: plt.figure(figsize=(12,5))
    ...: # First subplot
    ...: plt.subplot(1, 2, 1)
    ...: plt.hist(df['score'][y == 1], range=(0,1), color='gray', edgecolor='white')
    ...: plt.title('Figure 1.a. Scores (churners)')
    ...: plt.xlabel('Churn score')
    ...: # Second subplot
    ...: plt.subplot(1, 2, 2)
    ...: plt.hist(df['score'][y == 0], range=(0,1), color='gray', edgecolor='white')
    ...: plt.title('Figure 1.b. Scores (non-churners)')
    ...: plt.xlabel('Churn score');
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/06-1.png)

You can now imagine the threshold as a vertical line, and move it, right or left from the default threshold value 0.5. The customers falling on the right of that vertical line would be classified as positive, and those falling on the left as negative.

## Q3. Set a threshold for the churn scores

The method `.predict()` uses the default threshold. The predicted class can be obtained as:

```
In [12]: y_pred = clf.predict(X)
```

It is plainly seen in Figure 1.a, that, in this way, we are missing more than one half of the churners. So, in spite of its accuracy, our model would not be adequate for the actual business application. 

The **confusion matrix**, resulting from the cross tabulation of the actual and the predicted target values, confirms this visual intuition. Confusion matrices can be obtained in many ways. For instance, with the function `confusion_matrix` of the scikit-learn subpackage `metrics`:

```
In [13]: from sklearn.metrics import confusion_matrix
    ...: confusion_matrix(y, y_pred)
Out[13]: 
array([[3897,  135],
       [ 656,  312]], dtype=int64)
```

Alternatively, this matrix could be obtained with the Pandas function `crosstab()`. Note that scikit-learn returns the confusion matrix as a NumPy 2D array, while Pandas would have returned it as a Pandas data frame. 

The accuracy returned by the method `.score()` is the sum of the diagonal terms of this matrix divided by the sum of all terms of the matrix. It can also be calculated directly:

```
In [14]: (y == y_pred).mean().round(3)
Out[14]: 0.842
```

As we guessed from the histogram, our churn model is not capturing enough churners (304/968) for a business application. To predict more positives, we have to lower the threshold. Figure 1.a suggests that we have to go down to about 0.2 to make a real difference, while Figure 1.b warns us against lowering it further. So, let us try 0.2. The new vector of predicted clases is then obtained as:

```
In [15]: y_pred = (df['score'] > 0.2).astype(int)
```

The new confusion matrix is:

```
In [16]: confusion_matrix(y, y_pred)
Out[16]: 
array([[3164,  868],
       [ 343,  625]], dtype=int64)
```

Indeed, we are capturing now about 2/3 of the churners. This comes at the price of raising the false positives to 866, which affects the accuracy:

```
In [17]: (y == y_pred).mean().round(3)
Out[17]: 0.758
```

A clear way to summarize the evaluation of the model comes through the true positive and false positive rates. They can be extracted from the confusion matrix or calculated directly. The **true positive rate** is the proportion of predicted positives among the actual positives:

```
In [18]: y_pred[y == 1].mean().round(3)
Out[18]: 0.646
```

The **false positive rate** is the proportion of predicted positives among the actual negatives:

```
In [19]: y_pred[y == 0].mean().round(3)
Out[19]: 0.215
``` 

### Homework

1. In this data set, we find a mix of scales that can be visualized with the method `.describe()`. This slowns down convergence in the  method `.fit()`, so we had the increase the parameter `max_iter`. This is not relevant in a model as simple as the one used in this example, but it will be in the complex that will appear later in this course. Try different values for `max_iter` in the specification of the `LogisticRegression()` and examine how the maximum number of iterations affects the model accuracy in this case.

2. Rescale all the features which are not dummies and train the logistic regression classifier with the deafult number of iterations. Do you get a warning about non-convergence now?

3. Assume that the Omicron management plans to offer a **20% discount** to the customers that the model classifies as potential churners, and that this offer is going to have a 100% success, so the company will retain all the churners detected. Evaluate the benefit produced by this **retention policy** with the two models presented in this example.

4. Define a Python function which gives the benefit in terms of the threshold and find an **optimal threshold** for this retention policy.
