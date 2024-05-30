# [ML-03] Linear regression

## Linear regression

In machine learning, the term **regression** applies to the prediction of a numeric target ($Y$) in terms of a collection of features ($X_1, X_2, \dots, X_k$). In scikit-learn, the features must be numeric (or Boolean). We have already seen in the preceding lecture (ML-02) how to deal with categorical features. This will be illustrated in the example that follows (ML-04). 

Regression models are not necessarily related to a mathematical equation, as in statistical analysis, although an equation is the first idea that comes to our mind as a potential  predictive model. When the model is based on a linear equation, such as
$$Y = b_0 + b_1X_1 + b_2X_2 + \cdots + b_kX_k,$$
we have **linear regression**. Statisticians call $b_0$ intercept and $b_1, b_2, \dots, b_k$ regression coefficients. In machine learning, the bias is usually called **bias**, while the regression coefficients are called **weights**.

Though the predictions of a linear regression model can usually be improved with more advanced techniques, most analysts start there, because it helps them to understand the data. This may be interesting in some applications, in which the **interpretability** of the model is relevant. 

## Prediction error

In general, regression models are evaluated through their **prediction errors**. The basic schema is
$$\textrm{Prediction\ error} = \textrm{Actual\ value} - \textrm{Predicted\ value}.$$

In a linear regression context, prediction errors are called **residuals**. In the standard approach to linear regression, the regression coefficients are calculated so that the **mean squared error** (MSE) is minimum. This is called the **least squares method**. The errors of a linear equation obtained by means of the least squares method have an important property, that their sum is zero, which is no longer true in regression models obtained in other ways. 

## R-squared

The **coefficient of determination** is a popular metric for the evaluation of regression models. It is given by the formula
$$R^2 = 1 - \displaystyle \frac{\textrm{MSE}}{{\textrm{MSY}}}\thinspace,$$ 
in which $MSY$ is the average squared centered target value (subtracting the mean). It can be proved that $0 \le R^2 \le 1$. The maximum is $R^2 = 1$, which is equivalent to a null $MSE$, which happens when all the errors are zero, and all the predicted values are exactly equal to the corresponding actual values.

Statistics textbooks explain the coefficient of determination as the **percentage of variance** explained by the regression equation. Also, it coincides with the square of the correlation between the actual and the predicted values, called the **multiple correlation**. This explains the notation, and the name **R-squared statistic**, given to the coefficient of determination. Note that, for a given training data set, $MSY$ is fixed, so minimizing $MSE$ is equivalent to maximixing the correlation between actual and predicted values.

A word of caution. Correlation and R-squared are sensitive to extreme values, not unfrequent in real-world data. So, when the target has a skewed distribution, with a long right tail, a strong correlation may suggest that the model is better than it really is. The example that follows (ML-04) illustrates this point.

## Evaluating a regression model

The interpretation of the coefficient of determination as a squared correlation is no longer valid in regression models which are nonlinear or that, though being linear, are not obtained by the least squares method. Though it is presented as the standard metric for regression models in many sources, in a business application one should evaluate the prediction errors in a simple, direct way. Two popular metrics are the mean absolute error and the mean absolute percentage error, whose interpretation is completely straightforward.

Suppose that `y` is the target vector in the training data set, and `y_pred` is the predicted target vector. The **mean absolute error** can be obtained directly in Pandas as

```
mae = (y - y_pred).abs().mean()
```

The **mean absolute percentage error** would be

```
mape = ((y - y_pred)/y).abs().mean()
```

The mean can be replaced by the median to get a more robust metric. The scikit-learn subpackage `metrics` provides many functions to calculate regression metrics, in particular `mean_abolute_error()` and `mean_abolute_percentage_error()`.

A final word. These metrics allow for a quick comparison of regression models, which is needed in model selection. But a different question is whether the model selected would be good enough for the intended application. At that level of the analysis, a visualization of the prediction errors such as a histogram or a scatter plot can be useful.

