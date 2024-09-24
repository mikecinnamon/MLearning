# [ML-04] Example - House sales in King County

## Introduction

This example illustrates linear regression in scikit-learn. We develop a model for predicting **house sale prices** in King County (Washington), which includes Seattle. King is the most populous county in Washington (population 1,931,249 in the 2010 census), and the 13th-most populous in the United States. The data include the homes sold between May 2014 and May 2015.

## The data set

The data come in the file `king.csv`. It contains 13 house features plus the sale price and date, along with 21,613 observations.

The variables are:

* `id`, an identifier of the house.

* `date`, the date when the sale took place.

* `zipcode`, the ZIP code of the house.

* `lat`, the latitude of the house.

* `long`, the longitude of the house.

* `bedrooms`, the number of bedrooms.

* `bathrooms`, the number of bathrooms.

* `sqft_above`, the square footage of the house, discounting the basement.

* `sqft_basement`, the square footage of the basement.

* `sqft_lot`, the square footage of the lot.

* `floors`, the total floors (levels) in house.

* `waterfront`, a dummy for having a view to the waterfront.

* `condition`, a 1-5 rating.

* `yr_built`, the year when the house was built.

* `yr_renovated`, the year when the house was renovated.

* `price`, the sale price.

Source: Kaggle.

## Questions

Q1. How is the distribution of the sale price?

Q2. Develop a linear regression model for predicting the sale price in terms of the house features, leaving aside the zipcode. Evaluate this model.

Q3. Plot the actual price versus the price predicted by the model. What do you see?

Q4. Add a dummy for every zipcode to the feature collection and run the analysis again. What happened?

## Importing the data

Although scikit-learn is described in the technical documentation as managing the data in NumPy array format, you can equally input data in Pandas format. Nevertheless, even if scikit-learn estimators can take Pandas data containers, they always return NumPy arrays. 

Using Pandas format makes processing slower, but importing the data and adapting them for the learning process is easier. But, note that in machine learning preprocessing is a previous step. The learning step is made when the data are already clean and prepared (no missing values, no duplicates, binary columns for every possible outcome of a categorical feature, etc), so the Pandas tools are no longer needed. 

We use here the Pandas function `read_csv()` to import the data. First, we import the Pandas library:

```
In [1]: import pandas as pd
```

In the examples of this course, the source files are stored in a GitHub repository, so we can use a remote path to get access. The path will always the same.

```
In [2]: path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
```

In this case, we take the column `id` as the **index** (this is the role of the argument `index_col=0`). 

```
In [3]: df = pd.read_csv(path + 'king.csv', index_col=0)
````

## Exploring the data

`df` is a Pandas data frame. A report of the content, printed by the method `.info()`, does not show anything wrong. Note that there are no **missing values**. Also, note that `id` is not one the columns of this data frame (it is the index).

```
In [4]: df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 21613 entries, 7129300520 to 1523300157
Data columns (total 15 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   date           21613 non-null  object 
 1   zipcode        21613 non-null  int64  
 2   lat            21613 non-null  float64
 3   long           21613 non-null  float64
 4   bedrooms       21613 non-null  int64  
 5   bathrooms      21613 non-null  float64
 6   sqft_above     21613 non-null  int64  
 7   sqft_basement  21613 non-null  int64  
 8   sqft_lot       21613 non-null  int64  
 9   floors         21613 non-null  float64
 10  waterfront     21613 non-null  int64  
 11  condition      21613 non-null  int64  
 12  yr_built       21613 non-null  int64  
 13  yr_renovated   21613 non-null  int64  
 14  price          21613 non-null  int64  
dtypes: float64(4), int64(10), object(1)
memory usage: 2.6+ MB
```

A different view of the data is provided by the method `.head()`, which extracts the first (five) rows.

```
In [5]: df.head()
Out[5]: 
                       date  zipcode      lat     long  bedrooms  bathrooms   
id                                                                            
7129300520  20141013T000000    98178  47.5112 -122.257         3       1.00  \
6414100192  20141209T000000    98125  47.7210 -122.319         3       2.25   
5631500400  20150225T000000    98028  47.7379 -122.233         2       1.00   
2487200875  20141209T000000    98136  47.5208 -122.393         4       3.00   
1954400510  20150218T000000    98074  47.6168 -122.045         3       2.00   

            sqft_above  sqft_basement  sqft_lot  floors  waterfront   
id                                                                    
7129300520        1180              0      5650     1.0           0  \
6414100192        2170            400      7242     2.0           0   
5631500400         770              0     10000     1.0           0   
2487200875        1050            910      5000     1.0           0   
1954400510        1680              0      8080     1.0           0   

            condition  yr_built  yr_renovated   price  
id                                                     
7129300520          3      1955             0  221900  
6414100192          3      1951          1991  538000  
5631500400          3      1933             0  180000  
2487200875          5      1965             0  604000  
1954400510          3      1987             0  510000  
```
 
 We rescale the sale price to the thousands, to have simpler numbers. 

```
In [6]: df['price'] = df['price']/1000
```

## Q1. Distribution of the sale price

The distribution of a numeric series can be quickly explored in two ways. First, the method `.describe()` extracts a statistical summary. Here, the maximum price suggests that we may have a long right tail, which can be expected in real estate prices. This is a trait which statisticians call positive **skewness**.

```
In [7]: df['price'].describe()
Out[7]: 
count    21613.000000
mean       540.088142
std        367.127196
min         75.000000
25%        321.950000
50%        450.000000
75%        645.000000
max       7700.000000
Name: price, dtype: float64
```

Second, we can use a **histogram**. This histogram can be obtained directly in Pandas, with shorter code. Nevertheless, to maintain the uniformity along this course, we use `matplotlib.pyplot`. In some examples, this approach will allow us a better control of the graphical output. The histogram confirms our guess about the skewness of the distribution.

First, we import `matplotlib.pyplot`.

```
In [8]: from matplotlib import pyplot as plt
```

Now, we create the visualization. The size has been chosen for a good fit in a webpage. The argument `edgecolor='white'` creates the white lines separating the bars, which improves the visualization. The default of the function `hist()` takes the same value for both parameters `color` and `edgecolor`. The final semicolon stops some irrelevant output to be printed.

```
In [9]: plt.figure(figsize=(7,5))
   ...: plt.title('Figure 1. Actual price')
   ...: plt.hist(df['price'], color='gray', edgecolor='white')
   ...: plt.xlabel('Sale price (thousands)');
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/04-1.png)

## Q2. Linear regression equation

We use scikit-learn to obtain our regression models (not the only way in Python), so we create a **target vector** and a **feature matrix**. The target vector is the last column (`price`) and the feature matrix contains the other columns minus `date` and `zipcode`.

```
In [10]: y = df.iloc[:, -1]
    ...: X = df.iloc[:, 2:-1]
```

Alternatively, you can use the names of the columns, setting `y = df['price']` and `X = df.drop(columns=['date', 'zipcode', 'price'])`. Now, we import the **estimator class** `LinearRegression()` from the subpackage `linear_model`. 

```
In [11]: from sklearn.linear_model import LinearRegression
```

We create an instance of this class, calling it `reg`, to remind us of the job it does. 

```
In [12]: reg = LinearRegression()
```

The method `.fit()` calculates the optimal equation, that is, the parameter values for which the **loss** is minimum. Since we are using the default of `LinearRegression()`, which is **least squares** regression, the loss function is the MSE (mean squared error).

```
In [13]: reg.fit(X, y)
Out[13]: LinearRegression()
```

The predicted prices for the houses included in the training data set are then calculated with the method `.predict()`.

```
In [14]: y_pred = reg.predict(X)
```

Finally, we obtain a preliminary evaluation of the model with the method `.score()`.

```
In [15]: reg.score(X, y).round(3)
Out[15]: 0.646
```

This gives us a R-squared value of 0.646. Since this is least squares regression, we can interpret it as a squared correlation. So, the correlation between actual prices (`y`) and predicted prices (`y_pred`) is the square root, 0.804.

## Q3. Plot the actual price versus the price predicted by your model

We use again `matplolibt.pyplot` to create this **scatter plot**. The argument `s=2` controls the size of the dots. The choice of the size takes into account the number of dots, sometimes after a bit of trial and error.

```
In [16]: plt.figure(figsize=(5,5))
    ...: plt.title('Figure 2. Actual price vs predicted price')
    ...: plt.scatter(x=y_pred, y=y, color='black', s=1)
    ...: plt.xlabel('Predicted price (thousands)')
    ...: plt.ylabel('Actual price (thousands)');
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/04-2.png)

This type of visualization helps to understand the data, and to detect undesired effects. In this case, we see that, in spite of the strong correlation, the prediction error can be big. This could be expected, since the correlation only ensures an average predictive performance, and we have more than 20,000 data units.

Paying a bit more of attention, we can see that the biggest errors (in absolute value) happen in the most expensive houses. This is also a well known fact: the bigger what you measure, the bigger the measurement errors. We can visualize the situation with a scatter plot.

```
In [17]: plt.figure(figsize=(5,5))
    ...: plt.title('Figure 3. Absolute prediction error vs predicted price')
    ...: plt.scatter(x=y_pred, y=abs(y-y_pred), color='black', s=1)
    ...: plt.xlabel('Predicted price (thousands)')
    ...: plt.ylabel('Absolute predicted error (thousands)');
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/04-3.png)

Another issue is that some of the predicted prices are negative. We can count them:

```
In [18]: (y_pred < 0).sum()
Out[18]: 38
```

This may look pathological to you, but it is not rare in this type of data. Since the average error is null (this is a property of least squares), we have, more or less, the same amount of positive and negative errors. When a cheap house has a negative and substantial error, the predicted price can be negative. A different thing is the isolated point that we observe on the left of the two above figures. Something is wrong in this house.

## Q4. Dummies for the zipcodes

Since we are going to add the zipcode to the equation, we drop the longitude and the latitude, packing the remaining features in a matrix:

```
In [19]: X1 = df.iloc[:, 4:-1]
```

To create the dummies, we use the Pandas function `get_dummies()`, which returns the dummies as the columns of a data frame. The data type is `bool`. 

```
In [20]: X2 = pd.get_dummies(df['zipcode'])
```

Now, `X2` has 70 columns (as many as different zipcodes in the data set). The column names are the zipcode values. With `get_dummies()`, the dummy columns get names, so you know what is what. The drawback is that, when a categorical feature is numeric, the column names are numbers, which is not accepted by scikit-learn. We will handle this below.

```
In [21]: X2.head()
Out[21]: 
            98001  98002  98003  98004  98005  98006  98007  98008  98010   
id                                                                          
7129300520  False  False  False  False  False  False  False  False  False  \
6414100192  False  False  False  False  False  False  False  False  False   
5631500400  False  False  False  False  False  False  False  False  False   
2487200875  False  False  False  False  False  False  False  False  False   
1954400510  False  False  False  False  False  False  False  False  False   

            98011  ...  98146  98148  98155  98166  98168  98177  98178   
id                 ...                                                    
7129300520  False  ...  False  False  False  False  False  False   True  \
6414100192  False  ...  False  False  False  False  False  False  False   
5631500400  False  ...  False  False  False  False  False  False  False   
2487200875  False  ...  False  False  False  False  False  False  False   
1954400510  False  ...  False  False  False  False  False  False  False   

            98188  98198  98199  
id                               
7129300520  False  False  False  
6414100192  False  False  False  
5631500400  False  False  False  
2487200875  False  False  False  
1954400510  False  False  False  

[5 rows x 70 columns]
```

With the Pandas function `concat()`, we join the two parts of the new feature matrix (you can also do this with the methods `.merge()` or `.join()`). The argument `axis=1` indicates that the two submatrices are joined horizontally (the default is to join vertically).

```
In [22]: X = pd.concat([X1, X2], axis=1)
```

Indeed, the new matrix has the right shape:

```
In [23]: X.shape
Out[23]: (21613, 80)
```

To prevent the trouble with the column names, we convert `X` to a NumPy 2D array:

```
In [24]: X = X.values
```

Now, we fit a regression equation to the new data set. This replaces the former model by a new one, which takes 80 features instead of 12. We could instantiate a new estimator with a different name, keeping both models alive, but in this example we just update the existing estimator with the new feature matrix.

```
In [25]: reg.fit(X, y)
Out[25]: LinearRegression()
```

The new predictions are:

```
In [26]: y_pred = reg.predict(X)
```

And the new R-squared value:

```
In [27]: reg.score(X, y).round(3)
Out[27]: 0.785
```

This looks like a relevant improvement, compared to the former model. The scatter plot illustrates the improvement in the correlation:

```
In [28]: plt.figure(figsize=(5,5))
    ...: plt.title('Figure 4. Actual price vs predicted price')
    ...: plt.scatter(x=y_pred, y=y, color='black', s=1)
    ...: plt.xlabel('Predicted price (thousands)')
    ...: plt.ylabel('Actual price (thousands)');
```

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/04-4.png)

There are still negative predicted prices, though this has improved:

```
In [29]: (y_pred < 0).sum()
Out[29]: 16
```

## Homework

1. The role of longitude and latitude in the prediction of real estate prices is unclear. Do they really contribute to get better predictions in the first model of this example? If we keep them in the second model, do we get a better model? 

2. Evaluate in dollar terms the predictive performance of the two models presented in this example. For instance, you can use the mean (or median) absolute error. Setting a threshold which makes sense for housing prices (such as $100,000, or $200,000), can you make a statement about the percentage of houses whose price can be predicted with an error below the threshold?

3. Is it better to use the percentage error in the above assessment? Setting the threshold in percentage terms, can you make a statement about the percentage of houses with a prediction error below the threshold?

4. Can the strong correlation obtained for the models of this example be an artifact created by the extreme values? Trim the data set, dropping the houses beyond a certain threshold of price and/or size. Do you get a better model?

5. The distribution of the price is quite skewed, which is a fact of life in real state. The extreme values in the right tail of the distribution can exert an undesired influence on the regression coefficients. Develop and evaluate a model for predicting the price that is based on a linear regression equation which has the logarithm of the price on the left side. 
