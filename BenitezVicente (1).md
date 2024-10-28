# Vicente Benitez Lorente
# Assignment number 2

## Importing the data


```python
import pandas as pd
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'deposit.csv', index_col=0)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 45211 entries, 2065031284 to 2086934257
    Data columns (total 35 columns):
     #   Column               Non-Null Count  Dtype
    ---  ------               --------------  -----
     0   age                  45211 non-null  int64
     1   job_admin            45211 non-null  int64
     2   job_blue-collar      45211 non-null  int64
     3   job_entrepreneur     45211 non-null  int64
     4   job_housemaid        45211 non-null  int64
     5   job_management       45211 non-null  int64
     6   job_retired          45211 non-null  int64
     7   job_self-employed    45211 non-null  int64
     8   job_services         45211 non-null  int64
     9   job_student          45211 non-null  int64
     10  job_technician       45211 non-null  int64
     11  job_unemployed       45211 non-null  int64
     12  job_unknown          45211 non-null  int64
     13  marital_divorced     45211 non-null  int64
     14  marital_married      45211 non-null  int64
     15  marital_single       45211 non-null  int64
     16  education_primary    45211 non-null  int64
     17  education_secondary  45211 non-null  int64
     18  education_tertiary   45211 non-null  int64
     19  education_unknown    45211 non-null  int64
     20  default              45211 non-null  int64
     21  balance              45211 non-null  int64
     22  housing              45211 non-null  int64
     23  loan                 45211 non-null  int64
     24  channel_cellular     45211 non-null  int64
     25  channel_telephone    45211 non-null  int64
     26  channel_unknown      45211 non-null  int64
     27  duration             45211 non-null  int64
     28  pdays                45211 non-null  int64
     29  previous             45211 non-null  int64
     30  poutcome_failure     45211 non-null  int64
     31  poutcome_other       45211 non-null  int64
     32  poutcome_success     45211 non-null  int64
     33  poutcome_unknown     45211 non-null  int64
     34  deposit              45211 non-null  int64
    dtypes: int64(35)
    memory usage: 12.4 MB
    


```python
print(df.head())
```

                age  job_admin  job_blue-collar  job_entrepreneur  job_housemaid  \
    accnum                                                                         
    2065031284   58          0                0                 0              0   
    2051283096   44          0                0                 0              0   
    2029034586   33          0                0                 1              0   
    2070859436   47          0                1                 0              0   
    2098635102   33          0                0                 0              0   
    
                job_management  job_retired  job_self-employed  job_services  \
    accnum                                                                     
    2065031284               1            0                  0             0   
    2051283096               0            0                  0             0   
    2029034586               0            0                  0             0   
    2070859436               0            0                  0             0   
    2098635102               0            0                  0             0   
    
                job_student  ...  channel_telephone  channel_unknown  duration  \
    accnum                   ...                                                 
    2065031284            0  ...                  0                1       261   
    2051283096            0  ...                  0                1       151   
    2029034586            0  ...                  0                1        76   
    2070859436            0  ...                  0                1        92   
    2098635102            0  ...                  0                1       198   
    
                pdays  previous  poutcome_failure  poutcome_other  \
    accnum                                                          
    2065031284     -1         0                 0               0   
    2051283096     -1         0                 0               0   
    2029034586     -1         0                 0               0   
    2070859436     -1         0                 0               0   
    2098635102     -1         0                 0               0   
    
                poutcome_success  poutcome_unknown  deposit  
    accnum                                                   
    2065031284                 0                 1        0  
    2051283096                 0                 1        0  
    2029034586                 0                 1        0  
    2070859436                 0                 1        0  
    2098635102                 0                 1        0  
    
    [5 rows x 35 columns]
    


```python
df['deposit'].mean().round(3)
```




    0.117



## Q1. Logistic regression model


```python
y = df['deposit']
X = df.drop(columns='deposit')
```


```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=3500)
clf.fit(X, y)
```






```python
y_pred = clf.predict(X)
conf = pd.crosstab(y, y_pred)
print(conf)
```

    col_0        0     1
    deposit             
    0        38990   932
    1         3596  1693
    


```python
acc = (y == y_pred).mean().round(3)
```


```python
acc1 = y_pred[y == 1].mean().round(3)
acc0 = (1 - y_pred[y == 0]).mean().round(3)
```


```python
acc, acc1, acc0
```




    (0.9, 0.32, 0.977)



## Q1. Undersample


```python
df0, df1 = df[df['deposit'] == 0], df[df['deposit'] == 1]
```


```python
df1['deposit'].mean().round(3)
```




    1.0




```python
df0['deposit'].mean().round(3)
```




    0.0




```python
n0, n1 = df0.shape[0], df1.shape[0]
```


```python
df0_under = df0.sample(n1)
df_under = pd.concat([df0_under, df1])
df_under['deposit'].mean().round(3)
df_under.shape[0]

```




    10578




```python
y = df_under['deposit']
X = df_under.drop(columns='deposit')
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=8000)
clf.fit(X, y)
```




```python
y_pred = clf.predict(X)
conf = pd.crosstab(y, y_pred)
print(conf)
```

    col_0       0     1
    deposit            
    0        4417   872
    1        1149  4140
    


```python
acc = (y == y_pred).mean().round(3)
acc1 = y_pred[y == 1].mean().round(3)
acc0 = (1 - y_pred[y == 0]).mean().round(3)
acc, acc1, acc0
```




    (0.809, 0.783, 0.835)



In this undersampling we can see there is a loss on accuracy from 0.9 to .818, but we get a better positive accuracy with 0.821 vs 0.32... meaning that know the model is better at capturing the people that would accept the service, hence we will have to call les people to obtain the desire outcome. In this case the conversion is 81.67% vs an intial conversibon of 60%.

## Q2. Oversample


```python
df1_over = df1.sample(n0 - n1, replace=True)
```


```python
df_over = pd.concat([df, df1_over])
df_over['deposit'].mean().round(3)
df_over.shape[0]
```




    79844




```python
y = df_over['deposit']
X = df_over.drop(columns='deposit')
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=5000)
clf.fit(X, y)
```



```python
y_pred = clf.predict(X)
conf = pd.crosstab(y, y_pred)
print(conf)
```

    col_0        0      1
    deposit              
    0        33399   6523
    1         8747  31175
    


```python
acc = (y == y_pred).mean().round(3)
acc1 = y_pred[y == 1].mean().round(3)
acc0 = (1 - y_pred[y == 0]).mean().round(3)
acc, acc1, acc0
```




    (0.809, 0.781, 0.837)



In this oversampling we can see there is a loss on accuracy from 0.9 to .815, but is still good and slightly lowe than undersampling, but we get a better positive accuracy with 0.825 vs 0.32 and 0.821... meaning that know the model is better at capturing the people that would accept the service, hence we will have to call les people to obtain the desire outcome. In this case the conversion is 80.09% vs an intial conversibon of 60%.

## Q3. Compare the three

1.	Original Imbalanced Data:
o	True Negatives (TN): 38988
o	False Positives (FP): 934
o	False Negatives (FN): 3595
o	True Positives (TP): 1694
o	Precision: 0.644
o	Recall: 0.320

2.	Undersampling:
o	True Negatives (TN): 4315
o	False Positives (FP): 974
o	False Negatives (FN): 949
o	True Positives (TP): 4340
o	Precision: 0.817
o	Recall: 0.821

3.	Oversampling:
o	True Negatives (TN): 32132
o	False Positives (FP): 7790
o	False Negatives (FN): 6967
o	True Positives (TP): 32955
o	Precision: ≈0.809
o	Recall: ≈0.826

My conclussion is that both method are good at increasing the recall, which makes the model better at predicting the number of people that would accept that we need to call, however both method have their own drawbacks.

Both results are quite similar, in my opinion the undersampling looks better, because we have a slightly higher conversion and reduces the number of call we need to make due to having a lower false positives. The only things is that here we are lossing data that could be important and might risk having overfitting.
