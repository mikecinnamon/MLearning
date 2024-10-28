```python
import pandas as pd #import data#
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'deposit.csv', index_col=0)
```


```python
#Undersampling
df0,df1 = df[df['deposit'] == 0], df[df['deposit'] == 1]
```


```python
n0, n1 = df0.shape[0], df1.shape[0]
```


```python
#Question 01 Undersample the data
df0_under = df0.sample(n1)
```


```python
df_under = pd.concat([df0_under, df1]) #df_under is balanced, with n1 positive units and n1 negative units.
```


```python
#logistic regresssion
y1 = df_under['deposit']
x1 = df_under.drop(columns='deposit')
```


```python
def eval1(clf):
    y1_pred = clf.predict(x1)
    conf = pd.crosstab(y1, y1_pred)
    acc = (y1 == y1_pred).mean().round(3)
    TP = y1_pred[y1 == 1].mean().round(3) 
    TN = (1-y1_pred[y1 == 0]).mean().round(3) 
    return conf,acc,TP,TN
```


```python
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(max_iter=5000) #max_iter=2000 is not appliable here
clf1.fit(x1, y1)
eval1(clf1)
```




    (col_0       0     1
     deposit            
     0        4422   867
     1        1147  4142,
     0.81,
     0.783,
     0.836)




```python
#Question 02 oversample the data
df1_over = df1.sample(n0-n1,replace=True) 
```


```python
df_over = pd.concat([df, df1_over])
```


```python
#logistic regresssion
y2 = df_over['deposit']
x2 = df_over.drop(columns='deposit')
```


```python
def eval2(clf):
    y2_pred = clf.predict(x2)
    conf = pd.crosstab(y2, y2_pred)
    acc = (y2 == y2_pred).mean().round(3)
    TP = y2_pred[y2 == 1].mean().round(3) 
    TN = (1-y2_pred[y2 == 0]).mean().round(3)
    return conf,acc,TP,TN
```


```python
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(max_iter=5000)
clf2.fit(x2, y2)
eval2(clf2)
```




    (col_0        0      1
     deposit              
     0        33408   6514
     1         8584  31338,
     0.811,
     0.785,
     0.837)




```python
#Question 03 compare the data
```


```python
the confusion matrix in class: 
score        0     1
deposit             
0        32696  7226
1         1022  4267
```


```python
#when compared with Undersampling data
col_0       0     1
deposit            
 0        4422   867
 1        1147  4142

#and the oversampling data
col_0        0      1
deposit              
 0        33408   6514
 1         8584  31338

#when it comes to the Accuracy
original:0.818, 0.807, 0.819
Undersampling:0.81, 0.783, 0.836
Oversampling: 0.811, 0.785, 0.837

Based on the results, the conclusions are as follows:
Original Model: High accuracy, with overall good performance.
Undersampling Model:  Slightly lower accuracy and recall, meaning it may miss some positive cases (potential subscribers).
Oversampling Model: Improved accuracy and recall compared to the undersampling model, indicating it is more effective at identifying positive cases.
Overall, the oversampling model may be the best choice for balancing accuracy and recall, making it a better fit for this dataset.
```


      Cell In[96], line 4
        0        4422   867
        ^
    IndentationError: unexpected indent
    

