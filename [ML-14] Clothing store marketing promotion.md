# [ML-16] Example - Airline passenger satisfaction

## Introduction

The data for this example, published by Mysar Ahmad Bhat, provide details of customers who have already flown with an **airline company**. The feedback of the customers on various contexts and their flight data has been consolidated.

The main objective of the data could be to predict whether a future customer would be satisfied with the service given by this company, from the features evaluated in the study. A second objective could be to explore which aspects of the services offered have to be emphasized to generate more satisfied customers.

## The data set

The file `airsat.csv` contains data on 113,485 customers. The columns are:

* `female`, gender of the passenger (Female=1, Male=0).

* `age`, age of the passenger. Only passengers older than 15 were included in the data collection.

* `first`, type of airline customer (First-time=1, Returning=0).

* `business`, purpose of the flight (Business=1, Personal=0).

* `busclass`, travel class for the passenger seat (Business=1, Economy=0).

* `distance`, flight distance in miles.

* `depdelay`, flight departure delay in minutes.

* `arrdelay`, flight arrival delay in minutes.

* `time`, satisfaction with the convenience of the flight departure and arrival times from 1 (lowest) to 5 (highest). 

* `online_book`, satisfaction with the online booking experience from 1 (lowest) to 5 (highest). 

* `checkin`, satisfaction with the check-in service from 1 (lowest) to 5 (highest). 

* `online_board`, satisfaction with the online boarding experience from 1 (lowest) to 5 (highest). 

* `gate`, satisfaction with the gate location in the airport from 1 (lowest) to 5 (highest). 

* `on_board`, satisfaction with the on-boarding service in the airport from 1 (lowest) to 5 (highest).

* `seat`, satisfaction with the comfort of the airplane seat from 1 (lowest) to 5 (highest). 

* `leg_room`, satisfaction with the leg room of the airplane seat from 1 (lowest) to 5 (highest). 

* `clean`, satisfaction with the cleanliness of the airplane from 1 (lowest) to 5 (highest). 

* `food`, satisfaction with the food and drinks on the airplane from 1 (lowest) to 5 (highest). 

* `in_flight`, satisfaction with the in-flight service from 1 (lowest) to 5 (highest). 

* `wifi`, satisfaction with the in-flight Wifi service from 1 (lowest) to 5 (highest). 

* `entertain`, satisfaction with the in-flight entertainment from 1 (lowest) to 5 (highest). 

* `baggage`, satisfaction with the baggage handling from the airline from 1 (lowest) to 5 (highest). 

* `sat`, overall satisfaction with the airline (Satisfied=1, Neutral or unsatisfied=0).

Source of the data: Kaggle. Rows with missing data have been deleted. The option 'Not applicable' was available in the questionnaire for the satisfaction levels measured in 1-5 range, but it was encoded as 3.

## Questions

Q1. Develop a **random forest** model for predicting the passenger satisfaction.

Q2. The same for a **XGBoost** model. Is the improvement relevant?

Q3. Which features are most relevant for predicting the passenger satisfaction?

Q4. Try a **multilayer perceptron** model. Is it better than the ensemble models? 

Q5. Does the MLP model get better after **normalizing** the features?

## Importing the data

As in other examples, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. Since the passengers don't have an identifier, we let the index be a `RangeIndex`. 

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
   ...: df = pd.read_csv(path + 'airsat.csv')
```

## Exploring the data

The data report printed by the method `.info()` does not contradict the description given above. There are no null values.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 113485 entries, 0 to 113484
Data columns (total 23 columns):
 #   Column        Non-Null Count   Dtype
---  ------        --------------   -----
 0   female        113485 non-null  int64
 1   age           113485 non-null  int64
 2   first         113485 non-null  int64
 3   business      113485 non-null  int64
 4   busclass      113485 non-null  int64
 5   distance      113485 non-null  int64
 6   depdelay      113485 non-null  int64
 7   arrdelay      113485 non-null  int64
 8   time          113485 non-null  int64
 9   online_book   113485 non-null  int64
 10  checkin       113485 non-null  int64
 11  online_board  113485 non-null  int64
 12  gate          113485 non-null  int64
 13  on_board      113485 non-null  int64
 14  seat          113485 non-null  int64
 15  leg_room      113485 non-null  int64
 16  clean         113485 non-null  int64
 17  food          113485 non-null  int64
 18  in_flight     113485 non-null  int64
 19  wifi          113485 non-null  int64
 20  entertain     113485 non-null  int64
 21  baggage       113485 non-null  int64
 22  sat           113485 non-null  int64
dtypes: int64(23)
memory usage: 19.9 MB
```

The proportion of satisfied passengers is quite close to 50%, so class imbalance is not an issue here. We will use the accuracy to evaluate the models obtained.

```
In [3]: df['sat'].mean().round(3)
Out[3]: 0.467
```

## Target vector and features matrix

As in other examples of supervised learning, we create a target vector and a features matrix. The target vector is the last column (`sat`) and the features matrix is made of the other columns.

```
In [4]: y = df['sat']
   ...: X = df.drop(columns='sat')
```

## Train-test split

For validation, the data set is randomly split, keeping a 20% of the data units for testing. The argument `random_state=0` ensures the reproducibility of the split.

```
In [5]: from sklearn.model_selection import train_test_split
   ...: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## Q1. Random forest model

Our first predictive model is a random forest model. We instantiate an estimator from the scikit-learn class `ensemble.RandomForestClassifier()`, setting the number of trees to 200 and the maximum depth to 5. As usual, the model is trained on the training data.

```
In [6]: from sklearn.ensemble import RandomForestClassifier
   ...: rf = RandomForestClassifier(max_depth=5, n_estimators=200, random_state=0)
   ...: rf.fit(X_train, y_train)
Out[6]: RandomForestClassifier(max_depth=5, n_estimators=200, random_state=0)
```

Then, we evaluate the model on both the training and the test subsets, which comes easily in scikit-learn.

```
In [7]: round(rf.score(X_train, y_train), 3), round(rf.score(X_test, y_test), 3)
Out[7]: (0.913, 0.911)
```

The accuracy is about 91%. The test data do not provide evidence of overfitting.

## Q2. XGBoost model

We repeat the exercise with an XGBoost model, from the class `XGBClassifier()` of the package `xgboost`. We use the same arguments as in the preceding section, leaving the learning rate at the default value. 

```
In [8]: from xgboost import XGBClassifier
   ...: xgb = XGBClassifier(max_depth=5, n_estimators=200, random_state=0)
   ...: xgb.fit(X_train, y_train)
Out[8]: 
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=5, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=200, n_jobs=None,
              num_parallel_tree=None, random_state=0, ...)
```

Overfitting, whis is typical of gradient boosting models, is moderate in this case. We can take the accuracy 95% as the benchmark for other models. This is a **shallow model**, which takes the actual features as they come, with no **feature engineering**

```
In [9]: round(xgb.score(X_train, y_train), 3), round(xgb.score(X_test, y_test), 3)
Out[9]: (0.969, 0.953)
```

## Q3. Relevant features

In a predictive model based on decision trees, the relevance of the different features for predicting the target can be assessed with the attribute `.feature_importances_`, which works in `xgboost` as in scikit-learn. Since the outcome of this method is a plain 1D array, without index labels, we convert it to a Pandas series, using the column names as the index. Sorting by values, we get a clear report.

```
In [10]: pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
Out[10]: 
online_board    0.380
business        0.150
wifi            0.082
busclass        0.078
first           0.075
entertain       0.034
checkin         0.031
seat            0.023
clean           0.021
baggage         0.020
gate            0.018
leg_room        0.018
in_flight       0.015
on_board        0.013
time            0.008
age             0.008
online_book     0.007
arrdelay        0.005
food            0.004
distance        0.003
depdelay        0.003
female          0.002
dtype: float32
```

Online boarding looks that the most relevant feature. Also, the model reveals a difference between flying for business and the overall satisfaction, although it does not show in which direction. Popular wisdom tells us that people flying for personal issues (and also paying) are usually less tolerant with things not working properly. Indeed, this is what cross tabulation tells us:

```
In [11]: pd.crosstab(df['business'], df['sat'])
Out[11]: 
sat           0      1
business              
0         27387   3129
1         33150  49819
```

## Q4. MLP model

We try now a simple neural network, using the **package Keras**, with the default **TensorFlow** backend. We import the function `Input()` and the modules `models` and `layers`, which contain the resources needed for this example.

```
In [12]: from keras import Input, models, layers
```

Next, we specify the **network architecture**, as a sequence of transformations. This will be multilayer perceptron (MLP) with one hidden layer. The **input layer** contains one node for every feature. 

```
In [13]: input_tensor = Input(shape=(22,))
```

The **hidden layer** has 32 nodes (powers of 2 are commonly used in deep learning), and the activation function is the **rectified linear unit function** (ReLU). It is a **dense layer**, meaning that every node is connected to all nodes of the preceding layer. This involves 23 $\times$ 32 = 736 parameters.

```
In [14]: x = layers.Dense(32, activation='relu')(input_tensor)
```

The **output layer** has two nodes, since this model is a binary classifier. The activation function is here the **softmax function**, which ensures that the output is a vector of probabilities (positive numbers summing 1). It adds 33 $\times$ 2 = 66 parameters.

```
In [15]: output_tensor = layers.Dense(2, activation='softmax')(x)
```

The next step is instantiate an object of the class `models.Model()`. This works as in scikit-learn. We specify here the input and the output.

```
In [16]: mlp = models.Model(input_tensor, output_tensor)
```

Now, we **compile** the model, meaning the mathematical apparatus needed for fitting the model to the data gets ready. We have to specify the **optimization algorithm** (`optimizer='adam'`), the **loss function** (`loss='sparse_categorical_crossentropy'`) and the metrics used to evaluate the model performance (`metrics=['acc']`), in a list or dictionary.

```
In [17]: mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
```

We are ready now to apply the method `.fit()`. Note that the number of iterations is specified here, not when creating the model, as it was in scikit-learn. In every iteration, or **epoch**, the data set is randomly split in batches of size 32 (if you accept the deafult size). These batches are passed sequentially, and the parameter values are updated for evey batch. This means that they not are updated 50 times, but 50 times the number of batches (2,838). With `verbose=0` we stop a report of the progress being gradually displayed on the screen (we will see this later). Also, the semicolon stops some irrelevant output showing up (the same we do when plotting with Matplotlib). In this example, every epoch takes between one and two seconds in a regular laptop.

```
In [18]: mlp.fit(X_train, y_train, epochs=50, verbose=0);
```

Once the model has been trained, it is evaluated on the test data. The model does not improve the accuracy of the ensemble models tried before.  

```
In [19]: round(mlp.evaluate(X_test, y_test, verbose=0)[1], 3)
Out[19]: 0.907
```

## Q5. Multilayer perceptron model (normalized data)

Though scikit-learn has a method for normalizing all the columns of the feature matrix in one shot, it is not dificult to do it in Pandas. First we define a **min-max normalization** function:

```
In [20]: def normalize(x): 
    ...:     return (x - x.min())/(x.max() - x.min())
```

Now, we apply this function by column with the method `.apply()`.

```
In [21]: XN = X.apply(normalize)
```

We have now a new feature matrix, that we split exactly in the same way as we did with `X` and `y` (the argument `random_state=0` does the trick).

```
In [22]: XN_train, XN_test = train_test_split(XN, test_size=0.2, random_state=0)
```
We replicate the process of question Q4 with the normalized features. The improvement is quite clear, though the accuracy falls a bit short of that of the XGBoost model.

```
In [23]: mlp = models.Model(input_tensor, output_tensor)
    ...: mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    ...: mlp.fit(XN_train, y_train, epochs=50, verbose=0);
    ...: round(mlp.evaluate(XN_test, y_test, verbose=0)[1], 3)
Out[23]: 0.943
```
