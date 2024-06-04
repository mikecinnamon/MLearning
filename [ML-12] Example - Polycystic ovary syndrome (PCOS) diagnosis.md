# [ML-12] Example - Polycystic ovary syndrome (PCOS) diagnosis

## Introduction

The **polycystic ovary syndrome** (PCOS) is a disorder involving infrequent, irregular or prolonged menstrual periods, and often excess male hormone(androgen) levels. The ovaries develop numerous small collections of fluid (called follicles) and may fail to regularly release eggs.

The most common PCOS symptoms are missed, irregular, infrequent, or prolonged periods. Other symptoms include hair loss, or hair in places you don't want it, acne, darkened skin, mood changes, weight gain and others.

This example is concerned with the prediction of PCOS from a set of physical and clinical parameters. The data have been collected from 10 different hospitals across Kerala, India.

## The data set

The file `pcos.csv` contains data on 538 subjects, including an identifier, a label signalling the occurrence of PCOS, and 38 features. The columns are:

* `id`, a unique identifier for the subject.

* `pcos`, whether the subject has PCOS (1 = YES, 0 = NO).

* `age`, the subject's age (years).

* `weight`, the subject's weight (kg).

* `height`, the subject's height (cm).

* `blood`, the subject's blood group (A+ = 11, A- = 12, B+ = 13, B- = 14, O+ =15, O- = 16, AB+ =17, AB- = 18).			

* `pulse`, the subject's pulse rate (bpm).

* `rr`, the subject's respiratory rate (breaths/min).

* `hb`, the subject's hemoglobin level (g/dl).

* `cycle`, the subject's menstrual cycle (R/I).

* `cyclelen`, the subject's cycle length (days).

* `marriage`, time since the subject was married (years).

* `pregnant`, whether the subjects is pregnant (1 = YES, 0 = NO).

* `abortions`, the subject's number of abortions.

* `beta_hcg1`, the beta-hCG level (mIU/mL) in the first pregnancy test. Beta-hCG is a shorthand term for a blood test used to measure human chorionic gonadotropin (hCG) levels in early pregnancy. The hCG is a hormone for the maternal recognition of pregnancy, with two subunits, alpha and beta.

* `beta_hcg2`, the blood beta-HCG level (mIU/mL) in the second pregnancy test. When only one test exists, the value entered in this column has been copied from the previous column.

* `fsh`, the blood FSH level (miU/mL). The follicle-stimulating hormone (FSH) helps control the menstrual cycle and stimulates the growth of eggs in the ovaries. The FSH test can used to find the cause of infertility, to find the reason for irregular or stopped menstrual periods, or other purposes.

* `lh`, the blood LH level (mIU/mL).The luteinizing hormone (LH) plays an important role in sexual development and functioning. The LH test is used for the same purposes as the FSH test.

* `hip`, the hip perimeter (inch).

* 'waist', the waist perimeter (inch).

* `tsh`, the blood TSH level (miU/mL). The thyroid stimulating hormone (FSH) helps control the weight, body temperature, muscle strength, and other things. The TSH test can used to find  out how well the thyroid is working.

* `amh`, the blood AMH level (ng/mL). The anti-mullerian hormone (AMH) plays a key role in developing a baby’s sex organs while in the womb. The AMH test provides information on the number of remaining eggs and whether the ovaries might be aging too quickly.

* `prl`, the blood PRL level (ng/mL). When women are pregnant or have just given birth, their prolactin (PRL) levels increase so they can make breast milk.The  doctor may order a prolactin test given symptoms like irregular or no periods, infertility, breast milk discharge not being pregnant or nursing, etc.

* `d3`, the blood vitamin D3 level (ng/mL). The vitamin D test is used to screen for low levels of vitamin D so the patient can be treated with supplements to prevent health problems.

* `prg`, the blood PRG level (ng/mL). Progesterone (PRG) is a hormone that prepares the uterus for pregnancy. A PRG test may be used to help find the cause of female infertility, check whether fertility treatments are working, find out the risk of a miscarriage, and other purposes.

* `rbs`, the blood glucose level (mg/dL) measured in a random blood sugar test (RBS). The RBS test is a quick test that a doctor or nurse can carry out at short notice in their office or clinic. The patient does not need to fast beforehand.

* `weight_gain`, a dummy for weight gain.

* `hair_growth`, a dummy for hair growth.

* `skin_dark`, a dummy for skin darkening.

* `hair_loss`, a dummy for hair loss.

* `pimples`, a dummy for having pimples.

* `fastfood`, a dummy for fast food consumption.

* `exercise`, a dummy for doing regular exercise.

* `systolic`, systolic blood pressure (mmHg).

* `diastolic`, diastolic blood pressure (mmHg).

* `lfollicle`, number of follicles in the left ovary. The ovarian follicles are small sacs filled with fluid that are found inside a woman's ovaries. They secrete hormones which influence stages of the menstrual cycle and women begin puberty.

* `rfollicle`, number of follicles in the right ovary.

* `lsize`, average size of the follicles in the left ovary (mm).

* `fsize`, average size of the follicles in the left ovary (mm).

* `endometrium`, the thickness of the endometrium (mm).

## Questions

Q1. Develop a decision tree classifier for the diagnosis of PCOS. Which are the most relevant features?

Q2. There are three features which are used in this context: (a) the body mass index (BMI) is the body mass divided by the square of the body height (kg/m2), (b) the LH/FSH ratio, and (c) the hip/waist ratio. Does your model improve with these three additional features?

Q3. Perform a 3-fold cross-validation of your model. 

Q4. Reduce the size of the tree if you are not happy with the results obtained.

## Importing the data

As in the preceding examples, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. In this case, we take the column `id` as the index (`index_col=0`).

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
   ...: df = pd.read_csv(path + 'pcos.csv', index_col=0)
```

## Exploring the data

We print a report of the content of `df` with the method `.info()`. Everything is as expected, so far. There are no missing values.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 538 entries, 1 to 541
Data columns (total 39 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   pcos         538 non-null    int64  
 1   age          538 non-null    int64  
 2   weight       538 non-null    float64
 3   height       538 non-null    float64
 4   blood        538 non-null    int64  
 5   pulse        538 non-null    int64  
 6   rr           538 non-null    int64  
 7   hb           538 non-null    float64
 8   cycle        538 non-null    int64  
 9   cyclelen     538 non-null    int64  
 10  marriage     538 non-null    float64
 11  pregnant     538 non-null    int64  
 12  abortions    538 non-null    int64  
 13  beta_hcg1    538 non-null    float64
 14  beta_hcg2    538 non-null    float64
 15  fsh          538 non-null    float64
 16  lh           538 non-null    float64
 17  hip          538 non-null    int64  
 18  waist        538 non-null    int64  
 19  tsh          538 non-null    float64
 20  amh          538 non-null    float64
 21  prl          538 non-null    float64
 22  d3           538 non-null    float64
 23  prg          538 non-null    float64
 24  rbs          538 non-null    float64
 25  weight_gain  538 non-null    int64  
 26  hair_growth  538 non-null    int64  
 27  skin_dark    538 non-null    int64  
 28  hair_loss    538 non-null    int64  
 29  pimples      538 non-null    int64  
 30  fastfood     538 non-null    float64
 31  exercise     538 non-null    int64  
 32  systolic     538 non-null    int64  
 33  diastolic    538 non-null    int64  
 34  lfollicle    538 non-null    int64  
 35  rfollicle    538 non-null    int64  
 36  lsize        538 non-null    float64
 37  fsize        538 non-null    float64
 38  endometrium  538 non-null    float64
dtypes: float64(18), int64(21)
memory usage: 168.1 KB
```

The **incidence** of PCOS in this sample is 32.7%, so we don't have a serious case of imbalance here. Note that this incidence applies to a sample extracted from the hospital female population, so we have to be careful when drawing conclusions.

```
In [3]: df['pcos'].mean().round(3)
Out[3]: 0.327
```

## Feature matrix and target vector

We create a target vector and a feature matrix. The target vector is the first column (`pcos`). Among the features, `blood` is categorical, so we have to replace it by a collection of dummies. We create a submatrix integrates all the columns, except `blood`.

```
In [4]: y = df['pcos']
   ...: X1 = df.drop(columns=['blood', 'pcos'])
```

We apply the Pandas function `get_dummies()` to the categorical feature.

```
In [5]: X2 = pd.get_dummies(df['blood'])
```

```
In [6]: X2.columns
Out[6]: Index([11, 12, 13, 14, 15, 16, 17, 18], dtype='int64')
```

The column names of this matrix are integers, so we have to change that, which is easy.

```
In [7]: X2.columns = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
```

With the Pandas function `concat()`, we join the two parts of the new feature matrix, as we did in example ML-04.

```
In [8]: X = pd.concat([X1, X2], axis=1)
```

## Q1. Decision tree classifier

We train a decision tree classifier, as requested. Given the dimensions of the feature matrix, we allow for maximum depth 5.

```
In [9]: from sklearn.tree import DecisionTreeClassifier
   ...: clf = DecisionTreeClassifier(max_depth=5)
   ...: clf.fit(X, y)
Out[9]: DecisionTreeClassifier(max_depth=5)
```

The accuracy is quite high. Note this applies to a very specific population.

```
In [10]: clf.score(X, y).round(3)
Out[10]: 0.941
```

Nevertheless, the confusion matrix warns us that the accuracy is not that high for the PCOS patients (84.1%).

```
In [11]: y_pred = clf.predict(X)
    ...: from sklearn.metrics import confusion_matrix
    ...: confusion_matrix(y, y_pred)
Out[11]: 
array([[358,   4],
       [ 28, 148]])
```

The relevance of the features involved in this tree is obtained as in example ML-08.

```
In [12]: importance = pd.Series(clf.feature_importances_, index=X.columns)
    ...: importance[importance > 0].sort_values(ascending=False).round(3)
Out[12]: 
rfollicle      0.537
weight_gain    0.103
hair_growth    0.090
lfollicle      0.072
cyclelen       0.037
lh             0.028
lsize          0.022
fsize          0.019
waist          0.014
amh            0.014
endometrium    0.012
pimples        0.011
fastfood       0.011
tsh            0.010
marriage       0.009
prg            0.007
hip            0.002
dtype: float64
```

So far, it is pretty obvious which are the relevant features.

## Q2. Extra features

We add the extra features suggested to the current feature matrix.

```
In [13]: X['bmi'] = df['weight']/df['height']**2
    ...: X['lh_fsh'] = df['lh']/df['fsh']
    ...: X['hip_waist'] = df['hip']/df['waist']
```

We train again the decision tree classifier.

```
In [14]: clf.fit(X, y)
    ...: clf.score(X, y).round(3)
Out[14]: 0.952
```

So, the new model is a bit better. The hip waist seems to be reponsible for the improvement.

```
In [15]: importance = pd.Series(clf.feature_importances_, index=X.columns)
    ...: importance[importance > 0].sort_values(ascending=False).round(3)
Out[15]: 
rfollicle      0.529
weight_gain    0.101
hair_growth    0.088
lfollicle      0.061
lh_fsh         0.033
cyclelen       0.027
endometrium    0.024
lsize          0.022
fsize          0.019
marriage       0.017
height         0.016
hip_waist      0.014
lh             0.012
tsh            0.010
rbs            0.009
hb             0.008
amh            0.008
hip            0.002
dtype: float64
```

## Q3. 3-fold cross-validation

We address now the validation of the decision tree classifier. We use the function `cross_val_score()`, from the subpackage `model_selection`. We use a 3-fold partition, as suggested.

```
In [16]: from sklearn.model_selection import cross_val_score
    ...: cross_val_score(clf, X, y, cv=3).round(3)
Out[16]: array([0.689, 0.799, 0.827])
```

We have a clear case of **overfitting**. Let us try with a smaller tree.

## Q4. Reduce the size of the decision tree

We set now the maximum depth at 4, which will potentially halve the number of leaf nodes. The accuracy is just a bit lower.

```
In [17]: clf = DecisionTreeClassifier(max_depth=4)
    ...: clf.fit(X, y)
    ...: clf.score(X, y).round(3)
Out[17]: 0.931
```

We still have an overfitting problem here, though not so extreme. We leave further analysis for the homework.

```
In [18]: cross_val_score(clf, X, y, cv=3).round(3)
Out[18]: array([0.833, 0.754, 0.832])
```

## Homework

1. Try a new decision tree classifier, with less depth, to see whether you can so fix the overfitting problem.

2. Apply the same cross-validation approach to a logistic regression model. Do you get better results?

3. Cross-validation in scikit-learn uses a non-random specific splitting strategy called `StratifiedKFold`. So you get the same results across calls of `cross_val_score`. You change this easily by replacing `X` and `y` by "shuffled" versions, which can be created with the method `.sample()`. Try that and see what happens. Take care of shuffling `X` and `y` in the same way, by giving the same value to the parameter `random_state` of `.sample()`.