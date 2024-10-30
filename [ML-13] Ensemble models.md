# [ML-13] Ensemble models

## Ensemble models

Suppose that you ask a complex question to thousands of random people, and then aggregate their answers. In many cases, you will find that this aggregate answer is better than an expert's answer. This has been called the **wisdom of the crowds**.

**Ensemble learning** is based on a similar idea. If you aggregate the predictions of a group of regression or classification models, you will often get better predictions than with the best individual model. Let us be more specific: 

* Suppose that you have trained a few regression models, each one achieving a moderate R-squared statistic. A simple way to get better predictions could be to average the predictions of these models.

* In a classification context, you would average the predicted class probabilities. This is called **soft voting**. Alternatively, you can use **hard-voting**, which consists in picking the class with more votes. This course only covers soft voting.

The collection of models whose predictions are aggregated is called an **ensemble**. In scikit-learn, the subpackage `ensemble` offers plenty of choice. On top of the popularity ranking, we find random forest and gradient boosting models, both using ensembles of decision tree models.

## Random forests

One way to get a diverse ensemble is to use the same algorithm for every predictor, but training it on different random subsets of the training set. When these subsets are extracted by means of **sampling with replacement**, the method is called **bagging** (short for bootstrap aggregating). Bagging allows training instances to be sampled several times for the same predictor.

The star of the bagging ensemble methods is the **random forest** algorithm, which allows extra randomness when growing trees, by using just a random subset of the features at every split. This results in a greater tree diversity, generally yielding an overall better model. Despite its simplicity, random forest models are among the most powerful predictive models available. In general, they are trained faster and have less overfitting problems than other models.

Random forests can be used for both regression and classification. In the scikit-learn subpackage `ensemble`, this is provided by the estimator classes `RandomForestRegressor()` and `RandomForestClassifier()`. The growth of the trees is controlled with parameters such as `max_depth` and `max_leaf_nodes` (no defaults), as in individual tree models. You can also control the number of trees with the parameter `n_estimators` (the default is 100 trees), and the number of features that can be used at every split with the parameter `max_features` (look at the scikit-learn API Reference if you wish to play with this argument).

Here follows a classification example:

```
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_leaf_nodes=128, n_estimators=50)
```

In general, increasing the tree size or the number of trees leads to better prediction, but this may come at the price of overfitting the training data. A safe approach is to accept the default `n_estimators=100`, increasing gradually the tree size and testing overfitting at every step.

## Gradient boosting

The general idea of the **boosting** methodology is to train the models of the ensemble sequentially, each model trying to correct its predecessor. The star is here the **gradient boosting** algorithm, used in both regression and classification. As in the random forest algorithm, the models of the ensemble are based on decision trees, though, here, every tree model is trained on the errors made by the previous one.

The prediction of the ensemble model is obtained as a weighted average. The weights decrease at every step according to a parameter called **learning rate**. With a low learning rate, the weights decrease more slowly. There is a trade-off between the learning rate and the number of trees. With a low learning rate, you will probably need a higher number of trees. Some experts recommend to set a low learning rate (in the range from 0.001 to 0.01) and aim at a high number of trees (in the range from 3,000 to 10,000), but for that you may need huge computing power, since gradient boosting becomes a slow process when the number of features gets high.

In scikit-learn, gradient boosting is provided by the classes `GradientBoostingRegressor()` and `GradientBoostingClassifier()`, from the subpackage `ensemble`. The growth of the trees and the number of trees is controlled as for random forest models. Most practitioners accept the defaults `n_estimators=100` and `learning_rate=0.1`, but go beyond the default `max_depth=3`.

Here follows a classification example:

```
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)
```

**XGBoost** (extreme gradient boosting) is an implementation of gradient boosting designed for speed and performance. For many years, it has often been top-ranked in applied machine learning competitions. For Python, it is available in the package `xgboost`, which can be used as if it were a scikit-learn subpackage, with the **Scikit-Learn Estimator Interface** (see `https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html`). Gradient boosting optimization takes less time in `xgboost` than in scikit-learn. The defaults are `n_estimators=100`, `learning_rate=0.3` and `max_depth=6`.

The `xgboost` version of the preceding example would be:

```
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)
```

*Note*. `xgboost` can be installed from the shell or the console, with `pip install xgboost`. A few years ago, the installation raised conflicts of versions in many computers. These problems seem to have disappeared, and the last versions do not seem to present installation problems. It is already available in Google Colab, so no installation is needed there.
