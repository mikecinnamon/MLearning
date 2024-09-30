# [ML-07] Decision trees

## What is a decision tree?

A **decision tree** is a collection of **decision nodes**, connected by branches, extending downwards from the **root node**, until terminating in the **leaf nodes**. The usual visualization of a decision tree puts the root on top and the leaves at the bottom, as in Figures 1 and 2, which have been created with the scikit-learn function `plot_tree`.

Decision trees can be used for both regression and classification purposes. A decision tree creates a partition of the data set into a collection of subsets, one for each leaf. In a predictive model based on a decision tree, the predicted target value is the same for all the data units of the same leaf. More specifically:

* In a **decision tree regressor**, the predicted target value is the average target value in that leaf. 

* In a **decision tree classifier**, a predicted probability class is the proportion of that class in the leaf. Under the **default prediction rule**, the predicted class is the one that occurs more frequently in that leaf.

## Decision trees in scikit-learn

There are various ways to train a decision tree model. The top popular one is the **CART** (Classification And Regression Trees) algorithm. In scikit-learn, the subpackage `tree` provides the estimator classes `DecisionTreeRegressor()` and `DecisionTreeClassifier()`, both based on CART.

At every decision node, there is a **split**, based on one of the features and a cutoff value. CART chooses at every node the **optimal split**, that minimizes the **loss**. In decision tree regressors, as in linear regression, the default loss is the **mean square error** (MSE). Other choices are available through the parameter `criterion`. 

![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/07-1.png)

Figure 1 shows a decision tree regressor, developed to predict the price of a house (example ML-04). At every node, we find the number of data units (`samples`), the MSE (`squared_error`) and the predicted price (`value`), which is the average price in that leaf. The tree is optimal (meaning that it provides the minimum MSE) among those satisfying the conditions set by the arguments of `DecisionTreeRegressor()`. For every possible split, CART calculates the loss as the weighted average of the losses at the two branches, choosing the split that leads to the minimum loss. 


![](https://github.com/mikecinnamon/MLearning/blob/main/Figures/07-2.png)

In scikit-learn decision tree classifiers, the default loss function is **Gini impurity measure**. Nevertheless, to present a more consistent approach to classification models along this course, we use everywhere the **average cross-entropy**, which, in the class `DecisionTreeClassifier()`, is specified with the argument `criterion='entropy'`. Figure 2 shows a decision tree intended to be used in a spam filter (example ML-08). At every leaf, you find the number of units (`samples`), the average cross-entropy (`entropy`) and the number of negative and positive units (in alphabetical order, so negative first) in that leaf. As the predicted class probabilities in a leaf, the model takes the class proportions in that leaf. In a binary setting, we can say that the predicted score for a data unit is the proportion of positive units in the leaf where that unit falls. The tree is optimal in the sense that the overall average cross-entropy (the weighted average of the cross-entropies at the leaf nodes) is minimum.

## Controlling the growth of the tree

The predictive models based on decision trees are prone to **overfitting**. Even with a moderate number of features, a tree model whose growth is not stopped can lead to a complex model with overfitting problems. In scikit-learn, the classes `DecisionTreeRegressor()` and `DecisionTreeClassifier()` have several parameters for controlling the growth of the tree: `max_depth`, `max_leaf_nodes`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, etc. Only the first two will appear in these course:

* The parameter `max_depth` controls the **depth**, that is, the number of nodes in the longest branch. The trees of Figures 1 and 2 have been obtained by setting `max_depth=2`.

* The parameter `max_leaf_nodes` controls directly the **number of leaves**.

To obtain the tree of Figure 1, we would use:

```
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth=2)
reg.fit(X, y)
```

Then, we would plot the tree with:

```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(13,7))
plot_tree(treereg, fontsize=11);
```

To obtain the tree of Figure 2, we would use:

```
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
```

## Feature importance

One of the advantages of decision tree models is that it is very easy to get a report on the **feature importance**. The importance of a feature is computed as the contribution of that feature to the total loss reduction achieved by the model. In scikit-learn, the attribute `.feature_importances_` is a 1D array containing importance values for all the features. A zero value signals a feature that has not used in the tree. For the tree of Figure 1, this is would be obtained as follows.

```
reg.feature_importances_
```

## Homework

1. In Figure 1, the initial MSE is equal to 134,776.142. Calculate the final MSE as the weighted average of the MSE's in the four leaf nodes. Translate the MSE reduction into a R-squared value (what the method `.score()` would return for this model).

2. Check that the initial cross-entropy in Figure 2 is indeed 0.967. What is the final entropy, after the three splits? Remember that scikit-learn uses binary logs to calculate the cross-entropy. In Python, you get the binary log with `math.log(x, 2)`.

3. In Figure 2, suppose that the split in the root node has already been performed. The second split will be the one that brings a greater reduction of the cross-entropy. Is this the leftt one (`x[15] <= 0.135`) or the right one (`x[26] <= 0.08`)?  
