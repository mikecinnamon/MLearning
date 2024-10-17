## [ML-12] Example - Polycystic ovary syndrome (PCOS) diagnosis ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'pcos.csv', index_col=0)

# Exploring the data #
df.info()
df['pcos'].mean().round(3)

# Target vector and features matrix #
y = df['pcos']
X1 = df.drop(columns=['blood', 'pcos'])
X2 = pd.get_dummies(df['blood'])
X2.columns
X2.columns = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
X = pd.concat([X1, X2], axis=1)

# Q1. Decision tree classifier # 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X, y)
round(clf.score(X, y), 3)
y_pred = clf.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)
importance = pd.Series(clf.feature_importances_, index=X.columns)
importance[importance > 0].sort_values(ascending=False).round(3)

# Q2. Extra features #
X['bmi'] = df['weight']/df['height']**2
X['lh_fsh'] = df['lh']/df['fsh']
X['hip_waist'] = df['hip']/df['waist']
clf.fit(X, y)
round(clf.score(X, y), 3)
importance = pd.Series(clf.feature_importances_, index=X.columns)
importance[importance > 0].sort_values(ascending=False).round(3)

# Q3. 3-fold cross-validation #
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv=3).round(3)

# Q4. Reduce the size of the decision tree # 
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
clf.fit(X, y)
round(clf.score(X, y), 3)
cross_val_score(clf, X, y, cv=3).round(3)
clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=6)
clf.fit(X, y)
round(clf.score(X, y), 3)
importance = pd.Series(clf.feature_importances_, index=X.columns)
importance[importance > 0].sort_values(ascending=False).round(3)
cross_val_score(clf, X, y, cv=3).round(3)

# Plotting the tree #
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(13,7))
plot_tree(clf, fontsize=10)
plt.title('Figure 1. PCOS diagnosis tree', fontsize=16);
