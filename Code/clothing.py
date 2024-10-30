## [ML-14] Clothing store marketing promotion ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'clothing.csv', index_col=0)

# Exploring the data #
df.info()
df['resp'].mean().round(3)

# Train-test split #
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
df_train.shape, df_test.shape

# Target vectors and feature matrices #
y_train, X_train = df_train['resp'], df_train.drop(columns='resp')
y_test, X_test = df_test['resp'], df_test.drop(columns='resp')

## Q1a. Logistic regression model ##
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression(max_iter=5000)
logclf.fit(X_train, y_train)

## Q1b. Decision tree model ##
from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
treeclf.fit(X_train, y_train)

# Q1c. Random forest model #
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(criterion='entropy', max_depth=4, n_estimators=200)
rfclf.fit(X_train, y_train)

# Q1d. XGBoost model #
from xgboost import XGBClassifier
xgbclf = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=200)
xgbclf.fit(X_train, y_train)

# Q2a. Plotting function #
from matplotlib import pyplot as plt
def score_plot(mod):
    score = mod.predict_proba(X_train)[:, 1]
    # Set the size of the figure
    plt.figure(figsize=(12,5))
    # First subplot
    plt.subplot(1, 2, 1)
    plt.hist(score[y_train == 1], range=(0,1), color='gray', edgecolor='white')
    plt.title('Figure a. Scores (positives)')
    plt.xlabel('Predictive score')
    # Second subplot
    plt.subplot(1, 2, 2)
    plt.hist(score[y_train == 0], range=(0,1), color='gray', edgecolor='white')
    plt.title('Figure b. Scores (negatives)')
    plt.xlabel('Predictive score');

# Q2b. Comparing distributions #
score_plot(logclf)
score_plot(treeclf)
score_plot(rfclf)
score_plot(xgbclf)

# Q3a. Testing function #
def test(mod):
    score_train, score_test = mod.predict_proba(X_train)[:, 1], mod.predict_proba(X_test)[:, 1]
    y_pred_train, y_pred_test = score_train > 0.2, score_test > 0.2
    tp_train = y_pred_train[y_train == 1].mean().round(3)
    fp_train = y_pred_train[y_train == 0].mean().round(3)
    tp_test = y_pred_test[y_test == 1].mean().round(3)
    fp_test = y_pred_test[y_test == 0].mean().round(3)
    return (tp_train, fp_train), (tp_test, fp_test)

# Q3b. Comparing stats #
test(logclf)
test(rfclf)
test(xgbclf)
