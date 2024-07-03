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
df_train, df_test = train_test_split(df, test_size=0.2)
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

# Plotting function #
from matplotlib import pyplot as plt
def score_plot(score):
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

# Q2a. Logistic regression scores #
log_score = logclf.predict_proba(X_train)[:, 1]
score_plot(log_score)

# Q2b. Decision tree scores #
tree_score = treeclf.predict_proba(X_train)[:, 1]
score_plot(tree_score)

# Q2c. Random forest scores #
rf_score = rfclf.predict_proba(X_train)[:, 1]
score_plot(rf_score)

# Q2d. XGBoost scores #
xgb_score = xgbclf.predict_proba(X_train)[:, 1]
score_plot(xgb_score)

# Q3a. Testing the logistic regression model #
log_score_train, log_score_test = logclf.predict_proba(X_train)[:, 1], logclf.predict_proba(X_test)[:, 1]
y_pred_train, y_pred_test = log_score_train > 0.2, log_score_test > 0.2
conf_train, conf_test = pd.crosstab(y_train, y_pred_train), pd.crosstab(y_test, y_pred_test)
conf_train, conf_test

# Q3b. Testing the XGBoost model #
xgb_score_train, xgb_score_test = xgbclf.predict_proba(X_train)[:, 1], xgbclf.predict_proba(X_test)[:, 1]
y_pred_train, y_pred_test = xgb_score_train > 0.2, xgb_score_test > 0.2
conf_train, conf_test = pd.crosstab(y_train, y_pred_train), pd.crosstab(y_test, y_pred_test)
conf_train, conf_test


# XGBoost model #
from xgboost import XGBClassifier
xgbclf = XGBClassifier(max_depth=4, n_estimators=200)
xgbclf.fit(X_train, y_train)
y_pred = xgbclf.predict(X_test)
conf = pd.crosstab(y_test, y_pred)
conf

# Logistic regression classifier #
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression(max_iter=2000)
logclf.fit(X_train, y_train)
y_pred = logclf.predict(X_test)
conf = pd.crosstab(y_test, y_pred)
conf
