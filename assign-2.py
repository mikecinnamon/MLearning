## Assignment 2 ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/mikecinnamon/Data/main/'
df = pd.read_csv(path + 'deposit.csv', index_col=0)

# Q1a. Undersampling #
df_pos = df[df['deposit'] == 1]
df_neg = df[df['deposit'] == 0]
df_neg = df_neg.sample(n=len(df_pos), replace=False)
df_under = pd.concat([df_pos, df_neg])
df_under.shape
df_under['deposit'].mean()

# Q1b. Logistic regression classifier #
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
y_under, X_under = df_under['deposit'], df_under.drop(columns='deposit')
clf1 = LogisticRegression(max_iter=2000)
clf1.fit(X_under, y_under)
y_under_pred = clf1.predict(X_under)
conf_under = confusion_matrix(y_under, y_under_pred)
conf_under
acc1 = round(conf_under[1, 1]/sum(conf_under[1, :]), 3)
acc0 = round(conf_under[0, 0]/sum(conf_under[0, :]), 3)
acc1, acc0

# Q2a. Oversampling #
df_pos = df[df['deposit'] == 1]
df_neg = df[df['deposit'] == 0]
df_pos = df_pos.sample(n=len(df_neg), replace=True)
df_over = pd.concat([df_pos, df_neg])
df_over.shape
df_over['deposit'].mean()

# Q2b. Logistic regression classifier #
y_over, X_over = df_over['deposit'], df_over.drop(columns='deposit')
clf2 = LogisticRegression(max_iter=2000)
clf2.fit(X_over, y_over)
y_over_pred = clf2.predict(X_over)
conf_over = confusion_matrix(y_over, y_over_pred)
conf_over
acc1 = round(conf_over[1, 1]/sum(conf_over[1, :]), 3)
acc0 = round(conf_over[0, 0]/sum(conf_over[0, :]), 3)
acc1, acc0
