# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:03:44 2017

@author: dgmcl

bias_variance.py is a script used to plot learning 
and validation curves in order to visualise the
presence of bias or variance in the model
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("Source Data/df.csv")

def train_test(df):
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test(df)

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(penalty='l2', random_state=0))])

train_sizes, train_scores, test_scores =learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Tennis Dataset: Bias & Variance')
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.show()