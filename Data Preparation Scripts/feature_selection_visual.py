# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:17:12 2017

@author: dgmcl

feature_selection_visual.py is a script used in conjunction
with the SBS or Sqequential Backward Selection class to 
perform SBS on the dataset according to whichever algorithm
is chosen. A graph illustrating the performance of each algorithm
at each stage of the SBS process. 

"""
import pandas as pd
import matplotlib.pyplot as plt
from SBS import SBS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import svm

#function to create test and train split
def train_test(df):
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    return X_train, X_test, y_train, y_test

#different forms of the data read in 
df_raw = pd.read_csv("Source Data/df.csv")
df_stand = pd.read_csv("Source Data/df_stand.csv")
df_norm = pd.read_csv("Source Data/df_norm.csv")

#creating train and test splits for each data form
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test(df_raw)
X_train_stand, X_test_stand, y_train_stand, y_test_stand = train_test(df_stand)
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test(df_norm)

#initialising the different classifiers
rf = RandomForestClassifier(n_estimators = 100)
sv = svm.SVC(kernel='rbf', C=1, gamma=1) 
nn = MLPClassifier()

# selecting features
sbs = SBS(sv, k_features=5)
sbs.fit(X_train_norm, y_train_norm)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.85, 1.0])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.title('SVM Normalised', fontsize=14, fontweight='bold')
plt.show()