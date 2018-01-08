# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:19:10 2017

@author: dgmcl

evaluation.py is a script that is used to calculate
different metrics for the 3 different models created by the
3 different algorithms. Each of the models are trained 
and evaluated. Confusion Matrices and statistics on
accuracy, recall, precision and f1 are output.

"""

import prep_pred
import rankings
import tournaments
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

#function to create train and test sets
def train_test(df):
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test

#reading in different forms of the data for the different algorithms	
df = pd.read_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/df.csv")
df_stand = pd.read_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/df_stand.csv")
df_norm = pd.read_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/df_norm.csv")

#new dataset with features from feature selection process for random forest
df_rf = df[['The Winner', 'ATP', 'Round', 'Series', 'Tournament', 'Surface_Clay', 'Surface_Grass',
	'Surface_Hard', 'Court_Indoor', 'Court_Outdoor', 'Best of_3',
	'Best of_5', 'Points', 'Rank', 'ATP250', 'ATP500', 'Grand Slam',
	'Masters 1000', 'Outdoor', 'Hard', 'Clay', 'Grass', 'Win', 'Upset',
	'H2H']]

#new dataset with features from feature selection process for neural nets
df_nn = df_stand[['The Winner', 'Round', 'Series', 'Tournament', 'Surface_Clay', 'Surface_Grass',
       'Court_Outdoor', 'Best of_3', 'Best of_5', 'Points', 'Rank', 'ATP250',
       'Grand Slam', 'Masters Cup', 'Indoor', 'Outdoor', 'Hard', 'Clay',
       'Grass', 'H2H']]

#creating test and train sets for each of the different algorithms   
X_train, X_test, y_train, y_test = train_test(df_rf)
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test(df_norm)
X_train_stand, X_test_stand, y_train_stand, y_test_stand = train_test(df_nn) 

#initialising random forest classifier
rf  = RandomForestClassifier(n_estimators=500, criterion ='gini',  max_features = 'auto')
#fitting training data to create model
rf.fit(X_train, y_train)
#saving prediction made on test set
predicted_rf = rf.predict(X_test)
#getting accuracy of the predictions
accuracy_rf = accuracy_score(y_test, predicted_rf)
print("Random Forest Test Accuracy: ", accuracy_rf)
#creating confusion matrix and displaying in graph
confmat_rf = confusion_matrix(y_test,predicted_rf)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat_rf, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat_rf.shape[0]):
    for j in range(confmat_rf.shape[1]):
        ax.text(x=j, y=i, s=confmat_rf[i, j], va='center', ha='center')
plt.title('RF Confusion Matrix', y=1.2)
plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.savefig('C:/Users/dgmcl/rf_matrix.png', dpi=300)
plt.show()
#printing out classification report
print(classification_report(y_test,predicted_rf))
        
#initialising ANN classifier
#same processes as followed for random forest 
nn = MLPClassifier(hidden_layer_sizes=(100,100,100))
nn.fit(X_train_stand,y_train_stand)
predicted_nn = nn.predict(X_test_stand)
accuracy_nn = accuracy_score(y_test, predicted_nn)
print("ANN Test Accuracy: ", accuracy_nn)
confmat_nn = confusion_matrix(y_test,predicted_nn)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat_nn, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat_nn.shape[0]):
    for j in range(confmat_rf.shape[1]):
        ax.text(x=j, y=i, s=confmat_nn[i, j], va='center', ha='center')
plt.title('ANN Confusion Matrix', y=1.2)
plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.savefig('C:/Users/dgmcl/nn_matrix.png', dpi=300)
plt.show()
print(classification_report(y_test,predicted_nn))

#initialising SVM classifier
#same processes as followed for Random Forest 
sv = svm.SVC(kernel='poly', C=1, gamma=.8) 
sv.fit(X_train_norm, y_train_norm)
predicted_sv = sv.predict(X_test_norm)
accuracy_sv = accuracy_score(y_test, predicted_sv)
print("SVM Test Accuracy: ", accuracy_sv)
confmat_sv = confusion_matrix(y_test,predicted_sv)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat_sv, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat_rf.shape[0]):
    for j in range(confmat_sv.shape[1]):
        ax.text(x=j, y=i, s=confmat_sv[i, j], va='center', ha='center')
plt.title('SVM Confusion Matrix', y=1.2)
plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.savefig('C:/Users/dgmcl/sv_matrix.png', dpi=300)
plt.show()
print(classification_report(y_test,predicted_sv))