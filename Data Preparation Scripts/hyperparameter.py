# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:13:20 2017

@author: dgmcl

hyperparameter.py is a script that is used for hyperparameter
tuning of the Random Forest and SVM classifier.
Gridsearch is used to brute force the best combination of 
parameters. 

"""


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

#function to create test and train sets
def train_test(df):
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    return X_train, X_test, y_train, y_test

#function to gridsearch hyperparameters for random forest 
def rf_hyper(df):
	#creating train and test set
    X_train, X_test, y_train, y_test = train_test(df)
	#initialise random forest classifier
    rf = RandomForestClassifier()
	#setting the parameters to gridsearch
    parameters = {'n_estimators': [10, 100, 500], 
                  'max_features': ['log2', 'sqrt','auto'], 
                  'criterion': ['entropy', 'gini'],
                  'max_depth': [2, 20], 
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [1,8]
                 }
    
    acc_scorer = make_scorer(accuracy_score)
    #initialising the gridsearch 
    grid_obj = GridSearchCV(rf, parameters, scoring=acc_scorer, verbose = 2)
	#fitting the gridsearch 
    grid_obj = grid_obj.fit(X_train, y_train)
	#printing the best combination of hyperparameters 
    print(grid_obj.best_estimator_)
	#fitting the best hyperparameters to the classifier 
    rf = grid_obj.best_estimator_
	#training the model 
    rf.fit(X_train, y_train)
    #performing predictions on test set 
    predicted = rf.predict(X_test)
	#accuracy score to see if the performance has improved
    accuracy = accuracy_score(y_test, predicted)
    print(accuracy)

#function to gridsearch hyperparameters for SVM 	
def svm_hyper(df):
	#creating train and test set
    X_train, X_test, y_train, y_test = train_test(df)
	#initialise SVM classifier
    sv = svm.SVC()
	#setting the parameters to gridsearch
    parameters = {'Gamma': [0.01, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                  'C': [0.001, 0.01, 0.1, 1, 10], 
                  'kernel': ['poly', 'linear', 'rbf']
                 }
    
    acc_scorer = make_scorer(accuracy_score)
    #initialising the gridsearch 
    grid_obj = GridSearchCV(sv, parameters, scoring=acc_scorer, verbose = 2)
	#fitting the gridsearch 
    grid_obj = grid_obj.fit(X_train, y_train)
    #printing the best combination of hyperparameters 
    print(grid_obj.best_estimator_)
	#fitting the best hyperparameters to the classifier 
    sv = grid_obj.best_estimator_
	#training the model
    sv.fit(X_train, y_train)
    #performing predictions on test set
    predicted = sv.predict(X_test)
	#accuracy score to see if the performance has improved
    accuracy = accuracy_score(y_test, predicted)
    print(accuracy)




