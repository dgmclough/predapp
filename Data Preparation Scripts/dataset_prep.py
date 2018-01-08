# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:35:58 2017

@author: dgmcl

dataset_prep.py is a script that calls other scripts to 
1) prepare the dataset for training
2) create the rankings table for the application
3) create the tournaments table for the application
4) create the model and save it as a file

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

#calling script that creates rankings table
rankings.create_profiles()
#calling script that prepares the dataset for training the model
df = prep_pred.prep()
#calling the script that creates tournaments table
tournaments.create_tournaments()

#reading in the newly created dataset in raw form
df = pd.read_csv("Source Data/df.csv")

#new dataset with features from feature selection process
df_rf = df[['The Winner', 'ATP', 'Round', 'Series', 'Tournament', 'Surface_Clay', 'Surface_Grass',
'Surface_Hard', 'Court_Indoor', 'Court_Outdoor', 'Best of_3',
'Best of_5', 'Points', 'Rank', 'ATP250', 'ATP500', 'Grand Slam',
'Masters 1000', 'Outdoor', 'Hard', 'Clay', 'Grass', 'Win', 'Upset',
'H2H']]

#separating features and targets 
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

#initialising a Random Forest classifier with hyperparameters from tuning process
rf  = RandomForestClassifier(n_estimators=500, criterion ='gini',  max_features = 'auto')
#fitting the data to the classifier to train the model
rf.fit(X, y)
#saving the model    
joblib.dump(rf, 'Predatour/tmp/rf_model.sav')        

