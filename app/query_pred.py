# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:56:22 2017

@author: dgmcl

query_pred.py is a script that reads in a query from the web applicatin form
and then appends the missing features, cleans the data, transforms it and
shapes the dataset to fit for input into the model for prediction. It utilises 
already made functions from the prep_pred script that were for the training 
dataset preparation. 

"""
from app import prep_pred
import pandas as pd

#gathers the statistics for each player from the profile dataset and amends it to the query
def grab_stats(q, p):
    
    for index, row in q.iterrows():
        p1 = row.P1
        p2 = row.P2
        c =['Rank', 'Points', 'Grass', 'Clay', 
                'Hard', 'Indoor','Outdoor', 'ATP250', 'ATP500', 'Grand Slam',
                'Masters 1000', 'Masters Cup', 'Win', 'Upset', p2]
        p1d = p.loc[(p.Name == p1), c]
        for i,r in p1d.iterrows():
            for col in c:
                if(col == p2):
                    q.loc[index, 'P1 H2H'] = r[col]
                else: 
                    q.loc[index, 'P1 ' + str(col)] = r[col]
                    
        c2 =['Rank', 'Points', 'Grass', 'Clay', 
                'Hard', 'Indoor','Outdoor', 'ATP250', 'ATP500', 'Grand Slam',
                'Masters 1000', 'Masters Cup', 'Win', 'Upset', p1]
        
        p2d = p.loc[(p.Name == p2), c2]
        for i,r in p2d.iterrows():
            for col in c2:
                if(col == p1):
                    q.loc[index, 'P2 H2H'] = r[col]
                else: 
                    q.loc[index, 'P2 ' + str(col)] = r[col]
    return q

#gathers all information about the tournament and appends it to the query
def tournament(q, t):
    c =['ATP', 'Court', 'Series', 'Best of', 'Surface']
    for index, row in q.iterrows():
        tourn = row.Tournament
        t_i = t.loc[(t.Tournament == tourn), c]
        for i,r in t_i.iterrows():
            for col in c:
                q.loc[index, str(col)] = r[col]
    return q
        
#rearranges the order of the model to fit the model 
def rearrange(df):
    df = df[['The Winner', 'ATP', 'Round', 'Series', 'Tournament', 'Surface_Clay', 'Surface_Grass',
'Surface_Hard', 'Court_Indoor', 'Court_Outdoor', 'Best of_3',
'Best of_5', 'Points', 'Rank', 'ATP250', 'ATP500', 'Grand Slam',
'Masters 1000', 'Outdoor', 'Hard', 'Clay', 'Grass', 'Win', 'Upset',
'H2H']]
    return df

#reassigns max and min values after import so can access values easily
def max_min(f):
    data = pd.read_csv(f)
    df = pd.DataFrame(data)
    df = df.rename(index={0: 'Max', 1: 'Min'})
    return df

#reassigns mean and standard deviation values after import so can access values easily
def mean_std(f):
    data = pd.read_csv(f)
    df = pd.DataFrame(data)
    df = df.rename(index={0: 'Mean', 1: 'Std'})
    return df

#normalizes the query values according to the training sets max_min values
def normalize(df, mm):
    col_names = ['ATP', 'Round', 'Series', 'Tournament', 'Surface_Clay',
       'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor',
       'Best of_3', 'Best of_5', 'Points', 'Rank', 'ATP250', 'ATP500',
       'Grand Slam', 'Masters 1000', 'Masters Cup', 'Indoor', 'Outdoor',
       'Hard', 'Clay', 'Grass', 'Win', 'Upset', 'H2H']

    for index,r in df.iterrows(): 
        for i in col_names:
            df.loc[index, i] = ((r[i]) - mm.get_value('Min', i))/ (mm.get_value('Max', i) - (mm.get_value('Min', i)))
    return df

#standardises the query values according to the training sets mean and standard deviation  values
def standardize(df, ss):
    col_names = ['ATP', 'Round', 'Series', 'Tournament', 'Surface_Clay',
       'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor',
       'Best of_3', 'Best of_5', 'Points', 'Rank', 'ATP250', 'ATP500',
       'Grand Slam', 'Masters 1000', 'Masters Cup', 'Indoor', 'Outdoor',
       'Hard', 'Clay', 'Grass', 'Win', 'Upset', 'H2H']
    
    for index,r in df.iterrows(): 
        for i in col_names:
            df.loc[index, i] = (float(r[i]) - float(ss.get_value('Mean', i)))/ float((ss.get_value('Std', i)))

    return df

#improvised one hot encoding for the nominal features
def get_dummies(df):
    col = ['Surface', 'Court', 'Best of']
    df['Surface_Hard'] = 0
    df['Surface_Grass'] = 0
    df['Surface_Clay'] = 0
    df['Best of_5'] = 0
    df['Best of_3'] = 0
    df['Court_Indoor'] = 0
    df['Court_Outdoor'] = 0   
    
    for i in col:
        if(i == 'Surface'):
            for i,r in df.iterrows():
                if(r[i] == 'Hard'):
                    r['Surface_Hard'] = 1
                elif(r[i] == 'Grass'):
                    r['Surface_Grass'] = 1
                elif(r[i] == 'Clay'):
                    r['Surface_Clay'] = 1

        elif(i == 'Court'):
            for i,r in df.iterrows():
                if(r[i] == 'Indoor'):
                    r['Court_Indoor'] = 1
                if(r[i] == 'Outdoor'):
                    r['Court_Outdoor'] = 1
                     
        elif(i == 'Best of'):
            for i,r in df.iterrows():
                if(r[i] == 3.0):
                    r['Best of_3'] = 1
                if(r[i] == 5.0):
                    r['Best of _5.0'] = 1
                    
    return df
                  
#runs all of the functions to transform the query to prepare it for model prediction               
def transform(query):

    profiles = pd.read_csv("tmp/alt_profiles.csv")
    tournaments = pd.read_csv("tmp/tournaments.csv")
    query = grab_stats(query, profiles) 
    query = tournament(query, tournaments)
    query = prep_pred.mapping_series(query)     
    query = prep_pred.mapping_round(query)
    query = get_dummies(query)
    query = prep_pred.drop_features(query, ['Surface', 'Court', 'Best of'])
    query = prep_pred.mapping_tournament(query)
    query = prep_pred.feat_fix(query)
    query = rearrange(query)
                    
    return query
                
    
             

    





