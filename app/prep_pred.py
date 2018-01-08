# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:02:03 2017

@author: dgmcl

prep_pred.py is a script that contains various functions used
to prepare the data for training and for querying. The raw source 
data and gathered and cleaned initially. New features are then
created and transformed. Several csv files are returned that contain
the completed dataset in raw, normalised and standardised formats. 

"""

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

profiles = "C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/live_profiles.csv"

#appends the source data together
def create_dataset(filepath):
    df = pd.DataFrame()
    for f in glob.glob(filepath):
        data = pd.read_excel(f)
        d = pd.DataFrame(data)
        df = df.append(d, ignore_index=True)
    return df

#removes unnecessary features
def drop_features(df, c_drop):
    for c in c_drop:
        if c in df.columns:
            df.drop(c,axis=1,inplace = True)
    return df

#limits the dataset to players ranked within top 100 ATP
def top100(df):
    df = df[df['WRank'] < 150]
    df = df[df['LRank'] < 150]
    return df

#marks whether or not the match was an upset
def upset(df):
    df['Upset'] = np.where(df['WRank'] > df['LRank'], 1, 0)
    return df

#creates statistics for all the players for the dataset
#and adds their most recent statistics to their profile
def ratios(df, f, x ,y):
    alt_profiles = "C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/alt_profiles.csv"
    profiles = pd.read_csv(alt_profiles)

    for i in x:
        d = df.loc[(df.Winner == i) | (df.Loser == i)]
        for j in y:
            win = 0
            total = 0
            for index, row in d.iterrows():
                if((row[f] == j) & (row.Winner == i)):
                    win += 1
                    total += 1
                    df.loc[index, "P1 " + str(j)] = win/total
                elif((row[f] == j) & (row.Loser == i)):
                    total +=1
                    df.loc[index, "P2 " + str(j)] = win/total
                else:
                    if(row.Winner == i):
                        if(total == 0):
                            df.loc[index, "P1 " + str(j)] = 0
                        else:          
                            df.loc[index, "P1 " + str(j)] = win/total
                    elif(row.Loser == i):
                        if(total == 0):
                            df.loc[index, "P2 " + str(j)] = 0
                        else:          
                            df.loc[index, "P2 " + str(j)] = win/total
            if(total == 0):
                profiles.loc[(profiles.Name == i), str(j)] = 0
            else:
                profiles.loc[(profiles.Name == i), str(j)] = win/total  
                                   
        del d

    
    profiles.to_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/alt_profiles.csv", sep =',', index = False)  
    profiles.to_csv("C:/Users/dgmcl/Desktop/predapp/tmp/alt_profiles.csv", sep =',', index = False)
    return df

#creates win/loss/upset statistics for all the players for the dataset
#and adds their most recent statistics to their profile
def win_ratio(df, x):
    alt_profiles = "C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/alt_profiles.csv"
    profiles = pd.read_csv(alt_profiles)
    for i in x:
        d = df.loc[(df.Winner == i) | (df.Loser == i)]
        win = 0
        total = 0
        for index, row in d.iterrows():
            if(row.Winner == i):
                win += 1
                total += 1
                df.loc[index, "P1 Win"] = win/total
            elif(row.Loser == i):
                total += 1
                df.loc[index, "P2 Win"] = win/total
        if (total == 0):
            profiles.loc[(profiles.Name == i), "Win"] = 0
        else:
            profiles.loc[(profiles.Name == i), "Win"] = win/total
     
        
        upset = 0
        total2 = 0
        for index, row in d.iterrows():
            if((row.Winner == i) & (row['WRank'] < row['LRank'])):
                upset += 1
                total2 += 1
                df.loc[index, "P1 Upset"] = upset/total2
            elif((row.Loser == i) & (row['WRank'] < row['LRank'])):
                total2 += 1
                df.loc[index, "P2 Upset"] = upset/total2
            else:
                if(row.Winner == i):
                    if(total2 == 0):
                        df.loc[index, "P1 Upset" ] = 0
                    else:          
                        df.loc[index, "P1 Upset" ] = upset/total2
                elif(row.Loser == i):
                    if(total2 == 0):
                        df.loc[index, "P2 Upset"] = 0
                    else:          
                        df.loc[index, "P2 Upset"] = upset/total2
                    
        if(total2 == 0):
                profiles.loc[(profiles.Name == i), 'Upset'] = 0
        else:
                profiles.loc[(profiles.Name == i), 'Upset'] = upset/total2            
        
    del d
    profiles.to_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/alt_profiles.csv", sep =',', index = False)  
    profiles.to_csv("C:/Users/dgmcl/Desktop/predapp/tmp/alt_profiles.csv", sep =',', index = False)
    return df

#computes the head to head statistics for each player in each match
#creates a matrix of head to head results for all players
def H2H(df, players):
    alt_profiles = "C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/alt_profiles.csv"
    profiles = pd.read_csv(alt_profiles)
    
    for i in players:
        profiles[i] = 0
    
    df['P1 H2H'] = 0
    df['P2 H2H'] = 0
    list1 = []
    
    for index, row in df.iterrows():
        if (index == 0):
            list1.append((row.Winner, row.Loser))
        elif (((row.Winner,row.Loser) not in list1) & ((row.Loser, row.Winner) not in list1)):
            list1.append((row.Winner, row.Loser))
            
    
    for i in list1:
        p1w = 0
        p2w = 0
        count = 0
        p1 = i[0]  
        p2 = i[1]
        
        d = df.loc[(((df.Winner == p1) | (df.Winner == p2))) & (((df.Loser == p1) | (df.Loser == p2)))]
                                      
        for i,r in d.iterrows():
            if(r.Winner == p1):
                p1w += 1
                count += 1
                p2w= p2w
                if(p2w == 0):
                    df.loc[i, 'P1 H2H'] = p1w/count
                    df.loc[i, 'P2 H2H'] = 0
                else:
                    df.loc[i, 'P1 H2H'] = p1w/count
                    df.loc[i, 'P2 H2H'] = p2w/count
            elif(r.Winner == p2):
                p2w += 1
                count += 1
                p1w= p1w
                if(p1w == 0):
                    df.loc[i, 'P1 H2H'] = p2w/count
                    df.loc[i, 'P2 H2H'] = 0
                else:
                    df.loc[i, 'P1 H2H'] = p2w/count
                    df.loc[i, 'P2 H2H'] = p2w/count  
        
        profiles.loc[(profiles.Name == p1), p2] = p1w/count
        profiles.loc[(profiles.Name == p2), p1] = p2w/count       
        del d
    
    profiles.to_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/alt_profiles.csv", sep =',', index = False)    
    profiles.to_csv("C:/Users/dgmcl/Desktop/predapp/tmp/alt_profiles.csv", sep =',', index = False)
    return df

#creates new surface features for both players and calls ratios function
def surface_prep(df, players):
    surfaces = np.unique(df['Surface'])
    df['P1 Grass'] = 0
    df['P1 Clay'] = 0
    df['P1 Hard'] = 0
    df['P2 Grass'] = 0
    df['P2 Clay'] = 0
    df['P2 Hard'] = 0
    df = ratios(df, 'Surface', players, surfaces)
    return df

#creates new court features for both players and calls ratios function
def court_prep(df, players):
    court = np.unique(df['Court'])
    df['P1 Indoor'] = 0
    df['P2 Indoor'] = 0
    df['P1 Outdoor'] = 0
    df['P2 Outdoor'] = 0
    df = ratios(df, 'Court', players, court)
    return df

#creates new series features for both players and calls ratios function
def series_prep(df, players):
    series = np.unique(df['Series'])
    df['P1 ATP250'] = 0
    df['P1 ATP500'] = 0
    df['P1 Grand Slam'] = 0
    df['P1 Masters 1000'] = 0
    df['P1 Masters Cup'] = 0
    df['P2 ATP250'] = 0
    df['P2 ATP500'] = 0
    df['P2 Grand Slam'] = 0
    df['P2 Masters 1000'] = 0
    df['P2 Masters Cup'] = 0
    df = ratios(df, 'Series', players, series)
    return df

#removes all rows with Nan values in the dataset
def missing_values(df):
    df = df.dropna()
    return df

#transforms series values by mapping
def mapping_series(df):
    series_mapping = {'ATP250': 1,
                      'ATP500': 2, 
                      'Masters 1000': 3,
                      'Grand Slam' : 4,
                      'Masters Cup' : 5}
    df['Series'] = df['Series'].map(series_mapping)
    return df

#transforms round values by mapping
def mapping_round(df):
    round_mapping = {'Round Robin' : 1, '1st Round' : 2, '2nd Round' : 3, '3rd Round' : 4, '4th Round' : 5, 'Quarterfinals' : 6, 'Semifinals' : 7, 'The Final' : 8}
    df['Round'] = df['Round'].map(round_mapping)
    return df

#unused function that could add playing hand feature to dataset in future
def player_hand(df, profile, players):
    df['P1 Hand'] = "Unknown"
    df['P2 Hand'] = "Unknown"
    
    profiles = pd.read_csv(profile)
    
    for i in players:
        x = profiles.loc[(profiles.Name == i), 'Dominant Hand']
        for k,v in x.iteritems():
            dom_hand = v
            df.loc[(df.Winner == i), 'P1 Hand'] = dom_hand
            df.loc[(df.Loser == i), 'P2 Hand'] = dom_hand 
    return df

#preparing the player_hand feature for population
def mapping_hand(df):
    hand_mapping = {'Right-handed' : 1, 'Left-handed (two-handed backhand)' : 2, 'Unknown' : 3}
    df['P1 Hand'] = df['P1 Hand'].map(hand_mapping)
    df['P2 Hand'] = df['P2 Hand'].map(hand_mapping)
    return df

#one-hot encoding for nominal features
def get_dummies(df):
    columns = ['Surface', 'Court', 'Best of']
    df = pd.get_dummies(df, columns = columns)
    df.rename(columns={'Best of_3.0': 'Best of_3', 'Best of_5': 'Best of_5'})
    return df

#mapping surface to numemrical values
def mapping_surface(df):
    surface_mapping = {'Grass' : 1, 'Clay' : 2, 'Hard' : 3}
    df['Surface'] = df['Surface'].map(surface_mapping)
    return df

#mapping court to numerical values
def mapping_court(df):
    court_mapping = {'Indoor' : 1, 'Outdoor' : 2}
    df['Court'] = df['Court'].map(court_mapping)
    return df

#mapping tournament names to numerical values
def mapping_tournament(df):
    tournament_map = {label:idx for idx, label in enumerate(np.unique(df['Tournament']))}
    df['Tournament'] = df['Tournament'].map(tournament_map)
    return df

#mapping players to numerical values
def mapping_player(df, p):
    player_map = {}
    count = 0
    for i in p:
        player_map[i] = count
        count += 1
    df['P1'] = df['Winner'].map(player_map)
    df['P2'] = df['Loser'].map(player_map)
    return df

#standardising the dataset 
def standardize(df):
    col_names = ['ATP', 'Round', 'Series', 'Tournament', 'Surface_Clay',
       'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor',
       'Best of_3', 'Best of_5', 'Points', 'Rank', 'ATP250', 'ATP500',
       'Grand Slam', 'Masters 1000', 'Masters Cup', 'Indoor', 'Outdoor',
       'Hard', 'Clay', 'Grass', 'Win', 'Upset', 'H2H']
    features = df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df[col_names] = features
    return df

#normalising the dataset
def normalize(df):
    col_names = ['ATP', 'Round', 'Series', 'Tournament', 'Surface_Clay',
       'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor',
       'Best of_3', 'Best of_5', 'Points', 'Rank', 'ATP250', 'ATP500',
       'Grand Slam', 'Masters 1000', 'Masters Cup', 'Indoor', 'Outdoor',
       'Hard', 'Clay', 'Grass', 'Win', 'Upset', 'H2H']
    features = df[col_names]
    mms = MinMaxScaler().fit(features.values)
    features = mms.transform(features.values)
    df[col_names] = features
    return df

#removing the date feature
def drop_date(df):
    df.drop('Date', axis=1,inplace =True)
    return df

#rearranging the dataset to fit the model shape
def rearrange(df):

    df = df[['The Winner', 'ATP', 'Round', 'Series', 'Tournament', 'Surface_Clay', 'Surface_Grass',
       'Surface_Hard', 'Court_Indoor', 'Court_Outdoor', 'Best of_3',
       'Best of_5', 'Points', 'Rank', 'ATP250', 'ATP500',
       'Grand Slam', 'Masters 1000', 'Masters Cup', 'Indoor', 'Outdoor',
       'Hard', 'Clay', 'Grass', 'Win', 'Upset', 'H2H']]
    return df

#splitting the dataset into half P1 as winners and half P2 as winners
def split_winners(df):
    #changing feature names from Winner/Loser to Player1/Player2
    df.rename(columns={'WRank': 'P1 Rank', 'LRank': 'P2 Rank', 'WPts' : 'P1 Points', 'LPts' : 'P2 Points'}, inplace = True)
    
    #creating two seperate datasets to make half P1 as winner and other P2 as winner
    x,y = np.array_split(df, 2)
    x['The Winner'] = 'P1'
    y['The Winner'] = 'P2'
    
    #reassigning P1 stats as P2 and vice versa to match the switched assignment above
    y.columns.values[8] = 'P2 Rank'
    y.columns.values[2] = 'P1 Rank'
    y.columns.values[7] = 'P2 Points'
    y.columns.values[1] = 'P1 Points'
    y.columns.values[10] = 'P2 Grass'
    y.columns.values[11] = 'P2 Clay'    
    y.columns.values[12] = 'P2 Hard'    
    y.columns.values[13] = 'P1 Grass'    
    y.columns.values[14] = 'P1 Clay'            
    y.columns.values[15] = 'P1 Hard'    
    y.columns.values[16] = 'P2 Indoor'    
    y.columns.values[17] = 'P1 Indoor'    
    y.columns.values[18] = 'P2 Outdoor'    
    y.columns.values[19] = 'P1 Outdoor'       
    y.columns.values[20] = 'P2 ATP250'    
    y.columns.values[21] = 'P2 ATP500'    
    y.columns.values[22] = 'P2 Grand Slam'    
    y.columns.values[23] = 'P2 Masters 1000'    
    y.columns.values[24] = 'P2 Masters Cup'    
    y.columns.values[25] = 'P1 ATP250'    
    y.columns.values[26] = 'P1 ATP500'
    y.columns.values[27] = 'P1 Grand Slam'
    y.columns.values[28] = 'P1 Masters 1000'
    y.columns.values[29] = 'P1 Masters Cup'
    y.columns.values[30] = 'P1 Win'
    y.columns.values[31] = 'P2 Win'
    y.columns.values[32] = 'P1 Upset'
    y.columns.values[33] = 'P2 Upset'
    y.columns.values[35] = 'P1 H2H'
    y.columns.values[34] =   'P2 H2H'          
    
    y = y[['ATP', 'P2 Points', 'P2 Rank', 'Loser', 'Round', 'Series', 'Tournament',
       'P1 Points', 'P1 Rank', 'Winner', 'P1 Grass', 'P1 Clay', 'P1 Hard',
       'P2 Grass', 'P2 Clay', 'P2 Hard', 'P1 Indoor', 'P2 Indoor',
       'P1 Outdoor', 'P2 Outdoor', 'P1 ATP250', 'P1 ATP500', 'P1 Grand Slam',
       'P1 Masters 1000', 'P1 Masters Cup', 'P2 ATP250', 'P2 ATP500',
       'P2 Grand Slam', 'P2 Masters 1000', 'P2 Masters Cup', 'P2 Win',
       'P1 Win', 'P2 Upset', 'P1 Upset', 'P1 H2H', 'P2 H2H', 'Surface_Clay',
       'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor',
       'Best of_3', 'Best of_5', 'The Winner']]
    
    #bringing the datasets back together
    df = x.append(y, ignore_index=True)
    df.drop('Winner',axis=1,inplace = True)
    df.drop('Loser',axis=1,inplace = True)
    
    return df

#gathering the maximum and minimum value for each feature for normalisation use later
def max_min(df):
    col_names =  ['ATP', 'Round', 'Series', 'Tournament', 'Surface_Clay',
       'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor',
       'Best of_3', 'Best of_5', 'Points', 'Rank', 'ATP250', 'ATP500',
       'Grand Slam', 'Masters 1000', 'Masters Cup', 'Indoor', 'Outdoor',
       'Hard', 'Clay', 'Grass', 'Win', 'Upset', 'H2H']
    mm = {}
    for i in col_names:
        mm[i] = pd.Series([df[i].max(), df[i].min()], index =['Max', 'Min'])
    
    max_min = pd.DataFrame(mm) 
    max_min.to_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/max_min_values.csv", sep =',', index = False)

    ms = {}
    for i in col_names:
        ms[i] =  pd.Series([df[i].mean(), df[i].std()], index =['Mean', 'Std'])
        
    std = pd.DataFrame(ms)
    std.to_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/std_values.csv", sep =',', index = False)

#creating the new features from current features
def feat_fix(df):
    df['Points'] = df['P1 Points'] - df['P2 Points']
    df['Rank'] = df['P1 Rank'] - df['P2 Rank']
    df['ATP250'] = df['P1 ATP250'] - df['P2 ATP250']
    df['ATP500'] =  df['P1 ATP500'] - df['P2 ATP500']
    df['Grand Slam'] = df['P1 Grand Slam'] - df['P2 Grand Slam']
    df['Masters 1000'] = df['P1 Masters 1000'] -  df['P2 Masters 1000']
    df['Masters Cup'] = df['P1 Masters Cup'] -  df['P2 Masters Cup']
    df['Indoor'] = df['P1 Indoor'] - df['P2 Indoor']
    df['Outdoor'] = df['P1 Outdoor'] - df['P2 Outdoor']
    df['Hard'] = df['P1 Hard'] - df['P2 Hard']
    df['Clay'] = df['P1 Clay'] - df['P2 Clay']
    df['Grass'] = df['P1 Grass'] - df['P2 Grass']
    df['Win'] =  df['P1 Win'] -  df['P2 Win']
    df['Upset'] = df['P1 Upset'] - df['P2 Upset']
    df['H2H'] = df['P1 H2H'] - df['P2 H2H']
    
    df = drop_features(df, ['P1 Points', 'P2 Points', 'P1 Rank','P2 Rank','P1 ATP250', 'P2 ATP250','P1 ATP500', 'P2 ATP500', 'P1 Grand Slam','P2 Grand Slam',
                            'P1 Masters 1000', 'P2 Masters 1000', 'P1 Masters Cup', 'P2 Masters Cup','P1 Indoor', 'P2 Indoor', 'P1 Outdoor',
                            'P2 Outdoor', 'P1 Hard', 'P2 Hard', 'P1 Clay', 'P2 Clay', 'P1 Grass', 
                            'P2 Grass', 'P1 Win', 'P2 Win', 'P1 Upset', 'P2 Upset', 'P1 H2H', 'P2 H2H'])
    

    return df

def fix_profiles():
    alt_profiles = "C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/alt_profiles.csv"
    profiles = pd.read_csv(alt_profiles)
    group = ['Clay','Grass', 'Hard', 'Indoor', 'Outdoor', 'ATP250', 'ATP500', 'Grand Slam', 'Masters 1000']
    for i in group:
        median = profiles[i].median()
        profiles[i] = np.where(profiles[i] == 0, median, profiles[i])
        profiles[i]=profiles[i].fillna(median)
        
    profiles.to_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/alt_profiles.csv", sep =',', index = False)    
    profiles.to_csv("C:/Users/dgmcl/Desktop/predapp/tmp/alt_profiles.csv", sep =',', index = False)
    

#function to run the entire process of data prep
#creates 3 different versions of the dataset as Raw, Normalised and Standardised  and saves them externally
def prep():
    df = create_dataset("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/201*.xlsx")
    c_drop = ['Comment', 'Location','B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 'PSW', 'PSL', 'SJW', 'SJL','L1', 'L2','L3','L4','L5', 'Lsets', 'W1', 'W2','W3','W4','W5', 'Wsets', "AvgL", "AvgW", "MaxL", "MaxW"]
    df = top100(df)
    players = np.unique(df[['Winner', 'Loser']])
    df = drop_features(df, c_drop)
    df = missing_values(df)
    df = upset(df)
    df = surface_prep(df, players)
    df = court_prep(df, players)
    df = series_prep(df, players)
    df = win_ratio(df, players)
    df = H2H(df, players)
    df = mapping_series(df)
    df = mapping_round(df)
    df = get_dummies(df)
    df = mapping_tournament(df)
    df = drop_features(df, ['Date', 'Upset'])
    df = df.sample(frac=1)
    df = split_winners(df)
    df = feat_fix(df)
    df = rearrange(df)
    fix_profiles()
    max_min(df)
    df.to_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/df.csv", sep =',', index = False)
    df1 = df
    df = normalize(df)
    df = rearrange(df)
    df.to_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/df_norm.csv", sep =',', index = False)
    df = standardize(df)
    df = rearrange(df)
    df.to_csv("C:/Users/dgmcl/Desktop/Final Year Project/Tennis Data 1/df_stand.csv", sep =',', index = False)
    return df1
