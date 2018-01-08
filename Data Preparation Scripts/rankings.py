# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:19:10 2017

@author: dgmcl

rankings.py is a script that creates a player profile table 
in the form of a csv file. 
Source data is downloaded from www.openerarankings.com manually
and saved locally. The script takes information about players 
that is necessary to populate live queries from users. 

"""

import pandas as pd

#creates a row for each player in the order of the current rankings
#with informatin about their ranking and points at the time
#saves it in the form of a csv file 
def create_profiles():
	#file path to source data
    filepath = "Source Data/liveATP.xlsx"
    #reading in the up to date rankings spreadsheet
    data = pd.read_excel(filepath)
    df = pd.DataFrame(data)
	#removing unneccesary columns from the dataset
    df = df[['LAST UPDATE', 'Unnamed: 1', 'POINTS']]
    df = df.iloc[4:1000]
    df = df.reset_index(drop=True)
	#renaming the columns to match the tennis dataset
    df.rename(columns={'LAST UPDATE': 'Rank', 'Unnamed: 1': 'Name', 'POINTS': 'Points'}, inplace = True)
    
    #reformatting all of the names in the file to match their format in the tennis dataset
    for i,r in df.iterrows():
    
        pl = r.Name
        l = pl.split()
        if len(l) == 2:
            df.loc[i, 'Name'] = l[1] + " " + l[0][0] + "."
        elif len(l) == 3:
            df.loc[i, 'Name'] = l[1] + " " + l[2] + " " + l[0][0] + "."
        elif len(l) == 4:
            df.loc[i, 'Name'] = l[1] + " " + l[2] + " " + l[3] + " " + l[0][0] + "."
    
	#a few specific names need to be altered individually to ensure they are not missed 
    df.loc[(df.Name == 'Tsonga J.'), 'Name'] = 'Tsonga J.W.'
    df.loc[(df.Name == 'Martin Del Potro J.'), 'Name'] = 'Del Potro J.M.'
    df.loc[(df.Name == 'Struff J.'), 'Name'] = 'Struff J.L.'
    df.loc[(df.Name == 'Herbert P.'), 'Name'] = 'Herbert P.H.'
    df.loc[(df.Name == 'Stebe C.'), 'Name'] = 'Stebe C.M.'
    df.loc[(df.Name == 'Kuznetsov A.'), 'Name'] = 'Kuznetsov An.'
    
    #export to a csv file
    df.to_csv("alt_profiles.csv", sep =',', index = False)
	df.to_csv("Predatour/tmp/alt_profiles.csv", sep =',', index = False)

          