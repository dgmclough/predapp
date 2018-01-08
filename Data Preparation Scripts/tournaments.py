# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:18:29 2017

@author: dgmcl
"""

import pandas as pd
import glob

def create_tournaments():
    filepath = "Source Data/201*.xlsx"
    
    df = pd.DataFrame()
    for f in glob.glob(filepath):
        data = pd.read_excel(f)
        d = pd.DataFrame(data)
        df = df.append(d, ignore_index=True)
    
    tournaments = df[['ATP', 'Series', 'Tournament', 'Best of', 'Court', 'Surface']]
    tournaments = tournaments.drop_duplicates(subset='Tournament', keep='first', inplace=False)
    tournaments = tournaments.reset_index(drop=True)
    
    tournaments.to_csv("tournaments.csv", encoding='utf-8-sig', sep =',', index = False)
	tournaments.to_csv("Predatour/tmp/tournaments.csv", sep =',', index = False)
    return tournaments