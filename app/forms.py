# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:35:09 2017

@author: dgmcl

forms.py contains a Class, MatchForm. This sets out the unqiue
structure and contents for the form used on the web application.
It predefines and prepopulates the form with data taken from the 
datasets contained in the tmp folder. 

"""

from flask_wtf import Form
from wtforms import TextField, IntegerField, TextAreaField, SubmitField, RadioField, SelectField
import pandas as pd
import numpy as np
from wtforms import validators, ValidationError

#reading in player profiles dataset
f = "tmp/alt_profiles.csv"
data = pd.read_csv(f)
df = pd.DataFrame(data)
#only including the top 130 players
df = df.iloc[:130, :]
#place in list 
names = df.Name.tolist()
#aplhabetically sort  
names = sorted(names, key=str.lower)
#creating a list of tuples for insertion into dropdown
players = []
for i in names:
    players.append((i,i))

#reading in tournaments dataset
f2 = "tmp/tournaments.csv"
d = pd.read_csv(f2)
df1 = pd.DataFrame(d)
#placing tournament names into a list
t_names = df1.Tournament.tolist()
#removing a trailing Nan
t_names.remove(t_names[-1])
#aplbabetically sort
t_names = sorted(t_names, key=str.lower)
#creating a list of tuples for insertion into dropdown
tournaments = []
for i in t_names:
    tournaments.append((i,i))
    
#A class that receives a Flask WTForm and instantiates a new form  
#for rendering on the web page 
class MatchForm(Form):
   p1 = SelectField("Player 1", choices = players)
   p2 = SelectField("Player 2", choices = players)
   t = SelectField("Tournament", choices = tournaments)
   r = SelectField("Round",choices = [('1st Round', '1st Round'), ('2nd Round', '2nd Round'), ('2nd Round', '3rd Round'), ('4th Round', '4th Round'), ('Quarterfinals', 'Quarterfinals'), ('Semifinals', 'Semifinals'), ('The Final', 'The Final')])
   submit = SubmitField("Send")