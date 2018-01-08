# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:05:41 2017

@author: dgmcl
 
 views.py is the script that is responsible for the @route
 function of the flask application. It only has the '/' or index
 directory to route. When the application is initially run it
 initialises a MatchForm and passes it to the template for rendering. 
 When it receives a 'POST' request it gathers the data from the form, 
 transforms it using query_pred.py, loads the model and passes the 
 query to it for prediction. It gathers the returned data, calculates 
 probabalities, decimal odds and fractional odds for both players 
 and passes them to the template for rendering. 


"""

from app import app, forms, query_pred
import pandas as pd
from flask import render_template, request, flash
from sklearn.externals import joblib
import numpy as np
from fractions import Fraction

#route options for when the index page of the application is requested
#it has options for GET and POST HTTP Requests
@app.route('/', methods = ['GET', 'POST'])
def predict():
	#initialises a form to pass to template
    form = forms.MatchForm()
	#if the user has submitted a form 
    if request.method =='POST':
		#if the form is invalid, throw an error
        if form.validate() == False:
            flash('All fields are required')
            return render_template('index.html', form = form)
		#if it's a perfect form submission
        else:
			#placing all the form values into variables using request
            w = ''
            p1 = request.form['p1']
            p2 = request.form['p2']
            t = request.form['t']
            r = request.form['r']
			#creating a list of the data to create a dataframe from it
            q = [w, p1, p2, t , r]
			#create a dataframe with the appropriate column headers
            query = pd.DataFrame(data=[q], columns = ["The Winner", "P1", "P2", "Tournament", "Round"])
			#pass to script to transform before passing to model 
            query = query_pred.transform(query)
            #loading the model 
            rf = joblib.load('tmp/rf_model.sav')
			#separating the target from the features
            X, y = query.iloc[:, 1:].values, query.iloc[:, 0].values
			#asking the model to make a prediction
            prediction = rf.predict(X)
			#getting specific probability info from the model 
            prob = rf.predict_proba(X)
			#assigning probabilty info to different players
            prob1 = prob[0][0]
            prob2 = prob[0][1]
			#getting the models prediction (P1 or P2)
            p = prediction[0]
			#calculating player 1 decimal odds
            dec1 = 1/prob1
            dec1 = format(dec1, '.2f')
			#calculating payer 1 fractional odds
            frac1 = (1/prob1) - 1
            frac1 = format(frac1, '.2f')
            float(frac1)
            frac1 = Fraction(frac1).limit_denominator(50)
			#calculating player 2 decimal odds
            dec2 = 1/prob2
            dec2 = format(dec2, '.2f')
			#calculating payer 2 fractional odds
            frac2 = (1/prob2) - 1
            frac2 = format(frac2, '.2f')
            float(frac2)
            frac2 = Fraction(frac2).limit_denominator(50)
			
			#cecking to see which player was the winner to display their name
            if(p == 'P1'):
                result = p1

            elif(p == 'P2'):
                result = p2
			
    		#passing all of the information needed to the template for rendering the prediction and stats 
            return render_template('index.html', form = form, p1 = p1, p2 =p2, t=t, r=r, prediction= result, prob1 = prob1 , frac1 = frac1, dec1= dec1, prob2 = prob2,frac2 = frac2, dec2= dec2 )
    #as soon as the page initialises render page with form 
    elif request.method == 'GET':
        return render_template('index.html', form = form)
