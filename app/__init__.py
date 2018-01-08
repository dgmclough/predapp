# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:47:20 2017

@author: dgmcl
"""

from flask import Flask

app = Flask(__name__)
app.config.from_object('config')
from app import views
    