# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:06:25 2017

@author: dgmcl
"""

#!flask/bin/python
import os
from app import app
port = int(os.environ.get("PORT", 5000))
app.run(debug=True, host='0.0.0.0', port=port)