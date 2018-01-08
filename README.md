# predapp
Predapp (Predatour) is a python application hosted on a Flask webframework. It contains a predictive model made on scikit-learn that
predicts the outcome of Association of Tennis Professionals (ATP) men's matches from ATP 250 series level and upwards. Users can input 
queries via the Match Form, which are then passed to the Random Forest model and the models predictions are rendered to the screen. The
application also gives statistics such as implied probability, fractional and decimal odds for those to compare against bookmakers. 
The data was taken from www.tennis-data.co.uk and covers matches from 2014- present day. 

-Data Preparation Scripts folder includes all of the scripts for creating the dataset for training the model
and model creation. It saves the datasets locally and into the tmp folder in the application

-Predatour is the online application which includes the Flask files, the model, player profile and tournament
datasets, HTML templates, a requirements file for dependencies in the virtual environment and a Procfile for
running on Heroku

Steps to Prepare the Data and Train the Model
1. Ensure that the system you are running the scripts on have the dependecies which are Python 2.7 and 
the Python packages; NumPy, scikit-learn, pandas
2. Run the script: Data Preparation Scripts/dataset_prep.py

Steps to Run the web application locally
1. Ensure in the Flask file that the app.run() function is set to run on port 5000.
2. Ensure that the form action in the templates folder is set to localhost:5000.
3. Ensure that the local environment has virtualenv installed
4. Create a virtualenv file with the command: virtualenv <folder_name>
5. Place all of the Application contents in that folder
6. Activate the virtualenv with the command whilst in the app's directory: activate
7. Using pip, install all of the requirements from the requirements file
8. To get the application running use the command: python run.py
9. Open the browser and go to "localhost:5000"

The Application is also currently hosted on heroku at "https://predapp.herokuapp.com"

