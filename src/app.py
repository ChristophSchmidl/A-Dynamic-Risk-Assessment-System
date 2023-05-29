from flask import Flask, session, jsonify, request, Response
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
#import diagnosis 
#import predict_exited_from_saved_model
import json
import os
from scoring import score_model
from diagnostics import dataframe_summary, execution_time, outdated_packages_list, missing_data


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
test_data_path = os.path.join(config['test_data_path']) 


with open(os.path.join(os.getcwd(), prod_deployment_path, "trainedmodel.pkl"), 'rb') as file:
    prediction_model = pickle.load(file)


def read_pandas(filename):
    df = pd.read_csv(filename)
    return df


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    dataset = request.form.get('filename')
    dataset = os.path.join(os.getcwd(), dataset)

    if os.path.isfile(dataset):
        df = pd.read_csv(dataset)
        df = df.drop(columns=['corporation'])
        # Separate your data into features (X) and target (y)
        X = df.drop(columns=['exited'])
        y = df['exited']

        y_pred = prediction_model.predict(X)

        return Response(str(y_pred), content_type='text/plain', status=200)
    else:
        return "File does not exist", 404


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    return Response(str(score_model()), content_type='text/plain', status=200)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarize():
    df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, "testdata.csv"))
    df = df.drop(columns=['corporation'])
    # Separate your data into features (X) and target (y)
    X = df.drop(columns=['exited'])
    y = df['exited']

    return Response(str(dataframe_summary(X)), content_type='text/plain', status=200)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    #check timing, percent NA values, dependency check
    df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, "testdata.csv"))
    df = df.drop(columns=['corporation'])
    # Separate your data into features (X) and target (y)
    X = df.drop(columns=['exited'])
    y = df['exited']

    # timing
    exec_time_ingest, exec_time_train = execution_time()
    # percent NA values
    missing_percentage_list = missing_data(X)
    # dependency check
    outdated_list = outdated_packages_list().values.tolist()

    return Response([str(exec_time_ingest), str(exec_time_train), str(missing_percentage_list), str(outdated_list)], content_type='text/plain', status=200)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
