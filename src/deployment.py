from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path']) 


####################function for deployment
def store_model_into_pickle(model_name):

    if not os.path.exists(os.path.join(os.getcwd(), prod_deployment_path)):    
        os.makedirs(os.path.join(os.getcwd(), prod_deployment_path))

    # copy the latest pickle file (trainedmodel.pkl), the latestscore.txt value, 
    with open(os.path.join(os.getcwd(), output_model_path, model_name), 'rb') as file:
        model = pickle.load(file)

    pickle.dump(model, open(os.path.join(os.getcwd(), prod_deployment_path, model_name), 'wb'))

    with open(os.path.join(os.getcwd(), output_model_path, "latestscore.txt"),'rb') as file:
        latest_score = file.read()

    with open(os.path.join(os.getcwd(), prod_deployment_path, "latestscore.txt"), 'wb') as file:
        file.write(latest_score)

    # and the ingestedfiles.txt file into the deployment directory
    with open(os.path.join(os.getcwd(), dataset_csv_path, "ingestedfiles.txt"),'rb') as file:
        ingested_files = file.read()

    with open(os.path.join(os.getcwd(), prod_deployment_path, "ingestedfiles.txt"), 'wb') as file:
        file.write(ingested_files)

if __name__ == '__main__':
    store_model_into_pickle("trainedmodel.pkl")