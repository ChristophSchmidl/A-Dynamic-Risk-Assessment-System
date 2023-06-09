from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model(model_path, data_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    
    # 1. Load trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # 2. Load test data
    df_test = pd.read_csv(data_path)
    df_test = df_test.drop(columns=['corporation'])

    X_test = df_test.drop(columns=['exited'])
    y_test = df_test['exited']

    # 3. Calculate F1 score
    y_pred = model.predict(X_test)
    f1 = metrics.f1_score(y_test, y_pred)
    print('F1 score: ', f1)

    # 4. Write result to output_model_path + latestscore.txt
    if not os.path.exists(os.path.join(os.getcwd(), output_model_path)):    
        os.makedirs(os.path.join(os.getcwd(), output_model_path))

    latest_score = open(os.path.join(os.getcwd(), output_model_path, "latestscore.txt"),'w')
    latest_score.write(str(f1)+"\n")

    return f1


if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(), output_model_path, "trainedmodel.pkl")
    data_path = os.path.join(os.getcwd(), test_data_path, "testdata.csv")

    score_model(model_path, data_path)
