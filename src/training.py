from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    
    '''
    1. 
    Read in finaldata.csv using the pandas module. 
    The directory that you read from is specified in the 
    output_folder_path of your config.json starter file.
    '''
    df = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, "finaldata.csv"))
    df = df.drop(columns=['corporation'])
    
    # Separate your data into features (X) and target (y)
    X = df.drop(columns=['exited'])
    y = df['exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #print(X_train)
    #print(y_train)

    '''
    2. 
    Use the scikit-learn module to train an ML model 
    on your data. The training.py starter file already 
    contains a logistic regression model you should use for training.
    '''

    #use this logistic regression for training
    LR = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # Without StandardScaler: acc = 0.66
    # With StandardScaler: acc = 0.833
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', LR)
    ])
    
    #fit the logistic regression to your data
    model = pipeline.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy of the logistic regression model: ', accuracy)

    # Report contains precision, recall, f-1 score, support, accuracy
    print(metrics.classification_report(y_test, y_pred))

    '''
    3. 
    Write the trained model to your workspace, 
    in a file called trainedmodel.pkl. The directory 
    you'll save it in is specified in the 
    output_model_path entry of your config.json starter file.
    '''
    if not os.path.exists(os.path.join(os.getcwd(), model_path)):    
        os.makedirs(os.path.join(os.getcwd(), model_path))

    pickle.dump(model, open(os.path.join(os.getcwd(), model_path, "trainedmodel.pkl"), 'wb'))

if __name__ == '__main__':
    train_model()