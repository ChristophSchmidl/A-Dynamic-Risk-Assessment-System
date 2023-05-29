import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path']) 


##############Function for reporting
def score_model():
    df_test = pd.read_csv(os.path.join(os.getcwd(), test_data_path, "testdata.csv"))
    df_test = df_test.drop(columns=['corporation'])

    X_test = df_test.drop(columns=['exited'])
    y_test = df_test['exited']

    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    y_pred = model_predictions(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    plt.savefig(os.path.join(os.getcwd(), output_model_path, "confusionmatrix.png"))


if __name__ == '__main__':
    score_model()
