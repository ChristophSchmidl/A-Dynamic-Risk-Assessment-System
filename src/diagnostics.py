
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
from ingestion import merge_multiple_dataframe
from training import train_model

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(df_dataset):
    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(os.getcwd(), prod_deployment_path, "trainedmodel.pkl"), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(df_dataset)
    print(y_pred)
    return y_pred

##################Function to get summary statistics
def dataframe_summary(df):
    #calculate summary statistics here
    mean = df.mean()
    print("Mean:\n", mean)

    median = df.median()
    print("\nMedian:\n", median)

    std_dev = df.std()
    print("\nStandard Deviation:\n", std_dev)

    return [mean, median, std_dev]

##################Function to count missing data
def missing_data(df):
    missing_values = df.isna().sum()

    # Calculate the total number of entries in each column
    total_values = df.shape[0]

    # Calculate the percentage of missing values
    missing_percentage = (missing_values / total_values) * 100

    missing_percentage_list = missing_percentage.tolist()

    print(f"Missing percentage list: {missing_percentage_list}")

    return missing_percentage_list



##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    #return #return a list of 2 timing values in seconds
    
    start_ingest = timeit.default_timer()
    merge_multiple_dataframe()
    end_ingest = timeit.default_timer() 
    exec_time_ingest = end_ingest - start_ingest

    print(f"Execution time for ingestion: {exec_time_ingest} seconds")

    start_train = timeit.default_timer()
    train_model()
    end_train = timeit.default_timer() 
    exec_time_train = end_train - start_train

    print(f"Execution time for training: {exec_time_train} seconds")

    return[exec_time_ingest, exec_time_train]

##################Function to check dependencies
def outdated_packages_list():
    # Get the list of installed packages and their versions
    output_installed = subprocess.check_output(['pip', 'freeze']).decode()
    
    # Prepare an empty DataFrame
    df = pd.DataFrame(columns=['Package', 'Installed', 'Latest'])
    
    # Split the output into lines and iterate over them
    for line in output_installed.split('\n'):
        if line:
            # Split each line into the package name and installed version
            name, installed_version = line.split('==')
            
            # Use pip to find the latest version
            output_latest = subprocess.check_output(['pip', 'show', name]).decode()
            
            # Parse the output to find the version line
            latest_version = None
            for line_latest in output_latest.split('\n'):
                if line_latest.startswith('Version:'):
                    latest_version = line_latest.split(': ')[1]
            
            # Append the data to the DataFrame
            df = df.append({
                'Package': name,
                'Installed': installed_version,
                'Latest': latest_version
            }, ignore_index=True)
    
    print(df)
    return df


if __name__ == '__main__':

    df_test = pd.read_csv(os.path.join(os.getcwd(), test_data_path, "testdata.csv"))
    df_test = df_test.drop(columns=['corporation'])
    X_test = df_test.drop(columns=['exited'])
    y_test = df_test['exited']

    df_final = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, "finaldata.csv"))
    df_final = df_final.drop(columns=['corporation'])
    X_final = df_test.drop(columns=['exited'])
    y_final = df_test['exited']

    model_predictions(X_test)
    dataframe_summary(X_final)
    missing_data(X_final)
    execution_time()
    outdated_packages_list()





    
