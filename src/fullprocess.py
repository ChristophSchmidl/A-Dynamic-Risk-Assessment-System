from training import train_model
import scoring
from deployment import store_model_into_pickle
import reporting
from diagnostics import model_predictions
from ingestion import merge_multiple_dataframe
import apicalls
import json
import os
import pandas as pd
from sklearn import metrics
import subprocess


def read_txt_file(file_path):
    # Open the text file in read mode
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Strip newline characters and any leading/trailing whitespace from each line
    lines = [line.strip() for line in lines]

    return lines

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']

##################Check and read new data
#first, read ingestedfiles.txt
ingested_files = read_txt_file(os.path.join(os.getcwd(),prod_deployment_path, "ingestedfiles.txt" ))
print(f"Already ingested files: {ingested_files}")

new_files = []

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

allowed_suffix = [".csv"]
directories = [input_folder_path]

for directory in directories:
    filenames = os.listdir(os.path.join(os.getcwd(), directory))    
    for filename in filenames:
        _, file_extension = os.path.splitext(filename)
        if filename not in ingested_files and file_extension in allowed_suffix:
            new_files.append(filename)

print(f"Newly discovered files: {new_files}")


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) == 0:
    print("No new files discovered. Finishing process...")
    exit()

# Ingest new data
merge_multiple_dataframe()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
latest_score = read_txt_file(os.path.join(os.getcwd(), prod_deployment_path, "latestscore.txt"))[0]
print(f"Previous F1 score: {latest_score}")

model_path = os.path.join(os.getcwd(), prod_deployment_path, "trainedmodel.pkl")
data_path = os.path.join(os.getcwd(), output_folder_path, "finaldata.csv")
f1_new = scoring.score_model(model_path, data_path)
print('New F1 score: ', f1_new)


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if float(f1_new) >= float(latest_score):
    print(f"No need for re-training.")
    exit()
else:
    print(f"Model drift detected.")


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
train_model()
store_model_into_pickle("trainedmodel.pkl")

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
subprocess.call("python diagnostics.py", shell=True)
reporting.score_model()
subprocess.call("python apicalls.py", shell=True)








