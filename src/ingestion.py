import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():

    directories = [input_folder_path]
    allowed_suffix = [".csv"]

    ingested_files = []

    df_list = pd.DataFrame(columns=[
        'corporation',
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees',
        'exited']
    )

     # 1. Brosse input_folder_path dir for json or csv files
    for directory in directories:
        filenames = os.listdir(os.path.join(os.getcwd(), directory))

        for filename in filenames:
            #print(os.path.join(os.getcwd(), directory, filename))
            _, file_extension = os.path.splitext(filename)

            if file_extension in allowed_suffix:
                tmp_df = pd.read_csv(os.path.join(os.getcwd(), directory, filename))
                # 2. Merge the datasets together
                df_list=df_list.append(tmp_df)
                # Add ingested filenames
                ingested_files.append(os.path.join(os.getcwd(), directory, filename))

    # 3. Remove duplicates
    result = df_list.drop_duplicates()
    # 4. Write them to an output file "finaldata.csv" inside output_folder_path

    if not os.path.exists(os.path.join(os.getcwd(), output_folder_path)):    
        os.makedirs(os.path.join(os.getcwd(), output_folder_path))

    result.to_csv(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'), index=False)

    #current_datetime = datetime.now()
    #formatted_datetime = str(current_datetime.year)+ '/'+str(current_datetime.month)+ '/'+str(current_datetime.day)

    #allrecords=[sourcelocation,filename,len(data.index),thetimenow]

    ingestion_record = open(os.path.join(os.getcwd(), output_folder_path, "ingestedfiles.txt"),'w')
    for entry in ingested_files:
        ingestion_record.write(str(entry)+"\n")


if __name__ == '__main__':
    merge_multiple_dataframe()
