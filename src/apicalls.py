import requests
import json
import os


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"



#Call each API endpoint and store the responses
response1 = requests.post(
    url = URL + "/prediction", 
    data = {"filename": os.path.join(test_data_path, "testdata.csv")}
).content
print(f"Response1: {response1}")

response2 = requests.get(url = URL + "/scoring").content
print(f"Response2: {response2}")

response3 = requests.get(url = URL + "/summarystats").content
print(f"Response3: {response3}")

response4 = requests.get(url = URL + "/diagnostics").content
print(f"Response4: {response4}")

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace
with open(os.path.join(os.getcwd(), output_model_path, "apireturns.txt"), 'w') as file:
    for item in responses:
            file.write(str(item) + "\n")
    #file.write(responses)



