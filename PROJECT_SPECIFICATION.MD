# Project specification

## Data Ingestion

| Criteria  | Meets specifications  |
|---|---|
| Student will be able to update the training data for ML models to prepare for re-training | - The ingestion.py script should perform this step.<br>- Every file contained in the data folder needs to be read into Python.<br>- All files should be compiled into a pandas data frame and written to a csv file called “finaldata.csv”. De-dupe the compiled data frame before saving.<br>- Store the ingestion record in a file called “ingestedfiles.txt”. |

## Training, Scoring, and Deployment

| Criteria  | Meets specifications  |
|---|---|
| Student will be able to write a script that trains an ML model and writes the model to storage. | Students should write training.py to accomplish this. The model should be saved in the pickle format. |
| Student will be able to write a script that performs scoring of ML models | Students should write a scoring script in the scoring.py starter file. Scoring should be performed using the F1 score. |
| Student will be able to write ML model scores to persistent storage in files or databases. | The scoring.py script should write the F1 score to a .txt file called latestscore.txt. |
| Student will be able to regularly re-deploy models | The deployment.py script should copy the trained model, the F1 score, and the ingested file record to a production deployment directory. |

## Diagnostics

| Criteria  | Meets specifications  |
|---|---|
| Student will be able to write a script to automatically check the latency of model training and model prediction processes | - The diagnostics.py script should perform this step.<br>- Timing should be checked for both data ingestion and training in seconds.<br>- Summary statistics (means, medians, and modes) should be checked for each numeric column.<br>- Students will create a function for making predictions based on the deployed model and a dataset. |
| Student will be able to check data integrity and stability | - The diagnostics.py script should perform this step.<br>- Data integrity should be checked by measuring the percentage of NA values in each of the numeric dataset’s columns. | 
| Student will be able to check for dependency changes in scripts | - The diagnostics.py script should perform this step.<br>- All modules in requirements.txt need to have their latest versions and currently installed versions checked. |

## Reporting

| Criteria  | Meets specifications  |
|---|---|
| Student will be able to create API's that provide easy automated access to ML model scoring results. | The app.py script will perform this section.<br><br>- An endpoint for scoring needs to provide model scores based on test datasets and models (found in /testdata/).<br>- An endpoint for summary statistics needs to provide summary statistics for the ingested data (found in the directory specified by the output_folder_path in config.json)<br>- An endpoint for diagnostics needs to provide diagnostics for the ingested data (found in the directory specified by the output_folder_path in config.json). The diagnostics should include timing, dependency checks, and missing data checks.<br>- An endpoint for model predictions needs to return predictions from the deployed model (found in the directory specified in the prod_deployment path in config.json) for an input dataset (passed to the endpoint as an input)<br><br>Students will create a function in reporting.py that generates a confusion matrix that shows the accuracy of the model on test data (found in /testdata/). |
| Student will be able to call ML scoring API's and use their data in reports. | In apicalls.py, call API’s to get the model predictions, accuracy score, summary statistics, and diagnostics that are returned by the API endpoints. The apicalls.py script needs to combine these API outputs and write the combined outputs to the workspace, to a file called apireturns.txt. |

## Process Automation

| Criteria  | Meets specifications  |
|---|---|
| Student will be able to determine whether ML models need to be updated based on model scores. | - The fullprocess.py script will perform this section.<br>- Check for the presence of non-ingested data in the /sourcedata/ folder<br>- Check for whether the most recent model performs better than the previously deployed model |
| Student will be able to regularly re-deploy the latest models to production. | - In deployment.py, copy the model from its initial location to the final production deployment directory. The initial location is specified in the output_folder_path of config.json. The production deployment directory is specified in the prod_deployment_path of config.json.<br>- Students need to set up a cron job that regularly runs fullprocess.py.<br>- The fullprocess.py script should call the deployment.py script only under certain conditions: when there is new data ingested, AND when there is model drift. |


## Suggestions to make your project stand out!

1. Generate pdf reports that contain one or more plots, plus model diagnostics and summary statistics.
2. Store and analyze data related to time trends, including changes in latency and missing values percentages.
3. Store datasets and records in SQL databases instead of .csv and .txt files.