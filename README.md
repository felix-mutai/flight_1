In this task I have to analyse the given “Airline On-Time Performance” dataset
and build a predictive model that can predict the flight delay. Moreover, highlight some of the root causes of the flight delays.In addition, how we can enhance the
predictive power of the implemented model as well how these flight delays could be avoided or minimized.

## Requirements

* python version 3.6 or above
* conda 4.6.14
All the others requirements and packages that are needed for this project is listed in requirements.txt file

## For Testing
To start with testing the model on hidden dataset please follow the commands below.
After navigating to the task directory please run the following commands:
 ```
conda create --name <new_env> --file <requirements.txt>
conda activate <new_env>
python test_pipeline.py --input-csv-directory-path <your test data directory path> --input-csv-file-name <your test csv file name>
```
I assume that you have a single csv file with test data if not please merge all the files before testing by data_read_merge class. Moreover, 
results will be saved in the same data directory.

## For Re-training
After navigating to the task directory please run the following commands:
 ```
conda create --name <new_env> --file <requirements.txt>
conda activate <new_env>
python main.py --input-csv-directory-path <your test data directory path>
```
Moreover,results will be saved in the same data directory.

## Expected results

* Model performance evaluation graphs and Classification report.
* Model feature importance
*Analysis report that contains detailed data analysis, different model evaluations, root causes of flight delays, how to improve 
 the model performance and how to avoid or minimize the flight delays.   

## Report

For detailed report and discussion please find the Task_Report in ./Report/Task_Report

## Final Model

Pre-trained model which will be used for testing can be found in ./models


## Time need to done this task

* Data exploration and analysis - 1 hour
* Setup working environement and walking skeleton - 2 hours
* Development and deliverables
    * Architecture diagrams 20 minutes
    * Coding and model training - 3 hours
    * Readme - 20 minutes
    * Report writing - 1 hour
* Testing and bug fixing - 30 mins