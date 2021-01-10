

# Identifying the best potential applicants for a trucking company based on past length of employment

A major trucking company goes over thousands of applications per month.  These applications are tracked over time and are linked to a record that maintains employment status.  Taking the number of months of employment and trying to predict that value based on the input of the application may be able to identify key applications where the best recruiters should focus their time.  They currently have an AI model processing the data, but don't fully trust the results.

## Project Set Up and Installation
The dataset needed to be created from several tables kept within a database to join the data provided on the application with the employment dates.  The data of hire was subtracted from the current date if they are still employed, or from the last date of employment if they are not currently employed.  The data then was uploaded in csv format to an Azure ML workspace for further analysis.

## Dataset
The dataset does not include any personally identfying information.  Zip Code is provided as well as a fico score and responses to questions on driving history and driver license status.

### Overview
The dataset is from a major trucking company. severa tables were joined together to provide a consistent dataset.  The data is fairly lopsided as most applications are not hired, so a majority of the data has zero months of employment.

### Task
Identifying the applications that will give the highest return on investment is important as recruiting is a huge cost.  Turnover is fairly high as well.  Identifying an applicant that will continue to work for the company over time can help to optimize the investment in recruiting.  Using joined information that provides how long past applicants have worked for the company, along with the data supplied for the application can help identify the best applicants.

### Access
The dataset was uploaded to the machine learning workspace as part of the initial set up and analysis.  The dataset is not publicly available and will not be made available to others.

## Automated ML
I tried several different methods of data analysis to try and determine the best settings.  As the number of months is a number, a regression seemed the most logical initially.  The correlation coefficient was fairly low for regression however.  I also identified several factors that may have just been random chance and removed them from analysis, such as an application identification number.  I also removed columns that were scoring from other models

### Results
Using a categorization model targeting the weighted Area Under the Curve gave a final value of 98.6 and Accuracy of 97.4%

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
