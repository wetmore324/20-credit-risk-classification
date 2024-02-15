# Credit Risk Analysis Report

## Overview of the Analysis

In this Challenge, I used various techniques to train and evaluate a model based on loan risk. I used a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. The data included was loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, total debt and loan status.

Upon reviewing the data I determined loan status to be the target variable and seperated the data into labels and features.  From here I used y.value_counts() to check the balance of the target values.  This returned the following:
![image](https://github.com/wetmore324/20-credit-risk-classification/assets/136288855/4fd12e00-3962-45d8-b41a-4ad1ea0fea96)


Next I split the data into training and testing datasets by using train_test_split from sklearn.model_selection.  Then I checked the shape of X_train to see the full size of the data.  It returned (58152, 7).  Then I created a logistic regression model with the original data. Then the score of the model was determined.

![Screenshot 2024-02-15 113332](https://github.com/wetmore324/20-credit-risk-classification/assets/136288855/f23e5686-9e25-4162-bc56-40b38c4c3654)

After reviewing the testing scores predictions were made using the testing data. Last, I evaluated the model's performance by calculating the accuracy score of the model, generating a confusion matrix and then printing the classification report.

Upon evaluating the model's performance it was found necessary to predict a logistic regression model with resampled training data.  I used the RandomOverSampler module from the imbalanced-learn library to resample the data.  I then determined the distinct values of the sampled labels data by using numpy and calling np.unique(y_resampled, return_counts=True).  This gave the following result.

![Screenshot 2024-02-15 113556](https://github.com/wetmore324/20-credit-risk-classification/assets/136288855/28b703b1-674e-46ec-9dff-ec680972ddc6)

With this resampling it shows equal data from the healthy and unhealthy loans.  Once the data was resampled I then used the logistic regression classifier and fit it to the sampled data which allowed predictions to be made.  Once completed I was able to evaluate the accuracy score, generate a confusion matrix and print the classification report.

## Results

* Machine Learning Model 1: Logistic Regression Model
  
![Screenshot 2024-02-15 113651](https://github.com/wetmore324/20-credit-risk-classification/assets/136288855/0277cd16-af5f-4550-bf7a-62f34cd7509c)

** Balanced Accuracy Score : 0.9442676901753825
** Precision for healthy loan is 100% while precision for unhealthy loan is 87%.
** Recall for healthy loan is 100% while recall for unhealthy loan is 89%.


* Machine Learning Model 2: Logistic Regression Model utilizing RandomOverSampler
  
![Screenshot 2024-02-15 113828](https://github.com/wetmore324/20-credit-risk-classification/assets/136288855/68727300-88f7-4517-965c-8781c73d6037)

**Balanced Accuracy Score : 0.994180571103648
**Precision for healthy and unhealthy loans is 99%.
**Recall for healthy and unhealthy loans is 99%.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
