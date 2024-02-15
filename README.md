# Credit Risk Analysis Report

## Overview of the Analysis

In this Challenge, I used various techniques to train and evaluate a model based on loan risk. I used a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. The data included was loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, total debt and loan status.

Upon reviewing the data I determined loan status to be the target variable and seperated the data into labels and features.  From here I used y.value_counts() to check the balance of the target values.  This returned the following:
loan_status
0  75036
1   2500

Next I split the data into training and testing datasets by using train_test_split from sklearn.model_selection.  Then I checked the shape of X_train to see the full size of the data.  It returned (58152, 7).  Then I created a logistic regression model with the original data. Then the score of the model was determined.

Training Data Score: 0.9914878250103177
Testing Data Score: 0.9924164259182832

After reviewing the testing scores predictions were made using the testing data. Last, I evaluated the model's performance by calculating the accuracy score of the model, generating a confusion matrix and then printing the classification report.

Upon evaluating the model's performance it was found necessary to predict a logistic regression model with resampled training data.  I used the RandomOverSampler module from the imbalanced-learn library to resample the data.  I then determined the distinct values of the sampled labels data by using numpy and calling np.unique(y_resampled, return_counts=True).  This gave the following result.

ReSampled Labels:  [0 1]
Label Counts:  [56277 56277]

With this resampling it shows equal data from the healthy and unhealthy loans.  Once the data was resampled I then used the logistic regression classifier and fit it to the sampled data which allowed predictions to be made.  Once completed I was able to evaluate the accuracy score, generate a confusion matrix and print the classification report.

## Results

* Machine Learning Model 1: Logistic Regression Model
  
   **Classification Report**
  
                        precision    recall  f1-score   support
   0 (healthy loan)          1.00      1.00      1.00     18759
   1 (unhealthy loan)        0.87      0.89      0.88       625
              accuracy                           0.99     19384
             macro avg       0.94      0.94      0.94     19384
          weighted avg       0.99      0.99      0.99     19384

            **Balanced Accuracy Score : 0.9442676901753825
            **Precision for healthy loan is 100% while precision for unhealthy loan is 87%.
            **Recall for healthy loan is 100% while recall for unhealthy loan is 89%.


* Machine Learning Model 2: Logistic Regression Model utilizing RandomOverSampler
  
    **Classification Report**
  
                        precision    recall  f1-score   support
    0 (healthy loan)         0.99      0.99      0.99     56277
    1 (unhealthy loan)       0.99      0.99      0.99     56277
              accuracy                           0.99    112554
             macro avg       0.99      0.99      0.99    112554
          weighted avg       0.99      0.99      0.99    112554

            **Balanced Accuracy Score : 0.994180571103648
            **Precision for healthy and unhealthy loans is 99%.
            **Recall for healthy and unhealthy loans is 99%.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
