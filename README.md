# Credit_Risk_Analysis
To Complete Supervised Machine Learning Models from Module 17

## Project Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we will need to employ different techniques to train and evaluate models with unbalanced classes. We need to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. We will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. We will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk..

## Resources
- jupyter notebook, python, Machine Learning Models 

## Challenge Overview
Prerequisite:
1.  Download the credit_risk_resampling_starter_code.ipynb and credit_risk_ensemble_starter_code.ipynb
2.  Download the data in csv LoanStats_2019Q1.csv for this excercise 


## Deliverable 1:  Use Resampling Models to Predict Credit Risk

Using your knowledge of the imbalanced-learn and scikit-learn libraries, we will evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, we will use the oversampling RandomOverSampler and SMOTE algorithms, and then we will use the undersampling ClusterCentroids algorithm. Using these algorithms, we will resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

Follow the instructions below and use the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 1.

1.  Open the credit_risk_resampling_starter_code.ipynb file, rename it credit_risk_resampling.ipynb, and save it to your Credit_Risk_Analysis folder.
2.  Using the information we’ve provided in the starter code, create your training and target variables by completing the following steps:
3.  Create the training variables by converting the string values into numerical ones using the get_dummies() method.
4.  Create the target variables.
5.  Check the balance of the target variables.
6.  Begin resampling the training data. First, use the oversampling RandomOverSampler and SMOTE algorithms to resample the data, then use the undersampling 
    ClusterCentroids algorithm to resample the data. For each resampling algorithm, do the following:
7.  Use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
8.  Calculate the accuracy score of the model.
9.  Generate a confusion matrix.
10. Print out the imbalanced classification report.
11. Save your credit_risk_resampling.ipynb file to your Credit_Risk_Analysis folder.

## Deliverable 2:  Use the SMOTEENN Algorithm to Predict Credit Risk

Using your knowledge of the imbalanced-learn and scikit-learn libraries, we will use a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the SMOTEENN algorithm, we will resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report. 

Follow the instructions below and use the information in the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 2.

1.  Continue using your credit_risk_resampling.ipynb file where you have already created your training and target variables.
2.  Using the information we have provided in the starter code, resample the training data using the SMOTEENN algorithm.
3.  After the data is resampled, use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
4.  Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

## Deliverable 3:  Use Ensemble Classifiers to Predict Credit Risk

Follow the instructions below and use the information in the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 3.

1.  Open the credit_risk_ensemble_starter_code.ipynb file, rename it credit_risk_ensemble.ipynb, and save it to your Credit_Risk_Analysis folder.
2.  Using the information we have provided in the starter code, create your training and target variables by completing the following:
	    -   Create the training variables by converting the string values into numerical ones using the get_dummies() method.
	    -   Create the target variables.
	    -   Check the balance of the target variables.
3.  Resample the training data using the BalancedRandomForestClassifier algorithm with 100 estimators.
4.  After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
5.  Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.
6.  Resample the training data using the EasyEnsembleClassifier algorithm with 100 estimators.
7.  After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
8.  Save your credit_risk_ensemble.ipynb file to your Credit_Risk_Analysis folder.

## Credit Risk Analysis Results
  - Naive Random Oversampling balanced accuracy score, confusion matrix and imbalanced classification report
    ![image_name](https://github.com/raneymjohnGit/Credit_Risk_Analysis/blob/main/Resources/Naive_Random_Oversampling.png)
 
  - SMOTE Oversampling balanced accuracy score, confusion matrix and imbalanced classification report
    ![image_name](https://github.com/raneymjohnGit/Amazon_Vine_Analysis/blob/main/Resources/SMOTE_Oversampling.png)

  - ClusterCentroids Undersampling balanced accuracy score, confusion matrix and imbalanced classification report
    ![image_name](https://github.com/raneymjohnGit/Amazon_Vine_Analysis/blob/main/Resources/ClusterCentroids_Undersampling.png)

  - SMOTEENN combination sampling balanced accuracy score, confusion matrix and imbalanced classification report
    ![image_name](https://github.com/raneymjohnGit/Amazon_Vine_Analysis/blob/main/Resources/SMOTEENN_combination_sampling.png)
 
  - Balanced Random Forest Classifier balanced accuracy score, confusion matrix and imbalanced classification report
    ![image_name](https://github.com/raneymjohnGit/Amazon_Vine_Analysis/blob/main/Resources/Balanced_Random_Forest_Classifier.png)

  - Easy Ensemble AdaBoost Classifier balanced accuracy score, confusion matrix and imbalanced classification report
    ![image_name](https://github.com/raneymjohnGit/Amazon_Vine_Analysis/blob/main/Resources/Ensemble_AdaBoost_Classifier.png)

## Credit Risk Analysis Summary

1.  Based on the Analysis of six models, Ensemble alogorithms have more accuracy on the predictions.
2.  I would recommend to use Ensemble alogorithm "Easy Ensemble AdaBoost Classifier" because it has got 91% accuracy.
        
        - Out of 87 actual high risk, it predicted 78 correctly compared to others. 
        
        - Also the low risk 17118, low risk loan applications, it predicted only 975 as high risk, which is the lowset one compared to other models.
