# Credit_Risk_Analysis

## Project Overview

Using a credit card dataset from LendingClub, a peer-to-peer lending services company, I oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the ClusterCentroids algorithm. Then, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 

### Purpose

Evaluate the performance of the machine learning models I created and make a written recommendation on whether they should be used to predict credit risk.

## Resources

- Data Source: [Credit dataset from LendingClub](https://2u-data-curriculum-team.s3.amazonaws.com/dataviz-online/module_17/Module-17-Challenge-Resources.zip)
- Code: The code and results for this analysis is in the `credit_risk_ensemble.ipynb` and `credit_risk_resampling.ipynb` located within the `Notebook` folder.
- Software: JupyterNotebook, Python, Pandas
  
## Results


Balanced accuracy score and the precision and recall scores of all six machine learning models:

- **Naive Random Oversampling** 
  
  - high_risk precision score: 0.01
  - low_risk precision score: 1.00
  - high_risk recall score: 0.69
  - low_risk recall score: 0.60
  
    ![classification](Static/Images/OversamplingClassification.PNG)
  - Balanced accuracy score:
    
    ![accuracy](Static/Images/OversamplingBalancedAccuracy.PNG)

- **SMOTE Oversampling**
  
  - high_risk precision score: 0.01
  - low_risk precision score: 1.00
  - high_risk recall score: 0.63
  - low_risk recall score: 0.69
  
    ![classification](Static/Images/SMOTEClassification.PNG)
  - Balanced accuracy score:
    
    ![accuracy](Static/Images/SMOTEOversamplingAccuracy.PNG)

- **Cluster Centroids Undersampling**
  
  - high_risk precision score: 0.01
  - low_risk precision score: 1.00
  - high_risk recall score: 0.67
  - low_risk recall score: 0.42
  
    ![classification](Static/Images/UndersamplingClassification.PNG)
  - Balanced accuracy score:
    
    ![accuracy](Static/Images/ClusterUndersamplingAccuracy.PNG)

- **Combination (Over and Under) Sampling**
  
  - high_risk precision score: 0.01
  - low_risk precision score: 1.00
  - high_risk recall score: 0.72
  - low_risk recall score: 0.57
  
    ![classification](Static/Images/CombinationClassification.PNG)
  - Balanced accuracy score:
    
    ![accuracy](Static/Images/CombinationAccuracy.PNG)

- Balanced Random Forest Classifier
  
  - high_risk precision score: 0.03
  - low_risk precision score: 1.00
  - high_risk recall score: 0.70
  - low_risk recall score: 0.87
  
    ![classification](Static/Images/ForestClassification.PNG)
  - Balanced accuracy score:
    
    ![accuracy](Static/Images/ForestAccuracy.PNG)

- Easy Ensemble AdaBoost Classifier

  - high_risk precision score: 0.09
  - low_risk precision score: 1.00
  - high_risk recall score: 0.92
  - low_risk recall score: 0.94
  
    ![classification](Static/Images/EnsembleClassification.PNG)
  - Balanced accuracy score:
    
    ![accuracy](Static/Images/EnsembleAccurcay.PNG)


## Summary

There is a summary of the results (2 pt)
There is a recommendation on which model to use, or there is no recommendation with a justification (3 pt)

 Results show that the accuracy score for the majority of the models (SMOTE, Naive Random, Cluster, Combination) hovered around 0.65 or fell below that number. The Balanced Random Forest Classifier model's accuracy received a score of around 0.79 and the Easy Ensemble AdaBoost Classifer had an accuracy score of 0.93.

 The precision scores for the models 