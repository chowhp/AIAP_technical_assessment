# AIAP_technical_assessment

## Overview
The objective of the project is to predict the survival rate of patients suffering from coronary artery diseases
using the dataset provided to help doctors to formulate preemptive medical treatments.

A machine learning pipleline program [-- 'program.py' --] is created to process the given dataset and feed to
4 algorithm ['Logistic Regression', "GaussianNB', 'Decision Tree Classifier' & 'Random Forest Classifier'] whereby their
performance is then evaluated using confusion matrix.

By default, the dataset is splited into training set(80%) to test set(20%)


## Algorithms
1.) Logistic Regression
a.) Use
- efficient to train and made no assumptions about the distributiouns of feature classes
- less incline to over fitting and perform well when dataset is linearly seperable

b.) Performance
- confusion matrix F1 score


2.) GaussianNB
a.) Use
- works well less training data
- handle irrelevant features and support binary and support multi-class classiifcation
- expect features to be independent to each other

b.) Performance
- confusion matrix F1 score


3.) Decision Tree Classifer
a.) Use
- support non linearity, robust to outliers and can handle high dimension data
- performance classification without much computation
- prone to overfit and errors in cases of small training set and many class
- computationally expensive to train

b.) Performance
- confusion matrix F1 score


4.) Random Forest Classifier
a.) Use
- collection of decision trees and average/majority vote of the forest is selected as the predicted output.
- less prone to overfitting and gives a more generalized solution.
- computation expensive to train and requires large training sets

b.) Performance
- confusion matrix F1 score


## Confusion Matrix
The confusion matrix compares the actual target values with those predicted by the machine learning model.
It gives a overall view of how well the model is performing and what kinds of errors it is making.

It provides 4 different combinations of predicted and actual values:
1.) True Positives (TP):  models Correctly predict a Event values.
2.) True Negatives (TN):  models Correctly predict a No-Event values.
3.) False Positives (FP): models Incorrecly predict a Event values.
4.) False Negatives (FN): models Incorrectly predict a No-Event values.

From here, Accuracy, Precision, Recall, F1 score and more can be derived:
a.) Accuracy [from all classes (positive & negative), how many of them have been predicted correctly]
    Appplicable in cases where classes are well balance and not skewed.
    It should be as high as possible.
    Formula: (TP + TN) / (TP + TN + FP + FN)

b.) Precision [from all the classs predicted as positive, how many are actually positive]
    It should be as high as possible, at the expense of false negative prediction.
    Formula: (TP) / (TP + FP)

c.) Recall [from all the positive classes, how many are predicted correctly]
    Recall should be as high as possible, at the expense of false positive prediction.
    Formula: (TP) / (TP + FN)

d.) F1-score conveys the balance between the precision and the recall.
    Formula: (2 * Precision * Recall) / (Precision + Recall)

## Update
- Additional method to install the package and recreate its required environment via setup.py 
- Improve computation efficiency by replacing for loop in function using list comprehension
- Drop multiple columns - [Favorite color, Height & ID] in dataframe
- Add in docstrings and comments to help understand functions capabilities and improve overall code readability
- Rename functions to better reflect its underlying purpose 

## Execution    
- run bash shell script 'run.sh' which finds & installs packages listed in requirements.txt
- Or run 'pip install .' to install the required packages


## dir tree
AIAP_technical_assessment
├── src
│   ├── program.py
|   ├── __int__.py
├── data
│   ├── survive.db
├── eda.ipynb
├── run.sh
├── requirements.txt
├── setup.py
├── README.md

