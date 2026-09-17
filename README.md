
# Credit Card Fraud Detection Using XGBoost

This machine learning project focuses on detecting fraudulent credit card transactions using **XGBoost**, a powerful gradient boosting algorithm designed for classification and predictive modeling. The project uses a highly imbalanced transaction dataset, where legitimate transactions significantly outnumber fraudulent transactions. The main objective is to build a model that can accurately identify fraudulent transactions while minimizing incorrect predictions.

## Problem Statement

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions represent only a very small percentage of total transactions. This creates a **class imbalance problem**. If a model simply predicts every transaction as legitimate, it can achieve high accuracy while failing to detect most fraudulent transactions. Therefore, accuracy alone is not an appropriate metric for evaluating this type of model.

## Data Preprocessing

The dataset is first loaded and explored using **Pandas and NumPy**. The preprocessing stage includes checking the dataset structure, identifying missing values, examining the distribution of fraudulent and legitimate transactions, and preparing the features for model training.

The target variable, `Class`, represents whether a transaction is legitimate or fraudulent. A value of `0` represents a legitimate transaction, while `1` represents a fraudulent transaction. The dataset is divided into training and testing sets so that the model can be evaluated on previously unseen data.

## Handling Class Imbalance with SMOTE

To address the imbalance between legitimate and fraudulent transactions, the project uses **SMOTE (Synthetic Minority Over-sampling Technique)**. SMOTE generates synthetic examples of the minority class instead of simply duplicating existing fraudulent transactions.

SMOTE is applied only to the training data to prevent data leakage. This creates a more balanced training dataset and allows the XGBoost model to learn patterns associated with fraudulent transactions more effectively.

## XGBoost Model

The primary machine learning algorithm used is **XGBoost (Extreme Gradient Boosting)**. XGBoost builds multiple decision trees sequentially, where each new tree attempts to correct errors made by previous trees. It can capture complex and nonlinear relationships between transaction features.

The trained model produces a probability for each transaction indicating how likely it is to be fraudulent.

## Decision Threshold Tuning

Instead of automatically classifying transactions using the default probability threshold of 0.5, this project experiments with different decision thresholds. A lower threshold can increase fraud detection and recall but may also produce more false positives.

Threshold tuning is important because fraud detection involves a trade-off between missing fraudulent transactions and incorrectly flagging legitimate transactions. The appropriate threshold depends on the practical requirements of the application.

## Model Evaluation

The model is evaluated using several performance metrics, including **precision, recall, F1-score, ROC-AUC, and the confusion matrix**.

**Precision** measures how many transactions predicted as fraud are actually fraudulent. **Recall** measures how many actual fraudulent transactions are successfully detected. **F1-score** provides a balance between precision and recall, while **ROC-AUC** measures the model's ability to distinguish between fraudulent and legitimate transactions across different thresholds.

The confusion matrix provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

## Feature Importance

XGBoost feature importance scores are analyzed to understand which features have the greatest influence on the model's predictions. This provides interpretability and helps identify patterns that may be associated with fraudulent transactions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* Matplotlib
* SMOTE
* ROC-AUC
* Confusion Matrix

Overall, this project demonstrates a complete machine learning workflow for credit card fraud detection, covering data preprocessing, class imbalance handling, XGBoost classification, decision threshold optimization, model evaluation, and feature importance analysis.
