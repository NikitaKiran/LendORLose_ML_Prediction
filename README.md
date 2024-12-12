# Lend or Lose: Loan Default Prediction

This repository contains the code and resources for our ML project **"Lend or Lose: Loan Default Prediction"**. The project focuses on predicting loan defaults using machine learning models, exploring preprocessing techniques, and comparing model performances.

## Table of Contents
- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Preprocessing Steps](#preprocessing-steps)
- [Part 1: Model Comparisons](#part-1-model-comparisons)
- [Part 2: Advanced Models](#part-2-advanced-models)
- [Dependencies](#dependencies)
- [Results](#results)
- [Contributors](#contributors)

---

## Overview
Predicting loan defaults is crucial for financial institutions to minimize risks and optimize lending processes. This project aims to evaluate the effectiveness of various machine learning models, along with preprocessing and optimization techniques.

---

## Dataset Description
The dataset comprises **255,347 rows and 18 columns (features)**. Key details include:
- Numerical and categorical features.
- No missing or duplicate values were found.

---

## Preprocessing Steps
1. **Null Values**: None found.
2. **Duplicate Records**: None found.
3. **Feature Encoding**:
    - **One-Hot Encoding** (Part 1).
    - **Label Encoding** (Part 2 for better performance).
4. **Scaling**:
    - StandardScaler was used to standardize numerical features.
5. **Skew Reduction**:
    - Methods like SMOTE, Random Over-Sampling, and Random Under-Sampling were tested.
6. **Dimensionality Reduction**:
    - Correlation heatmaps were used to analyze relationships between features.

---

## Part 1 (Final_code.ipynb): Model Comparisons
The following models were trained and tested on the dataset with hyperparameter tuning using GridSearchCV:

| Model                        | Training Accuracy | Testing Accuracy |
|------------------------------|-------------------|------------------|
| Decision Trees              | 88.52%            | 88.16%           |
| Random Forest               | 88.59%            | 88.33%           |
| XGBoost                     | 88.68%            | **88.48%**       |
| AdaBoost                    | 88.61%            | 88.35%           |
| KNN                         | 88.42%            | 88.23%           |
| Gaussian Naive Bayes        | 88.52%            | 88.28%           |
| Gradient Boosting Classifier| 88.64%            | 88.42%           |

**Best Model**: XGBoost achieved the highest testing accuracy.

---

## Part 2 (Final_Code_2.ipynb): Advanced Models
The following advanced models were explored:

### Logistic Regression
- **Best Parameters**: `C=0.1`, `Solver=liblinear`
- **Accuracy on Kaggle**: 88.588%

### Support Vector Machine (SVM)
- **Kernel**: RBF
- **Best Parameter**: `C=10`
- **Accuracy on Kaggle**: 88.637%

### Neural Network
- **Architecture**: Sequential model with 2 hidden layers
    - Hidden Layer 1: 64 neurons
    - Hidden Layer 2: 512 neurons
    - Activation: ReLU
- **Best Learning Rate**: 0.01
- **Accuracy on Kaggle**: **88.754%**

**Best Model**: Neural Network achieved the highest Kaggle score.

---

## Dependencies
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `keras`, `tensorflow`, `imblearn`

---

## Results
- **Best Model (Part 1)**: XGBoost with 88.821% Kaggle accuracy.
- **Best Model (Part 2)**: Neural Network with 88.754% Kaggle accuracy.

---

## Contributors
- **Nikita Kiran** ([Email](mailto:Nikita.Kiran@iiitb.ac.in))
- **Samyak Jain** ([Email](mailto:Samyak.Jain@iiitb.ac.in))
- **Nupur Dhananjay Patil** ([Email](mailto:Nupur.Patil@iiitb.ac.in))

---

Explore the full report and visualizations in the respective notebooks!
