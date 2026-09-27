# Cardiovascular Disease Prediction

A machine learning classification project exploring the prediction of cardiovascular disease using demographic, clinical, and lifestyle features.

## Overview

This project evaluates several machine learning approaches for predicting whether a patient has cardiovascular disease.

The analysis follows a complete machine learning workflow, including:

* Exploratory data analysis
* Data cleaning
* Feature engineering
* Baseline modeling
* Feature importance analysis
* Feature selection
* Hyperparameter tuning
* Model evaluation
* ROC curve analysis
* Model interpretation

## Dataset

This project uses the **Cardiovascular Disease Dataset** by Svetlana Ulianova, available through Kaggle:

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

The dataset contains **70,000 patient observations** and 13 original variables.

The target variable is:

* `cardio = 0` — No cardiovascular disease
* `cardio = 1` — Cardiovascular disease

Features include:

* Age
* Gender
* Height
* Weight
* Systolic blood pressure (`ap_hi`)
* Diastolic blood pressure (`ap_lo`)
* Cholesterol
* Glucose
* Smoking
* Alcohol consumption
* Physical activity

The dataset is **not redistributed in this repository**. Please obtain it directly from the original Kaggle source and review the applicable dataset terms before using it.

## Setup

After downloading the dataset from Kaggle, place `cardio_train.csv` in the same directory as the Jupyter notebook:

```text
cardiovascular-disease-prediction/
├── cardiovascular_disease_prediction.ipynb
├── cardio_train.csv
└── README.md
```

The notebook expects the dataset to be named:

```text
cardio_train.csv
```

The dataset is loaded using:

```python
pd.read_csv("cardio_train.csv", sep=";")
```

## Data Cleaning & Feature Engineering

The analysis begins by checking the dataset for missing values, distributions, and potential outliers.

Several features were engineered to improve the representation of the data:

* `age_years` — age converted from days to years
* `bmi` — Body Mass Index calculated from height and weight
* `bp_difference` — difference between systolic and diastolic blood pressure

The original `id` and `age` columns were removed after feature engineering.

Implausible measurements for height, weight, and blood pressure were also removed.

After cleaning, **68,706 observations** remained.

## Machine Learning Models

The project evaluates several classification models:

### Logistic Regression

Used as a baseline linear classification model and provides an interpretable reference point for the more complex models.

### Random Forest

Used to capture nonlinear relationships and interactions between features.

### HistGradientBoosting

A gradient boosting model used to further capture nonlinear relationships in the data.

Hyperparameter tuning was performed using cross-validation to identify improved model configurations.

## Evaluation

Models were evaluated using:

* **Accuracy** — proportion of predictions that were correct
* **Precision** — proportion of positive predictions that were correct
* **Recall** — proportion of actual positive cases identified
* **F1 Score** — balance between precision and recall
* **ROC-AUC** — ability to distinguish between the two classes across classification thresholds

### Final Model Performance

| Model                      | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| -------------------------- | -------: | --------: | -----: | -----: | ------: |
| Tuned Logistic Regression  |   72.44% |    75.03% | 66.41% | 70.46% |   0.789 |
| Tuned Random Forest        |   73.46% |    76.13% | 67.54% | 71.58% |   0.801 |
| HistGradientBoosting       |   73.33% |    75.46% | 68.32% | 71.71% |   0.802 |
| Tuned HistGradientBoosting |   73.24% |    75.15% | 68.62% | 71.73% |   0.802 |

The final models achieved approximately **73% accuracy** and ROC-AUC values around **0.80**.

## ROC Curve Analysis

ROC curves were used to examine model performance across different classification thresholds.

The final models produced ROC-AUC values between approximately **0.789 and 0.802**, indicating meaningful discrimination between patients with and without cardiovascular disease within this dataset.

## Model Interpretation

Feature importance was examined to understand which variables the tree-based models relied on most heavily when making predictions.

Age, BMI, systolic and diastolic blood pressure, height, and weight were among the features that contributed substantially to model predictions.

Because several variables are correlated, particularly height, weight, and BMI, feature importance should not be interpreted as evidence of independent causal effects.

The feature importance analysis is therefore intended to describe **model behavior and predictive relationships**, rather than establish medical causation.

## Limitations

Several limitations should be considered:

* The models were evaluated on a single dataset.
* Predictive performance does not establish causal relationships.
* Model performance does not demonstrate clinical usefulness.
* Correlated features can make feature-importance interpretation difficult.
* External validation would be necessary to evaluate generalization to other populations.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## Repository Structure

```text
cardiovascular-disease-prediction/
│
├── cardiovascular_disease_prediction.ipynb
├── README.md
└── .gitignore
```

The dataset is intentionally excluded from the repository and must be downloaded separately from Kaggle.

## Disclaimer

This project is for educational and research purposes. The models developed in this project are not intended for medical diagnosis or clinical decision-making.
