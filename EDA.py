import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## 3. Exploratory Data Analysis
#
# Before training any models, I examined the dataset's dimensions, variables, missing values, and target distribution.
#
# The dataset contains 70,000 observations with no missing values. The target variable is also approximately balanced between the two classes, making accuracy a useful metric while still motivating the use of precision, recall, F1 score, and ROC-AUC for a more complete evaluation.
#

df = pd.read_csv("cardio_train.csv", sep=";")
#copying the dataset
data = df.copy()

# converting age from days to years for more readability
data["age_years"] = data["age"] / 365.25

# adding a body mass index column
data["bmi"] = data["weight"] / (data["height"] / 100) ** 2

# Adding a pulse difference column
data["bp_difference"] = data["ap_hi"] - data["ap_lo"]

#removing redundant columns
data = data.drop(columns=["id", "age"])
data = data[
    (data["height"] >= 120) & (data["height"] <= 220) &
    (data["weight"] >= 30) & (data["weight"] <= 200) &
    (data["ap_hi"] >= 70) & (data["ap_hi"] <= 240) &
    (data["ap_lo"] >= 40) & (data["ap_lo"] <= 160)
]

#EDA:
# proportion of CVD vs no CVD
data["cardio"].value_counts(normalize=True)
sns.countplot(data=data, x="cardio")
plt.title("Cardiovascular Disease Distribution")
plt.xlabel("Cardiovascular Disease")
plt.ylabel("Number of Patients")
plt.show()

#Age and CVD
sns.histplot(data=data, x="age_years", hue="cardio", bins=30, kde=True)
plt.title("Age Distribution by Cardiovascular Disease Status")
plt.xlabel("Age (years)")
plt.ylabel("Number of Patients")
plt.show()
data.groupby("cardio")["age_years"].mean()
#conclusion: correlated, matrix gives us 0.24

#BMI and CVD:
sns.boxplot(data=data, x="cardio", y="bmi")
plt.title("BMI by Cardiovascular Disease Status")
plt.xlabel("Cardiovascular Disease")
plt.ylabel("BMI")
plt.show()
data.groupby("cardio")["bmi"].mean()
#conclusion: correlation matrix gives 0.19, will test

#Blood pressure and CVD:
sns.boxplot(data=data, x="cardio", y="ap_hi")
plt.title("Systolic Blood Pressure by CVD Status")
plt.xlabel("Cardiovascular Disease")
plt.ylabel("Systolic Blood Pressure")
plt.show()
data.groupby("cardio")["ap_hi"].mean()
#conclusion: correlated, matrix gives us 0.43

#Cholersterol and CVD, note, cholesterol data is categorical: 1 = normal, 2 = above normal, 3 = well above normal
cholesterol_cvd = pd.crosstab(data["cholesterol"], data["cardio"], normalize="index") * 100

cholesterol_cvd.plot(kind="bar", stacked=True, figsize=(8, 5))
plt.title("CVD Prevalence by Cholesterol Category")
plt.xlabel("Cholesterol Category")
plt.ylabel("Percentage of Patients")
plt.legend(title="CVD", labels=["No CVD", "CVD"])
plt.xticks(rotation=0)
plt.show()
pd.crosstab(data["cholesterol"], data["cardio"], normalize="index") * 100
#conclusion: from crosstab, we can see that high cholesterol is strongly associated with CVD, despite low score in correlation matrix

#Smoking and CVD, note 0 = non-smoker and 1 = smoker:
sns.countplot(data=data, x="smoke", hue="cardio")
plt.title("Smoking Status by CVD Status")
plt.xlabel("Smoker")
plt.ylabel("Number of Patients")
plt.show()
#conclusion: little to no correlation (-0.02 from matrix)

#Correlation matrix (heatmap):
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.title("Correlation Matrix of Cardiovascular Risk Variables")
plt.show()
#note useless correlations such as weight and BMI and ap_hi and bp
