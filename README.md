# Stroke_Prediction
**Overview**
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.

In our project we want to predict stroke using machine learning classification algorithms, evaluate and compare their results. We did the following tasks:

Performance Comparison using Machine Learning Classification Algorithms on a Stroke Prediction dataset.
using visualization libraries, ploted various plots like pie chart, count plot, curves, etc.
Used various Data Preprocessing techniques.
Handle class imbalanced.
Build various machine learning models
Optimized SVM and Random Forest Classifiers using RandomizedSearchCV to reach the best model.
Domain: Deep Learning.

**Installing Python libraries and packages**

The required python libraries and packages are,
pandas

Numpy

sklearn

matplotlib

seaborn

**Features of the Dataset**

Dataset contains 5111 rows. Each row in the data provides relevant information about the patient.

gender: "Male", "Female" or "Other"

age: age of the patient

hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension

heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease

ever_married: "No" or "Yes"

work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"

Residence_type: "Rural" or "Urban"

avg_glucose_level: average glucose level in blood

bmi: body mass index

smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*

stroke: 1 if the patient had a stroke or 0 if not
