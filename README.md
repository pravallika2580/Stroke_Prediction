# Stroke_Prediction

**Overview**

According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. Over 12 million people worldwide will have their first stroke this year and 6.5 million will die as a result. Over 100 million people in the world have experienced stroke. The incidence of stroke increases significantly with age, however over 60% of strokes happen to people under the age of 70 and 16% happen to those under the age of 50. This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.

In our project, we want to predict stroke using machine learning classification algorithms, and evaluate and compare their results. We did the following tasks:

Performance Comparison using Machine Learning Classification Algorithms on a Stroke Prediction Dataset.
Using visualization libraries, I plotted various plots like pie charts, count plots, curves, etc.
Used various Data Preprocessing techniques.
Handle class imbalance.
Build various machine learning models
Optimized SVM and Random Forest Classifiers using RandomizedSearchCV to reach the best model.
Domain: Deep Learning.

**Installing Python libraries and packages**

The required Python libraries and packages are,

* pandas
* Numpy
* sklearn
* matplotlib
* seaborn

# Features of the Dataset

The dataset contains medical and demographic information for 5,110 patients, aimed at identifying factors related to the occurrence of strokes. The dataset consists of 12 columns, each representing a feature or characteristic related to the patient’s health or lifestyle.
Dataset contains 5111 rows. 

Each row in the data provides relevant information about the patient. 
* gender: "Male", "Female" or "Other"
* age: age of the patient
* hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
* heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
* ever_married: "No" or "Yes"
* work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
* Residence_type: "Rural" or "Urban"
* avg_glucose_level: average glucose level in blood
* bmi: body mass index
* smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
* stroke: 1 if the patient had a stroke or 0 if not 

There are 4 columns with numeric data (age, avg_glucose_level, bmi, id) and several categorical or binary columns (e.g., gender, smoking_status). Additionally, the BMI column has some missing values.

This dataset can be used for predictive modeling to assess the likelihood of stroke occurrence based on various health and lifestyle factors.
