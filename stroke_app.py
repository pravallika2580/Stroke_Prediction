import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogistic

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Preprocess the dataset
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df['gender_Male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['gender_Female'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)
df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
df['work_type_Private'] = df['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
df['work_type_Self_employed'] = df['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)
df['work_type_Govt_job'] = df['work_type'].apply(lambda x: 1 if x == 'Govt_job' else 0)
df['work_type_children'] = df['work_type'].apply(lambda x: 1 if x == 'children' else 0)
df['work_type_Never_worked'] = df['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
df['smoking_status_formerly_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
df['smoking_status_never_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
df['smoking_status_smokes'] = df['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)
df['smoking_status_Unknown'] = df['smoking_status'].apply(lambda x: 1 if x == 'Unknown' else 0)

# Prepare the dataset for training
df_model = df.copy()
df_model.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1, inplace=True)
X = df_model.drop('stroke', axis=1)
y = df_model['stroke']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Train a logistic regression model
logistic_reg = SklearnLogistic()
logistic_reg.fit(X_train, y_train)

# Streamlit UI
st.title("Stroke Risk Prediction")

st.header("Enter the Input Features")

# Input fields in the main area
age = st.slider('Age', min_value=18, max_value=100, value=30)
gender = st.selectbox('Gender', ['Male', 'Female'])
hypertension = st.selectbox('Hypertension (0: No, 1: Yes)', [0, 1])
heart_disease = st.selectbox('Heart Disease (0: No, 1: Yes)', [0, 1])
ever_married = st.selectbox('Ever Married (0: No, 1: Yes)', [0, 1])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=20.0)
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Button to trigger prediction
predict_button = st.button('Predict Stroke Risk')

# Prepare user data for prediction
user_data = {
    'age': age,
    'gender_Male': 1 if gender == 'Male' else 0,
    'gender_Female': 1 if gender == 'Female' else 0,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'work_type_Private': 1 if work_type == 'Private' else 0,
    'work_type_Self_employed': 1 if work_type == 'Self-employed' else 0,
    'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
    'work_type_children': 1 if work_type == 'children' else 0,
    'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
    'Residence_type': 1 if residence_type == 'Urban' else 0,
    'smoking_status_formerly_smoked': 1 if smoking_status == 'formerly smoked' else 0,
    'smoking_status_never_smoked': 1 if smoking_status == 'never smoked' else 0,
    'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0,
    'smoking_status_Unknown': 1 if smoking_status == 'Unknown' else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi
}

# Convert user data to DataFrame to match the model's input format
user_input_df = pd.DataFrame([user_data], columns=X.columns)

# Fill missing values if any
user_input_df = user_input_df.fillna(X.median())

# Make prediction
if predict_button:
    stroke_prediction = logistic_reg.predict(user_input_df)

    # Display prediction
    stroke_prob = logistic_reg.predict_proba(user_input_df)[0][1] * 100
    st.subheader(f"Stroke Probability: {stroke_prob:.2f}%")
    if stroke_prediction == 1:
        st.error("High risk of stroke detected! Please consult a doctor.")
    else:
        st.success("Low risk of stroke detected.")

