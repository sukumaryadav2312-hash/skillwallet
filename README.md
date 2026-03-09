# Hypertension Stage Prediction using Machine Learning

## Overview
Hypertension (high blood pressure) is one of the most common cardiovascular conditions and a major risk factor for heart disease and stroke. Early detection and monitoring can significantly reduce the risk of severe complications.

This project builds a **Machine Learning–based prediction system** that classifies the **stage of hypertension** using clinical parameters and lifestyle information.

The system analyzes patient data and provides:
- Hypertension stage prediction
- Risk score estimation
- Personalized lifestyle recommendations

 This project is designed as a **decision-support tool only** and should not replace professional medical diagnosis.

---

# Project Objectives

The primary goals of this project are:

- Predict hypertension stages using clinical and lifestyle data
- Identify patterns associated with hypertension risk
- Support preventive screening and monitoring
- Provide lifestyle recommendations based on risk level
- Build a user-friendly prediction interface

---

# Technologies Used

## Programming Language
- Python

## Data Processing
- Pandas
- NumPy

## Machine Learning
- Scikit-learn
- XGBoost (optional)

## Visualization
- Matplotlib
- Seaborn

## Interface / Deployment
- Streamlit
- Flask

---

# Development Environment Setup

## 1 Install Python

Download Python from:

https://www.python.org/downloads/

Recommended version:

Python 3.9 or higher

---

## 2 Create Virtual Environment

Run the following command:

python -m venv hypertension_env

Activate the environment:

### Windows

hypertension_env\Scripts\activate

### Mac/Linux

source hypertension_env/bin/activate

---

## 3 Install Required Libraries

pip install pandas numpy scikit-learn matplotlib seaborn streamlit flask xgboost

---

# Dataset

The dataset should contain the following features:

- Age
- Gender
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Heart Rate
- Body Mass Index (BMI)
- Smoking Habits
- Alcohol Consumption
- Physical Activity Level
- Medication Status
- Family Medical History
- Symptoms

## Target Variable

Hypertension Stage

Possible classes include:

- Normal
- Elevated
- Stage 1 Hypertension
- Stage 2 Hypertension

Datasets can be obtained from:

- UCI Machine Learning Repository
- Kaggle
- Open healthcare datasets

---

# Data Preprocessing

The preprocessing pipeline includes:

## Data Cleaning
- Removing duplicate records
- Correcting inconsistent entries

## Handling Missing Values
- Mean or median imputation for numeric features
- Mode replacement for categorical features

## Encoding Categorical Variables
- Label Encoding
- One-Hot Encoding

## Feature Scaling
- Standardization
- Normalization

---

# Exploratory Data Analysis (EDA)

EDA helps understand the relationship between features and hypertension stages.

Techniques used include:

- Feature distribution plots
- Correlation heatmaps
- Pairplots
- Boxplots
- Outlier detection

---

# Feature Selection

Feature selection helps identify the most relevant variables for prediction.

Methods used:

- Correlation analysis
- Recursive Feature Elimination (RFE)
- Feature importance from tree-based models

This step improves accuracy and reduces overfitting.

---

# Machine Learning Models

The following supervised learning models are implemented:

## Logistic Regression
Baseline classification model.

## Decision Tree
Captures nonlinear relationships between features.

## Random Forest
An ensemble model that improves stability and prediction accuracy.

## Gradient Boosting
A high-performance boosting algorithm for classification tasks.

---

# Model Training

Steps involved:

1. Split the dataset into training and testing sets

Training Data: 80%  
Testing Data: 20%

2. Train machine learning models using the training dataset.

3. Optimize model performance using hyperparameter tuning with:

- GridSearchCV
- RandomizedSearchCV

---

# Model Evaluation

Models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics help determine the most effective model.

---

# Prediction Pipeline

The prediction system processes new patient data through the following steps:

1. Input validation
2. Data preprocessing
3. Feature transformation
4. Model prediction
5. Hypertension stage classification
6. Risk score generation

Example output:

Predicted Stage: Stage 1 Hypertension  
Risk Score: 72%

---

# Recommendation System

Based on prediction results, the system generates personalized health recommendations such as:

- Reduce salt intake
- Increase physical activity
- Maintain healthy body weight
- Monitor blood pressure regularly
- Reduce alcohol consumption
- Manage stress levels

---

# User Interface

A simple web interface can be built using **Streamlit** or **Flask**.

Features include:

- Patient data input form
- Prediction button
- Hypertension stage display
- Risk score visualization
- Lifestyle recommendations

Run the application using:

streamlit run app.py

---

# Project Structure

hypertension-prediction/

data/
    hypertension_dataset.csv

notebooks/
    eda_analysis.ipynb

models/
    trained_model.pkl

src/
    preprocessing.py
    feature_selection.py
    train_model.py
    prediction_pipeline.py

app/
    app.py

requirements.txt
README.md

---

# System Testing

The system can be tested under different scenarios:

## Preventive Screening
Evaluating hypertension risk for healthy individuals.

## Hypertension Monitoring
Tracking patients already diagnosed with hypertension.

## Emergency Triage
Identifying patients with severe hypertension risk.

---

# Deployment

The system can be deployed using:

- Streamlit Cloud
- Render
- Heroku
- AWS

Example deployment command:

streamlit run app.py

---

# Ethical Considerations

Important notes:

- This system is not a medical diagnosis tool.
- It is intended for educational and decision-support purposes only.
- Medical decisions must always be made by qualified healthcare professionals.

---

# Future Improvements

Possible improvements include:

- Integration with wearable health devices
- Mobile app development
- Real-time blood pressure monitoring
- Deep learning models
- Explainable AI for medical transparency
- Integration with hospital databases

---

# Conclusion

This project demonstrates how **machine learning can assist in early hypertension detection and preventive healthcare monitoring**. By combining data analysis, predictive modeling, and an accessible interface, the system can support better health awareness and decision-making.

---

# License

This project is released under the MIT License.
