# app.py

import streamlit as st
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# -----------------------------
# Model Accuracy
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Logistic Regression Classification App")

st.write("## Breast Cancer Prediction")

st.write(f"### Model Accuracy : {accuracy:.2f}")

st.write("## Enter Input Values")

# -----------------------------
# User Inputs
# -----------------------------
mean_radius = st.number_input("Mean Radius", value=14.0)

mean_texture = st.number_input("Mean Texture", value=20.0)

mean_perimeter = st.number_input("Mean Perimeter", value=90.0)

mean_area = st.number_input("Mean Area", value=600.0)

mean_smoothness = st.number_input("Mean Smoothness", value=0.1)

# -----------------------------
# Input Data
# -----------------------------
input_data = pd.DataFrame({
    'mean radius': [mean_radius],
    'mean texture': [mean_texture],
    'mean perimeter': [mean_perimeter],
    'mean area': [mean_area],
    'mean smoothness': [mean_smoothness]
})

# Add Missing Columns
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Arrange Columns Properly
input_data = input_data[X.columns]

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    st.write("### Prediction Probability")
    st.write(probability)

    if prediction[0] == 1:
        st.success("Prediction : Benign Cancer")
    else:
        st.error("Prediction : Malignant Cancer")
