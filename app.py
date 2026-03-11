import streamlit as st
from src.prediction import Insurance_Prediction

st.title("Insurance Prediction App")
st.write("This app predicts the insurance premium based on user details.")

Age = st.number_input("Enter Age", min_value=18, max_value=100, step=1)
Annual_Income_LPA = st.number_input("Enter Annual Income (LPA)", min_value=0.0)
Policy_Term_Years = st.number_input("Enter Policy Term (Years)", min_value=1)
Sum_Assured_Lakhs = st.number_input("Enter Sum Assured (Lakhs)", min_value=0.0)

if st.button("Predict"):

    model = Insurance_Prediction()

    result = model.prediction(
        Age,
        Annual_Income_LPA,
        Policy_Term_Years,
        Sum_Assured_Lakhs
    )

    st.success(f"Predicted Insurance Value: {result}")