import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load saved model and encoders
model = load_model('regression_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Estimated Salary Prediction")
st.write("Enter customer details below:")

# Inputs
credit_score = st.slider("Credit Score", 350, 850, 600)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", value=50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary_dummy = 0  # Just for consistency in original dataset
geo = st.selectbox("Geography", onehot_encoder_geo.categories_[0].tolist())

# Encode inputs
gender_encoded = label_encoder_gender.transform([gender])[0]
has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
is_active_member_encoded = 1 if is_active_member == "Yes" else 0
geo_encoded = onehot_encoder_geo.transform([[geo]]).toarray()[0]

# Final input array
input_data = np.array([[credit_score, gender_encoded, age, tenure, balance,
                        num_of_products, has_cr_card_encoded, is_active_member_encoded,
                        estimated_salary_dummy]])
input_data = np.concatenate([input_data, geo_encoded.reshape(1, -1)], axis=1)

# Scale the input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_scaled)[0][0]
    st.success(f"Predicted Estimated Salary: ₹{prediction:,.2f}")
