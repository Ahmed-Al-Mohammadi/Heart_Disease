import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


try:
    model = joblib.load("best_heart_model.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    st.error("Model or Scaler not found. Please run the `save_model.py` script first from your notebook.")
    st.stop()


st.sidebar.header("Patient Data Input")

sex_options = {0: "Female", 1: "Male"}
cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
fbs_options = {0: "False (< 120 mg/dl)", 1: "True (> 120 mg/dl)"}
restecg_options = {0: "Normal", 1: "ST-T wave abnormality", 2: "Probable or definite left ventricular hypertrophy"}
exang_options = {0: "No", 1: "Yes"}
slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
thal_options = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"} 

def user_input_features():
    st.sidebar.markdown("---")
    age = st.sidebar.slider("Age", 20, 100, 50)
    
    sex_label = st.sidebar.selectbox("Sex", options=list(sex_options.values()))
    sex = list(sex_options.keys())[list(sex_options.values()).index(sex_label)]

    cp_label = st.sidebar.selectbox("Chest Pain Type", options=list(cp_options.values()))
    cp = list(cp_options.keys())[list(cp_options.values()).index(cp_label)]

    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 120, 570, 240)

    fbs_label = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=list(fbs_options.values()))
    fbs = list(fbs_options.keys())[list(fbs_options.values()).index(fbs_label)]

    restecg_label = st.sidebar.selectbox("Resting Electrocardiographic Results", options=list(restecg_options.values()))
    restecg = list(restecg_options.keys())[list(restecg_options.values()).index(restecg_label)]

    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 70, 220, 150)

    exang_label = st.sidebar.selectbox("Exercise Induced Angina", options=list(exang_options.values()))
    exang = list(exang_options.keys())[list(exang_options.values()).index(exang_label)]

    oldpeak = st.sidebar.slider("ST depression induced by exercise relative to rest", 0.0, 6.2, 1.0)

    slope_label = st.sidebar.selectbox("Slope of the peak exercise ST segment", options=list(slope_options.values()))
    slope = list(slope_options.keys())[list(slope_options.values()).index(slope_label)]

    ca = st.sidebar.slider("Number of major vessels (0-3) colored by flourosopy", 0, 3, 0)

    # ملاحظة: في بياناتك، قيم thal هي 3,6,7 لكن النموذج تدرب غالباً على قيم مختلفة.
    # سنستخدم القيم الموجودة في ملف البيانات الخام 1,2,3. تأكد من أن هذا صحيح.
    thal_label = st.sidebar.selectbox("Thal", options=list(thal_options.values()))
    thal = list(thal_options.keys())[list(thal_options.values()).index(thal_label)]

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.title("🫀 Heart Disease Prediction App")
st.markdown("This application uses a Machine Learning model to predict the likelihood of a patient having heart disease based on their medical data.")
st.markdown("---")

st.subheader("Patient Data")
st.write(input_df)

if st.sidebar.button("Predict"):
    data_scaled = scaler.transform(input_df)
    
    prediction = model.predict(data_scaled)
    prediction_proba = model.predict_proba(data_scaled)

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("**Result: High Likelihood of Heart Disease**", icon="⚠️")
        st.write(f"Confidence: **{prediction_proba[0][1]*100:.2f}%**")
        st.warning("Please consult a doctor for a professional diagnosis.")
    else:
        st.success("**Result: Low Likelihood of Heart Disease**", icon="✅")
        st.write(f"Confidence: **{prediction_proba[0][0]*100:.2f}%**")
        st.info("This is a prediction based on data and not a medical diagnosis. Regular check-ups are always recommended.")
