import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("💓 Heart Disease Prediction App")
st.markdown("#### Powered by Multiple Machine Learning Models")

# Load all models
models = {
    'Logistic Regression': joblib.load('lr_model.pkl'),
    'Decision Tree': joblib.load('dt_model.pkl'),
    'Gradient Boosting': joblib.load('gb_model.pkl'),
    'Random Forest': joblib.load('rf_model.pkl'),
    'Support Vector Machine': joblib.load('svm_model.pkl')
}

# Load saved column structure
model_columns = joblib.load('model_columns.pkl')

st.sidebar.header("🧠 Select a Model")
selected_model_name = st.sidebar.selectbox("Choose model to predict", list(models.keys()))
selected_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.write("🔒 Your data is 100% private and stays on your device.")

st.markdown("### 📝 Enter Patient Information:")

# User inputs
age = st.number_input('Age', min_value=1, max_value=120, value=45)
sex = st.radio('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure', value=120)
chol = st.number_input('Serum Cholesterol (mg/dl)', value=200)
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
restecg = st.selectbox('Resting ECG Results', ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
thalach = st.number_input('Max Heart Rate Achieved', value=150)
exang = st.radio('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.number_input('ST Depression Induced by Exercise', value=1.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment', ['upsloping', 'flat', 'downsloping'])

# Input dictionary
input_dict = {
    'age': age,
    'sex': 1 if sex == 'Male' else 0,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': 1 if fbs == 'Yes' else 0,
    'restecg': restecg,
    'thalach': thalach,
    'exang': 1 if exang == 'Yes' else 0,
    'oldpeak': oldpeak,
    'slope': slope,
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encode input
input_encoded = pd.get_dummies(input_df)

# Reindex to match model training columns
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button("💡 Predict"):
    prediction = selected_model.predict(input_encoded)[0]
    proba = None
    if hasattr(selected_model, "predict_proba"):
        proba = selected_model.predict_proba(input_encoded)[0][1]

    st.markdown("### 🧾 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ **High Risk**: The patient may have heart disease.")
    else:
        st.success("✅ **Low Risk**: The patient likely does not have heart disease.")

    if proba is not None:
        st.markdown(f"**Prediction Confidence:** `{proba * 100:.2f}%`")
