import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background-color: #1e293b;
            color: #f8fafc;
        }
        h1, h3, h4 {
            color: #38bdf8;
        }
        .block-container {
            padding: 2rem 1rem;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            padding: 0.5em 1em;
            border-radius: 10px;
        }
        .stRadio>div>label {
            color: #f8fafc !important;
        }
        label:has(> div:contains("Chest Pain Type")) {
            color: #fef08a !important;  /* soft yellow */
        }
    </style>
""", unsafe_allow_html=True)


# App title
st.title("💓 Heart Disease Predictor")
st.markdown("##### Predict your risk using multiple ML models")

# Load models and column names
models = {
    'Logistic Regression': joblib.load('lr_model.pkl'),
    'Decision Tree': joblib.load('dt_model.pkl'),
    'Gradient Boosting': joblib.load('gb_model.pkl'),
    'Random Forest': joblib.load('rf_model.pkl'),
    'Support Vector Machine': joblib.load('svm_model.pkl')
}

model_columns = joblib.load('model_columns.pkl')

# Sidebar model selector
st.sidebar.title("🔧 Settings")
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_choice]

st.sidebar.markdown("🧪 Your data is not stored or shared.")

# User input fields
st.subheader("📋 Patient Information")

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

# Prepare input
input_data = {
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
    'slope': slope
}

# Create dataframe
input_df = pd.DataFrame([input_data])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Prediction
st.subheader("🔎 Prediction")

if st.button("Run Prediction"):
    pred = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1] if hasattr(model, "predict_proba") else None

    if pred == 1:
        st.error("⚠️ High Risk: Patient may have heart disease.")
    else:
        st.success("✅ Low Risk: Patient likely does not have heart disease.")

    if proba is not None:
        st.info(f"📈 Confidence: `{proba * 100:.2f}%`")

