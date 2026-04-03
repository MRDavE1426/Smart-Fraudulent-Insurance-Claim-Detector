import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
from PIL import Image

# ===============================
# STEP 1: TITLE
# ===============================
st.set_page_config(page_title="AI Fraud Detection", layout="wide")

st.title("🚨 Smart Fraudulent Insurance Claim Detector")
st.markdown("Hybrid Fraud Detection using ML + Text + Image Analysis")

# ===============================
# STEP 2: LOAD DATA
# ===============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("carclaims.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

# ===============================
# STEP 3: CLEANING
# ===============================
if df is not None:
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

# ===============================
# STEP 4: ENCODING
# ===============================
encoders = {}

if df is not None:
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    for col in cat_cols:
        if df[col].nunique() < 50:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

# ===============================
# STEP 5: MODEL TRAIN / LOAD
# ===============================
model = None
X = None

if df is not None:
    # 🔥 FIXED: remove target column
    if 'FraudFound' in df.columns:
        X = df.drop('FraudFound', axis=1)
    else:
        X = df.copy()

    if not X.empty:
        if os.path.exists("fraud_model.pkl"):
            with open("fraud_model.pkl", "rb") as f:
                model = pickle.load(f)
        else:
            X_train, _ = train_test_split(X, test_size=0.2, random_state=42)

            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X_train)

            with open("fraud_model.pkl", "wb") as f:
                pickle.dump(model, f)

        df['anomaly'] = model.predict(X)
    else:
        df['anomaly'] = 1

# ===============================
# STEP 6: TEXT ANALYSIS
# ===============================
if df is not None:
    if 'Description' in df.columns:
        df['text_flag'] = df['Description'].apply(
            lambda x: 1 if "damage" in str(x).lower() else 0
        )
    else:
        df['text_flag'] = 0

# ===============================
# USER INPUT
# ===============================
st.sidebar.header("Enter Claim Details")

claim_amount = st.sidebar.number_input("Claim Amount", min_value=0)
vehicle_price = st.sidebar.number_input("Vehicle Price", min_value=0)
age = st.sidebar.slider("Age", 18, 100, 30)
severity = st.sidebar.selectbox("Severity", ["Minor", "Major", "Total Loss"])

description = st.sidebar.text_area("Claim Description")

uploaded_file = st.sidebar.file_uploader("Upload Image")

image_flag = 0

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, use_container_width=True)

    if img.size[0] < 200 or img.size[1] < 200:
        image_flag = 1
        st.warning("Low quality image detected")

# ===============================
# ML PREDICTION (FIXED)
# ===============================
ml_prediction = 1

if model is not None and X is not None:
    try:
        input_data = pd.DataFrame({
            'ClaimAmount': [claim_amount],
            'VehiclePrice': [vehicle_price],
            'Age': [age]
        })

        # match columns
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[X.columns]

        ml_prediction = model.predict(input_data)[0]

    except:
        ml_prediction = 1

# ===============================
# TEXT FLAG
# ===============================
text_flag = 0
if description and "damage" in description.lower():
    text_flag = 1

# ===============================
# RISK LOGIC
# ===============================
risk_score = 0

if claim_amount > vehicle_price:
    risk_score += 2
if claim_amount > 50000:
    risk_score += 1
if severity == "Total Loss":
    risk_score += 2
if age < 25:
    risk_score += 1

if ml_prediction == -1:
    risk_score += 2

if text_flag == 1:
    risk_score += 1

if image_flag == 1:
    risk_score += 1

# ===============================
# RESULT
# ===============================
if st.sidebar.button("Analyze Claim"):

    st.subheader("Fraud Result")
    st.metric("Risk Score", risk_score)

    if risk_score >= 4:
        st.error("High Risk Fraud 🚨")
    elif risk_score >= 2:
        st.warning("Medium Risk ⚠️")
    else:
        st.success("Low Risk ✅")

# ===============================
# VISUALIZATION
# ===============================
st.subheader("Analytics Dashboard")

if df is not None and 'anomaly' in df.columns:
    fraud_counts = df['anomaly'].value_counts()

    chart_df = pd.DataFrame({
        "Type": ["Fraud", "Not Fraud"],
        "Count": [fraud_counts.get(-1, 0), fraud_counts.get(1, 0)]
    })

    st.bar_chart(chart_df.set_index("Type"))

    fig = px.pie(chart_df, names="Type", values="Count")
    st.plotly_chart(fig)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("Developed by Devarshi 🚀")