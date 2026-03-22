import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Accounting Fraud Detection", layout="wide")

st.title("AI Accounting Fraud Detection Dashboard")
st.write("Multi-class accounting record classification using machine learning.")

# Load data and model
df = pd.read_excel("data/data.xlsx", engine="openpyxl")
model = joblib.load("models/fraud_model.pkl")

# Prepare data
X = df.drop("Category", axis=1)
y = df["Category"]

# Predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

results = df.copy()
results["Predicted_Category"] = predictions
results["Confidence"] = probabilities.max(axis=1)

# Summary metrics
st.subheader("Model Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(results))
col2.metric("Features", X.shape[1])
col3.metric("Classes", y.nunique())

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(results.head(20), use_container_width=True)

# Predicted category counts
st.subheader("Predicted Category Distribution")
st.bar_chart(results["Predicted_Category"].value_counts().sort_index())

# Highest-confidence predictions
st.subheader("Top High-Confidence Predictions")
high_conf = results.sort_values(by="Confidence", ascending=False)
st.dataframe(high_conf.head(20), use_container_width=True)

# Explain one row
st.subheader("Inspect a Record")
row_index = st.number_input(
    "Choose row index",
    min_value=0,
    max_value=len(results) - 1,
    value=0,
    step=1
)

selected_row = results.iloc[row_index]
st.write(selected_row)

st.subheader("Basic Explanation")
st.write(
    f"This record was predicted as Category {selected_row['Predicted_Category']} "
    f"with confidence {selected_row['Confidence']:.2f}. "
    "The model used the 12 accounting-related features to make this classification."
)