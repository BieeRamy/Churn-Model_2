import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
model = joblib.load("churn_model.pkl")
data = pd.read_csv("cleaned_churn_data.csv")

# Title
st.title("🔍 Customer Churn Prediction Dashboard")

# Sidebar
st.sidebar.header("📤 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Use uploaded data or sample
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded file successfully!")
else:
    df = data.copy()
    st.info("Using sample cleaned dataset.")

# Predict
X = df.drop(columns=["churn"], errors="ignore")
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]

# Add predictions to data
df["churn_prediction"] = preds
df["churn_probability"] = np.round(probs, 3)

# Show Data
st.subheader("📄 Preview of Data with Predictions")
st.dataframe(df.head(20))

# Churn distribution
st.subheader("📊 Churn Prediction Summary")
churn_counts = df["churn_prediction"].value_counts().rename(index={0: "No", 1: "Yes"})
st.bar_chart(churn_counts)

# Confusion matrix
if "churn" in df.columns:
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(df["churn"], df["churn_prediction"])
    st.subheader("📉 Confusion Matrix")
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    st.pyplot()

    # Metrics
    st.subheader("📌 Classification Report")
    report = classification_report(df["churn"], df["churn_prediction"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# SHAP Explanation (first 10 rows for speed)
st.subheader("🔍 SHAP Explanation for First 10 Predictions")
explainer = shap.TreeExplainer(model.named_steps["classifier"])
X_transformed = model.named_steps["preprocessor"].transform(X[:10])
shap_values = explainer(X_transformed)

# Display beeswarm plot
st.set_option('deprecation.showPyplotGlobalUse', False)
sh

# Show Data
st.subheader("📄 Preview of Data with Predictions")
st.dataframe(df.head(20))

# 📥 Download predictions
st.download_button(
    label="📥 Download Predictions as CSV",
    data=df.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv"
)ap.summary_plot(shap_values, X_transformed, show=False)
st.pyplot()
