# app.py - Streamlit Heart Disease Prediction App with Full EDA, Summary, Pairplot, and Conclusion

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Load dataset and model
df = pd.read_csv("heart.csv")
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💓 Heart Disease Prediction Project")
st.markdown("This app includes full EDA, prediction, and conclusions.")

# Section 1: EDA
st.header("📊 Exploratory Data Analysis (EDA)")

# Target distribution
st.subheader("Target Variable Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='target', data=df, ax=ax1)
st.pyplot(fig1)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# Boxplot for numeric features
st.subheader("Boxplot for Numeric Features")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df[['age', 'chol', 'thalach', 'trestbps']], ax=ax3)
st.pyplot(fig3)

# Gender vs Heart Disease
st.subheader("Heart Disease by Gender")
fig4, ax4 = plt.subplots()
sns.countplot(x='sex', hue='target', data=df, ax=ax4)
st.pyplot(fig4)

# Chest pain type vs Target
st.subheader("Chest Pain Type vs Heart Disease")
fig5, ax5 = plt.subplots()
sns.countplot(x='cp', hue='target', data=df, ax=ax5)
st.pyplot(fig5)

# Summary statistics table
st.subheader("📋 Summary Statistics Table")
st.dataframe(df.describe())

# Pairplot saved as image
st.subheader("🔄 Pairplot of Selected Features")

pairplot_path = "pairplot.png"
if not os.path.exists(pairplot_path):
    pair = sns.pairplot(df[['age', 'thalach', 'chol', 'target']], hue='target')
    pair.savefig(pairplot_path)
    plt.close()

st.image(pairplot_path, caption="Pairplot of age, thalach, chol vs target", use_column_width=True)

# Section 2: Prediction
st.header("🧠 Heart Disease Prediction")

st.markdown("Fill the form below and click Predict:")

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.radio("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("💖 Heart Disease Detected!")
    else:
        st.success("✅ No Heart Disease Detected.")

# Section 3: Conclusion
st.header("📝 Final Conclusion")

st.markdown("""
Based on EDA, we can conclude:
- Chest Pain Type and Max Heart Rate (thalach) are strong indicators of heart disease.
- People with higher oldpeak (ST depression) are more likely to have heart issues.
- Heart disease appears more common in males in this dataset.
- Prediction model uses 13 features to predict disease with high accuracy.

This concludes our project with EDA, machine learning, and live prediction app.
""")
