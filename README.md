# heart-disease-streamlit-project
# 💓 Heart Disease Prediction Streamlit App

This project predicts whether a person has heart disease using medical attributes.  
It includes full data analysis, a machine learning model, and a live Streamlit app.

## 🔍 Dataset
- Source: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Features: Age, Sex, Chest Pain, Cholesterol, etc.
- Target: 0 = No disease, 1 = Heart disease

## 📊 Project Pipeline
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Training using Random Forest
4. Evaluation using precision, recall, F1-score
5. Streamlit App Deployment

## 🖼️ EDA Sample Code
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
