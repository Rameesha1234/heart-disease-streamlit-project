# 💓 Heart Disease Prediction Streamlit App

This project predicts whether a person has heart disease based on clinical features.  
It includes full exploratory data analysis (EDA), model training, and a live Streamlit web app.

---

## 🔍 Dataset

- **Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Rows:** 303  
- **Target:** `target` (0 = No disease, 1 = Heart disease)  
- **Features Used:**
  - `age` – Age in years
  - `sex` – Gender (1 = male, 0 = female)
  - `cp` – Chest pain type (0-3)
  - `trestbps` – Resting blood pressure
  - `chol` – Serum cholesterol (mg/dl)
  - `fbs` – Fasting blood sugar > 120 mg/dl
  - `restecg` – Resting ECG results (0-2)
  - `thalach` – Max heart rate achieved
  - `exang` – Exercise-induced angina
  - `oldpeak` – ST depression
  - `slope` – Slope of peak ST segment
  - `ca` – Number of major vessels (0-3)
  - `thal` – Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)

---

## 📊 Project Pipeline

1. **EDA** (heatmaps, countplots, boxplots, pairplot)
2. **Preprocessing** (scaling, train-test split with stratify)
3. **Model Training** (RandomForestClassifier)
4. **Evaluation** (precision, recall, F1-score, confusion matrix)
5. **Deployment** with Streamlit app
6. **Conclusion** with feature insights

---

## 🖼️ EDA Sample Code

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
