# 💓 Heart Disease Prediction Streamlit App

This project predicts whether a person has heart disease based on clinical features.
It includes full exploratory data analysis (EDA), model training, and a live Streamlit web app.

---

## 🔍 Dataset

* **Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
* **Rows:** 303
* **Target:** `target` (0 = No disease, 1 = Heart disease)
* **Features Used:**

  * `age` – Age in years
  * `sex` – Gender (1 = male, 0 = female)
  * `cp` – Chest pain type (0-3)
  * `trestbps` – Resting blood pressure
  * `chol` – Serum cholesterol (mg/dl)
  * `fbs` – Fasting blood sugar > 120 mg/dl
  * `restecg` – Resting ECG results (0-2)
  * `thalach` – Max heart rate achieved
  * `exang` – Exercise-induced angina
  * `oldpeak` – ST depression
  * `slope` – Slope of peak ST segment
  * `ca` – Number of major vessels (0-3)
  * `thal` – Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)

---

## 📊 Project Pipeline

1. **EDA** (heatmaps, countplots, boxplots, pairplot)
2. **Preprocessing** (scaling, train-test split with stratify)
3. **Model Training** (RandomForestClassifier)
4. **Evaluation** (precision, recall, F1-score, confusion matrix)
5. **Deployment** with Streamlit app
6. **Conclusion** with feature insights

---

## 🗈️ EDA Sample Code

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```

---

## 📊 Countplot Code

```python
sns.countplot(x='sex', hue='target', data=df)
plt.title("Heart Disease by Gender")
plt.show()
```

---

## 🫠 Model Training Code

```python
from sklearn.ensemble import RandomForestClassifier

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

---

## 🚀 How to Run the App Locally

Follow these steps to run the Streamlit app on your computer:

### 1. Install Required Packages

Make sure you have Python installed. Then open terminal or command prompt and run:

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn joblib
```

### 2. Run the App

Go to the folder where your `app.py` is saved and run:

```bash
streamlit run app.py
```

### 3. Use the App

Your default browser will open at:

```
http://localhost:8501
```

You'll see:

* 📊 EDA visualizations
* 🫠 A form for prediction
* ✅ Real-time result
* 📋 Conclusion section

---

## 📂 Files Included

| File              | Description                         |
| ----------------- | ----------------------------------- |
| `heart.csv`       | Input dataset                       |
| `project.ipynb`   | Full EDA, preprocessing, model code |
| `app.py`          | Streamlit frontend code             |
| `heart_model.pkl` | Trained Random Forest model         |
| `scaler.pkl`      | StandardScaler for input features   |
| `pairplot.png`    | Auto-generated EDA image            |
| `Viva_Notes.txt`  | Viva questions and answers          |
| `README.md`       | This readme file                    |

---

## 📸 App Preview Features

* 📊 EDA Visualizations (heatmap, boxplot, pairplot)
* 📋 Data summary table
* 🫠 Real-time prediction form (13 inputs)
* ✅ Result shown as “No Heart Disease” or “Heart Disease Detected”
* 📝 Final insight/conclusion section

---

## 📌 External Links

* 📜 **Kaggle Notebook:** [Click here](https://www.kaggle.com/code/Rameesha1234/heart-disease-prediction)

---

## 👤 Author

* **Name:** Rameesha
* **University:** PUCIT
* **Course:** Introduction to Data Science – Final Term Project
* **GitHub:** [https://github.com/Rameesha1234/heart-disease-streamlit-project](https://github.com/Rameesha1234/heart-disease-streamlit-project)

---
