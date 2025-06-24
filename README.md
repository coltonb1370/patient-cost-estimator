# 🏥 Transparent Treatment Cost Estimator

The **Transparent Treatment Cost Estimator** is a Streamlit web application built to help patients and healthcare professionals estimate the cost of hospital visits based on patient demographics, visit details, and comorbidities. The app incorporates an XGBoost regression model trained on synthetic Midwest healthcare data and provides model explanations via SHAP and data drift detection using Evidently AI.

---

## 🚀 Features

- **Interactive Cost Estimation**: Input patient age, gender, reason for visit, urgency level, etc.
- **Model Explainability**: View SHAP visualizations of key cost drivers.
- **Data Drift Monitoring**: Detect shifts in patient demographics and visit patterns over time.

---

## 🧰 Technologies Used

- **Python 3.10+**
- **Streamlit**
- **XGBoost**
- **Pandas / NumPy**
- **Joblib**
- **Evidently AI**
- **Matplotlib / SHAP** (for visualizations)

---

## 📂 Repository Structure

```
├── app.py                           # Main Streamlit app
├── xgb_midwest_cost_model.pkl      # Trained XGBoost model
├── reference_data.csv              # Dataset used as baseline for data drift
├── current_data.csv                # Dataset used as live input for drift monitoring
├── shap_summary_cleaned.png        # Static SHAP image
├── requirements.txt                # Python dependencies
├── full_midwest_patient_cost_dataset.csv  # Full dataset
└── README.md                       # Project documentation
```

## 📄 License

This project is for academic and educational purposes.


