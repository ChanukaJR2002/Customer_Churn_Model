
# Telco Customer Churn Predictor

A machine learning web app that predicts whether a telecom customer is likely to **churn** (cancel their service), based on their account details, services subscribed, and billing information. Built with **XGBoost** and served through an interactive **Streamlit** UI.

---

## Project Structure

```
Customer_Churn_Model/
│
├── Telco_Customer_Churn.ipynb        # Notebook: EDA, preprocessing, model training & evaluation
├── app.py                            # Streamlit web app for inference
├── Telco_Churn_Model.pkl             # Final tuned XGBoost model
├── scaler.pkl                        # StandardScaler fitted on training data
├── feature_columns.pkl               # Exact one-hot encoded column order expected by the model
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Training dataset (IBM Telco Customer Churn dataset)
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Problem Statement

Customer churn — when a subscriber leaves a telecom provider — is costly to acquire-and-replace. This project trains a binary classifier to flag customers at high risk of churning, using account, billing, and service-subscription features, so retention efforts can be targeted proactively.

**Dataset:** [IBM/Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) — ~7,000 customer records with demographic info, account tenure, subscribed services, billing details, and a churn label.

---

## Model & Performance

Several models were trained and compared in the notebook, with hyperparameter tuning via `GridSearchCV` (5-fold CV, optimized for F1-score). **XGBoost** was selected as the final model based on the highest ROC-AUC on the test set.

| Model (tuned)         | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Decision Tree          | 0.737    | 0.503     | 0.786  | 0.614    | 0.753   |
| Random Forest          | 0.765    | 0.542     | 0.735  | 0.624    | 0.842   |
| Logistic Regression    | 0.738    | 0.504     | 0.783  | 0.614    | 0.841   |
| **XGBoost (final model)** | **0.743** | **0.510** | **0.805** | **0.624** | **0.844** |


Key preprocessing steps (replicated in the app):
- One-hot encoding of categorical features (services, contract type, payment method, etc.)
- Feature scaling via `StandardScaler`
- Class imbalance handled with `scale_pos_weight` in XGBoost

---

## App Features

- Clean Streamlit form for entering customer account details, demographics, phone/internet services, and billing info
- **Smart conditional fields:**
  - Setting *Internet Service = "No"* automatically disables and sets all six dependent services (Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies) to `"No internet service"`
  - Setting *Phone Service = "No"* automatically disables and sets *Multiple Lines* to `"No phone service"`
- **Input sanity check:** flags inconsistent Tenure / Monthly Charges / Total Charges combinations and disables prediction until resolved
- One-hot encodes and scales inputs to match the exact feature order used at training time
- Displays churn prediction with probability score and a visual progress bar
- **Reset** button to restore all fields to default values

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ChanukaJR2002/Customer_Churn_Model.git
cd Customer_Churn_Model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
Make sure `Telco_Churn_Model.pkl`, `scaler.pkl`, and `feature_columns.pkl` are in the same folder as `app.py`, then run:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Training

The full pipeline — data cleaning, exploratory data analysis, encoding, scaling, model comparison (Decision Tree, Random Forest, Logistic Regression, XGBoost), hyperparameter tuning, and final model export — is documented in **`Telco_Customer_Churn.ipynb`**.

```bash
jupyter notebook Telco_Customer_Churn.ipynb
```

---

## Known Limitations

- Precision is moderate (~0.51), meaning the model produces a fair number of false positives (customers flagged as "likely to churn" who actually stay) — by design, since recall was prioritized.
- Trained on a single static, historical dataset; performance on a different customer base or time period may vary and should be re-validated.
- `scaler.pkl` reproduces the exact preprocessing pipeline used in training; any changes to the input feature set require regenerating both the scaler and `feature_columns.pkl`.

---

## Future Improvements

- Explore additional models (CatBoost, LightGBM) and ensembling
- Add SHAP-based explainability to show which factors drove each prediction
- Threshold tuning / cost-sensitive evaluation to balance precision vs. recall based on business cost of false positives vs. false negatives
- Deploy publicly (Streamlit Community Cloud / Hugging Face Spaces) instead of running locally only
- Add batch prediction support (upload a CSV of customers, get churn predictions for all)

---

## Acknowledgements

Trained on the IBM/Kaggle Telco Customer Churn dataset. Final model: tuned XGBoost classifier.

---

## Author Details
---------------
## Name - Chanuka Rajapaksa
## GitHub - https://github.com/ChanukaJR2002
## Linkedin - https://www.linkedin.com/in/chanuka-rajapaksa-14b9533a1/
=======
# Customer_Churn_Model
Predict telecom customer churn risk from account and service details using XGBoost.

