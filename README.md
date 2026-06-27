# Telco Customer Churn Predictor

A machine learning web app that predicts whether a telecom customer is likely to **churn** (cancel their service), based on their account details, services subscribed, and billing information. Built with **XGBoost** and served through an interactive **Streamlit** UI.

---

## Project Structure

```
Customer_Churn_Model/
│
├── Telco_Customer_Churn.ipynb            # Notebook: EDA, preprocessing, model training & evaluation
├── app.py                                # Streamlit web app for inference
├── Telco_Churn_Model.pkl                 # Final tuned XGBoost model
├── scaler.pkl                            # StandardScaler fitted on training data
├── feature_columns.pkl                   # Exact one-hot encoded column order expected by the model
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Training dataset (IBM Telco Customer Churn dataset)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## Problem Statement

Customer churn — when a subscriber leaves a telecom provider — is costly to acquire and replace. This project trains a binary classifier to flag customers at high risk of churning, using account, billing, and service-subscription features, so retention efforts can be targeted proactively.

**Dataset:** [IBM/Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) — ~7,000 customer records with demographic info, account tenure, subscribed services, billing details, and a churn label.

---

## Preprocessing Pipeline

The following steps are applied in the notebook and replicated in the app:

- Dropped `customerID` and `gender` columns as they are not predictive
- Binary encoding of `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, and `Churn` (Yes → 1, No → 0)
- `TotalCharges` converted to numeric with missing values filled as `0`
- One-hot encoding of remaining categorical features using `pd.get_dummies(drop_first=True)`
- Stratified train/test split (80/20, `random_state=42`)
- Feature scaling using `StandardScaler` fitted on training data only
- Class imbalance handled with `scale_pos_weight=2.77` in XGBoost

---

## Model & Performance

Four models were trained and compared with hyperparameter tuning via `GridSearchCV` (5-fold CV, optimised for F1-score). **XGBoost** was selected as the final model based on the highest ROC-AUC on the test set.

| Model (tuned)             | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------------|----------|-----------|--------|----------|---------|
| Decision Tree             | 0.737    | 0.503     | 0.786  | 0.614    | 0.753   |
| Random Forest             | 0.765    | 0.542     | 0.735  | 0.624    | 0.842   |
| Logistic Regression       | 0.738    | 0.504     | 0.783  | 0.614    | 0.841   |
| **XGBoost (final model)** | **0.743**| **0.510** | **0.805** | **0.624** | **0.844** |

Recall was prioritised by design — it is more costly to miss a churning customer than to flag one who stays.

---

## App Features

- Clean Streamlit form for entering customer account details, demographics, phone/internet services, and billing info
- **Smart conditional fields:**
  - Setting *Internet Service = "No"* automatically disables and sets all six dependent services (Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies) to `"No internet service"`
  - Setting *Phone Service = "No"* automatically disables and sets *Multiple Lines* to `"No phone service"`
- **Input sanity check:** flags inconsistent Tenure / Monthly Charges / Total Charges combinations and disables prediction until resolved
- One-hot encodes inputs and aligns to the exact feature column order used at training time via `feature_columns.pkl`
- Scales inputs using the saved `StandardScaler` before passing to the model
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

### 3. Generate the model files

Open the notebook and run all cells:
```bash
jupyter notebook Telco_Customer_Churn.ipynb
```

Click **Kernel → Restart & Run All** and wait until the last cell prints:
```
Model saved: XGBClassifier
Scaler saved: StandardScaler
Feature columns saved: 29 columns
```

This generates `Telco_Churn_Model.pkl`, `scaler.pkl`, and `feature_columns.pkl` in the project folder.

### 4. Run the app

Make sure `Telco_Churn_Model.pkl`, `scaler.pkl`, and `feature_columns.pkl` are in the same folder as `app.py`, then run:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Training

The full pipeline — data cleaning, exploratory data analysis, binary encoding, one-hot encoding, feature scaling, model comparison (Decision Tree, Random Forest, Logistic Regression, XGBoost), hyperparameter tuning, ROC curve comparison, and final model export — is documented in **`Telco_Customer_Churn.ipynb`**.

---

## Known Limitations

- Precision is moderate (~0.51), meaning the model produces a fair number of false positives (customers flagged as "likely to churn" who actually stay) — by design, since recall was prioritised.
- Trained on a single static, historical dataset; performance on a different customer base or time period may vary and should be re-validated.
- `scaler.pkl` and `feature_columns.pkl` must be regenerated together with `Telco_Churn_Model.pkl` whenever the notebook is retrained — using mismatched pkl files will cause errors.
- Only supports individual customer predictions — batch prediction from a CSV is not yet supported.

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

**Name** — Chanuka Rajapaksa

**GitHub** — https://github.com/ChanukaJR2002

**LinkedIn** — https://www.linkedin.com/in/chanuka-rajapaksa-14b9533a1/
