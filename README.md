# 💰 Financial Risk Analysis — Loan Default Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow?style=flat&logo=powerbi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-green?style=flat&logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

> End-to-end financial risk analysis project predicting loan default using Python ML models and an interactive 3-page Power BI dashboard to help banks identify high-risk borrowers.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tools & Technologies](#-tools--technologies)
- [Analysis Steps](#-analysis-steps)
- [Key Findings](#-key-findings)
- [ML Model Results](#-ml-model-results)
- [Business Recommendations](#-business-recommendations)
- [Dashboard Preview](#-dashboard-preview)
- [How to Run](#-how-to-run)

---

## 📌 Project Overview

Loan defaults cost banks and financial institutions billions every year. This project:

- Analyses 28,632 loan records to identify default patterns
- Identifies which factors most strongly predict loan default
- Builds ML models to predict which borrowers will default
- Flags high-risk borrowers for bank risk management teams
- Delivers a 3-page interactive Power BI dashboard

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | Credit Risk Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) |
| **Rows** | 28,632 records (after cleaning) |
| **Columns** | 12 features |
| **Target** | `loan_status` (0=Paid, 1=Default) |
| **Default Rate** | 21.7% |

**Key Features:**
- `loan_grade` — Credit grade A (best) to G (worst)
- `loan_percent_income` — Loan amount as % of income
- `person_income` — Annual income
- `loan_int_rate` — Interest rate
- `loan_amnt` — Loan amount requested
- `cb_person_default_on_file` — Previous default history
- `person_home_ownership` — Rent / Own / Mortgage

---

## 📁 Project Structure
```
financial_risk/
│
├── data/
│   └── credit_risk_dataset.csv
│
├── output/
│   ├── loan_cleaned.csv
│   ├── loan_predictions.csv
│   ├── 01_default_distribution.png
│   ├── 02_grade_default.png
│   ├── 03_income_default.png
│   ├── 04_interest_default.png
│   ├── 05_prev_default.png
│   ├── 06_model_comparison.png
│   ├── 07_roc_curve.png
│   ├── 08_confusion_matrix.png
│   └── 09_feature_importance.png
│
├── Screenshots/
│   ├── Loan risk overview.png
│   ├── High risk customers.png
│   └── Risk insights.png
│
├── PowerBI/
│   └── Financial_Risk_Dashboard.pbix
│
├── loan_risk_analysis.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Data cleaning, EDA, ML models |
| **pandas / numpy** | Data manipulation |
| **matplotlib / seaborn** | Visualisations |
| **scikit-learn** | Logistic Regression, Random Forest |
| **Power BI Desktop** | Interactive 3-page dashboard |

---

## 🔄 Analysis Steps

### Step 1 — Data Loading & Cleaning
- Loaded 32,581 raw records
- Removed missing values and outliers
- Removed ages over 100 and employment length over 60 years
- Final clean dataset: **28,632 records**

### Step 2 — Exploratory Analysis (5 Charts)
- Overall loan default distribution
- Default rate by loan grade (A–G)
- Income distribution: Paid vs Default
- Interest rate: Paid vs Default
- Previous default history impact

### Step 3 — Machine Learning Models
- Encoded categorical features using LabelEncoder
- Split data: 80% train / 20% test (random_state=42)
- Trained Logistic Regression and Random Forest
- Evaluated with Accuracy, ROC-AUC, Confusion Matrix
- Generated default probability scores per borrower
- Assigned Risk Level: Low / Medium / High

### Step 4 — Power BI Dashboard
- 3-page interactive dashboard
- Page 1: Loan Risk Overview
- Page 2: High Risk Loan Customers
- Page 3: Risk Insights & Recommendations

---

## 🔍 Key Findings

| Finding | Detail |
|---|---|
| Overall default rate | **21.7%** — 6,202 out of 28,632 loans defaulted |
| Loan grade risk | Grade A = **9.6%** default vs Grade G = **98.3%** default |
| Income gap | Defaulters earn median **$42K/year** vs **$60K** for paid |
| Interest rate | Defaulters pay median **13.5%** vs **10.6%** for paid |
| Previous default | No history = **18.2%** vs Has history = **37.7%** |
| Top predictor | `loan_percent_income` — loan as % of income |

---

## 🤖 ML Model Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 83.5% | 0.831 |
| **Random Forest** | **92.9%** | **0.933** |

> Random Forest significantly outperformed Logistic Regression — the complex non-linear relationships between loan features are better captured by ensemble methods.

### Confusion Matrix (Random Forest)

| | Predicted Paid | Predicted Default |
|---|---|---|
| **Actually Paid** | 4,448 ✅ | 40 ❌ |
| **Actually Defaulted** | 364 ❌ | 875 ✅ |

### Top 10 Default Predictors
1. 💳 loan_percent_income (0.225)
2. 💰 person_income (0.138)
3. 📋 loan_grade (0.122)
4. 📈 loan_int_rate (0.118)
5. 🏠 person_home_ownership (0.095)
6. 🎯 loan_intent (0.074)
7. 💵 loan_amnt (0.072)
8. 💼 person_emp_length (0.065)
9. 🎂 person_age (0.044)
10. 📅 cb_person_cred_hist_length (0.035)

---

## 💡 Business Recommendations

### Loan Approval Strategy
- ✅ Automatically reject Grade F and G loan applications — default rate above 70%
- ✅ Require additional collateral for Grade D and E applicants
- ✅ Set maximum loan_percent_income threshold at 30%

### Interest Rate Policy
- ✅ Cap interest rates for borrowers already at financial risk
- ✅ High interest rates increase default probability — a vicious cycle
- ✅ Offer rate reduction incentives for on-time payment history

### Income Verification
- ✅ Mandate income verification for all loans above $10,000
- ✅ Flag applications where loan exceeds 40% of annual income
- ✅ Prioritise applicants with income above $60,000

### Previous Default History
- ✅ Applicants with previous defaults should face stricter criteria
- ✅ Require 2+ years clean credit history before approving new loans
- ✅ Use ML model monthly to rescore existing borrowers for early warning

---

## 📊 Dashboard Preview

### Page 1 — Loan Risk Overview
![Loan Risk Overview](financial_risk/Screenshots/Loan%20risk%20overview.png)

### Page 2 — High Risk Loan Customers
![High Risk Customers](financial_risk/Screenshots/High%20risk%20customer.png)

### Page 3 — Risk Insights & Recommendations
![Risk Insights](financial_risk/Screenshots/Risk%20insights.png)

---

## ▶️ How to Run

### 1. Clone the repository
```
git clone https://github.com/PrabinPokhrel/financial-risk-analysis.git
cd financial-risk-analysis
```

### 2. Install required libraries
```
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to https://www.kaggle.com/datasets/laotse/credit-risk-dataset
- Download `credit_risk_dataset.csv`
- Place it inside the `data/` folder

### 4. Run the analysis
```
python loan_risk_analysis.py
```

### 5. View the Power BI dashboard
- Open `PowerBI/Financial_Risk_Dashboard.pbix` in Power BI Desktop
- Power BI Desktop is free — download from microsoft.com

---

## 👤 Author

**Prabin Pokhrel**
Master's in Business Intelligence — Dalarna University
- GitHub: [@PrabinPokhrel](https://github.com/PrabinPokhrel)

---

*⭐ If you found this project helpful, please give it a star!*
```

