
# FINANCIAL RISK ANALYSIS — LOAN DEFAULT PREDICTION
# Step 4: Machine Learning Models
# Author: Prabin Pokhrel


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay,
                              roc_auc_score, roc_curve)

sns.set_theme(style="whitegrid")

OUTPUT_PATH = r'C:\Users\WELCOME\Desktop\DataAnalysis_Projects\financial_risk\output'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Load cleaned data ─────────────────────
df = pd.read_csv(os.path.join(OUTPUT_PATH, 'loan_cleaned.csv'))
print("✅ Cleaned data loaded!")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())


# ════════════════════════════════════════════
# 4.1 Prepare Features
# ════════════════════════════════════════════

print("\n" + "=" * 60)
print("4.1 PREPARING FEATURES")
print("=" * 60)

# Keep original text columns for Power BI export
df_powerbi = df.copy()

# Encode categorical columns for ML
le = LabelEncoder()
cat_cols = ['loan_grade', 'loan_intent',
            'person_home_ownership',
            'cb_person_default_on_file']

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("✅ Categorical columns encoded!")

# Features and target
features = ['person_age', 'person_income', 'person_emp_length',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'loan_grade', 'loan_intent', 'person_home_ownership',
            'cb_person_default_on_file', 'cb_person_cred_hist_length']

X = df[features]
y = df['loan_status']

print(f"\n✅ Features shape: {X.shape}")
print(f"✅ Target shape  : {y.shape}")
print(f"\nDefault rate in dataset: {y.mean()*100:.1f}%")


# ════════════════════════════════════════════
# 4.2 Split Data
# ════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n✅ Training set : {X_train.shape[0]:,} rows")
print(f"✅ Testing set  : {X_test.shape[0]:,} rows")


# ════════════════════════════════════════════
# 4.3 Model 1 — Logistic Regression
# ════════════════════════════════════════════

print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]
lr_acc  = accuracy_score(y_test, lr_pred) * 100
lr_auc  = roc_auc_score(y_test, lr_prob)

print(f"✅ Accuracy : {lr_acc:.1f}%")
print(f"✅ ROC-AUC  : {lr_auc:.3f}")
print(classification_report(y_test, lr_pred,
      target_names=['Paid', 'Default']))


# ════════════════════════════════════════════
# 4.4 Model 2 — Random Forest
# ════════════════════════════════════════════

print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_acc  = accuracy_score(y_test, rf_pred) * 100
rf_auc  = roc_auc_score(y_test, rf_prob)

print(f"✅ Accuracy : {rf_acc:.1f}%")
print(f"✅ ROC-AUC  : {rf_auc:.3f}")
print(classification_report(y_test, rf_pred,
      target_names=['Paid', 'Default']))


# ════════════════════════════════════════════
# 4.5 Charts
# ════════════════════════════════════════════

# ── Chart 6: Model Comparison ─────────────
fig, ax = plt.subplots(figsize=(7, 5))
models     = ['Logistic Regression', 'Random Forest']
accuracies = [lr_acc, rf_acc]
colors_m   = ['#3498DB', '#2ECC71']
bars = ax.bar(models, accuracies,
              color=colors_m, edgecolor='white', width=0.5)
ax.set_title('Model Accuracy Comparison',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(75, 95)
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center',
            fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '06_model_comparison.png'), dpi=150)
plt.show()
print("✅ Chart 6 saved!")


# ── Chart 7: ROC Curve ────────────────────
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(lr_fpr, lr_tpr, color='#3498DB', lw=2,
        label=f'Logistic Regression (AUC={lr_auc:.3f})')
ax.plot(rf_fpr, rf_tpr, color='#2ECC71', lw=2,
        label=f'Random Forest (AUC={rf_auc:.3f})')
ax.plot([0,1],[0,1], color='gray', lw=1,
        linestyle='--', label='Random Guess')
ax.fill_between(lr_fpr, lr_tpr, alpha=0.05, color='#3498DB')
ax.fill_between(rf_fpr, rf_tpr, alpha=0.05, color='#2ECC71')
ax.set_title('ROC Curve — Model Comparison',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '07_roc_curve.png'), dpi=150)
plt.show()
print("✅ Chart 7 saved!")


# ── Chart 8: Confusion Matrix ─────────────
cm = confusion_matrix(y_test, rf_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Paid', 'Default'])
disp.plot(cmap='Blues', ax=ax)
ax.set_title('Random Forest — Confusion Matrix',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '08_confusion_matrix.png'), dpi=150)
plt.show()
print("✅ Chart 8 saved!")


# ── Chart 9: Feature Importance ───────────
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top10    = feat_imp.sort_values(ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top10.index, top10.values,
               color='#3498DB', edgecolor='white')
ax.set_title('Top 10 Factors That Predict Loan Default',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Importance Score', fontsize=12)
for bar, val in zip(bars, top10.values):
    ax.text(val + 0.002,
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '09_feature_importance.png'), dpi=150)
plt.show()
print("✅ Chart 9 saved!")


# ════════════════════════════════════════════
# 4.6 Save Predictions for Power BI
# ════════════════════════════════════════════

print("\n" + "=" * 60)
print("4.6 SAVING PREDICTIONS FOR POWER BI")
print("=" * 60)

# Get test set with original text columns
result = df_powerbi.loc[X_test.index].copy()
result['Actual_Default']       = y_test.values
result['Predicted_Default']    = rf_pred
result['Default_Probability']  = (rf_prob * 100).round(1)
result['Risk_Level']           = pd.cut(
    result['Default_Probability'],
    bins=[0, 30, 60, 100],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Remove nulls
result = result[result['Default_Probability'] > 0]
result = result.dropna(subset=['Risk_Level'])

result.to_csv(
    os.path.join(OUTPUT_PATH, 'loan_predictions.csv'),
    index=False, encoding='utf-8-sig')

print("✅ loan_predictions.csv saved!")
print(f"\nTotal rows     : {len(result):,}")
print(f"High Risk      : {len(result[result['Risk_Level']=='High Risk']):,}")
print(f"Medium Risk    : {len(result[result['Risk_Level']=='Medium Risk']):,}")
print(f"Low Risk       : {len(result[result['Risk_Level']=='Low Risk']):,}")

print(f"\n🎉 Step 4 Complete!")
print(f"Logistic Regression → Accuracy: {lr_acc:.1f}% | AUC: {lr_auc:.3f}")
print(f"Random Forest       → Accuracy: {rf_acc:.1f}% | AUC: {rf_auc:.3f}")
print(f"Top predictor       : {feat_imp.idxmax()}")
