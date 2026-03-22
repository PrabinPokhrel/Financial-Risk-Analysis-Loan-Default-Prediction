
# FINANCIAL RISK ANALYSIS — LOAN DEFAULT PREDICTION
# Steps 2 & 3: Load, Clean & Key Analysis
# Author: Prabin Pokhrel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

# ── Paths ─────────────────────────────────
DATA_PATH   = r'C:\Users\WELCOME\Desktop\DataAnalysis_Projects\financial_risk\data\credit_risk_dataset.csv'
OUTPUT_PATH = r'C:\Users\WELCOME\Desktop\DataAnalysis_Projects\financial_risk\output'
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ════════════════════════════════════════════
# STEP 2: LOAD & CLEAN DATA
# ════════════════════════════════════════════

print("=" * 60)
print("STEP 2: LOADING & CLEANING DATA")
print("=" * 60)

# ── Load ──────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# ── Drop missing values ───────────────────
df.dropna(inplace=True)
print(f"\n✅ After dropping nulls: {df.shape[0]:,} rows")

# ── Remove outliers ───────────────────────
# Remove unrealistic ages (> 100)
df = df[df['person_age'] <= 100]
# Remove unrealistic employment length (> 60 years)
df = df[df['person_emp_length'] <= 60]
print(f"✅ After removing outliers: {df.shape[0]:,} rows")

# ── Check target column ───────────────────
print(f"\n📊 Loan Status Distribution:")
print(df['loan_status'].value_counts())
default_rate = df['loan_status'].mean() * 100
print(f"\n📊 Overall Default Rate: {default_rate:.1f}%")

# ── Save cleaned data ─────────────────────
df.to_csv(os.path.join(OUTPUT_PATH, 'loan_cleaned.csv'),
          index=False, encoding='utf-8-sig')
print("✅ loan_cleaned.csv saved!")


# ════════════════════════════════════════════
# STEP 3: KEY ANALYSIS — 5 CHARTS
# ════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 3: KEY ANALYSIS")
print("=" * 60)

# ── Chart 1: Overall Default Distribution ─
fig, ax = plt.subplots(figsize=(7, 5))
counts = df['loan_status'].value_counts()
labels = ['Paid (0)', 'Default (1)']
colors = ['#2ECC71', '#E74C3C']
bars = ax.bar(labels, counts.values,
              color=colors, edgecolor='white', width=0.5)
ax.set_title('Loan Default Distribution',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Loan Status', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 100,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '01_default_distribution.png'), dpi=150)
plt.show()
print("✅ Chart 1 saved!")


# ── Chart 2: Default Rate by Loan Grade ───
grade_default = df.groupby('loan_grade')['loan_status'].mean() * 100
grade_default = grade_default.sort_index()
print(f"\n📊 Default Rate by Loan Grade:")
print(grade_default.round(1))

fig, ax = plt.subplots(figsize=(8, 5))
colors_grade = ['#2ECC71','#A8D5A2','#F7DC6F',
                '#E67E22','#E74C3C','#C0392B','#922B21']
bars = ax.bar(grade_default.index, grade_default.values,
              color=colors_grade[:len(grade_default)],
              edgecolor='white', width=0.6)
ax.set_title('Default Rate by Loan Grade (%)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Loan Grade (A=Best → G=Worst)', fontsize=12)
ax.set_ylabel('Default Rate (%)', fontsize=12)
ax.set_ylim(0, 100)
for bar, val in zip(bars, grade_default.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{val:.1f}%', ha='center',
            fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '02_grade_default.png'), dpi=150)
plt.show()
print("✅ Chart 2 saved!")


# ── Chart 3: Income vs Default (Box Plot) ─
print(f"\n📊 Average Income by Loan Status:")
print(df.groupby('loan_status')['person_income'].mean().round(0))

fig, ax = plt.subplots(figsize=(7, 5))
paid     = df[df['loan_status']==0]['person_income']
default  = df[df['loan_status']==1]['person_income']
bp = ax.boxplot([paid, default],
                labels=['Paid', 'Default'],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#2ECC71')
bp['boxes'][1].set_facecolor('#E74C3C')
ax.set_title('Income Distribution: Paid vs Default',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Loan Status', fontsize=12)
ax.set_ylabel('Annual Income ($)', fontsize=12)
ax.text(1, paid.median(),
        f'  Median: ${paid.median():,.0f}',
        va='center', fontsize=10, color='#27AE60')
ax.text(2, default.median(),
        f'  Median: ${default.median():,.0f}',
        va='center', fontsize=10, color='#C0392B')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '03_income_default.png'), dpi=150)
plt.show()
print("✅ Chart 3 saved!")


# ── Chart 4: Interest Rate vs Default ─────
print(f"\n📊 Average Interest Rate by Loan Status:")
print(df.groupby('loan_status')['loan_int_rate'].mean().round(2))

fig, ax = plt.subplots(figsize=(7, 5))
paid_int    = df[df['loan_status']==0]['loan_int_rate']
default_int = df[df['loan_status']==1]['loan_int_rate']
bp = ax.boxplot([paid_int, default_int],
                labels=['Paid', 'Default'],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#2ECC71')
bp['boxes'][1].set_facecolor('#E74C3C')
ax.set_title('Interest Rate: Paid vs Default',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Loan Status', fontsize=12)
ax.set_ylabel('Interest Rate (%)', fontsize=12)
ax.text(1, paid_int.median(),
        f'  Median: {paid_int.median():.1f}%',
        va='center', fontsize=10, color='#27AE60')
ax.text(2, default_int.median(),
        f'  Median: {default_int.median():.1f}%',
        va='center', fontsize=10, color='#C0392B')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '04_interest_default.png'), dpi=150)
plt.show()
print("✅ Chart 4 saved!")


# ── Chart 5: Previous Default History ─────
prev_default = df.groupby('cb_person_default_on_file')['loan_status'].mean() * 100
print(f"\n📊 Default Rate by Previous Default History:")
print(prev_default.round(1))

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(['No Previous Default\n(N)',
               'Has Previous Default\n(Y)'],
              prev_default.values,
              color=['#2ECC71', '#E74C3C'],
              edgecolor='white', width=0.5)
ax.set_title('Default Rate by Previous Default History (%)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Default Rate (%)', fontsize=12)
ax.set_ylim(0, 60)
for bar, val in zip(bars, prev_default.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center',
            fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '05_prev_default.png'), dpi=150)
plt.show()
print("✅ Chart 5 saved!")


print("\n🎉 Steps 2 & 3 Complete! All files saved to output folder.")
