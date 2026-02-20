"""
Fine-tuned LLM for In-Hospital Mortality Prediction
====================================================
SIMPLIFIED VERSION - Works without PyTorch/Transformers

This script provides:
1. Data preprocessing and feature engineering
2. A simulation of LLM-style text processing using TF-IDF and embeddings
3. Comparison-ready metrics format
4. Fairness analysis

For the FULL LLM fine-tuning, use the Jupyter notebook with a proper
PyTorch environment (Python 3.10-3.12 recommended).
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data processing and ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, 
    mean_squared_error, mean_absolute_error, r2_score
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("MORTALITY PREDICTION - LLM-Style Text Processing")
print("=" * 70)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n[1/7] Loading data...")

patients = pd.read_csv('patients_table.csv', index_col=0)
admissions = pd.read_csv('patient_admissions.csv', index_col=0)
diagnoses = pd.read_csv('disease_diagnosis_code.csv', index_col=0)

print(f"Patients: {len(patients)} records")
print(f"Admissions: {len(admissions)} records")
print(f"Diagnoses: {len(diagnoses)} records")

# =============================================================================
# 2. ICD CODE HARMONIZATION
# =============================================================================
print("\n[2/7] Harmonizing ICD codes...")

ICD9_TO_ICD10_MAP = {
    '4280': 'I50.9', '4271': 'I49.9', '42833': 'I50.33', '42731': 'I48.91',
    '4019': 'I10', '41401': 'I25.10', '486': 'J18.9', '51881': 'J96.01',
    '5070': 'J69.0', '5849': 'N17.9', '5859': 'N18.9', '2764': 'E87.1',
    '2859': 'D64.9', '99591': 'A41.9', '99592': 'R65.20', '2930': 'F05',
    '33394': 'G47.33', '5609': 'K56.60', '5781': 'K92.1'
}

def harmonize_icd_codes(diagnoses_df):
    def convert(row):
        if row['icd_version'] == 10:
            return row['icd_code']
        icd9 = str(row['icd_code'])
        if icd9 in ICD9_TO_ICD10_MAP:
            return ICD9_TO_ICD10_MAP[icd9]
        try:
            prefix = int(icd9[:3])
            if 390 <= prefix <= 459: return f"I{icd9}"
            elif 460 <= prefix <= 519: return f"J{icd9}"
            elif 520 <= prefix <= 579: return f"K{icd9}"
            elif 580 <= prefix <= 629: return f"N{icd9}"
            elif 240 <= prefix <= 279: return f"E{icd9}"
        except: pass
        return f"UNK_{icd9}"
    
    diagnoses_df['icd_code_harmonized'] = diagnoses_df.apply(convert, axis=1)
    return diagnoses_df

diagnoses = harmonize_icd_codes(diagnoses.copy())
print(f"Harmonized {len(diagnoses)} diagnosis codes")

# =============================================================================
# 3. ICD DESCRIPTIONS FOR CLINICAL TEXT
# =============================================================================
ICD_DESCRIPTIONS = {
    'I50': 'heart failure', 'I49': 'cardiac arrhythmia', 'I48': 'atrial fibrillation',
    'I10': 'hypertension', 'I25': 'coronary artery disease', 'J18': 'pneumonia',
    'J96': 'respiratory failure', 'J69': 'aspiration pneumonia', 'N17': 'acute kidney injury',
    'N18': 'chronic kidney disease', 'E87': 'electrolyte imbalance', 'D64': 'anemia',
    'A41': 'sepsis', 'R65': 'severe sepsis', 'F05': 'delirium', 'G47': 'sleep apnea',
    'K56': 'bowel obstruction', 'K92': 'GI bleeding', 'E11': 'diabetes'
}

def get_diagnosis_description(icd_code):
    code = str(icd_code)
    for prefix, desc in ICD_DESCRIPTIONS.items():
        if code.startswith(prefix):
            return desc
    return f"diagnosis_{code[:3]}"

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
print("\n[3/7] Feature engineering...")

# Convert datetime columns
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

# Calculate Length of Stay
admissions['los_days'] = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / (24 * 3600)

# Temporal features
admissions['admit_hour'] = admissions['admittime'].dt.hour
admissions['admit_dayofweek'] = admissions['admittime'].dt.dayofweek
admissions['is_weekend'] = admissions['admit_dayofweek'].isin([5, 6]).astype(int)
admissions['is_night_admission'] = ((admissions['admit_hour'] >= 22) | (admissions['admit_hour'] <= 6)).astype(int)

# Merge tables
merged = admissions.merge(patients, on='subject_id', how='left')

# Aggregate diagnoses
diag_grouped = diagnoses.groupby('hadm_id').agg({
    'icd_code_harmonized': lambda x: list(x),
    'seq_num': 'count'
}).reset_index()
diag_grouped.columns = ['hadm_id', 'diagnosis_codes', 'num_diagnoses']

merged = merged.merge(diag_grouped, on='hadm_id', how='left')
merged['num_diagnoses'] = merged['num_diagnoses'].fillna(0)
merged['diagnosis_codes'] = merged['diagnosis_codes'].apply(lambda x: x if isinstance(x, list) else [])

print(f"Merged dataset: {len(merged)} records")
print(f"Mortality rate: {merged['hospital_expire_flag'].mean()*100:.2f}%")

# =============================================================================
# 5. CREATE CLINICAL TEXT (LLM-style input)
# =============================================================================
print("\n[4/7] Creating clinical text representations...")

def create_clinical_text(row):
    age = row.get('anchor_age', 'unknown')
    gender = 'female' if row.get('gender', '') == 'F' else 'male'
    race = str(row.get('race', 'unknown')).lower()
    admission_type = str(row.get('admission_type', 'unknown')).lower()
    admission_location = str(row.get('admission_location', 'unknown')).lower()
    insurance = str(row.get('insurance', 'unknown')).lower()
    marital_status = str(row.get('marital_status', 'unknown')).lower()
    
    diag_codes = row.get('diagnosis_codes', [])
    if isinstance(diag_codes, list) and len(diag_codes) > 0:
        descriptions = list(set([get_diagnosis_description(c) for c in diag_codes[:10]]))
        diagnoses_text = ', '.join(descriptions[:7])
    else:
        diagnoses_text = 'no documented diagnoses'
    
    is_weekend = 'weekend' if row.get('is_weekend', 0) == 1 else 'weekday'
    is_night = 'night' if row.get('is_night_admission', 0) == 1 else 'day'
    num_diag = int(row.get('num_diagnoses', 0))
    
    text = f"""Patient is a {age} year old {gender} of {race} ethnicity. 
Admission type: {admission_type} from {admission_location}. 
Insurance: {insurance}. Marital status: {marital_status}.
Admitted on a {is_weekend} during {is_night} hours.
Number of diagnoses: {num_diag}.
Primary diagnoses include: {diagnoses_text}."""
    
    return text.strip()

merged['clinical_text'] = merged.apply(create_clinical_text, axis=1)

# Sample clinical text
print("\n" + "=" * 50)
print("SAMPLE CLINICAL TEXT")
print("=" * 50)
print(merged['clinical_text'].iloc[0])
print("=" * 50)

# Prepare final dataset
llm_df = merged[['hadm_id', 'subject_id', 'clinical_text', 'hospital_expire_flag', 
                 'los_days', 'race', 'gender', 'anchor_age', 'num_diagnoses']].copy()
llm_df['hospital_expire_flag'] = llm_df['hospital_expire_flag'].fillna(0).astype(int)
llm_df = llm_df.dropna(subset=['clinical_text', 'hospital_expire_flag', 'los_days'])

# =============================================================================
# 6. MODEL TRAINING (TF-IDF + Gradient Boosting as LLM proxy)
# =============================================================================
print("\n[5/7] Training models...")

# Split data
train_df, temp_df = train_test_split(llm_df, test_size=0.3, random_state=SEED,
                                     stratify=llm_df['hospital_expire_flag'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED,
                                   stratify=temp_df['hospital_expire_flag'])

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# TF-IDF Vectorization (simulates transformer tokenization)
print("Creating TF-IDF embeddings (simulating transformer embeddings)...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_text = tfidf.fit_transform(train_df['clinical_text'])
X_val_text = tfidf.transform(val_df['clinical_text'])
X_test_text = tfidf.transform(test_df['clinical_text'])

# --- MORTALITY MODEL ---
print("\nTraining mortality prediction model (Gradient Boosting on text embeddings)...")
mortality_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED
)
mortality_model.fit(X_train_text, train_df['hospital_expire_flag'])

# Predictions
y_pred_mortality = mortality_model.predict(X_test_text)
y_prob_mortality = mortality_model.predict_proba(X_test_text)[:, 1]

# Metrics
mortality_results = {
    'auc_roc': roc_auc_score(test_df['hospital_expire_flag'], y_prob_mortality),
    'precision': precision_score(test_df['hospital_expire_flag'], y_pred_mortality),
    'recall': recall_score(test_df['hospital_expire_flag'], y_pred_mortality),
    'f1_score': f1_score(test_df['hospital_expire_flag'], y_pred_mortality)
}

print("\n" + "=" * 50)
print("MORTALITY PREDICTION RESULTS (Test Set)")
print("=" * 50)
print(f"AUC-ROC:   {mortality_results['auc_roc']:.4f}")
print(f"Precision: {mortality_results['precision']:.4f}")
print(f"Recall:    {mortality_results['recall']:.4f}")
print(f"F1-Score:  {mortality_results['f1_score']:.4f}")

print("\nClassification Report:")
print(classification_report(test_df['hospital_expire_flag'], y_pred_mortality,
                          target_names=['Survived', 'Expired']))

# --- LOS MODEL ---
print("\nTraining Length of Stay prediction model...")
los_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED
)
los_model.fit(X_train_text, train_df['los_days'])

y_pred_los = los_model.predict(X_test_text)

los_results = {
    'mse': mean_squared_error(test_df['los_days'], y_pred_los),
    'rmse': np.sqrt(mean_squared_error(test_df['los_days'], y_pred_los)),
    'mae': mean_absolute_error(test_df['los_days'], y_pred_los),
    'r2': r2_score(test_df['los_days'], y_pred_los)
}

print("\n" + "=" * 50)
print("LENGTH OF STAY PREDICTION RESULTS (Test Set)")
print("=" * 50)
print(f"MSE:  {los_results['mse']:.4f}")
print(f"RMSE: {los_results['rmse']:.4f} days")
print(f"MAE:  {los_results['mae']:.4f} days")
print(f"R²:   {los_results['r2']:.4f}")

# =============================================================================
# 7. FAIRNESS ANALYSIS
# =============================================================================
print("\n[6/7] Fairness analysis...")

def standardize_race(race):
    race = str(race).upper()
    if 'WHITE' in race: return 'White'
    elif 'BLACK' in race or 'AFRICAN' in race: return 'Black'
    elif 'HISPANIC' in race or 'LATINO' in race: return 'Hispanic'
    elif 'ASIAN' in race: return 'Asian'
    elif 'UNKNOWN' in race or 'UNABLE' in race: return 'Unknown'
    else: return 'Other'

test_df = test_df.copy()
test_df['race_group'] = test_df['race'].apply(standardize_race)

fairness_results = {}
for race_group in test_df['race_group'].unique():
    if race_group == 'Unknown':
        continue
    
    race_mask = test_df['race_group'] == race_group
    if race_mask.sum() < 30:
        continue
    
    X_race = X_test_text[race_mask.values]
    y_race = test_df.loc[race_mask, 'hospital_expire_flag']
    
    y_pred = mortality_model.predict(X_race)
    y_prob = mortality_model.predict_proba(X_race)[:, 1]
    
    try:
        auc = roc_auc_score(y_race, y_prob)
    except:
        auc = np.nan
    
    fairness_results[race_group] = {
        'n_samples': race_mask.sum(),
        'mortality_rate': y_race.mean(),
        'auc_roc': auc,
        'precision': precision_score(y_race, y_pred, zero_division=0),
        'recall': recall_score(y_race, y_pred, zero_division=0),
        'f1_score': f1_score(y_race, y_pred, zero_division=0)
    }

fairness_df = pd.DataFrame(fairness_results).T.sort_values('n_samples', ascending=False)

print("\n" + "=" * 70)
print("FAIRNESS ANALYSIS: Model Performance by Ethnic Group")
print("=" * 70)
print(fairness_df.round(4).to_string())

auc_values = fairness_df['auc_roc'].dropna()
if len(auc_values) > 1:
    disparity = auc_values.max() - auc_values.min()
    print(f"\nAUC-ROC Disparity (max - min): {disparity:.4f}")
    print(f"Mean AUC across groups: {auc_values.mean():.4f}")

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================
print("\n[7/7] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Confusion Matrix
cm = confusion_matrix(test_df['hospital_expire_flag'], y_pred_mortality)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Survived', 'Expired'],
            yticklabels=['Survived', 'Expired'])
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_title('Mortality Prediction - Confusion Matrix')

# 2. LOS Predictions
axes[0, 1].scatter(test_df['los_days'], y_pred_los, alpha=0.5, s=20)
max_val = max(test_df['los_days'].max(), max(y_pred_los))
axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual LOS (days)')
axes[0, 1].set_ylabel('Predicted LOS (days)')
axes[0, 1].set_title(f'LOS Prediction (R² = {los_results["r2"]:.3f})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Fairness - AUC by Race
if len(fairness_df) > 0:
    colors = plt.cm.Set2(np.linspace(0, 1, len(fairness_df)))
    bars = axes[1, 0].bar(range(len(fairness_df)), fairness_df['auc_roc'].values, 
                          color=colors, edgecolor='black', alpha=0.8)
    axes[1, 0].set_xticks(range(len(fairness_df)))
    axes[1, 0].set_xticklabels(fairness_df.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('Model Performance by Ethnic Group')
    axes[1, 0].axhline(y=fairness_df['auc_roc'].mean(), color='red', linestyle='--',
                       label=f'Mean: {fairness_df["auc_roc"].mean():.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Results Summary
axes[1, 1].axis('off')
summary_text = f"""
SUMMARY OF RESULTS
==================

MORTALITY PREDICTION
--------------------
AUC-ROC:   {mortality_results['auc_roc']:.4f}
Precision: {mortality_results['precision']:.4f}
Recall:    {mortality_results['recall']:.4f}
F1-Score:  {mortality_results['f1_score']:.4f}

LENGTH OF STAY PREDICTION
-------------------------
RMSE: {los_results['rmse']:.2f} days
MAE:  {los_results['mae']:.2f} days
R²:   {los_results['r2']:.4f}

FAIRNESS ANALYSIS
-----------------
Groups: {len(fairness_df)}
AUC Disparity: {disparity:.4f}

MODEL: TF-IDF + Gradient Boosting
(Simulating LLM text embeddings)
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.savefig('mortality_results_simplified.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mortality_results_simplified.png")

# Save results
results_df = pd.DataFrame({
    'Metric': ['AUC-ROC', 'Precision', 'Recall', 'F1-Score', 'LOS RMSE', 'LOS MAE', 'LOS R²'],
    'Value': [mortality_results['auc_roc'], mortality_results['precision'],
              mortality_results['recall'], mortality_results['f1_score'],
              los_results['rmse'], los_results['mae'], los_results['r2']]
})
results_df.to_csv('model_results_simplified.csv', index=False)
fairness_df.to_csv('fairness_analysis_simplified.csv')

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print("""
Results saved:
  - mortality_results_simplified.png
  - model_results_simplified.csv
  - fairness_analysis_simplified.csv

NOTE: This is a simplified version using TF-IDF + Gradient Boosting.
For the FULL LLM fine-tuning with BioClinicalBERT:
1. Set up Python 3.10-3.12 environment
2. Install: pip install torch transformers datasets
3. Run: mortality_llm_finetuning.ipynb
""")
