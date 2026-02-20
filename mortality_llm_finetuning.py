"""
Fine-tuned LLM for In-Hospital Mortality Prediction
====================================================
This script implements a fine-tuned transformer model for predicting:
1. In-hospital mortality (hospital_expire_flag)
2. Length of hospital stay

Uses BioClinicalBERT as the base model, which is pre-trained on clinical notes.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data processing and ML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

# Deep Learning / Transformers
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW

# Hugging Face Transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# 1. DATA LOADING AND INTEGRATION
# =============================================================================

def load_data(data_path='.'):
    """Load all three data tables."""
    patients = pd.read_csv(f'{data_path}/patients_table.csv', index_col=0)
    admissions = pd.read_csv(f'{data_path}/patient_admissions.csv', index_col=0)
    diagnoses = pd.read_csv(f'{data_path}/disease_diagnosis_code.csv', index_col=0)
    
    print(f"Patients: {len(patients)} records")
    print(f"Admissions: {len(admissions)} records")
    print(f"Diagnoses: {len(diagnoses)} records")
    
    return patients, admissions, diagnoses


# =============================================================================
# 2. ICD CODE HARMONIZATION (ICD-9 to ICD-10 mapping)
# =============================================================================

# Common ICD-9 to ICD-10 mapping for cardiovascular, respiratory, and metabolic conditions
ICD9_TO_ICD10_MAP = {
    # Cardiovascular
    '4280': 'I50.9',    # Heart failure
    '4271': 'I49.9',    # Cardiac dysrhythmias
    '42833': 'I50.33',  # Acute on chronic diastolic heart failure
    '42832': 'I50.32',  # Chronic diastolic heart failure
    '42731': 'I48.91',  # Atrial fibrillation
    '42781': 'I49.9',   # Cardiac dysrhythmia
    '4019': 'I10',      # Hypertension
    '41401': 'I25.10',  # Coronary atherosclerosis
    '4240': 'I34.0',    # Mitral valve disorders
    '4589': 'I95.9',    # Hypotension
    '4279': 'I51.9',    # Heart disease
    '42789': 'I49.9',   # Other cardiac dysrhythmias
    
    # Respiratory
    '486': 'J18.9',     # Pneumonia
    '51881': 'J96.01',  # Acute respiratory failure
    '5070': 'J69.0',    # Aspiration pneumonia
    '5119': 'J91.8',    # Pleural effusion
    '5180': 'J98.4',    # Lung disorders
    '5185': 'J96.90',   # Respiratory failure
    
    # Renal
    '5849': 'N17.9',    # Acute kidney failure
    '5859': 'N18.9',    # Chronic kidney disease
    '585': 'N18.9',     # Chronic kidney disease
    
    # Metabolic/Endocrine
    '2764': 'E87.1',    # Hyperosmolality
    '2767': 'E87.5',    # Hyperkalemia
    '2763': 'E87.0',    # Hyperosmolality and hypernatremia
    '2859': 'D64.9',    # Anemia
    '25000': 'E11.9',   # Diabetes mellitus
    '2449': 'E03.9',    # Hypothyroidism
    
    # Infectious
    '99591': 'A41.9',   # Sepsis
    '99592': 'R65.20',  # Severe sepsis
    '0389': 'A49.9',    # Septicemia
    '5990': 'N39.0',    # Urinary tract infection
    
    # Neurological
    '2930': 'F05',      # Delirium
    '33394': 'G47.33',  # Obstructive sleep apnea
    '43491': 'I63.9',   # Cerebral infarction
    
    # GI
    '5609': 'K56.60',   # Intestinal obstruction
    '56210': 'K63.1',   # Diverticulum of intestine
    '5781': 'K92.1',    # Melena
    '1539': 'C18.9',    # Colon cancer
    
    # Musculoskeletal
    '71590': 'M25.50',  # Joint pain
    '73300': 'M81.0',   # Osteoporosis
    '73730': 'M62.81',  # Muscle weakness
    
    # Other
    '45981': 'R60.9',   # Edema
    '3004': 'F41.1',    # Anxiety disorder
    'V850': 'Z68.1',    # BMI
}


def harmonize_icd_codes(diagnoses):
    """
    Harmonize ICD-9 codes to ICD-10.
    For unmapped codes, create a generic mapping based on code prefix.
    """
    def convert_icd9_to_icd10(row):
        if row['icd_version'] == 10:
            return row['icd_code']
        
        icd9_code = str(row['icd_code'])
        
        # Check direct mapping first
        if icd9_code in ICD9_TO_ICD10_MAP:
            return ICD9_TO_ICD10_MAP[icd9_code]
        
        # Generic mapping based on code prefix for unmapped codes
        prefix = icd9_code[:3] if len(icd9_code) >= 3 else icd9_code
        
        # Cardiovascular (390-459 -> I)
        if prefix.isdigit() and 390 <= int(prefix[:3] if prefix[:3].isdigit() else 0) <= 459:
            return f"I{prefix}"
        # Respiratory (460-519 -> J)
        elif prefix.isdigit() and 460 <= int(prefix[:3] if prefix[:3].isdigit() else 0) <= 519:
            return f"J{prefix}"
        # Digestive (520-579 -> K)
        elif prefix.isdigit() and 520 <= int(prefix[:3] if prefix[:3].isdigit() else 0) <= 579:
            return f"K{prefix}"
        # Genitourinary (580-629 -> N)
        elif prefix.isdigit() and 580 <= int(prefix[:3] if prefix[:3].isdigit() else 0) <= 629:
            return f"N{prefix}"
        # Endocrine (240-279 -> E)
        elif prefix.isdigit() and 240 <= int(prefix[:3] if prefix[:3].isdigit() else 0) <= 279:
            return f"E{prefix}"
        # Mental disorders (290-319 -> F)
        elif prefix.isdigit() and 290 <= int(prefix[:3] if prefix[:3].isdigit() else 0) <= 319:
            return f"F{prefix}"
        # Nervous system (320-389 -> G)
        elif prefix.isdigit() and 320 <= int(prefix[:3] if prefix[:3].isdigit() else 0) <= 389:
            return f"G{prefix}"
        else:
            return f"UNK_{icd9_code}"
    
    diagnoses['icd_code_harmonized'] = diagnoses.apply(convert_icd9_to_icd10, axis=1)
    return diagnoses


# =============================================================================
# 3. ICD CODE DESCRIPTIONS (for natural language representation)
# =============================================================================

ICD_DESCRIPTIONS = {
    # Heart conditions
    'I50': 'heart failure',
    'I49': 'cardiac arrhythmia',
    'I48': 'atrial fibrillation',
    'I10': 'hypertension',
    'I25': 'coronary artery disease',
    'I34': 'mitral valve disease',
    'I95': 'hypotension',
    'I51': 'heart disease',
    'I63': 'stroke',
    
    # Respiratory
    'J18': 'pneumonia',
    'J96': 'respiratory failure',
    'J69': 'aspiration pneumonia',
    'J91': 'pleural effusion',
    'J98': 'respiratory disorder',
    
    # Kidney
    'N17': 'acute kidney injury',
    'N18': 'chronic kidney disease',
    'N39': 'urinary tract infection',
    
    # Metabolic
    'E87': 'electrolyte imbalance',
    'D64': 'anemia',
    'E11': 'diabetes',
    'E03': 'hypothyroidism',
    
    # Infectious
    'A41': 'sepsis',
    'A49': 'bacterial infection',
    'R65': 'severe sepsis',
    
    # Neurological
    'F05': 'delirium',
    'G47': 'sleep apnea',
    
    # GI
    'K56': 'bowel obstruction',
    'K63': 'diverticulosis',
    'K92': 'gastrointestinal bleeding',
    'C18': 'colon cancer',
    
    # Musculoskeletal
    'M25': 'joint pain',
    'M81': 'osteoporosis',
    'M62': 'muscle weakness',
    
    # Other
    'R60': 'edema',
    'F41': 'anxiety',
    'Z68': 'body mass index finding',
}


def get_diagnosis_description(icd_code):
    """Get human-readable description for ICD code."""
    code = str(icd_code)
    
    # Try exact prefix match
    for prefix, desc in ICD_DESCRIPTIONS.items():
        if code.startswith(prefix):
            return desc
    
    # Try first 3 characters
    prefix_3 = code[:3]
    for prefix, desc in ICD_DESCRIPTIONS.items():
        if prefix_3.startswith(prefix[:2]):
            return desc
    
    return f"diagnosis code {code}"


# =============================================================================
# 4. FEATURE ENGINEERING AND DATA PREPARATION
# =============================================================================

def prepare_merged_dataset(patients, admissions, diagnoses):
    """
    Merge all tables and create features.
    """
    # Convert datetime columns
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    admissions['deathtime'] = pd.to_datetime(admissions['deathtime'])
    
    # Calculate Length of Stay (LOS) in days
    admissions['los_days'] = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / (24 * 3600)
    
    # Extract temporal features
    admissions['admit_hour'] = admissions['admittime'].dt.hour
    admissions['admit_dayofweek'] = admissions['admittime'].dt.dayofweek
    admissions['admit_month'] = admissions['admittime'].dt.month
    admissions['is_weekend'] = admissions['admit_dayofweek'].isin([5, 6]).astype(int)
    admissions['is_night_admission'] = ((admissions['admit_hour'] >= 22) | (admissions['admit_hour'] <= 6)).astype(int)
    
    # Merge patients with admissions
    merged = admissions.merge(patients, on='subject_id', how='left')
    
    # Harmonize ICD codes
    diagnoses_harmonized = harmonize_icd_codes(diagnoses.copy())
    
    # Create diagnosis features per admission
    # Get top diagnoses (by sequence number) for each admission
    diag_grouped = diagnoses_harmonized.groupby('hadm_id').agg({
        'icd_code_harmonized': lambda x: list(x),
        'seq_num': 'count'
    }).reset_index()
    diag_grouped.columns = ['hadm_id', 'diagnosis_codes', 'num_diagnoses']
    
    # Merge diagnoses
    merged = merged.merge(diag_grouped, on='hadm_id', how='left')
    merged['num_diagnoses'] = merged['num_diagnoses'].fillna(0)
    merged['diagnosis_codes'] = merged['diagnosis_codes'].apply(lambda x: x if isinstance(x, list) else [])
    
    return merged


def create_clinical_text(row):
    """
    Create a natural language clinical summary for each patient admission.
    This text will be input to the LLM for fine-tuning.
    """
    # Demographics
    age = row.get('anchor_age', 'unknown')
    gender = 'female' if row.get('gender', '') == 'F' else 'male' if row.get('gender', '') == 'M' else 'unknown gender'
    race = str(row.get('race', 'unknown')).lower()
    
    # Admission details
    admission_type = str(row.get('admission_type', 'unknown')).lower()
    admission_location = str(row.get('admission_location', 'unknown')).lower()
    
    # Insurance and social factors
    insurance = str(row.get('insurance', 'unknown')).lower()
    marital_status = str(row.get('marital_status', 'unknown')).lower()
    
    # Get diagnoses descriptions
    diag_codes = row.get('diagnosis_codes', [])
    if isinstance(diag_codes, list) and len(diag_codes) > 0:
        # Get unique descriptions for top diagnoses
        descriptions = []
        seen = set()
        for code in diag_codes[:10]:  # Top 10 diagnoses
            desc = get_diagnosis_description(code)
            if desc not in seen:
                descriptions.append(desc)
                seen.add(desc)
        diagnoses_text = ', '.join(descriptions[:7])  # Limit to 7 unique diagnoses
    else:
        diagnoses_text = 'no documented diagnoses'
    
    # Temporal features
    los = row.get('los_days', 0)
    is_weekend = 'weekend' if row.get('is_weekend', 0) == 1 else 'weekday'
    is_night = 'night' if row.get('is_night_admission', 0) == 1 else 'day'
    num_diag = int(row.get('num_diagnoses', 0))
    
    # Construct clinical text
    text = f"""Patient is a {age} year old {gender} of {race} ethnicity. 
Admission type: {admission_type} from {admission_location}. 
Insurance: {insurance}. Marital status: {marital_status}.
Admitted on a {is_weekend} during {is_night} hours.
Number of diagnoses: {num_diag}.
Primary diagnoses include: {diagnoses_text}."""

    return text.strip()


def prepare_llm_dataset(merged_df):
    """
    Prepare dataset for LLM fine-tuning.
    Creates clinical text summaries and prepares labels.
    """
    # Create clinical text for each admission
    merged_df['clinical_text'] = merged_df.apply(create_clinical_text, axis=1)
    
    # Ensure target variables
    merged_df['hospital_expire_flag'] = merged_df['hospital_expire_flag'].fillna(0).astype(int)
    
    # Create features dataframe
    features_df = merged_df[['hadm_id', 'subject_id', 'clinical_text', 'hospital_expire_flag', 
                            'los_days', 'race', 'gender', 'anchor_age', 'num_diagnoses',
                            'admission_type', 'insurance']].copy()
    
    # Remove any rows with missing critical data
    features_df = features_df.dropna(subset=['clinical_text', 'hospital_expire_flag'])
    
    return features_df


# =============================================================================
# 5. CUSTOM DATASET CLASS FOR TRANSFORMERS
# =============================================================================

class MortalityDataset(Dataset):
    """Custom PyTorch dataset for mortality prediction."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LOSDataset(Dataset):
    """Custom PyTorch dataset for Length of Stay prediction (regression)."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


# =============================================================================
# 6. MODEL ARCHITECTURE - TRANSFORMER FOR CLINICAL PREDICTION
# =============================================================================

class ClinicalBERTClassifier(nn.Module):
    """
    Fine-tuned BERT model for mortality prediction.
    Uses a pre-trained clinical/biomedical BERT as base.
    """
    
    def __init__(self, model_name, num_labels=2, dropout=0.3):
        super(ClinicalBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class ClinicalBERTRegressor(nn.Module):
    """
    Fine-tuned BERT model for Length of Stay prediction (regression).
    """
    
    def __init__(self, model_name, dropout=0.3):
        super(ClinicalBERTRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        output = self.regressor(pooled_output)
        return output.squeeze(-1)


# =============================================================================
# 7. TRAINING FUNCTIONS
# =============================================================================

def train_mortality_model(model, train_loader, val_loader, optimizer, scheduler, 
                         num_epochs=5, device='cpu'):
    """Train the mortality prediction model."""
    
    criterion = nn.CrossEntropyLoss()
    best_val_auc = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.5
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def train_los_model(model, train_loader, val_loader, optimizer, scheduler,
                   num_epochs=5, device='cpu'):
    """Train the Length of Stay prediction model."""
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_mae = mean_absolute_error(all_labels, all_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss (MSE): {avg_train_loss:.4f}")
        print(f"  Val Loss (MSE): {avg_val_loss:.4f}, Val MAE: {val_mae:.4f} days")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# =============================================================================
# 8. EVALUATION FUNCTIONS
# =============================================================================

def evaluate_mortality_model(model, test_loader, device='cpu'):
    """Comprehensive evaluation of mortality prediction model."""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    results = {
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }
    
    return results


def evaluate_los_model(model, test_loader, device='cpu'):
    """Comprehensive evaluation of LOS prediction model."""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    results = {
        'mse': mean_squared_error(all_labels, all_preds),
        'rmse': np.sqrt(mean_squared_error(all_labels, all_preds)),
        'mae': mean_absolute_error(all_labels, all_preds),
        'r2': r2_score(all_labels, all_preds),
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return results


def fairness_analysis(df, model, tokenizer, device='cpu', max_length=256):
    """
    Analyze model performance across different ethnic groups.
    """
    # Standardize race categories
    race_mapping = {
        'WHITE': 'White',
        'BLACK/AFRICAN AMERICAN': 'Black',
        'BLACK/CARIBBEAN ISLAND': 'Black',
        'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic',
        'HISPANIC/LATINO - GUATEMALAN': 'Hispanic',
        'HISPANIC/LATINO - CUBAN': 'Hispanic',
        'HISPANIC/LATINO - MEXICAN': 'Hispanic',
        'HISPANIC/LATINO - DOMINICAN': 'Hispanic',
        'HISPANIC/LATINO - SALVADORAN': 'Hispanic',
        'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic',
        'HISPANIC OR LATINO': 'Hispanic',
        'ASIAN': 'Asian',
        'ASIAN - CHINESE': 'Asian',
        'ASIAN - ASIAN INDIAN': 'Asian',
        'ASIAN - VIETNAMESE': 'Asian',
        'ASIAN - KOREAN': 'Asian',
        'OTHER': 'Other',
        'UNKNOWN': 'Unknown',
        'UNABLE TO OBTAIN': 'Unknown',
        'PATIENT DECLINED TO ANSWER': 'Unknown',
    }
    
    df = df.copy()
    df['race_group'] = df['race'].map(lambda x: race_mapping.get(str(x).upper(), 'Other'))
    
    fairness_results = {}
    
    for race_group in df['race_group'].unique():
        if pd.isna(race_group) or race_group == 'Unknown':
            continue
            
        race_df = df[df['race_group'] == race_group]
        
        if len(race_df) < 50:  # Skip groups with too few samples
            continue
        
        # Create dataset for this group
        dataset = MortalityDataset(
            texts=race_df['clinical_text'].tolist(),
            labels=race_df['hospital_expire_flag'].tolist(),
            tokenizer=tokenizer,
            max_length=max_length
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Get predictions
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = [1 if p > 0.5 else 0 for p in all_probs]
        
        # Calculate metrics
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = np.nan
        
        fairness_results[race_group] = {
            'n_samples': len(race_df),
            'mortality_rate': np.mean(all_labels),
            'auc_roc': auc,
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, zero_division=0)
        }
    
    return pd.DataFrame(fairness_results).T


# =============================================================================
# 9. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_training_history(history, title="Training History"):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metric plot (AUC for classification, MAE for regression)
    if 'val_auc' in history:
        axes[1].plot(history['val_auc'], label='Val AUC-ROC', marker='o', color='green')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_title(f'{title} - AUC-ROC')
    elif 'val_mae' in history:
        axes[1].plot(history['val_mae'], label='Val MAE', marker='o', color='orange')
        axes[1].set_ylabel('MAE (days)')
        axes[1].set_title(f'{title} - Mean Absolute Error')
    
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(labels, predictions, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Survived', 'Expired'],
                yticklabels=['Survived', 'Expired'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_fairness_analysis(fairness_df):
    """Plot fairness analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['auc_roc', 'precision', 'recall', 'f1_score']
    titles = ['AUC-ROC by Race', 'Precision by Race', 'Recall by Race', 'F1-Score by Race']
    
    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        data = fairness_df[metric].dropna()
        bars = ax.bar(range(len(data)), data.values, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.axhline(y=data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, data.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fairness_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_los_predictions(labels, predictions):
    """Plot Length of Stay prediction results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(labels, predictions, alpha=0.5, s=20)
    max_val = max(max(labels), max(predictions))
    axes[0].plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    axes[0].set_xlabel('Actual LOS (days)')
    axes[0].set_ylabel('Predicted LOS (days)')
    axes[0].set_title('Predicted vs Actual Length of Stay')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    residuals = np.array(predictions) - np.array(labels)
    axes[1].hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', label='Zero Error')
    axes[1].set_xlabel('Prediction Error (days)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Prediction Errors')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('los_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("=" * 70)
    print("FINE-TUNED LLM FOR IN-HOSPITAL MORTALITY PREDICTION")
    print("=" * 70)
    
    # ---------------------------
    # STEP 1: Load and prepare data
    # ---------------------------
    print("\n[1/8] Loading data...")
    patients, admissions, diagnoses = load_data('.')
    
    print("\n[2/8] Preparing merged dataset...")
    merged_df = prepare_merged_dataset(patients, admissions, diagnoses)
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Mortality rate: {merged_df['hospital_expire_flag'].mean()*100:.2f}%")
    print(f"Average LOS: {merged_df['los_days'].mean():.2f} days")
    
    print("\n[3/8] Creating clinical text representations...")
    llm_df = prepare_llm_dataset(merged_df)
    print(f"LLM dataset shape: {llm_df.shape}")
    
    # Sample clinical text
    print("\nSample clinical text:")
    print("-" * 50)
    print(llm_df['clinical_text'].iloc[0])
    print("-" * 50)
    
    # ---------------------------
    # STEP 2: Initialize tokenizer and model
    # ---------------------------
    print("\n[4/8] Initializing model and tokenizer...")
    
    # Use BioBERT or ClinicalBERT (fallback to DistilBERT if not available)
    try:
        MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Loaded: {MODEL_NAME}")
    except:
        try:
            MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            print(f"Loaded: {MODEL_NAME}")
        except:
            MODEL_NAME = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            print(f"Loaded fallback: {MODEL_NAME}")
    
    # ---------------------------
    # STEP 3: Prepare train/val/test splits
    # ---------------------------
    print("\n[5/8] Preparing data splits...")
    
    # Stratified split for mortality prediction
    train_df, temp_df = train_test_split(
        llm_df, test_size=0.3, random_state=SEED,
        stratify=llm_df['hospital_expire_flag']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED,
        stratify=temp_df['hospital_expire_flag']
    )
    
    print(f"Train: {len(train_df)} samples (mortality rate: {train_df['hospital_expire_flag'].mean()*100:.2f}%)")
    print(f"Val: {len(val_df)} samples (mortality rate: {val_df['hospital_expire_flag'].mean()*100:.2f}%)")
    print(f"Test: {len(test_df)} samples (mortality rate: {test_df['hospital_expire_flag'].mean()*100:.2f}%)")
    
    # Create datasets
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    
    train_dataset = MortalityDataset(
        texts=train_df['clinical_text'].tolist(),
        labels=train_df['hospital_expire_flag'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    val_dataset = MortalityDataset(
        texts=val_df['clinical_text'].tolist(),
        labels=val_df['hospital_expire_flag'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    test_dataset = MortalityDataset(
        texts=test_df['clinical_text'].tolist(),
        labels=test_df['hospital_expire_flag'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ---------------------------
    # STEP 4: Train Mortality Model
    # ---------------------------
    print("\n[6/8] Training mortality prediction model...")
    print(f"Using device: {device}")
    
    mortality_model = ClinicalBERTClassifier(MODEL_NAME, num_labels=2, dropout=0.3).to(device)
    
    # Optimizer and scheduler
    NUM_EPOCHS = 5
    optimizer = AdamW(mortality_model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    mortality_model, mortality_history = train_mortality_model(
        mortality_model, train_loader, val_loader, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, device=device
    )
    
    # ---------------------------
    # STEP 5: Evaluate Mortality Model
    # ---------------------------
    print("\n[7/8] Evaluating mortality prediction model...")
    
    mortality_results = evaluate_mortality_model(mortality_model, test_loader, device)
    
    print("\n" + "=" * 50)
    print("MORTALITY PREDICTION RESULTS (Test Set)")
    print("=" * 50)
    print(f"AUC-ROC:   {mortality_results['auc_roc']:.4f}")
    print(f"Precision: {mortality_results['precision']:.4f}")
    print(f"Recall:    {mortality_results['recall']:.4f}")
    print(f"F1-Score:  {mortality_results['f1_score']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        mortality_results['labels'], 
        mortality_results['predictions'],
        target_names=['Survived', 'Expired']
    ))
    
    # ---------------------------
    # STEP 6: Train and Evaluate LOS Model
    # ---------------------------
    print("\n[8/8] Training Length of Stay prediction model...")
    
    # Create LOS datasets
    train_los_dataset = LOSDataset(
        texts=train_df['clinical_text'].tolist(),
        labels=train_df['los_days'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    val_los_dataset = LOSDataset(
        texts=val_df['clinical_text'].tolist(),
        labels=val_df['los_days'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    test_los_dataset = LOSDataset(
        texts=test_df['clinical_text'].tolist(),
        labels=test_df['los_days'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    train_los_loader = DataLoader(train_los_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_los_loader = DataLoader(val_los_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_los_loader = DataLoader(test_los_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    los_model = ClinicalBERTRegressor(MODEL_NAME, dropout=0.3).to(device)
    
    optimizer_los = AdamW(los_model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler_los = get_linear_schedule_with_warmup(
        optimizer_los, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    los_model, los_history = train_los_model(
        los_model, train_los_loader, val_los_loader, optimizer_los, scheduler_los,
        num_epochs=NUM_EPOCHS, device=device
    )
    
    # Evaluate LOS model
    los_results = evaluate_los_model(los_model, test_los_loader, device)
    
    print("\n" + "=" * 50)
    print("LENGTH OF STAY PREDICTION RESULTS (Test Set)")
    print("=" * 50)
    print(f"MSE:  {los_results['mse']:.4f}")
    print(f"RMSE: {los_results['rmse']:.4f} days")
    print(f"MAE:  {los_results['mae']:.4f} days")
    print(f"R²:   {los_results['r2']:.4f}")
    
    # ---------------------------
    # STEP 7: Fairness Analysis
    # ---------------------------
    print("\n" + "=" * 50)
    print("FAIRNESS ANALYSIS BY ETHNIC GROUP")
    print("=" * 50)
    
    fairness_df = fairness_analysis(test_df, mortality_model, tokenizer, device, MAX_LENGTH)
    print("\nPerformance metrics by race/ethnicity:")
    print(fairness_df.round(4).to_string())
    
    # Calculate fairness metrics
    auc_values = fairness_df['auc_roc'].dropna()
    if len(auc_values) > 1:
        auc_disparity = auc_values.max() - auc_values.min()
        print(f"\nAUC-ROC Disparity (max - min): {auc_disparity:.4f}")
        print(f"This indicates {'significant' if auc_disparity > 0.1 else 'minimal'} performance variation across groups.")
    
    # ---------------------------
    # STEP 8: Generate visualizations
    # ---------------------------
    print("\nGenerating visualizations...")
    
    try:
        plot_training_history(mortality_history, "Mortality Model Training")
        plot_confusion_matrix(mortality_results['labels'], mortality_results['predictions'],
                            "Mortality Prediction - Confusion Matrix")
        plot_los_predictions(los_results['labels'], los_results['predictions'])
        if len(fairness_df) > 0:
            plot_fairness_analysis(fairness_df)
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # ---------------------------
    # Summary
    # ---------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Model Architecture: Fine-tuned {MODEL_NAME}
Task 1 - Mortality Prediction:
    - AUC-ROC: {mortality_results['auc_roc']:.4f}
    - F1-Score: {mortality_results['f1_score']:.4f}
    
Task 2 - Length of Stay Prediction:
    - RMSE: {los_results['rmse']:.4f} days
    - MAE: {los_results['mae']:.4f} days
    - R²: {los_results['r2']:.4f}
    
Task 3 - Fairness Analysis:
    - Analyzed {len(fairness_df)} ethnic groups
    - Performance variation: {'Significant' if auc_disparity > 0.1 else 'Minimal'} across groups
""")
    
    return {
        'mortality_model': mortality_model,
        'los_model': los_model,
        'mortality_results': mortality_results,
        'los_results': los_results,
        'fairness_results': fairness_df,
        'history': {'mortality': mortality_history, 'los': los_history}
    }


if __name__ == "__main__":
    results = main()
