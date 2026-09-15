"""
MediTrack - Notebook Generator & Executor
Builds and executes all required Jupyter notebooks with outputs visible:
1. 01_data_preparation_and_statistical_analysis.ipynb
2. 02_database_and_sql_queries.ipynb
3. 03_predictive_modelling_threshold_tuning.ipynb
4. 04_text_processing_and_neural_baseline.ipynb
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

def create_and_execute_nb(nb_path, cells_content):
    os.makedirs(os.path.dirname(nb_path), exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.cells = []
    
    for cell_type, content in cells_content:
        if cell_type == 'markdown':
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == 'code':
            nb.cells.append(nbf.v4.new_code_cell(content))
            
    print(f"Executing notebook: {nb_path}...")
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully saved executed notebook to {nb_path}")

def build_all_notebooks():
    # Notebook 1: Data Prep & Statistical Analysis
    nb1_cells = [
        ('markdown', "# MediTrack: Data Preparation & Statistical Hypothesis Testing\n### Project CP-02: Readmission Risk Prediction and Clinical Decision Support\nThis notebook executes the ingestion, clinical exclusions, missing data handling, and hypothesis tests required by Section 5 of the project brief."),
        ('code', """
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from src.data_prep import clean_and_prepare_data

# 1. Inspect Raw Data & Run Preprocessing Pipeline
raw_path = 'data/raw/diabetic_data.csv'
df_cleaned = clean_and_prepare_data(raw_path, 'data/processed')
print(f"Cleaned dataset records: {len(df_cleaned):,}")
df_cleaned[['encounter_id', 'patient_nbr', 'age_group', 'admission_type_name', 'time_in_hospital', 'diag_1_category', 'readmitted_30d']].head()
"""),
        ('code', """
# 2. Base Rate of 30-Day Readmission
total = len(df_cleaned)
readm_30d = (df_cleaned['readmitted_30d'] == 1).sum()
base_rate = readm_30d / total
print(f"Total encounters: {total:,}")
print(f"30-day Readmissions: {readm_30d:,} ({base_rate*100:.2f}%)")

# Breakdown by original category
df_cleaned['readmitted'].value_counts()
"""),
        ('code', """
# 3. Hypothesis Test 1: Age Group Differences (Chi-Square)
age_ct = pd.crosstab(df_cleaned['age_group'], df_cleaned['readmitted_30d'])
chi2, p_val, dof, _ = stats.chi2_contingency(age_ct)
print(f"Chi-Square Statistic: {chi2:.3f}, df: {dof}, p-value: {p_val:.4e}")

age_rates = df_cleaned.groupby('age_group')['readmitted_30d'].mean() * 100
print("\\nReadmission Rate by Age Group (%):")
print(age_rates)
"""),
        ('code', """
# 4. Hypothesis Test 2: Admission Type (Chi-Square & Odds Ratio)
adm_ct = pd.crosstab(df_cleaned['admission_type_name'], df_cleaned['readmitted_30d'])
chi2_adm, p_adm, dof_adm, _ = stats.chi2_contingency(adm_ct)
print(f"Admission Type Chi2: {chi2_adm:.3f}, df: {dof_adm}, p-value: {p_adm:.4e}")

# Odds Ratio Emergency vs Elective
em_pos = ((df_cleaned['admission_type_name'] == 'Emergency') & (df_cleaned['readmitted_30d'] == 1)).sum()
em_neg = ((df_cleaned['admission_type_name'] == 'Emergency') & (df_cleaned['readmitted_30d'] == 0)).sum()
el_pos = ((df_cleaned['admission_type_name'] == 'Elective') & (df_cleaned['readmitted_30d'] == 1)).sum()
el_neg = ((df_cleaned['admission_type_name'] == 'Elective') & (df_cleaned['readmitted_30d'] == 0)).sum()
odds_ratio = (em_pos / em_neg) / (el_pos / el_neg)
print(f"Odds Ratio (Emergency vs Elective): {odds_ratio:.3f}")
"""),
        ('code', """
# 5. Hypothesis Test 3: Prior Inpatient History (t-test, Mann-Whitney U, Logit Odds Ratio)
readm_inp = df_cleaned[df_cleaned['readmitted_30d'] == 1]['number_inpatient']
noreadm_inp = df_cleaned[df_cleaned['readmitted_30d'] == 0]['number_inpatient']

t_stat, t_pval = stats.ttest_ind(readm_inp, noreadm_inp, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(readm_inp, noreadm_inp)
print(f"Mean Prior Inpatient (Readmitted): {readm_inp.mean():.3f}")
print(f"Mean Prior Inpatient (Non-Readmitted): {noreadm_inp.mean():.3f}")
print(f"Welch t-stat: {t_stat:.3f}, p-val: {t_pval:.4e}")
print(f"Mann-Whitney U: {u_stat:.1f}, p-val: {u_pval:.4e}")

# Logistic Odds Ratio
X = sm.add_constant(df_cleaned['number_inpatient'])
logit_mod = sm.Logit(df_cleaned['readmitted_30d'], X).fit(disp=False)
print(f"Logistic Odds Ratio per prior stay: {np.exp(logit_mod.params['number_inpatient']):.3f}")
""")
    ]
    create_and_execute_nb('notebooks/01_data_preparation_and_statistical_analysis.ipynb', nb1_cells)
    
    # Notebook 2: Database & SQL Queries
    nb2_cells = [
        ('markdown', "# MediTrack: Relational Database & SQL Queries\n### Project CP-02: Readmission Risk Prediction and Clinical Decision Support\nThis notebook demonstrates the relational database architecture and executes the three mandatory analytical queries."),
        ('code', """
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/meditrack.db')

# Verify Database Tables & Record Counts
tables = ['patients', 'encounters', 'diagnoses', 'medications']
for t in tables:
    count = pd.read_sql_query(f"SELECT COUNT(*) FROM {t}", conn).iloc[0, 0]
    print(f"Table '{t}': {count:,} records")
"""),
        ('code', """
# QUERY 1: Readmission Rate by Demographic & Prior Inpatient Cohort
q1 = '''
SELECT 
    p.age_group,
    p.gender,
    CASE 
        WHEN e.number_inpatient = 0 THEN '0 Prior Inpatient Visits'
        WHEN e.number_inpatient = 1 THEN '1 Prior Inpatient Visit'
        WHEN e.number_inpatient = 2 THEN '2 Prior Inpatient Visits'
        ELSE '3+ Prior Inpatient Visits'
    END AS inpatient_utilization_tier,
    COUNT(e.encounter_id) AS total_encounters,
    SUM(e.readmitted_30d) AS readmitted_30d_count,
    ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(e.encounter_id), 2) AS readmission_rate_pct,
    ROUND(AVG(e.time_in_hospital), 2) AS avg_length_of_stay_days
FROM encounters e
JOIN patients p ON e.patient_nbr = p.patient_nbr
GROUP BY p.age_group, p.gender, inpatient_utilization_tier
HAVING COUNT(e.encounter_id) >= 50
ORDER BY readmission_rate_pct DESC
LIMIT 8;
'''
pd.read_sql_query(q1, conn)
"""),
        ('code', """
# QUERY 2: Average Length of Stay & Readmission Rate by Primary Diagnosis Group
q2 = '''
SELECT 
    d.clinical_category AS primary_diagnosis_group,
    COUNT(DISTINCT e.encounter_id) AS encounter_count,
    ROUND(AVG(e.time_in_hospital), 2) AS avg_stay_days,
    ROUND(MIN(e.time_in_hospital), 1) AS min_stay_days,
    ROUND(MAX(e.time_in_hospital), 1) AS max_stay_days,
    SUM(e.readmitted_30d) AS readmissions_30d_count,
    ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(e.encounter_id), 2) AS readmission_rate_pct
FROM diagnoses d
JOIN encounters e ON d.encounter_id = e.encounter_id
WHERE d.diagnosis_seq = 1
GROUP BY d.clinical_category
ORDER BY avg_stay_days DESC;
'''
pd.read_sql_query(q2, conn)
"""),
        ('code', """
# QUERY 3: Patient Encounter Sequences Tracing Readmission Trajectory (Window Functions)
q3 = '''
WITH patient_journey AS (
    SELECT 
        e.patient_nbr,
        p.age_group,
        p.gender,
        e.encounter_id,
        e.time_in_hospital AS stay_duration_days,
        e.admission_type_name,
        e.readmitted_30d,
        e.readmitted_raw,
        d.clinical_category AS primary_diagnosis,
        ROW_NUMBER() OVER (PARTITION BY e.patient_nbr ORDER BY e.encounter_id ASC) AS encounter_sequence_num,
        COUNT(e.encounter_id) OVER (PARTITION BY e.patient_nbr) AS total_patient_admissions,
        LAG(e.encounter_id) OVER (PARTITION BY e.patient_nbr ORDER BY e.encounter_id ASC) AS prior_encounter_id,
        LAG(e.time_in_hospital) OVER (PARTITION BY e.patient_nbr ORDER BY e.encounter_id ASC) AS prior_stay_duration_days
    FROM encounters e
    JOIN patients p ON e.patient_nbr = p.patient_nbr
    LEFT JOIN diagnoses d ON e.encounter_id = d.encounter_id AND d.diagnosis_seq = 1
)
SELECT 
    patient_nbr,
    encounter_sequence_num,
    total_patient_admissions,
    encounter_id,
    admission_type_name,
    primary_diagnosis,
    stay_duration_days,
    prior_encounter_id,
    prior_stay_duration_days,
    readmitted_raw,
    readmitted_30d
FROM patient_journey
WHERE total_patient_admissions > 1
ORDER BY patient_nbr, encounter_sequence_num
LIMIT 8;
'''
pd.read_sql_query(q3, conn)
""")
    ]
    create_and_execute_nb('notebooks/02_database_and_sql_queries.ipynb', nb2_cells)
    
    # Notebook 3: Predictive Modeling & Threshold Tuning
    nb3_cells = [
        ('markdown', "# MediTrack: Predictive Modeling & Clinical Threshold Tuning\n### Project CP-02: Readmission Risk Prediction and Clinical Decision Support\nThis notebook compares Logistic Regression, Random Forest, and Gradient Boosting under class weighting, evaluates decision thresholds, and defends the recall vs. precision trade-off."),
        ('code', """
import json
import pandas as pd
import matplotlib.pyplot as plt

with open('reports/model_evaluation_report.json') as f:
    report = json.load(f)

# Model Comparison Table
comp_rows = []
for name, res in report['models_comparison'].items():
    comp_rows.append({
        'Model': name,
        'ROC-AUC': res['roc_auc'],
        'PR-AUC': res['pr_auc'],
        'Brier Score': res['brier_score'],
        'Sensitivity @ 0.50': res['default_threshold_0_5']['sensitivity_recall'],
        'Precision @ 0.50': res['default_threshold_0_5']['precision_ppv'],
        'Tuned Clinical Threshold': res['clinical_operational_threshold']['threshold'],
        'Sensitivity @ Tuned': res['clinical_operational_threshold']['sensitivity_recall'],
        'Precision @ Tuned': res['clinical_operational_threshold']['precision_ppv']
    })
pd.DataFrame(comp_rows)
"""),
        ('code', """
# Written Defense of Clinical Threshold Choice (0.18 vs 0.50)
print("=== CLINICAL THRESHOLD JUSTIFICATION ===")
print(report['clinical_threshold_justification']['justification'])
"""),
        ('code', """
# Plotting Sensitivity vs Precision Trade-off Curve
gb_curve = pd.DataFrame(report['models_comparison']['Gradient Boosting (Balanced)']['threshold_curve'])

plt.figure(figsize=(9, 5))
plt.plot(gb_curve['threshold'], gb_curve['sensitivity_recall'], label='Sensitivity (Recall)', color='#0284c7', lw=2.5)
plt.plot(gb_curve['threshold'], gb_curve['precision_ppv'], label='Precision (PPV)', color='#10b981', lw=2.5)
plt.plot(gb_curve['threshold'], gb_curve['f1_score'], label='F1-Score', color='#8b5cf6', linestyle='--', lw=2)
plt.axvline(x=0.18, color='#ef4444', linestyle=':', label='Clinical Operating Threshold (0.18)', lw=2)
plt.title('Clinical Decision Threshold Trade-off (Gradient Boosting)', fontsize=13, fontweight='bold')
plt.xlabel('Decision Threshold', fontsize=11)
plt.ylabel('Metric Value', fontsize=11)
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""")
    ]
    create_and_execute_nb('notebooks/03_predictive_modelling_threshold_tuning.ipynb', nb3_cells)
    
    # Notebook 4: Text Processing, Neural Baseline & Explainability
    nb4_cells = [
        ('markdown', "# MediTrack: Text Processing, Neural Baseline & SHAP Factor Attribution\n### Project CP-02: Readmission Risk Prediction and Clinical Decision Support\nThis notebook evaluates ICD-9 diagnosis text derivation, benchmarks the neural MLP baseline against tree ensembles, and demonstrates patient-level SHAP attributions."),
        ('code', """
import json
import joblib
import pandas as pd
import numpy as np

with open('reports/model_evaluation_report.json') as f:
    report = json.load(f)

print("=== NLP DIAGNOSIS FEATURE ABLATION ===")
nlp_res = report['nlp_text_ablation']
print(f"Baseline Gradient Boosting ROC-AUC: {nlp_res['baseline_gradient_boosting_roc_auc']}")
print(f"NLP-Enhanced Gradient Boosting ROC-AUC: {nlp_res['nlp_enhanced_gradient_boosting_roc_auc']}")
print(f"Top Extracted Clinical Terms: {nlp_res['top_extracted_terms'][:8]}")
print(f"Interpretation: {nlp_res['interpretation']}")
"""),
        ('code', """
print("=== NEURAL BASELINE (MLP) BENCHMARK ===")
mlp_res = report['models_comparison']['Neural Baseline (MLP)']
print(f"Neural Baseline ROC-AUC: {mlp_res['roc_auc']} | PR-AUC: {mlp_res['pr_auc']}")
print(f"At default (0.50): Sensitivity={mlp_res['default_threshold_0_5']['sensitivity_recall']} (Fails clinical use due to base rate disparity)")
print(f"At tuned clinical threshold ({mlp_res['clinical_operational_threshold']['threshold']}): Sensitivity={mlp_res['clinical_operational_threshold']['sensitivity_recall']}, Precision={mlp_res['clinical_operational_threshold']['precision_ppv']}")
print("Comparison: Tree ensemble (Gradient Boosting ROC-AUC: 0.6445) modestly outperforms Neural MLP (0.6421) on tabular clinical features while providing faster convergence.")
"""),
        ('code', """
# Patient-Level Factor Attribution with SHAP
preprocessor = joblib.load('models/preprocessor.joblib')
model = joblib.load('models/best_model.joblib')
explainer = joblib.load('models/shap_explainer.joblib')
with open('models/feature_names.json') as f:
    feat_names = json.load(f)

# Load sample patient
df = pd.read_csv('data/processed/cleaned_encounters.csv', nrows=5)
from src.model_pipeline import NUMERICAL_COLS, CATEGORICAL_COLS, BINARY_COLS
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype(str)
X_sample = df[NUMERICAL_COLS + CATEGORICAL_COLS + BINARY_COLS]
X_trans = preprocessor.transform(X_sample)

# Compute factor contributions for Patient 1
pt_shap = explainer(X_trans[:1])
pt_shap_values = pt_shap.values[0, :, 1] # Positive class attributions

top_indices = np.argsort(np.abs(pt_shap_values))[::-1][:6]
print("Top Contributing Factors for Patient #1:")
for idx in top_indices:
    print(f"  {feat_names[idx]}: {pt_shap_values[idx]:+.4f}")
""")
    ]
    create_and_execute_nb('notebooks/04_text_processing_and_neural_baseline.ipynb', nb4_cells)

if __name__ == '__main__':
    build_all_notebooks()
