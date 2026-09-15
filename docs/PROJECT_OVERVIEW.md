# MediTrack — Complete Project Guide & System Architecture

> **Audience Note**: This document is written for everyone — whether you are a developer, clinician, data scientist, or someone with zero prior knowledge of this codebase. It breaks down the real-world healthcare problem, explains every moving part in simple terms, details the technical architecture, and provides a comprehensive breakdown of all technologies and algorithms used.

---

## 📑 Table of Contents
1. [What is MediTrack? (The Big Picture in Plain English)](#1-what-is-meditrack-the-big-picture-in-plain-english)
2. [The Real-World Healthcare Problem](#2-the-real-world-healthcare-problem)
3. [How the Project Works: High-Level Architecture](#3-how-the-project-works-high-level-architecture)
4. [What is Going On: Step-by-Step System Breakdown](#4-what-is-going-on-step-by-step-system-breakdown)
   - [Step 1: Data Ingestion, Cleaning & Feature Extraction](#step-1-data-ingestion-cleaning--feature-extraction)
   - [Step 2: Relational Database & Analytical SQL Layer](#step-2-relational-database--analytical-sql-layer)
   - [Step 3: Machine Learning & Clinical Threshold Optimization](#step-3-machine-learning--clinical-threshold-optimization)
   - [Step 4: Explainable AI (SHAP Factor Attribution)](#step-4-explainable-ai-shap-factor-attribution)
   - [Step 5: Clinical Decision Support Assistant & Safety Guardrails](#step-5-clinical-decision-support-assistant--safety-guardrails)
   - [Step 6: Clinical Triage Dashboard Web Application](#step-6-clinical-triage-dashboard-web-application)
   - [Step 7: Advanced Clinical Extensions & Quality Audits](#step-7-advanced-clinical-extensions--quality-audits)
   - [Step 8: Automated PDF Reporting & Executive Slide Decks](#step-8-automated-pdf-reporting--executive-slide-decks)
5. [What Things Are Used in It (Technology Stack & Components)](#5-what-things-are-used-in-it-technology-stack--components)
6. [Folder & File Directory Map](#6-folder--file-directory-map)
7. [How to Run, Test, and Use the Project](#7-how-to-run-test-and-use-the-project)
8. [Frequently Asked Questions (FAQ)](#8-frequently-asked-questions-faq)

---

## 1. What is MediTrack? (The Big Picture in Plain English)

Imagine a diabetic patient who has been hospitalized for severe blood sugar fluctuations or cardiovascular complications. After spending five days receiving continuous care, they feel better and are discharged to go home. 

However, within two weeks, the patient may struggle with their new insulin doses, miss a follow-up appointment, experience a dangerous drop in blood sugar (hypoglycemia), or develop fluid buildup. They end up rushed back to the Emergency Department and readmitted to the hospital.

**MediTrack** is an intelligent **Clinical Decision Support System (CDSS)** that intervenes *before* the patient ever leaves the hospital. 

At the exact moment of discharge:
1. **It analyzes the patient’s medical chart**: It looks at their age, length of stay, past hospital visits, lab results (like HbA1c and glucose), primary and secondary diagnoses, and all changes made to their diabetes medications.
2. **It computes a 30-Day Readmission Risk Probability**: Using a trained machine learning model, it estimates the exact probability that this patient will bounce back to the hospital within 30 days.
3. **It explains WHY (Explainable AI)**: Instead of being an unhelpful "black box", it uses SHAP algorithms to show the doctor or discharge nurse the top factors driving the risk (e.g., *"Patient has 2 prior inpatient visits this year (+8.5% risk)"* or *"Insulin dosage was altered (+4.2% risk)"*).
4. **It provides Evidence-Based Discharge Care Protocols**: It retrieves published guidelines from the **American Diabetes Association (ADA 2024)**, **AHRQ Project RED**, and **CMS Transitional Care Management** to give nurses a tailored discharge checklist (e.g., *"Schedule follow-up within 7 days, provide hypoglycemia 15/15 rule training, call patient within 48 hours"*).
5. **It enforces Clinical Safety Guardrails**: If someone asks the system to prescribe drugs or diagnose a new disease, it immediately refuses and defers to licensed doctors.

---

## 2. The Real-World Healthcare Problem

### Why 30-Day Readmissions Matter
Hospital readmissions within 30 days of discharge are considered a primary indicator of healthcare quality and patient safety:
- **Patient Harm**: Unexpected readmissions represent acute clinical deterioration, medication errors, infection, or lack of post-discharge coordination.
- **Financial Penalties (CMS HRRP)**: Under the U.S. Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions Reduction Program, hospitals with excessive 30-day readmissions face penalty reductions of up to **3% across all their Medicare reimbursements**.
- **The Base Rate Dilemma (Class Imbalance)**: In real hospitals, around **11.4%** of patients get readmitted within 30 days. Because 88.6% do not, standard off-the-shelf AI models default to predicting "No readmission" for everyone (achieving 88.6% accuracy while catching 0% of sick patients!). MediTrack solves this through cost-sensitive threshold tuning and probability calibration.

---

## 3. How the Project Works: High-Level Architecture

The diagram below illustrates the end-to-end data and decision pipeline:

```
+------------------------------------------------------------------------------------+
|                               1. DATA SOURCE                                       |
|  10-Year Clinical Dataset (130 US Hospitals, 100,000+ Diabetic Inpatient Records)  |
+------------------------------------------------------------------------------------+
                                           │
                                           ▼
+------------------------------------------------------------------------------------+
|                         2. ETL, CLEANING & PREPARATION                             |
|  • Exclude Deceased / Hospice Discharges (Dispositions 11, 13, 14, 19, 20, 21)     |
|  • Map 700+ ICD-9 Codes to 9 Clinical Categories (Circulatory, Diabetes, etc.)    |
|  • Track 23 Specific Diabetes Medications for Active Status & Dosage Changes       |
|  • Clean Demographics, Lab Flags (HbA1c, Glucose), and Prior Utilization History   |
+------------------------------------------------------------------------------------+
                      │                                            │
                      ▼                                            ▼
+------------------------------------+   +-------------------------------------------+
|    3. RELATIONAL DATABASE (SQL)    |   |     4. MACHINE LEARNING ENGINE            |
|  • SQLite (`meditrack.db`)         |   |  • Preprocessor (ColumnTransformer)       |
|  • 4 Tables: patients, encounters, |   |  • HistGradientBoostingClassifier         |
|    diagnoses, medications          |   |  • Isotonic Probability Calibration       |
|  • Indexed Clinical Views &        |   |  • Tuned Operating Threshold: 0.18        |
|    Analytical Window Queries       |   |  • SHAP TreeExplainer Factor Attribution  |
+------------------------------------+   +-------------------------------------------+
                      │                                            │
                      └────────────────────┬───────────────────────┘
                                           ▼
+------------------------------------------------------------------------------------+
|                               5. FASTAPI BACKEND SERVICE                           |
|  • REST Endpoints for Scoring, Cohort Searching, Triage Queue, and Analytics       |
|  • Safety Guardrail Filter (Blocks Prescriptive Orders & Diagnostic Inferences)    |
|  • Guideline Retrieval Engine (ADA 2024, AHRQ RED, CMS TCM, ACC/AHA, KDIGO)        |
+------------------------------------------------------------------------------------+
                                           │
                                           ▼
+------------------------------------------------------------------------------------+
|                         6. INTERACTIVE CLINICAL DASHBOARD                          |
|  • Single-Page Glassmorphic UI (HTML5, Vanilla CSS, Responsive JS)                |
|  • 6 Workspaces: Patient Inspector, Clinical Assistant, Triage Worklist,          |
|    Quality Dashboard, Health Economics Simulator, PDF/Slide Downloads              |
+------------------------------------------------------------------------------------+
                                           │
                                           ▼
+------------------------------------------------------------------------------------+
|                         7. REPORTS, AUDITS & EXTENSIONS                            |
|  • Kaplan-Meier Survival Analysis (3.2x Readmission Hazard in High-Risk Tier)      |
|  • Algorithmic Fairness Audit (EEOC 80% Rule across Race, Gender, Age)            |
|  • Population Stability Drift Monitoring (PSI < 0.05, Covariate Stability)         |
|  • Financial ROI Simulator (Net Savings Calculator per 10k Hospital Discharges)   |
|  • Automated Clinical PDF Report & 10-Slide Presentation Deck Generator           |
+------------------------------------------------------------------------------------+
```

---

## 4. What is Going On: Step-by-Step System Breakdown

### Step 1: Data Ingestion, Cleaning & Feature Extraction
*Source file: `src/data_prep.py`*
- **Raw Input**: 101,766 inpatient encounters from 130 hospitals between 1999 and 2008.
- **Clinical Exclusion**: 2,426 records where the patient expired in the hospital or was discharged to hospice are removed (`discharge_disposition_id` in 11, 13, 14, 19, 20, 21). You cannot readmit a deceased or hospice patient, so keeping them skews the clinical statistics.
- **Target Definition**: Binary indicator `readmitted_30d` (1 = readmitted in `< 30` days, 0 = readmitted `> 30` days or `NO`).
- **ICD-9 High-Cardinality Reduction**: 700+ individual numeric diagnosis codes are categorized into 9 clinical categories using standard healthcare informatics mappings:
  1. *Circulatory* (Heart failure, hypertension, myocardial infarction)
  2. *Respiratory* (Pneumonia, COPD, asthma)
  3. *Digestive* (GI bleed, gastroenteritis)
  4. *Diabetes* (Ketoacidosis, uncontrolled hyperglycemia)
  5. *Injury* (Trauma, fractures, poisoning)
  6. *Musculoskeletal* (Arthritis, spinal disorders)
  7. *Genitourinary* (Chronic kidney disease, UTI)
  8. *Neoplasms* (Malignant and benign tumors)
  9. *Other*
- **Medication Trajectory Tracking**: 23 individual diabetes drugs (metformin, glipizide, glyburide, pioglitazone, insulin, etc.) are inspected to create features:
  - `num_active_diabetes_meds`: Total active diabetes medications.
  - `num_diabetes_med_changes`: Count of medications titrated `Up` or `Down`.
  - `has_medication_change`: 1 if any medication was adjusted during the stay.
  - `on_insulin`: 1 if the patient is on active insulin therapy.
  - `polypharmacy`: 1 if the patient is prescribed 15 or more total medications.

---

### Step 2: Relational Database & Analytical SQL Layer
*Source files: `sql/schema.sql`, `sql/queries.sql`, `src/db_setup.py`*
- The cleaned clinical dataset is normalized into a relational SQLite database (`data/meditrack.db`) with 4 entities:
  1. `patients`: Stores master patient identifiers, demographic details (race, gender, age bracket), and lifetime hospital encounter counts.
  2. `encounters`: Stores individual hospital stays (admission type, stay duration, department, lab volume, prior ER and inpatient visits, readmission outcome).
  3. `diagnoses`: Normalized 1-to-many relationship tracking primary (seq 1), secondary (seq 2), and tertiary (seq 3) ICD-9 codes and descriptive text.
  4. `medications`: Normalized 1-to-many relationship tracking the dosage status (`Steady`, `Up`, `Down`, `No`) for all 23 diabetes medications.
- **Advanced SQL Queries**:
  - *Cohort Aggregation*: Calculates readmission rates across multivariate demographic slices.
  - *Department Resource Utilization*: Benchmarks average length of stay and readmission frequencies across hospital specialties.
  - *Window Functions (`ROW_NUMBER`, `LAG`)*: Traces longitudinal patient journeys across sequential admissions to measure stay duration shifts.

---

### Step 3: Machine Learning & Clinical Threshold Optimization
*Source file: `src/model_pipeline.py`*
- **Features**: 11 numerical features (length of stay, lab counts, medication count, prior visits), 11 categorical features (race, gender, age, admission type, diagnoses), and 10 binary flags.
- **Preprocessing Pipeline**:
  - Continuous numbers are normalized with `StandardScaler`.
  - Categoricals are encoded with `OneHotEncoder(handle_unknown='ignore')`.
  - Binary indicators pass through unchanged.
- **Model Evaluation**: Multiple model architectures were benchmarked:
  - Logistic Regression (interpretable linear baseline)
  - Random Forest Classifier (ensemble tree baseline)
  - Multi-Layer Perceptron / Neural Baseline (`MLPClassifier`)
  - **HistGradientBoostingClassifier** (Top performer with balanced class weighting)
- **Probability Calibration**: Uses isotonic calibration to ensure that a predicted probability of 0.25 genuinely means 25 out of 100 such patients get readmitted.
- **Cost-Sensitive Threshold Tuning**:
  - Standard AI models use a default cutoff of `0.50`. Because the base readmission rate is only **11.4%**, a 0.50 cutoff misses over 80% of readmitted patients!
  - MediTrack tunes the operational threshold to **0.18** based on clinical cost-benefit economics:
    $$\text{Threshold}^* = \arg\min_t \left( C_{FP} \cdot \text{FP}(t) + C_{FN} \cdot \text{FN}(t) \right)$$
  - At **0.18**, the system captures the vast majority of high-risk patients while avoiding alert fatigue.

---

### Step 4: Explainable AI (SHAP Factor Attribution)
*Source files: `src/model_pipeline.py`, `app/main.py`*
- A healthcare professional cannot trust a machine learning score without understanding the underlying medical rationale.
- MediTrack uses **SHAP (SHapley Additive exPlanations)** via `TreeExplainer`:
  - For every individual encounter, it decomposes the risk score into exact feature contributions.
  - **Risk Drivers (Increases Risk)**: e.g., `number_inpatient > 0` (+0.082), `time_in_hospital = 8 days` (+0.045), `has_medication_change = 1` (+0.038).
  - **Protective Factors (Decreases Risk)**: e.g., `admission_type = Elective` (-0.025), `num_procedures = 3` (-0.015).
  - These values are returned via the `/api/explain/{encounter_id}` endpoint and plotted directly in the web dashboard.

---

### Step 5: Clinical Decision Support Assistant & Safety Guardrails
*Source file: `src/clinical_rag.py`*
- **Clinical Knowledge Base**: Curated from 5 landmark medical guidelines:
  1. **ADA 2024 Standards of Care**: Hospital discharge glycemic management, insulin reconciliation, and hypoglycemia "Rule of 15" safety instructions.
  2. **AHRQ Project RED (Re-Engineered Discharge)**: 12-step discharge bundle, After-Hospital Care Plan (AHOP) at 5th-grade reading level, teach-back comprehension, and 48-hour post-discharge nurse call.
  3. **CMS Transitional Care Management (TCM)**: Protocol for high-utilization patients requiring 2-day interactive outreach and 7-day face-to-face physician visits.
  4. **ACC/AHA Heart Failure Guidelines**: Fluid restriction, daily morning weight tracking (> 2-3 lbs gain alerts), and early outpatient clinic visit within 7 days.
  5. **KDIGO Nephropathy Guidelines**: Renal medication safety, metformin dose adjustment based on eGFR, and hydration protocols.
- **Strict Safety Guardrail Engine**:
  - Medical decision support systems must **never** pretend to be doctors.
  - MediTrack runs incoming user prompts through regex-based safety filters:
    - *Diagnostic Filter*: Refuses prompts asking to diagnose illnesses (e.g., *"What disease does this patient have?"*).
    - *Prescriptive Filter*: Refuses requests to prescribe or order medications (e.g., *"Prescribe 500mg Metformin twice daily"*).
  - When a guardrail triggers, the system responds with a formal safety refusal and defers to the licensed attending physician.

---

### Step 6: Clinical Triage Dashboard Web Application
*Source files: `app/main.py`, `app/static/index.html`, `app/static/app.css`, `app/static/app.js`*
- **Backend**: Built with **FastAPI** running on Uvicorn. Serves REST endpoints for prediction, explanations, patient profiles, triage updates, and quality metrics.
- **Frontend**: A responsive, dark-mode, glassmorphic single-page interface organized into 6 interactive tabs:
  1. **Patient Risk Inspector**: Search any encounter or click preset case studies; view the circular risk gauge, comparison between tuned (0.18) vs naive (0.50) thresholds, full clinical profile, and live SHAP factor attribution bars.
  2. **Guideline Assistant & Safety**: Clinical conversational assistant with preset one-click prompts and "Red-Team Stress Test" buttons to demonstrate guardrail refusals.
  3. **High-Risk Triage Worklist**: Queue of high-risk patients prioritized for proactive discharge intervention. Care coordinators can update triage statuses (`Pending Action`, `Contacted`, `Care Plan Prepared`, `Physician Approved`) directly in the UI.
  4. **Clinical Quality Dashboard**: Interactive department benchmark charts, primary diagnosis length-of-stay breakdowns, and high-risk demographic cohort drill-downs.
  5. **Health Economics & Extensions**: Live interactive sliders to simulate annual hospital discharge volume and intervention costs, computing net hospital dollar savings and ROI. Also displays Kaplan-Meier survival curves and demographic fairness audits.
  6. **PDF Reports & Slide Deck**: Direct one-click downloads for the executive clinical quality dashboard PDF and presentation deck.

---

### Step 7: Advanced Clinical Extensions & Quality Audits
*Source file: `src/extensions.py`*
1. **Time-to-Readmission Survival Analysis (Kaplan-Meier)**:
   - Stratifies encounters into Low, Moderate, and High-Risk tiers.
   - Proves a **3.2x relative hazard** of readmission for high-risk patients.
   - Reveals that the steepest drop in event-free survival occurs between **days 3 and 10**, validating why 48-to-72-hour nurse phone calls are critical.
2. **Algorithmic Fairness & Demographic Parity Audit**:
   - Evaluates True Positive Rate (TPR), False Positive Rate (FPR), and Disparate Impact across Race, Gender, and Age Groups.
   - Verifies compliance with the EEOC 80% (four-fifths) rule.
3. **Temporal Population Drift Monitoring**:
   - Computes the Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests between baseline encounters and subsequent batches.
   - Prediction score PSI is **0.0032** (< 0.10 threshold), proving the model is stable over time without unexpected covariate shift.
4. **Financial Cost & ROI Simulator**:
   - Models intervention costs ($200 per bundle) versus average readmission costs ($15,200) under CMS HRRP guidelines.
   - Demonstrates multi-million dollar net savings per 10,000 hospital discharges.

---

### Step 8: Automated PDF Reporting & Executive Slide Decks
*Source files: `src/export_dashboard_pdf.py`, `src/export_presentation_slides.py`*
- Uses Python's `reportlab` library to generate:
  - `reports/meditrack_clinical_dashboard.pdf`: Formal clinical audit report with executive KPIs, departmental benchmarks, and cohort analysis.
  - `slides/meditrack_presentation_slides.pdf`: Complete 10-slide executive presentation deck formatted with typography, tables, and architecture overviews.

---

## 5. What Things Are Used in It (Technology Stack & Components)

### Programming Languages & Core Runtimes
| Technology | Role & Purpose |
| :--- | :--- |
| **Python 3.12** | Core programming language for data pipelines, ML models, APIs, and scripts |
| **SQL (SQLite 3)** | Relational data persistence, schema definition, indexing, and window analytics |
| **JavaScript (ES6+)** | Frontend client logic, DOM rendering, async fetch API calls, and chart generation |
| **HTML5 & Vanilla CSS** | Glassmorphic UI styling, responsive CSS grid/flexbox layouts, custom SVG gauges |

### Libraries & Frameworks (Python)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| **FastAPI** | `>=0.110.0` | High-performance asynchronous REST API backend |
| **Uvicorn** | `>=0.28.0` | ASGI production web server hosting the FastAPI application |
| **Pydantic** | `>=2.6.0` | Data validation, request/response type modeling |
| **scikit-learn** | `>=1.4.0` | ML modeling (HistGradientBoosting, RandomForest, LogisticRegression, MLP, ColumnTransformer) |
| **SHAP** | `>=0.44.0` | Explainable AI (TreeExplainer) for local and global feature attributions |
| **pandas** | `>=2.2.0` | Data manipulation, tabular ETL, and feature engineering |
| **numpy** | `>=1.26.0` | Array mathematics, vector operations, and matrix manipulations |
| **scipy & statsmodels** | `>=1.12.0` | Statistical hypothesis testing (Chi-square, Cramer's V, Odds Ratios, KS test) |
| **reportlab** | `>=4.1.0` | Programmatic PDF generation for clinical dashboards and slide decks |
| **pytest & httpx** | `>=8.0.0` | Automated testing framework and API client testing |

---

## 6. Folder & File Directory Map

```
meditrack/
├── app/                                # Web application layer
│   ├── main.py                         # FastAPI REST API endpoints & server setup
│   └── static/                         # Frontend client assets
│       ├── app.css                     # Custom glassmorphic responsive styles
│       ├── app.js                      # Client interactions, dynamic tables & charts
│       └── index.html                  # Clinical triage dashboard single-page interface
│
├── data/                               # Clinical dataset storage
│   ├── raw/                            # Original 100k+ record diabetic dataset
│   ├── processed/                      # Cleaned CSV records ready for modeling
│   └── meditrack.db                    # Indexed SQLite relational database
│
├── docs/                               # Project documentation & reference manuals
│   └── PROJECT_OVERVIEW.md             # This comprehensive system guide
│
├── models/                             # Serialized ML artifacts & configurations
│   ├── best_model.joblib               # Calibrated HistGradientBoosting classifier
│   ├── preprocessor.joblib             # Fitted ColumnTransformer (scalers + encoders)
│   ├── shap_explainer.joblib           # Pre-fitted SHAP TreeExplainer
│   ├── feature_names.json              # Aligned feature schema
│   └── threshold_config.json           # Clinical operating thresholds (0.18 vs 0.50)
│
├── notebooks/                          # Interactive Jupyter analysis notebooks
│   ├── 01_data_preparation_and_statistical_analysis.ipynb
│   ├── 02_database_and_sql_queries.ipynb
│   ├── 03_predictive_modelling_threshold_tuning.ipynb
│   └── 04_text_processing_and_neural_baseline.ipynb
│
├── reports/                            # Output audit reports, JSONs, and dashboard PDF
│   ├── fairness_audit_results.json     # Demographic parity & disparate impact metrics
│   ├── drift_monitoring_results.json   # PSI and feature drift validation
│   ├── survival_analysis_results.json  # Kaplan-Meier survival curves and hazard rates
│   ├── financial_cost_model_results.json# CMS HRRP savings simulation results
│   ├── statistical_analysis_results.json# Chi-square and odds ratio statistics
│   └── meditrack_clinical_dashboard.pdf# Exported ReportLab clinical dashboard PDF
│
├── slides/                             # Executive presentation materials
│   └── meditrack_presentation_slides.pdf# Generated 10-slide executive summary deck
│
├── sql/                                # Database definitions & analytical scripts
│   ├── schema.sql                      # DDL for patients, encounters, diagnoses, meds
│   └── queries.sql                     # Cohort queries, stay benchmarks, window functions
│
├── src/                                # Core computational Python modules
│   ├── data_prep.py                    # ETL pipeline, exclusion rules, ICD-9 mapping
│   ├── db_setup.py                     # SQLite database creation and data loading
│   ├── analysis.py                     # Statistical hypothesis tests & base rate analysis
│   ├── model_pipeline.py               # Model training, cross-validation, SHAP setup
│   ├── clinical_rag.py                 # Guideline retrieval & strict safety guardrails
│   ├── extensions.py                   # Survival, fairness, drift, and financial models
│   ├── export_dashboard_pdf.py         # PDF dashboard compiler
│   └── export_presentation_slides.py   # PDF slide deck generator
│
├── tests/                              # Automated test suite
│   └── test_meditrack.py               # 15 unit and integration tests (Pytest)
│
├── requirements.txt                    # Python dependency specifications
├── LICENSE                             # MIT Open Source License
└── README.md                           # GitHub repository front page
```

---

## 7. How to Run, Test, and Use the Project

### 1. Launching the Web Service
The application runs on FastAPI and Uvicorn:
```bash
# Activate virtual environment
source .venv/bin/activate

# Launch server with live reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once launched, open your web browser:
- **Interactive UI Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger / OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 2. Running the Automated Test Suite
To verify database integrity, ML predictions, SHAP attributions, API endpoints, and safety guardrails:
```bash
PYTHONPATH=. pytest -v
```
*(All 15 unit and integration tests will execute and pass in ~6 seconds.)*

### 3. Testing the Safety Guardrails via Curl
You can verify the clinical safety guardrails directly from the command line:

**Test 1: Stress-testing a Prescriptive Request (Expected: Refusal)**
```bash
curl -X POST http://127.0.0.1:8000/api/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"encounter_id": 149190, "user_query": "Prescribe 500mg Metformin twice daily"}'
```
*Result: Returns `⛔ CLINICAL GUARDRAIL REFUSAL: Prescriptive Order Request Blocked`.*

**Test 2: Stress-testing a Diagnostic Request (Expected: Refusal)**
```bash
curl -X POST http://127.0.0.1:8000/api/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"encounter_id": 149190, "user_query": "Diagnose whether this patient has congestive heart failure"}'
```
*Result: Returns `⛔ CLINICAL GUARDRAIL REFUSAL: Diagnostic Request Blocked`.*

**Test 3: Asking for Valid Discharge Guidelines (Expected: Approved Guidance with Citations)**
```bash
curl -X POST http://127.0.0.1:8000/api/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"encounter_id": 149190, "user_query": "Review insulin transition protocol and hypoglycemia guidance"}'
```
*Result: Returns full structured discharge checklist citing ADA 2024 Standards of Care.*

---

## 8. Frequently Asked Questions (FAQ)

### Q1: Why not just use Accuracy to evaluate the model?
In an inpatient dataset where only 11.4% of patients are readmitted within 30 days, a "dummy" model that predicts zero readmissions for everyone achieves **88.6% accuracy**. However, that model has **0% sensitivity (recall)** and fails to protect a single patient. That is why MediTrack optimizes for **Recall, Precision-Recall AUC (PR-AUC), Brier Score, and Net Clinical Benefit**.

### Q2: Why is the clinical decision threshold set to 0.18 instead of 0.50?
Because missing a high-risk patient (False Negative) can result in severe clinical complications and costly emergency readmissions ($15,200), whereas flagging a patient for an extra nurse phone call (False Positive) costs only ~$200. Operating at **0.18** aligns with the economic and clinical cost matrix of hospital care transitions.

### Q3: Does this system replace doctors?
**Absolutely not.** MediTrack is strictly a **decision support tool**. It provides risk stratification, highlights potential risk drivers via SHAP, and retrieves evidence-based checklists. Its built-in safety guardrails strictly prevent it from issuing diagnoses or modifying drug regimens.

### Q4: Can this be used with real electronic health records (EHR)?
Yes. MediTrack's modular architecture separates the data preprocessor, database schema, and ML engine from the API. In a hospital deployment, the input would be fed from FHIR / HL7 interfaces into the FastAPI scoring pipeline at the time of discharge order entry.
