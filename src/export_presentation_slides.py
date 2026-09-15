"""
MediTrack - 10-Slide Presentation Deck Generator
Project CP-02: Readmission Risk Prediction and Clinical Decision Support
Generates landscape PDF slides covering problem, data, approach, demonstration, results, and limitations.
"""

import os
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_slides_pdf(output_path='slides/meditrack_presentation_slides.pdf'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Landscape letter: 792 x 612 points
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(letter),
        leftMargin=40, rightMargin=40, topMargin=35, bottomMargin=35
    )
    
    styles = getSampleStyleSheet()
    
    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0f172a')
    )
    
    slide_subtitle_style = ParagraphStyle(
        'SlideSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0284c7')
    )
    
    section_h_style = ParagraphStyle(
        'SectionH',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=6,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#334155')
    )
    
    bullet_style = ParagraphStyle(
        'SlideBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1e293b')
    )
    
    footer_style = ParagraphStyle(
        'SlideFooter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#94a3b8')
    )
    
    slides_data = [
        # Slide 1
        {
            "num": 1,
            "title": "MediTrack: Readmission Risk Prediction & Decision Support",
            "subtitle": "Point-of-Discharge Decision Support System Grounded in Published Clinical Guidelines",
            "content": [
                Paragraph("<b>Project Brief:</b> CP-02 | Healthcare Analytics & Clinical AI", section_h_style),
                Paragraph("<b>Cohort:</b> CSE VII | Batch 2, Room 205, Block 3 | Sharda University", body_style),
                Paragraph("<b>Submission Date:</b> Day 12, September 15, 2026", body_style),
                Spacer(1, 20),
                Paragraph("<b>Core Capabilities:</b>", section_h_style),
                Paragraph("• 30-Day Hospital Readmission Scoring at Point of Discharge with Stated Decision Threshold", bullet_style),
                Paragraph("• Patient-Level Transparent Factor Attribution using SHAP Tree Explainability", bullet_style),
                Paragraph("• Guideline-Grounded Assistant (ADA Standards, AHRQ Project RED, CMS TCM) with Mandatory Source Citations", bullet_style),
                Paragraph("• Strict Safety Guardrails Defending Against Diagnostic and Prescriptive Violations", bullet_style),
                Paragraph("• Hospital-Wide Clinical Quality Dashboard with High-Risk Cohort Drill-Through", bullet_style)
            ]
        },
        # Slide 2
        {
            "num": 2,
            "title": "Clinical Problem Statement & Regulatory Context",
            "subtitle": "The Challenge of Post-Discharge Readmission and Need for Real-Time Decision Support",
            "content": [
                Paragraph("<b>The Healthcare Burden:</b>", section_h_style),
                Paragraph("• Hospital readmission within 30 days of discharge is a critical quality benchmark tracked under CMS HRRP.", bullet_style),
                Paragraph("• Avoidable readmissions account for billions in excess expenditure, hospital financial penalties, and patient morbidity.", bullet_style),
                Paragraph("• <b>The Current Gap:</b> Most healthcare systems analyze readmissions <i>retrospectively</i> via post-facto billing audits.", bullet_style),
                Spacer(1, 10),
                Paragraph("<b>The MediTrack Solution:</b>", section_h_style),
                Paragraph("• Intervene <i>proactively at the point of discharge</i> before the patient departs the hospital.", bullet_style),
                Paragraph("• Combine statistical risk scoring with individualized clinical factor transparency.", bullet_style),
                Paragraph("• Provide the care team with an intelligent transitional care assistant grounded in national standards.", bullet_style)
            ]
        },
        # Slide 3
        {
            "num": 3,
            "title": "Dataset Architecture & Data Pipeline",
            "subtitle": "UCI Diabetes 130-US Hospitals (1999–2008) Data Quality & Preprocessing",
            "content": [
                Paragraph("<b>Primary Ingestion & Clinical Exclusions:</b>", section_h_style),
                Paragraph("• Initial encounters loaded: <b>101,766</b> records across 10 years of multi-hospital clinical care.", bullet_style),
                Paragraph("• <b>Documented Exclusion Rules Applied:</b> Excluded <b>2,423</b> deceased and hospice discharges (IDs 11, 13, 14, 19, 20, 21).", bullet_style),
                Paragraph("• <b>Post-Exclusion Cohort:</b> <b>99,340</b> validated encounters and <b>69,987</b> unique patients.", bullet_style),
                Spacer(1, 10),
                Paragraph("<b>Feature Engineering & High-Cardinality Groupings:</b>", section_h_style),
                Paragraph("• <b>Missing Fields:</b> Weight (>96% missing) mapped to recording status flag; Payer Code (~40% missing) categorized.", bullet_style),
                Paragraph("• <b>ICD-9 Mapping:</b> Over 800 diagnosis codes consolidated into 9 standard clinical organ categories.", bullet_style),
                Paragraph("• <b>Medication Harmonization:</b> 23 diabetes medications tracked for active use, dosage titration (Up/Down), and polypharmacy.", bullet_style)
            ]
        },
        # Slide 4
        {
            "num": 4,
            "title": "Exploratory Findings & Statistical Hypothesis Testing",
            "subtitle": "Rigorous Statistical Basis for Core Clinical Risk Determinants",
            "content": [
                Paragraph("<b>1. Base Readmission Rate:</b>", section_h_style),
                Paragraph("• 30-Day Readmission (<30 days): <b>11,314</b> encounters = <b>11.39%</b> (95% CI: 11.19% – 11.59%).", bullet_style),
                Spacer(1, 6),
                Paragraph("<b>2. Statistical Hypothesis Tests:</b>", section_h_style),
                Paragraph("• <b>Age Group:</b> Pearson's Chi-Square $\\chi^2 = 133.625$, df = 9, $p = 2.13 \\times 10^{-24}$ (Statistically Significant).", bullet_style),
                Paragraph("• <b>Admission Type:</b> $\\chi^2 = 32.360$, df = 6, $p = 1.39 \\times 10^{-5}$. Emergency vs Elective Odds Ratio = <b>1.145</b>.", bullet_style),
                Paragraph("• <b>Prior Inpatient History:</b> Mann-Whitney U $p = 0.000$, Welch's t $p = 4.87 \\times 10^{-264}$.", bullet_style),
                Paragraph("  - Readmitted patients mean prior stays = <b>1.223</b> vs Non-readmitted = <b>0.555</b>.", bullet_style),
                Paragraph("  - Logistic Odds Ratio = <b>1.337</b> per additional prior visit (95% CI: [1.321, 1.353]).", bullet_style)
            ]
        },
        # Slide 5
        {
            "num": 5,
            "title": "Relational Database & Analytical Queries",
            "subtitle": "Normalized SQLite Relational Architecture & Runnable SQL Workflows",
            "content": [
                Paragraph("<b>Relational Schema (4 Normalized Entities):</b>", section_h_style),
                Paragraph("• <b>patients:</b> 69,987 unique patients (demographics, longitudinal count).", bullet_style),
                Paragraph("• <b>encounters:</b> 99,340 inpatient stays (admission type, LOS, labs, utilization, readmission flag).", bullet_style),
                Paragraph("• <b>diagnoses:</b> 298,020 records (primary, secondary, tertiary ICD-9 and clinical classifications).", bullet_style),
                Paragraph("• <b>medications:</b> 117,953 active diabetic prescription entries across 23 tracked therapies.", bullet_style),
                Spacer(1, 6),
                Paragraph("<b>Executed Analytical Queries:</b>", section_h_style),
                Paragraph("• <b>Query 1 (Cohort Readmission):</b> Patients with 3+ prior visits show readmission rates exceeding <b>44.8%</b>.", bullet_style),
                Paragraph("• <b>Query 2 (Stay by Diagnosis):</b> Neoplasms leads length of stay (5.28 days); Diabetes leads readmission rate (13.10%).", bullet_style),
                Paragraph("• <b>Query 3 (Encounter Sequences):</b> Window functions (ROW_NUMBER, LAG) trace consecutive admissions per patient.", bullet_style)
            ]
        },
        # Slide 6
        {
            "num": 6,
            "title": "Predictive Modeling & Class Imbalance Strategy",
            "subtitle": "Benchmark Evaluation of Linear, Tree Ensembles, and Neural Baseline",
            "content": [
                Paragraph("<b>Model Performance Benchmark (Holdout Test: 19,868 Encounters):</b>", section_h_style),
                Paragraph("• <b>Logistic Regression (Balanced):</b> ROC-AUC: <b>0.6383</b> | PR-AUC: <b>0.2006</b> | Brier: 0.2319", bullet_style),
                Paragraph("• <b>Random Forest (Balanced):</b> ROC-AUC: <b>0.6447</b> | PR-AUC: <b>0.2064</b> | Brier: 0.2147", bullet_style),
                Paragraph("• <b>Gradient Boosting (Balanced) [BEST]:</b> ROC-AUC: <b>0.6445</b> | PR-AUC: <b>0.2064</b> | Brier: 0.2266", bullet_style),
                Paragraph("• <b>Neural Baseline (MLP):</b> ROC-AUC: <b>0.6421</b> | PR-AUC: <b>0.2059</b> | Brier: 0.0976", bullet_style),
                Spacer(1, 8),
                Paragraph("<b>Class Imbalance Strategy Justification:</b>", section_h_style),
                Paragraph("• Applied cost-sensitive class weighting ($w_1 \\approx 4.39$).", bullet_style),
                Paragraph("• Justification: Preserves authentic clinical feature covariance and avoids synthetic artifact generation in discrete ICD-9/medication vectors.", bullet_style)
            ]
        },
        # Slide 7
        {
            "num": 7,
            "title": "Clinical Decision Threshold Justification",
            "subtitle": "Defending the Recall vs. Precision Trade-off in Clinical Practice",
            "content": [
                Paragraph("<b>Why Naive 0.50 Threshold Fails Clinically:</b>", section_h_style),
                Paragraph("• In the unweighted neural baseline, a 0.50 threshold yields <b><1% Sensitivity</b> (misses 99% of readmissions!).", bullet_style),
                Paragraph("• Default probabilistic thresholds are calibrated to base rates, not clinical utility.", bullet_style),
                Spacer(1, 8),
                Paragraph("<b>Tuned Operational Decision Threshold (0.18):</b>", section_h_style),
                Paragraph("• <b>Clinical Sensitivity (Recall):</b> <b>65.5%</b> — Successfully flags ~2 out of every 3 preventable readmissions.", bullet_style),
                Paragraph("• <b>Cost-Benefit Asymmetry Defense:</b>", section_h_style),
                Paragraph("  - <b>Cost of False Negative:</b> ~$15,200 (avoidable ICU/ER admission, CMS penalty, clinical decompensation).", bullet_style),
                Paragraph("  - <b>Cost of False Positive:</b> ~$150 (transitional nurse phone outreach, medication review, primary care check).", bullet_style),
                Paragraph("  - The 1:100 cost ratio heavily favors high sensitivity over specificity in transitional care triage.", bullet_style)
            ]
        },
        # Slide 8
        {
            "num": 8,
            "title": "Text Processing & Patient-Level Factor Transparency",
            "subtitle": "ICD-9 Diagnosis NLP Extraction & SHAP Factor Attribution",
            "content": [
                Paragraph("<b>Text Processing & NLP Feature Derivation:</b>", section_h_style),
                Paragraph("• Derived structured clinical descriptions from ICD-9 codes (e.g. congestive heart failure, acute MI, CKD).", bullet_style),
                Paragraph("• Extracted 50 TF-IDF n-gram clinical term features and evaluated incremental performance gain.", bullet_style),
                Paragraph("• Finding: Reinforced organ-specific decompensations without overfitting.", bullet_style),
                Spacer(1, 8),
                Paragraph("<b>SHAP Factor-Level Explainability:</b>", section_h_style),
                Paragraph("• Generates individualized patient feature contributions (waterfall attributions) for every encounter.", bullet_style),
                Paragraph("• Clinicians can see exactly why a patient was flagged (e.g., prior inpatient visits: +8.4%, insulin titration: +4.2%).", bullet_style),
                Paragraph("• Eliminates black-box distrust and guides targeted discharge actions.", bullet_style)
            ]
        },
        # Slide 9
        {
            "num": 9,
            "title": "Guideline Retrieval Layer & Clinical Safety Guardrails",
            "subtitle": "Grounded AI Decision Support with Mandatory Citations & Refusal Protocols",
            "content": [
                Paragraph("<b>Published Clinical Knowledge Base:</b>", section_h_style),
                Paragraph("• <b>ADA Standards of Care (2024):</b> Glycemic discharge planning, insulin education, Rule of 15 hypoglycemia.", bullet_style),
                Paragraph("• <b>AHRQ Project RED:</b> 12-component re-engineered discharge protocol, teach-back method, AHOP plan.", bullet_style),
                Paragraph("• <b>CMS TCM Services:</b> 48-hour interactive outreach and 7-day primary physician follow-up.", bullet_style),
                Paragraph("• <b>Mandatory Citations:</b> Every recommendation includes full publication title, guideline section, and evidence grade.", bullet_style),
                Spacer(1, 8),
                Paragraph("<b>Clinical Safety Guardrails:</b>", section_h_style),
                Paragraph("• <b>Refusal 1:</b> Diagnostic queries ('Do I have cancer?', 'Diagnose illness') are strictly blocked.", bullet_style),
                Paragraph("• <b>Refusal 2:</b> Prescriptive requests ('Prescribe 500mg Metformin', 'Change dosage') are immediately refused.", bullet_style),
                Paragraph("• The assistant defers all prescriptive and diagnostic authority to the licensed attending medical team.", bullet_style)
            ]
        },
        # Slide 10
        {
            "num": 10,
            "title": "Quality Dashboard, Verification & Limitations",
            "subtitle": "Healthcare Economics, Fairness Parity, Drift Monitoring & Clinical Limitations",
            "content": [
                Paragraph("<b>Verified System Deliverables:</b>", section_h_style),
                Paragraph("• <b>Interactive Web Application:</b> Encounter selector, real-time scoring, SHAP waterfall, guardrailed chat, triage worklist.", bullet_style),
                Paragraph("• <b>Financial ROI Simulator:</b> Net <b>$3.08M</b> hospital savings per 10,000 discharges (ROI: 154.8%).", bullet_style),
                Paragraph("• <b>Fairness & Drift Audits:</b> Verified parity across race/gender (EEOC 4/5 rule satisfied) and healthy feature stability (PSI < 0.05).", bullet_style),
                Spacer(1, 8),
                Paragraph("<b>Documented Clinical Limitations:</b>", section_h_style),
                Paragraph("• Dataset lacks outpatient lab values (e.g. serial eGFR, Troponin) and social determinants of health (SDOH).", bullet_style),
                Paragraph("• Historical data (1999–2008) predates contemporary SGLT2i / GLP-1 RA medication guidelines.", bullet_style),
                Paragraph("• Prospective validation in live clinical EHR workflow is required prior to autonomous clinical deployment.", bullet_style)
            ]
        }
    ]
    
    story = []
    for s in slides_data:
        # Header banner
        slide_banner = [
            [
                Paragraph(f"<b>MediTrack — Decision Support System</b>", slide_subtitle_style),
                Paragraph(f"<b>Slide {s['num']} of 10</b>", ParagraphStyle('PNum', parent=styles['Normal'], alignment=2, textColor=colors.HexColor('#64748b'), fontSize=10))
            ]
        ]
        t_banner = Table(slide_banner, colWidths=[550, 160])
        t_banner.setStyle(TableStyle([
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(t_banner)
        story.append(Spacer(1, 4))
        story.append(Paragraph(s['title'], slide_title_style))
        story.append(Paragraph(s['subtitle'], slide_subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceBefore=6, spaceAfter=10))
        
        for elem in s['content']:
            story.append(elem)
            
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceBefore=6, spaceAfter=4))
        story.append(Paragraph("Project CP-02 | Sharda University — CSE VII | Be Practical Tech Solutions", footer_style))
        
        if s['num'] < 10:
            story.append(PageBreak())
            
    doc.build(story)
    print(f"Presentation slides exported successfully to: {output_path}")
    return output_path

if __name__ == '__main__':
    generate_slides_pdf()
