"""
MediTrack - Clinical Quality Dashboard PDF Exporter
Project CP-02: Readmission Risk Prediction and Clinical Decision Support

Generates a publication-grade PDF report covering:
1. Hospital Readmission Metrics & KPIs
2. Readmission Rate by Medical Specialty / Department
3. Length of Stay Distribution & Diagnosis Categories
4. High-Risk Cohort Drill-Down Analysis
5. Clinical Guideline Transition Implementation
"""

import os
import json
import sqlite3
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_dashboard_pdf(db_path='data/meditrack.db', output_pdf_path='reports/meditrack_clinical_dashboard.pdf'):
    """
    Builds and exports the comprehensive MediTrack Clinical Quality Dashboard PDF.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # Load KPI summary data
    total_encounters = pd.read_sql_query("SELECT COUNT(*) FROM encounters", conn).iloc[0, 0]
    total_readm = pd.read_sql_query("SELECT SUM(readmitted_30d) FROM encounters", conn).iloc[0, 0]
    overall_rate = round(100.0 * total_readm / total_encounters, 2)
    avg_los = round(pd.read_sql_query("SELECT AVG(time_in_hospital) FROM encounters", conn).iloc[0, 0], 2)
    
    # Query 1: Department / Medical Specialty
    dept_sql = """
    SELECT 
        CASE 
            WHEN medical_specialty = 'Missing_or_Unknown' THEN 'General Inpatient / Unspecified'
            ELSE medical_specialty 
        END AS department_name,
        COUNT(encounter_id) AS encounters,
        ROUND(AVG(time_in_hospital), 1) AS avg_stay_days,
        SUM(readmitted_30d) AS readmissions,
        ROUND(100.0 * SUM(readmitted_30d) / COUNT(encounter_id), 1) AS readm_rate_pct
    FROM encounters
    GROUP BY department_name
    HAVING COUNT(encounter_id) >= 200
    ORDER BY encounters DESC
    LIMIT 10;
    """
    dept_df = pd.read_sql_query(dept_sql, conn)
    
    # Query 2: Primary Diagnosis Group
    diag_sql = """
    SELECT 
        d.clinical_category AS diagnosis_category,
        COUNT(e.encounter_id) AS encounters,
        ROUND(AVG(e.time_in_hospital), 1) AS avg_los,
        SUM(e.readmitted_30d) AS readmissions,
        ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(e.encounter_id), 1) AS readm_rate_pct
    FROM diagnoses d
    JOIN encounters e ON d.encounter_id = e.encounter_id
    WHERE d.diagnosis_seq = 1
    GROUP BY d.clinical_category
    ORDER BY encounters DESC;
    """
    diag_df = pd.read_sql_query(diag_sql, conn)
    
    # Query 3: High Risk Cohort Drill-Through
    cohort_sql = """
    SELECT 
        p.age_group,
        p.gender,
        CASE 
            WHEN e.number_inpatient = 0 THEN '0 Prior Inpatient'
            WHEN e.number_inpatient = 1 THEN '1 Prior Inpatient'
            ELSE '2+ Prior Inpatient'
        END AS prior_utilization,
        COUNT(e.encounter_id) AS cohort_size,
        SUM(e.readmitted_30d) AS readmitted_30d,
        ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(e.encounter_id), 1) AS readm_pct
    FROM encounters e
    JOIN patients p ON e.patient_nbr = p.patient_nbr
    GROUP BY p.age_group, p.gender, prior_utilization
    HAVING COUNT(e.encounter_id) >= 150
    ORDER BY readm_pct DESC
    LIMIT 8;
    """
    cohort_df = pd.read_sql_query(cohort_sql, conn)
    
    conn.close()
    
    # Construct Document
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569')
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=10,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    
    kpi_val_style = ParagraphStyle(
        'KpiVal',
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        alignment=1, # Center
        textColor=colors.HexColor('#0284c7')
    )
    
    kpi_lbl_style = ParagraphStyle(
        'KpiLbl',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        alignment=1,
        textColor=colors.HexColor('#64748b')
    )
    
    story = []
    
    # 1. Header & Branding
    story.append(Paragraph("MediTrack — Clinical Quality & Readmission Analytics", title_style))
    story.append(Paragraph("Hospital-Wide Quality Assurance, Department Benchmarking & Decision Support Audit Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceBefore=8, spaceAfter=12))
    
    # 2. Executive KPI Cards
    kpi_data = [
        [
            Paragraph(f"<b>{total_encounters:,}</b>", kpi_val_style),
            Paragraph(f"<b>{overall_rate}%</b>", kpi_val_style),
            Paragraph("<b>0.18 (Tuned)</b>", kpi_val_style),
            Paragraph(f"<b>{avg_los} Days</b>", kpi_val_style),
            Paragraph("<b>$3.08M / 10k</b>", kpi_val_style)
        ],
        [
            Paragraph("Evaluated Encounters", kpi_lbl_style),
            Paragraph("30d Readmission Base Rate", kpi_lbl_style),
            Paragraph("Clinical Decision Threshold", kpi_lbl_style),
            Paragraph("Avg Length of Stay", kpi_lbl_style),
            Paragraph("Simulated Net Savings", kpi_lbl_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[108, 108, 108, 108, 108])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))
    
    # 3. Department & Medical Specialty Benchmarks
    story.append(Paragraph("1. Department & Medical Specialty Readmission Performance", section_style))
    story.append(Paragraph(
        "Clinical departments with the highest encounter volumes and corresponding 30-day readmission rates. "
        "Units exceeding the 11.4% hospital average are prioritized for dedicated discharge nurse transitional outreach.",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    dept_table_data = [["Department / Specialty", "Encounters", "Avg Stay (Days)", "30d Readmissions", "Readmission Rate"]]
    for _, row in dept_df.iterrows():
        dept_table_data.append([
            str(row['department_name']),
            f"{row['encounters']:,}",
            f"{row['avg_stay_days']}",
            f"{row['readmissions']:,}",
            f"{row['readm_rate_pct']}%"
        ])
    dept_table = Table(dept_table_data, colWidths=[200, 85, 85, 85, 85])
    dept_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(dept_table)
    story.append(Spacer(1, 14))
    
    # 4. Primary Diagnosis Category Utilization & Length of Stay Distribution
    story.append(Paragraph("2. Primary Diagnosis Categories & Length of Stay (LOS) Distribution", section_style))
    diag_table_data = [["Primary Diagnosis Category", "Encounters", "Avg Stay (Days)", "Readmissions", "30d Readm %"]]
    for _, row in diag_df.iterrows():
        diag_table_data.append([
            str(row['diagnosis_category']),
            f"{row['encounters']:,}",
            f"{row['avg_los']}",
            f"{row['readmissions']:,}",
            f"{row['readm_rate_pct']}%"
        ])
    diag_table = Table(diag_table_data, colWidths=[190, 85, 85, 90, 90])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 14))
    
    # Page Break for Drill-Through Analysis
    story.append(PageBreak())
    
    # 5. High-Risk Cohort Drill-Through
    story.append(Paragraph("3. High-Risk Cohort Drill-Through (Care Team Triage Worklist)", section_style))
    story.append(Paragraph(
        "Multivariate cohort stratification cross-referencing age bracket, gender, and prior inpatient admissions. "
        "Patients with repeated prior inpatient hospitalizations exhibit acute readmission risk spikes approaching 30-45%.",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    cohort_table_data = [["Age Group", "Gender", "Prior Inpatient Tier", "Cohort Size", "30d Readmissions", "Readmission Rate"]]
    for _, row in cohort_df.iterrows():
        cohort_table_data.append([
            str(row['age_group']),
            str(row['gender']),
            str(row['prior_utilization']),
            f"{row['cohort_size']:,}",
            f"{row['readmitted_30d']:,}",
            f"{row['readm_pct']}%"
        ])
    cohort_table = Table(cohort_table_data, colWidths=[90, 80, 140, 75, 75, 80])
    cohort_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(cohort_table)
    story.append(Spacer(1, 14))
    
    # 6. Guideline Implementation & Clinical Safety Summary
    story.append(Paragraph("4. Clinical Decision Support & Guideline Implementation Audit", section_style))
    story.append(Paragraph(
        "<b>AHRQ Project RED Compliance:</b> All patients with risk scores exceeding the operational threshold (0.18) "
        "automatically receive a structured After-Hospital Care Plan (AHOP) and 48-hour phone call priority.<br/>"
        "<b>ADA Glycemic Transitions:</b> Inpatients on insulin therapy or with acute dosage changes are scheduled for "
        "ambulatory medication reconciliation within 7-14 days with hypoglycemia teach-back education.<br/>"
        "<b>Safety Guardrail Enforcement:</b> The embedded assistant strictly blocks diagnostic and prescriptive inquiries, "
        "preserving clinical accountability and directing all diagnostic/dosing authority to licensed attending physicians.",
        body_style
    ))
    story.append(Spacer(1, 16))
    
    # Footer Notice
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#94a3b8'), spaceBefore=4, spaceAfter=8))
    story.append(Paragraph(
        "<b>MediTrack Decision Support System (CP-02)</b> — Generated for Clinical Quality Review. "
        "Confidential & De-Identified Healthcare Information under HIPAA Safe Harbor.",
        subtitle_style
    ))
    
    doc.build(story)
    print(f"Clinical Dashboard PDF exported successfully to: {output_pdf_path}")
    return output_pdf_path

if __name__ == '__main__':
    generate_dashboard_pdf()
