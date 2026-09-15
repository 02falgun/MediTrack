"""
MediTrack - Statistical Analysis & Hypothesis Testing
Project CP-02: Readmission Risk Prediction and Clinical Decision Support
"""

import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

def perform_statistical_analysis(cleaned_csv_path, output_dir='reports'):
    """
    Executes rigorous statistical hypothesis tests on readmission differences
    by age group, admission type, and prior inpatient history.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(cleaned_csv_path, low_memory=False)
    n_total = len(df)
    
    # 1. Establish Readmission Base Rate
    readm_counts = df['readmitted'].value_counts()
    readm_30d_count = int((df['readmitted_30d'] == 1).sum())
    readm_30d_rate = float(readm_30d_count / n_total)
    
    # Wilson score confidence interval for base rate
    z = 1.96
    p_hat = readm_30d_rate
    ci_lower = (p_hat + z**2 / (2*n_total) - z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n_total))/n_total)) / (1 + z**2/n_total)
    ci_upper = (p_hat + z**2 / (2*n_total) + z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n_total))/n_total)) / (1 + z**2/n_total)
    
    base_rate_summary = {
        'total_encounters': n_total,
        'readmitted_under_30d_count': readm_30d_count,
        'readmitted_under_30d_rate': round(readm_30d_rate, 4),
        'readmitted_under_30d_pct': round(readm_30d_rate * 100, 2),
        'ci_95_pct': [round(float(ci_lower) * 100, 2), round(float(ci_upper) * 100, 2)],
        'category_breakdown': {k: int(v) for k, v in readm_counts.to_dict().items()}
    }
    
    print("\n=== 1. READMISSION BASE RATE ===")
    print(f"Total encounters evaluated: {n_total:,}")
    print(f"30-day Readmission count: {readm_30d_count:,} ({base_rate_summary['readmitted_under_30d_pct']}%)")
    print(f"95% Confidence Interval: [{base_rate_summary['ci_95_pct'][0]}%, {base_rate_summary['ci_95_pct'][1]}%]")
    
    # 2. Hypothesis Test 1: Age Group Differences
    # Contingency table
    age_contingency = pd.crosstab(df['age_group'], df['readmitted_30d'])
    chi2_age, p_val_age, dof_age, expected_age = stats.chi2_contingency(age_contingency)
    cramer_v_age = np.sqrt(chi2_age / (n_total * (min(age_contingency.shape) - 1)))
    
    # Age group specific rates
    age_rates = df.groupby('age_group')['readmitted_30d'].agg(['count', 'sum', 'mean']).reset_index()
    age_rates.columns = ['age_group', 'encounters', 'readmissions', 'rate']
    age_rates['rate_pct'] = (age_rates['rate'] * 100).round(2)
    
    age_summary = {
        'test_name': "Pearson's Chi-Square Test of Independence",
        'null_hypothesis': "30-day readmission rate is independent of patient age group",
        'chi2_statistic': round(float(chi2_age), 3),
        'degrees_of_freedom': int(dof_age),
        'p_value': float(p_val_age),
        'cramers_v': round(float(cramer_v_age), 4),
        'statistically_significant': bool(p_val_age < 0.05),
        'rates_by_age': age_rates.to_dict(orient='records')
    }
    
    print("\n=== 2. HYPOTHESIS TEST: AGE GROUP ===")
    print(f"Chi2: {age_summary['chi2_statistic']}, df: {age_summary['degrees_of_freedom']}, p-value: {p_val_age:.4e}")
    print(f"Cramer's V: {age_summary['cramers_v']} (Significant: {age_summary['statistically_significant']})")
    
    # 3. Hypothesis Test 2: Admission Type Differences
    adm_contingency = pd.crosstab(df['admission_type_name'], df['readmitted_30d'])
    chi2_adm, p_val_adm, dof_adm, expected_adm = stats.chi2_contingency(adm_contingency)
    cramer_v_adm = np.sqrt(chi2_adm / (n_total * (min(adm_contingency.shape) - 1)))
    
    adm_rates = df.groupby('admission_type_name')['readmitted_30d'].agg(['count', 'sum', 'mean']).reset_index()
    adm_rates.columns = ['admission_type', 'encounters', 'readmissions', 'rate']
    adm_rates['rate_pct'] = (adm_rates['rate'] * 100).round(2)
    
    # Odds ratio: Emergency vs Elective
    em_readm = int(((df['admission_type_name'] == 'Emergency') & (df['readmitted_30d'] == 1)).sum())
    em_noreadm = int(((df['admission_type_name'] == 'Emergency') & (df['readmitted_30d'] == 0)).sum())
    el_readm = int(((df['admission_type_name'] == 'Elective') & (df['readmitted_30d'] == 1)).sum())
    el_noreadm = int(((df['admission_type_name'] == 'Elective') & (df['readmitted_30d'] == 0)).sum())
    
    odds_em = (em_readm / em_noreadm) if em_noreadm > 0 else 0
    odds_el = (el_readm / el_noreadm) if el_noreadm > 0 else 0
    or_em_vs_el = odds_em / odds_el if odds_el > 0 else 0
    
    adm_summary = {
        'test_name': "Pearson's Chi-Square Test of Independence & Odds Ratio",
        'null_hypothesis': "30-day readmission rate is independent of admission type",
        'chi2_statistic': round(float(chi2_adm), 3),
        'degrees_of_freedom': int(dof_adm),
        'p_value': float(p_val_adm),
        'cramers_v': round(float(cramer_v_adm), 4),
        'odds_ratio_emergency_vs_elective': round(float(or_em_vs_el), 3),
        'statistically_significant': bool(p_val_adm < 0.05),
        'rates_by_admission_type': adm_rates.to_dict(orient='records')
    }
    
    print("\n=== 3. HYPOTHESIS TEST: ADMISSION TYPE ===")
    print(f"Chi2: {adm_summary['chi2_statistic']}, df: {adm_summary['degrees_of_freedom']}, p-value: {p_val_adm:.4e}")
    print(f"Odds Ratio (Emergency vs Elective): {adm_summary['odds_ratio_emergency_vs_elective']}")
    
    # 4. Hypothesis Test 3: Prior Inpatient History Differences
    readm_inp = df[df['readmitted_30d'] == 1]['number_inpatient']
    noreadm_inp = df[df['readmitted_30d'] == 0]['number_inpatient']
    
    t_stat, t_pval = stats.ttest_ind(readm_inp, noreadm_inp, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(readm_inp, noreadm_inp, alternative='two-sided')
    
    # Binned prior inpatient rate
    df['inpatient_tier'] = pd.cut(df['number_inpatient'], bins=[-1, 0, 1, 2, 100], labels=['0 visits', '1 visit', '2 visits', '3+ visits'])
    inp_rates = df.groupby('inpatient_tier', observed=False)['readmitted_30d'].agg(['count', 'sum', 'mean']).reset_index()
    inp_rates.columns = ['inpatient_tier', 'encounters', 'readmissions', 'rate']
    inp_rates['rate_pct'] = (inp_rates['rate'] * 100).round(2)
    
    # Univariate Logistic Regression to get exact Odds Ratio per inpatient stay
    X = sm.add_constant(df['number_inpatient'])
    logit_mod = sm.Logit(df['readmitted_30d'], X).fit(disp=False)
    or_per_inpatient = float(np.exp(logit_mod.params['number_inpatient']))
    ci_or = np.exp(logit_mod.conf_int().loc['number_inpatient']).values
    
    inpatient_summary = {
        'test_name': "Welch's Two-Sample t-Test & Mann-Whitney U Test & Logistic Odds Ratio",
        'null_hypothesis': "Prior inpatient visit counts do not differ between 30-day readmitted and non-readmitted patients",
        'mean_prior_inpatient_readmitted': round(float(readm_inp.mean()), 3),
        'mean_prior_inpatient_non_readmitted': round(float(noreadm_inp.mean()), 3),
        'welch_t_statistic': round(float(t_stat), 3),
        'welch_p_value': float(t_pval),
        'mann_whitney_u_stat': round(float(u_stat), 1),
        'mann_whitney_p_value': float(u_pval),
        'odds_ratio_per_prior_visit': round(or_per_inpatient, 3),
        'odds_ratio_95_ci': [round(float(ci_or[0]), 3), round(float(ci_or[1]), 3)],
        'statistically_significant': bool(u_pval < 0.05),
        'rates_by_inpatient_tier': inp_rates.to_dict(orient='records')
    }
    
    print("\n=== 4. HYPOTHESIS TEST: PRIOR INPATIENT HISTORY ===")
    print(f"Mean prior inpatient visits (Readmitted): {inpatient_summary['mean_prior_inpatient_readmitted']}")
    print(f"Mean prior inpatient visits (Non-Readmitted): {inpatient_summary['mean_prior_inpatient_non_readmitted']}")
    print(f"Mann-Whitney U p-value: {u_pval:.4e}, Welch t p-value: {t_pval:.4e}")
    print(f"Odds Ratio per additional prior inpatient visit: {inpatient_summary['odds_ratio_per_prior_visit']} (95% CI: {inpatient_summary['odds_ratio_95_ci']})")
    
    complete_report = {
        'base_rate': base_rate_summary,
        'age_test': age_summary,
        'admission_type_test': adm_summary,
        'prior_inpatient_test': inpatient_summary
    }
    
    report_json_path = os.path.join(output_dir, 'statistical_analysis_results.json')
    with open(report_json_path, 'w') as f:
        json.dump(complete_report, f, indent=2)
        
    print(f"\nAll statistical test results saved to {report_json_path}")
    return complete_report

if __name__ == '__main__':
    perform_statistical_analysis('data/processed/cleaned_encounters.csv')
