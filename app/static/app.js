/**
 * MediTrack - Clinical Decision Support Client Application Logic
 * Integrates risk scoring, SHAP factors, guardrailed RAG assistant, triage, and dashboard.
 */

let currentEncounterId = 149190;
let currentProfile = null;
let currentPrediction = null;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initQuickButtons();
  initSearch();
  initChat();
  initEconomicsCalculator();

  // Load initial patient encounter #149190
  loadPatientEncounter(currentEncounterId);
  loadDashboardAnalytics();
  loadTriageWorklist();
  loadExtensionsData();
});

// 1. Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add('active');
    });
  });
}

// 2. Patient Search & Quick Pickers
function initQuickButtons() {
  const btns = document.querySelectorAll('.quick-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const encId = parseInt(btn.getAttribute('data-enc'));
      loadPatientEncounter(encId);
    });
  });
}

function initSearch() {
  const searchInput = document.getElementById('patient-search-input');
  if (!searchInput) return;

  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      const val = searchInput.value.trim();
      if (val) {
        // If numeric, load directly
        const num = parseInt(val);
        if (!isNaN(num)) {
          loadPatientEncounter(num);
        } else {
          // Query API for match
          fetch(`/api/patients?search=${encodeURIComponent(val)}&limit=1`)
            .then(r => r.json())
            .then(data => {
              if (data.encounters && data.encounters.length > 0) {
                loadPatientEncounter(data.encounters[0].encounter_id);
              } else {
                alert(`No encounter found matching "${val}"`);
              }
            });
        }
      }
    }
  });
}

// 3. Load & Render Patient Profile, Prediction & SHAP
async function loadPatientEncounter(encounterId) {
  try {
    currentEncounterId = encounterId;

    // Fetch profile, prediction, and explanations concurrently
    const [profileRes, predRes, explainRes] = await Promise.all([
      fetch(`/api/patient/${encounterId}`),
      fetch(`/api/predict/${encounterId}`),
      fetch(`/api/explain/${encounterId}`)
    ]);

    if (!profileRes.ok) {
      alert(`Encounter #${encounterId} not found.`);
      return;
    }

    currentProfile = await profileRes.json();
    currentPrediction = await predRes.json();
    const explainData = await explainRes.json();

    renderPatientDetails(currentProfile);
    renderRiskScore(currentPrediction);
    renderShapFactors(explainData);

  } catch (err) {
    console.error("Error loading patient encounter:", err);
  }
}

function renderPatientDetails(p) {
  document.getElementById('display-encounter-id').textContent = `#${p.encounter_id}`;
  document.getElementById('display-patient-nbr').textContent = `Patient Nbr: ${p.patient_nbr}`;

  document.getElementById('vital-age').textContent = p.age_group || 'Unknown';
  document.getElementById('vital-gender').textContent = p.gender || 'Unknown';
  document.getElementById('vital-race').textContent = p.race || 'Unknown';
  document.getElementById('vital-admission').textContent = p.admission_type_name || 'Emergency';
  document.getElementById('vital-los').textContent = `${p.time_in_hospital} days`;
  document.getElementById('vital-labs').textContent = `${p.num_lab_procedures} procedures`;
  document.getElementById('vital-inpatient').textContent = `${p.number_inpatient} previous stay(s)`;
  document.getElementById('vital-emergency').textContent = `${p.number_emergency} ER visits`;
  document.getElementById('vital-a1c').textContent = p.a1c_result || 'Not Tested';
  document.getElementById('vital-glucose').textContent = p.glucose_result || 'Not Tested';

  // Primary Diagnosis
  const primaryDiag = p.diagnoses && p.diagnoses.length > 0 ? p.diagnoses[0] : null;
  document.getElementById('vital-primary-diag').textContent = primaryDiag ? `${primaryDiag.clinical_category} (ICD ${primaryDiag.icd9_code})` : 'Unspecified';

  // Diagnoses tags
  const diagContainer = document.getElementById('diagnoses-list');
  diagContainer.innerHTML = '';
  if (p.diagnoses && p.diagnoses.length > 0) {
    p.diagnoses.forEach(d => {
      const tag = document.createElement('div');
      tag.className = 'factor-item';
      tag.innerHTML = `
        <span class="factor-name">Seq #${d.diagnosis_seq}: <b>${d.clinical_category}</b></span>
        <span style="color: #94a3b8; font-size: 11px;">${d.clinical_description || d.icd9_code}</span>
      `;
      diagContainer.appendChild(tag);
    });
  }

  // Active Medications
  const medsContainer = document.getElementById('medications-list');
  medsContainer.innerHTML = '';
  if (p.active_medications && p.active_medications.length > 0) {
    p.active_medications.forEach(m => {
      const tag = document.createElement('span');
      const isShift = m.has_dosage_change;
      tag.className = `prompt-chip ${isShift ? 'red-team' : ''}`;
      tag.style.marginRight = '6px';
      tag.style.marginBottom = '6px';
      tag.style.display = 'inline-block';
      tag.textContent = `${m.medication_name}: ${m.dosage_status} ${isShift ? '⚡ (Titrated)' : ''}`;
      medsContainer.appendChild(tag);
    });
  } else {
    medsContainer.innerHTML = '<span style="color: #64748b; font-size: 12px;">No active diabetic medications recorded</span>';
  }
}

function renderRiskScore(pred) {
  const pct = pred.risk_percentage;
  document.getElementById('risk-score-pct').textContent = `${pct.toFixed(1)}%`;

  // Update gauge circle SVG stroke-dashoffset
  // Circumference = 2 * PI * r = 2 * 3.14159 * 48 = 301.6
  const circle = document.getElementById('gauge-progress-circle');
  const circumference = 301.6;
  const offset = circumference - (pct / 100) * circumference;
  circle.style.strokeDashoffset = offset;

  // Color according to risk tier
  const tierBadge = document.getElementById('risk-tier-badge');
  tierBadge.className = 'risk-tier-badge';

  if (pred.risk_tier === 'High Risk') {
    tierBadge.classList.add('badge-high');
    tierBadge.textContent = 'HIGH RISK (30-Day Readmission Alert)';
    circle.style.stroke = '#f43f5e';
  } else if (pred.risk_tier === 'Moderate Risk') {
    tierBadge.classList.add('badge-moderate');
    tierBadge.textContent = 'MODERATE RISK (Elevated Vulnerability)';
    circle.style.stroke = '#f59e0b';
  } else {
    tierBadge.classList.add('badge-low');
    tierBadge.textContent = 'LOW RISK (Standard Discharge Protocol)';
    circle.style.stroke = '#10b981';
  }

  // Threshold Comparison Box
  document.getElementById('thresh-tuned-status').innerHTML = pred.flagged_for_intervention_clinical
    ? '<span style="color: #fb7185; font-weight: 700;">🚨 FLAGGED FOR INTERVENTION</span>'
    : '<span style="color: #34d399; font-weight: 600;">STANDARD DISCHARGE</span>';

  document.getElementById('thresh-default-status').innerHTML = pred.flagged_default_0_5
    ? '<span style="color: #fb7185; font-weight: 700;">FLAGGED</span>'
    : '<span style="color: #94a3b8;">NOT FLAGGED (Missed by 0.50 Cutoff)</span>';

  document.getElementById('threshold-justification-text').textContent = pred.threshold_justification;
}

function renderShapFactors(explainData) {
  const container = document.getElementById('shap-factors-container');
  container.innerHTML = '';

  const factors = explainData.top_contributing_factors || [];
  if (factors.length === 0) {
    container.innerHTML = '<p style="color: #64748b; font-size: 12px;">No significant feature attributions detected.</p>';
    return;
  }

  // Find maximum absolute impact for bar scaling
  const maxImp = Math.max(...factors.map(f => Math.abs(f.impact_pct)), 1.0);

  factors.forEach(f => {
    const isPos = f.attribution > 0;
    const widthPct = Math.min((Math.abs(f.impact_pct) / maxImp) * 100, 100);

    const row = document.createElement('div');
    row.className = 'factor-item';
    row.innerHTML = `
      <div class="factor-name">
        <span style="color: ${isPos ? '#fb7185' : '#34d399'}; font-size: 14px;">${isPos ? '▲' : '▼'}</span>
        <span>${formatFeatureName(f.feature)}</span>
      </div>
      <div class="factor-bar-wrapper">
        <div class="factor-bar-track">
          <div class="${isPos ? 'factor-bar-fill-pos' : 'factor-bar-fill-neg'}" style="width: ${widthPct}%;"></div>
        </div>
        <div class="factor-impact-val" style="color: ${isPos ? '#fb7185' : '#34d399'};">
          ${isPos ? '+' : ''}${f.impact_pct.toFixed(1)}%
        </div>
      </div>
    `;
    container.appendChild(row);
  });
}

function formatFeatureName(name) {
  const map = {
    'number_inpatient': 'Prior Inpatient Admissions',
    'num_medications': 'Total Medication Volume',
    'time_in_hospital': 'Hospital Length of Stay',
    'has_prior_inpatient': 'Has Previous Hospitalization',
    'num_diabetes_med_changes': 'Diabetic Medication Changes',
    'insulin_dosage_change': 'Insulin Dose Titration (Up/Down)',
    'on_insulin': 'Active Insulin Prescription',
    'polypharmacy': 'Polypharmacy Flag (>=15 Meds)',
    'number_emergency': 'Prior Emergency Department Visits',
    'number_outpatient': 'Outpatient Clinic Visits',
    'age_approx': 'Patient Age Scale',
    'diag_1_category_Circulatory': 'Primary: Circulatory / Heart',
    'diag_1_category_Diabetes': 'Primary: Diabetes Complication',
    'diag_1_category_Respiratory': 'Primary: Respiratory / Pulmonary',
    'admission_type_name_Emergency': 'Admission Type: Emergency'
  };
  return map[name] || name.replace(/_/g, ' ');
}

// 4. Assistant Chat & Guardrail Testing
function initChat() {
  const sendBtn = document.getElementById('chat-send-btn');
  const chatInput = document.getElementById('chat-user-input');

  sendBtn.addEventListener('click', () => handleChatSubmit());
  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleChatSubmit();
  });

  // Prompt chips
  const chips = document.querySelectorAll('.prompt-chip[data-prompt]');
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      const p = chip.getAttribute('data-prompt');
      chatInput.value = p;
      handleChatSubmit();
    });
  });
}

async function handleChatSubmit() {
  const chatInput = document.getElementById('chat-user-input');
  const query = chatInput.value.trim();
  if (!query) return;

  appendChatBubble('user', query);
  chatInput.value = '';

  try {
    const res = await fetch('/api/assistant/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        encounter_id: currentEncounterId,
        user_query: query
      })
    });

    const data = await res.json();

    if (data.status === 'REFUSED') {
      appendChatBubble('refusal', data.content);
    } else {
      appendChatBubble('assistant', data.content, data.citations);
    }
  } catch (err) {
    appendChatBubble('refusal', 'Error connecting to clinical assistant service.');
  }
}

function appendChatBubble(type, text, citations = []) {
  const history = document.getElementById('chat-history');
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${type}`;

  // Format simple markdown
  let formatted = text
    .replace(/### (.*?)\n/g, '<h3 style="font-size: 14px; font-weight: 700; margin-bottom: 6px; color: #38bdf8;">$1</h3>')
    .replace(/#### (.*?)\n/g, '<h4 style="font-size: 13px; font-weight: 600; margin-top: 10px; margin-bottom: 4px; color: #f8fafc;">$1</h4>')
    .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
    .replace(/\*(.*?)\*/g, '<i>$1</i>')
    .replace(/• (.*?)\n/g, '<div style="margin-left: 8px; margin-bottom: 4px;">• $1</div>')
    .replace(/\n\n/g, '<br/>');

  bubble.innerHTML = formatted;

  // Render citations if present
  if (citations && citations.length > 0) {
    const citBox = document.createElement('div');
    citBox.style.marginTop = '12px';
    citBox.style.paddingTop = '8px';
    citBox.style.borderTop = '1px solid rgba(148, 163, 184, 0.15)';
    citBox.innerHTML = '<div style="font-size: 11px; font-weight: 700; color: #06b6d4; text-transform: uppercase; margin-bottom: 6px;">Mandatory Source Citations:</div>';

    citations.forEach(c => {
      const cCard = document.createElement('div');
      cCard.className = 'citation-card';
      cCard.innerHTML = `
        <div class="auth">${c.authority} — <i>${c.evidence_level}</i></div>
        <div style="font-size: 10.5px; color: #94a3b8; margin-top: 2px;">${c.citation}</div>
      `;
      citBox.appendChild(cCard);
    });
    bubble.appendChild(citBox);
  }

  history.appendChild(bubble);
  history.scrollTop = history.scrollHeight;
}

// 5. Clinical Quality Dashboard
async function loadDashboardAnalytics() {
  try {
    const res = await fetch('/api/dashboard/metrics');
    const data = await res.json();

    renderDepartmentChart(data.departments);
    renderDiagnosisTable(data.diagnoses);
    renderCohortDrilldown(data.high_risk_cohorts);
  } catch (err) {
    console.error("Error loading dashboard metrics:", err);
  }
}

function renderDepartmentChart(depts) {
  const container = document.getElementById('dept-chart-container');
  if (!container) return;
  container.innerHTML = '';

  const maxRate = Math.max(...depts.map(d => d.readm_rate_pct), 15.0);

  depts.forEach(d => {
    const barItem = document.createElement('div');
    barItem.className = 'chart-bar-item';
    const heightPct = (d.readm_rate_pct / maxRate) * 100;

    barItem.innerHTML = `
      <div class="chart-bar-rect" style="height: ${heightPct}%;">
        <div class="chart-bar-val">${d.readm_rate_pct}%</div>
      </div>
      <div class="chart-bar-lbl" title="${d.department}">${d.department}</div>
    `;
    container.appendChild(barItem);
  });
}

function renderDiagnosisTable(diagnoses) {
  const tbody = document.getElementById('diagnosis-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  diagnoses.forEach(d => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><b>${d.diagnosis_group}</b></td>
      <td>${d.encounters.toLocaleString()}</td>
      <td>${d.avg_stay_days} days</td>
      <td><span style="color: ${d.readm_rate_pct >= 12 ? '#fb7185' : '#38bdf8'}; font-weight: 600;">${d.readm_rate_pct}%</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderCohortDrilldown(cohorts) {
  const tbody = document.getElementById('cohort-drill-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  cohorts.forEach(c => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><b>${c.age_group}</b></td>
      <td>${c.gender}</td>
      <td><span class="badge-status-select" style="background: rgba(244, 63, 94, 0.15); color: #fb7185;">${c.prior_utilization}</span></td>
      <td>${c.cohort_count.toLocaleString()}</td>
      <td><b style="color: #fb7185;">${c.readm_rate_pct}%</b></td>
      <td>
        <button class="quick-btn" onclick="drillIntoCohort('${c.age_group}', '${c.gender}')">Inspect Cohort</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function drillIntoCohort(age, gender) {
  alert(`Drill-through active: Filtering high-risk triage queue for ${age} ${gender} patients.`);
  // Switch to triage worklist tab
  document.querySelector('.nav-tab-btn[data-tab="tab-triage"]').click();
}

// 6. Care Team High-Risk Triage Worklist
async function loadTriageWorklist() {
  try {
    const res = await fetch('/api/cohort/worklist?limit=15');
    const data = await res.json();
    const tbody = document.getElementById('triage-worklist-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    data.worklist.forEach(item => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><b>#${item.encounter_id}</b></td>
        <td>${item.age_group} (${item.gender[0]})</td>
        <td>${item.primary_diagnosis}</td>
        <td><span style="color: #fb7185; font-weight: 700;">${item.number_inpatient} visits</span></td>
        <td>${item.time_in_hospital} days</td>
        <td><span class="badge-high">${(item.estimated_risk_score * 100).toFixed(1)}%</span></td>
        <td>
          <select class="badge-status-select" onchange="updateTriageStatus(${item.encounter_id}, this.value)">
            <option value="Pending Action" ${item.triage_status === 'Pending Action' ? 'selected' : ''}>Pending Action</option>
            <option value="Contacted (48h)" ${item.triage_status === 'Contacted (48h)' ? 'selected' : ''}>Contacted (48h)</option>
            <option value="Care Plan Prepared" ${item.triage_status === 'Care Plan Prepared' ? 'selected' : ''}>Care Plan Prepared</option>
            <option value="Physician Approved" ${item.triage_status === 'Physician Approved' ? 'selected' : ''}>Physician Approved</option>
          </select>
        </td>
        <td>
          <button class="quick-btn" onclick="loadPatientEncounter(${item.encounter_id}); document.querySelector('.nav-tab-btn[data-tab=\\'tab-inspector\\']').click();">Inspect</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading triage worklist:", err);
  }
}

async function updateTriageStatus(encId, newStatus) {
  try {
    await fetch('/api/cohort/worklist/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ encounter_id: encId, status: newStatus })
    });
    console.log(`Updated triage status for encounter #${encId} to ${newStatus}`);
  } catch (err) {
    console.error("Failed to update triage status:", err);
  }
}

// 7. Health Economics & ROI Calculator
function initEconomicsCalculator() {
  const volSlider = document.getElementById('slider-volume');
  const costSlider = document.getElementById('slider-intervention-cost');
  if (!volSlider || !costSlider) return;

  const updateCalc = () => {
    const vol = parseInt(volSlider.value);
    const cost = parseInt(costSlider.value);

    document.getElementById('vol-display').textContent = `${vol.toLocaleString()} Discharges`;
    document.getElementById('cost-display').textContent = `$${cost} / Patient`;

    // Sensitivity at 0.18 threshold: 65.5% of readmissions flagged
    // Base rate: 11.39% -> 113.9 readmissions per 1,000 discharges
    const totalReadmissions = (vol * 0.1139);
    const flaggedReadmissions = totalReadmissions * 0.655;

    // Intervention rate: ~35% of total discharges flagged
    const totalInterventions = vol * 0.35;
    const totalInterventionCost = totalInterventions * cost;

    // Averted readmissions: 30% relative risk reduction (AHRQ Project RED trial standard)
    const avertedReadmissions = flaggedReadmissions * 0.30;
    const grossSavings = avertedReadmissions * 15200; // $15,200 CMS readmission cost
    const netSavings = grossSavings - totalInterventionCost;
    const roi = (netSavings / totalInterventionCost) * 100;

    document.getElementById('calc-averted-readmissions').textContent = `${Math.round(avertedReadmissions).toLocaleString()} Avoided`;
    document.getElementById('calc-program-cost').textContent = `$${Math.round(totalInterventionCost).toLocaleString()}`;
    document.getElementById('calc-net-savings').textContent = `$${Math.round(netSavings).toLocaleString()}`;
    document.getElementById('calc-roi').textContent = `${roi.toFixed(1)}%`;
  };

  volSlider.addEventListener('input', updateCalc);
  costSlider.addEventListener('input', updateCalc);
  updateCalc();
}

// 8. Extensions Data Loading
async function loadExtensionsData() {
  try {
    const res = await fetch('/api/extensions');
    const data = await res.json();

    // Fairness audit table
    if (data.fairness && data.fairness.metrics_by_attribute) {
      const raceAudit = data.fairness.metrics_by_attribute.race_clean || {};
      const fBody = document.getElementById('fairness-audit-body');
      if (fBody) {
        fBody.innerHTML = '';
        for (const [grp, m] of Object.entries(raceAudit)) {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><b>${grp}</b></td>
            <td>${m.sample_size.toLocaleString()}</td>
            <td>${(m.selection_rate * 100).toFixed(1)}%</td>
            <td><b style="color: #10b981;">${(m.tpr_sensitivity * 100).toFixed(1)}%</b></td>
            <td>${(m.fpr * 100).toFixed(1)}%</td>
            <td><span class="badge-status-select">${m.disparate_impact_ratio}</span></td>
          `;
          fBody.appendChild(tr);
        }
      }
      const fFindings = document.getElementById('fairness-findings-text');
      if (fFindings) fFindings.textContent = data.fairness.written_findings;
    }
  } catch (err) {
    console.error("Error loading extensions data:", err);
  }
}
