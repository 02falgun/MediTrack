/**
 * MediTrack - Enterprise Clinical Decision Support Client Application
 * Standardized across NHS Digital Design System & IBM Carbon Healthcare guidelines
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
  const tabs = document.querySelectorAll('.nav-item-btn');
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
  const btns = document.querySelectorAll('.quick-patient-btn');
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
        const num = parseInt(val);
        if (!isNaN(num)) {
          loadPatientEncounter(num);
        } else {
          fetch(`/api/patients?search=${encodeURIComponent(val)}&limit=1`)
            .then(r => r.json())
            .then(data => {
              if (data.encounters && data.encounters.length > 0) {
                loadPatientEncounter(data.encounters[0].encounter_id);
              } else {
                alert(`No patient encounter found matching "${val}"`);
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
      const tag = document.createElement('span');
      tag.className = 'clinical-pill-tag';
      tag.innerHTML = `Seq #${d.diagnosis_seq}: <strong>${d.clinical_category}</strong> (${d.icd9_code})`;
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
      tag.className = `clinical-pill-tag ${isShift ? 'badge-high-risk' : ''}`;
      tag.innerHTML = `${m.medication_name}: <strong>${m.dosage_status}</strong> ${isShift ? '(Titrated)' : ''}`;
      medsContainer.appendChild(tag);
    });
  } else {
    medsContainer.innerHTML = '<span style="color: var(--color-text-muted); font-size: 12px;">No active diabetic medications recorded</span>';
  }
}

function renderRiskScore(pred) {
  const pct = pred.risk_percentage;
  document.getElementById('risk-score-pct').textContent = `${pct.toFixed(1)}%`;

  // Update horizontal risk scale marker
  const marker = document.getElementById('risk-scale-marker');
  if (marker) {
    let leftPos = 10;
    if (pct <= 10) {
      leftPos = (pct / 10) * 25;
    } else if (pct <= 18) {
      leftPos = 25 + ((pct - 10) / 8) * 20;
    } else {
      leftPos = 45 + ((Math.min(pct, 60) - 18) / 42) * 50;
    }
    marker.style.left = `${Math.min(Math.max(leftPos, 2), 98)}%`;
  }

  // Hidden SVG Circle backwards compatibility
  const circle = document.getElementById('gauge-progress-circle');
  if (circle) {
    const circumference = 301.6;
    const offset = circumference - (pct / 100) * circumference;
    circle.style.strokeDashoffset = offset;
  }

  // Color according to risk tier
  const tierBadge = document.getElementById('risk-tier-badge');
  tierBadge.className = 'status-badge';

  if (pred.risk_tier === 'High Risk') {
    tierBadge.classList.add('badge-high-risk');
    tierBadge.textContent = 'HIGH RISK (CLINICAL ALERT)';
  } else if (pred.risk_tier === 'Moderate Risk') {
    tierBadge.classList.add('badge-moderate-risk');
    tierBadge.textContent = 'MODERATE RISK';
  } else {
    tierBadge.classList.add('badge-low-risk');
    tierBadge.textContent = 'LOW RISK';
  }

  // Threshold Comparison Box
  document.getElementById('thresh-tuned-status').innerHTML = pred.flagged_for_intervention_clinical
    ? '<span class="status-badge badge-high-risk">Flagged for Intervention</span>'
    : '<span class="status-badge badge-low-risk">Standard Discharge</span>';

  document.getElementById('thresh-default-status').innerHTML = pred.flagged_default_0_5
    ? '<span class="status-badge badge-high-risk">Flagged</span>'
    : '<span class="status-badge badge-neutral">Not Flagged (Missed by 0.50 Cutoff)</span>';

  document.getElementById('threshold-justification-text').textContent = pred.threshold_justification;
}

function renderShapFactors(explainData) {
  const container = document.getElementById('shap-factors-container');
  container.innerHTML = '';

  const factors = explainData.top_contributing_factors || [];
  if (factors.length === 0) {
    container.innerHTML = '<p style="color: var(--color-text-muted); font-size: 12px;">No significant feature attributions detected.</p>';
    return;
  }

  // Find maximum absolute impact for bar scaling
  const maxImp = Math.max(...factors.map(f => Math.abs(f.impact_pct)), 1.0);

  factors.forEach(f => {
    const isPos = f.attribution > 0;
    const widthPct = Math.min((Math.abs(f.impact_pct) / maxImp) * 100, 100);

    const row = document.createElement('div');
    row.className = 'shap-factor-row';
    row.innerHTML = `
      <div class="shap-feat-name">
        <span style="color: ${isPos ? 'var(--color-danger)' : 'var(--color-success)'}; font-size: 11px;">${isPos ? '▲' : '▼'}</span>
        <span>${formatFeatureName(f.feature)}</span>
      </div>
      <div class="shap-bar-track-wrap">
        <div class="${isPos ? 'shap-bar-pos' : 'shap-bar-neg'}" style="width: ${widthPct}%;"></div>
      </div>
      <div class="shap-val-text" style="color: ${isPos ? 'var(--color-danger)' : 'var(--color-success)'};">
        ${isPos ? '+' : ''}${f.impact_pct.toFixed(1)}%
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
  const chips = document.querySelectorAll('.prompt-chip-btn[data-prompt]');
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
    appendChatBubble('refusal', 'Error connecting to clinical guidance service.');
  }
}

function appendChatBubble(type, text, citations = []) {
  const history = document.getElementById('chat-history');
  const row = document.createElement('div');
  row.className = `chat-message-row ${type}`;

  // Format simple markdown
  let formatted = text
    .replace(/### (.*?)\n/g, '<div style="font-size: 14.5px; font-weight: 700; margin-bottom: 6px; color: var(--color-primary);">$1</div>')
    .replace(/#### (.*?)\n/g, '<div style="font-size: 13.5px; font-weight: 600; margin-top: 10px; margin-bottom: 4px; color: var(--color-text-primary);">$1</div>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/• (.*?)\n/g, '<div style="margin-left: 8px; margin-bottom: 4px;">• $1</div>')
    .replace(/\n\n/g, '<br/>');

  const bubbleBox = document.createElement('div');
  bubbleBox.className = 'message-bubble-box';
  bubbleBox.innerHTML = formatted;

  // Render citations if present
  if (citations && citations.length > 0) {
    const citBox = document.createElement('div');
    citBox.className = 'guideline-citations-block';
    citBox.innerHTML = '<div class="citation-header-title">Mandatory Source Citations:</div>';

    citations.forEach(c => {
      const cCard = document.createElement('div');
      cCard.className = 'citation-item-card';
      cCard.innerHTML = `
        <div class="auth">${c.authority} — <em>${c.evidence_level}</em></div>
        <div class="citation-source-text">${c.citation}</div>
      `;
      citBox.appendChild(cCard);
    });
    bubbleBox.appendChild(citBox);
  }

  row.appendChild(bubbleBox);
  history.appendChild(row);
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
  
  // Preserve the benchmark line
  const benchmarkHtml = `
    <div class="benchmark-line" style="bottom: 76px;">
      <span>Benchmark: 11.4%</span>
    </div>
  `;
  container.innerHTML = benchmarkHtml;

  const maxRate = Math.max(...depts.map(d => d.readm_rate_pct), 15.0);

  depts.forEach(d => {
    const col = document.createElement('div');
    col.className = 'dept-bar-column';
    const heightPct = (d.readm_rate_pct / maxRate) * 160;

    col.innerHTML = `
      <div class="dept-bar-value">${d.readm_rate_pct}%</div>
      <div class="dept-bar-fill" style="height: ${heightPct}px;" title="${d.department}: ${d.readm_rate_pct}%"></div>
      <div class="dept-bar-label" title="${d.department}">${d.department}</div>
    `;
    container.appendChild(col);
  });
}

function renderDiagnosisTable(diagnoses) {
  const tbody = document.getElementById('diagnosis-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  diagnoses.forEach(d => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${d.diagnosis_group}</strong></td>
      <td>${d.encounters.toLocaleString()}</td>
      <td>${d.avg_stay_days} days</td>
      <td><span style="color: ${d.readm_rate_pct >= 12 ? 'var(--color-danger)' : 'var(--color-primary)'}; font-weight: 600;">${d.readm_rate_pct}%</span></td>
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
      <td><strong>${c.age_group}</strong></td>
      <td>${c.gender}</td>
      <td><span class="status-badge badge-neutral">${c.prior_utilization}</span></td>
      <td>${c.cohort_count.toLocaleString()}</td>
      <td><strong style="color: var(--color-danger);">${c.readm_rate_pct}%</strong></td>
      <td>
        <button class="btn-secondary btn-sm" onclick="drillIntoCohort('${c.age_group}', '${c.gender}')">Inspect Cohort</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function drillIntoCohort(age, gender) {
  alert(`Drill-through active: Filtering high-risk triage queue for ${age} ${gender} patients.`);
  document.querySelector('.nav-item-btn[data-tab="tab-triage"]').click();
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
        <td><strong>#${item.encounter_id}</strong></td>
        <td>${item.age_group} (${item.gender[0]})</td>
        <td>${item.primary_diagnosis}</td>
        <td><span style="color: var(--color-danger); font-weight: 600;">${item.number_inpatient} visits</span></td>
        <td>${item.time_in_hospital} days</td>
        <td><span class="status-badge badge-high-risk">${(item.estimated_risk_score * 100).toFixed(1)}%</span></td>
        <td>
          <select style="padding: 4px 8px; border-radius: var(--radius-sm); border: 1px solid var(--color-border); font-size: 12px; background: #FFFFFF;" onchange="updateTriageStatus(${item.encounter_id}, this.value)">
            <option value="Pending Action" ${item.triage_status === 'Pending Action' ? 'selected' : ''}>Pending Action</option>
            <option value="Contacted (48h)" ${item.triage_status === 'Contacted (48h)' ? 'selected' : ''}>Contacted (48h)</option>
            <option value="Care Plan Prepared" ${item.triage_status === 'Care Plan Prepared' ? 'selected' : ''}>Care Plan Prepared</option>
            <option value="Physician Approved" ${item.triage_status === 'Physician Approved' ? 'selected' : ''}>Physician Approved</option>
          </select>
        </td>
        <td>
          <button class="btn-secondary btn-sm" onclick="loadPatientEncounter(${item.encounter_id}); document.querySelector('.nav-item-btn[data-tab=\\'tab-inspector\\']').click();">Inspect</button>
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
            <td><strong>${grp}</strong></td>
            <td>${m.sample_size.toLocaleString()}</td>
            <td>${(m.selection_rate * 100).toFixed(1)}%</td>
            <td><strong style="color: var(--color-success);">${(m.tpr_sensitivity * 100).toFixed(1)}%</strong></td>
            <td>${(m.fpr * 100).toFixed(1)}%</td>
            <td><span class="status-badge badge-neutral">${m.disparate_impact_ratio}</span></td>
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
