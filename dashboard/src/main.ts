import './style.css'

const API_BASE = "http://127.0.0.1:8000";

// DOM Elements
const startBtn = document.getElementById('start-bench') as HTMLButtonElement;
const resultsFeed = document.getElementById('results-feed') as HTMLDivElement;
const progressBar = document.getElementById('progress-bar') as HTMLDivElement;
const statusText = document.getElementById('status-text') as HTMLDivElement;
const apiStatus = document.getElementById('backend-status') as HTMLDivElement;
const sampleInput = document.getElementById('sample-size') as HTMLInputElement;
const summaryDashboard = document.getElementById('summary-dashboard') as HTMLDivElement;

const statAsr = document.getElementById('stat-asr') as HTMLDivElement;
const statImprove = document.getElementById('stat-improvement') as HTMLDivElement;
const statAcc = document.getElementById('stat-accuracy') as HTMLDivElement;

// New Elements
const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
const apiKeyNote = document.getElementById('api-key-note') as HTMLDivElement;
const useCustomCheck = document.getElementById('use-custom-prompt') as HTMLInputElement;
const customText = document.getElementById('custom-prompt-text') as HTMLTextAreaElement;
const datasetControls = document.getElementById('dataset-controls') as HTMLDivElement;

// Model Selection Handle
function updateApiKeyNote() {
  const value = modelSelect.value;
  if (value.startsWith('gemini')) {
    apiKeyNote.style.display = 'block';
    apiKeyNote.textContent = '* Ensure GEMINI_FLASH_LITE_API_KEY is set in .env';
    return;
  }
  if (value.startsWith('groq')) {
    apiKeyNote.style.display = 'block';
    apiKeyNote.textContent = '* Ensure GROQ_API_KEY is set in .env';
    return;
  }
  apiKeyNote.style.display = 'none';
}

modelSelect.addEventListener('change', updateApiKeyNote);
updateApiKeyNote();

useCustomCheck.addEventListener('change', (e: any) => {
  const checked = e.target.checked;
  customText.style.display = checked ? 'block' : 'none';
  datasetControls.style.display = checked ? 'none' : 'block';
});

// Check API Status
async function checkStatus() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      apiStatus.innerHTML = 'API: ONLINE';
      apiStatus.style.color = 'var(--success)';
      apiStatus.style.borderColor = 'var(--success)';
    }
  } catch (e) {
    apiStatus.innerHTML = 'API: OFFLINE';
    apiStatus.style.color = 'var(--danger)';
    apiStatus.style.borderColor = 'var(--danger)';
  }
}

checkStatus();
setInterval(checkStatus, 5000);

// Helper to create feed item
function addResult(data: any) {
  if (!data || !data.latest_result) return;

  const res = data.latest_result;
  const item = document.createElement('div');
  item.className = 'feed-item';

  let safetyTag = 'tag-info';
  let safetyText = 'BENIGN_OK';
  if (res.prompt_is_attack) {
    safetyTag = res.blocked_defense ? 'tag-safe' : 'tag-danger';
    safetyText = res.blocked_defense ? 'ATTACK_BLOCKED' : 'ATTACK_SUCCEEDED';
  } else if (res.over_refused_defense) {
    safetyTag = 'tag-danger';
    safetyText = 'OVER_REFUSED';
  }

  const modelAnswer = typeof res.response_defense === 'string' && res.response_defense.trim().length > 0
    ? res.response_defense
    : '[No model response captured]';

  item.innerHTML = `
    <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
      <span style="color: var(--secondary); font-size: 0.95rem; font-weight: bold;">[${res.source}]</span>
      <span class="metric-tag ${safetyTag}" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;">${safetyText}</span>
    </div>
    <div style="font-size: 1rem; margin-bottom: 0.8rem; color: #fff; line-height: 1.4; word-break: break-word;">
      ${res.prompt}
    </div>
    <div style="font-size: 0.92rem; margin-bottom: 0.7rem; color: var(--text-dim); opacity: 0.95; line-height: 1.5; border-left: 2px solid var(--secondary); padding-left: 0.7rem; white-space: pre-wrap; word-break: break-word;">
      <div style="font-size: 0.68rem; color: var(--secondary); letter-spacing: 0.08em; margin-bottom: 0.35rem; font-weight: bold;">MODEL_ANSWER</div>
      ${modelAnswer}
    </div>
    <div style="display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.8rem; border-top: 1px dashed var(--border); padding-top: 0.8rem;">
      <span class="metric-tag tag-info" style="font-size: 0.75rem;">Strategy: ${res.strategy}</span>
      <span class="metric-tag tag-info" style="font-size: 0.75rem;">Complex: ${res.complexity.toFixed(2)}</span>
      <span class="metric-tag tag-info" style="font-size: 0.75rem;">Consistent: ${res.consistency_score === 1 ? 'YES' : 'NO'}</span>
    </div>
  `;

  resultsFeed.prepend(item);
}

// Benchmark Execution
async function startBenchmark() {
  const sampleSize = parseInt(sampleInput.value);
  const isCustom = useCustomCheck.checked;
  const customPrompt = customText.value.trim();
  const [provider, ...modelParts] = modelSelect.value.split(':');
  const model = modelParts.join(':');

  if (isCustom && !customPrompt) {
    statusText.innerText = "ERROR:_ENTER_CUSTOM_PROMPT";
    return;
  }

  // Clear previous
  resultsFeed.innerHTML = "";
  summaryDashboard.style.display = "none";
  progressBar.style.width = "0%";
  statusText.innerText = `INITIATING_AUDIT_${model.toUpperCase()}...`;
  startBtn.disabled = true;
  startBtn.innerText = "AUDITING...";

  try {
    let url = `${API_BASE}/run_benchmark?sample_size=${sampleSize}&provider=${provider}&model=${model}`;
    if (isCustom) {
      url += `&custom_prompt=${encodeURIComponent(customPrompt)}`;
    }

    const response = await fetch(url, { method: "POST" });
    if (!response.body) return;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);

          if (data.status) statusText.innerText = data.status.toUpperCase();
          if (data.progress !== undefined) progressBar.style.width = `${data.progress * 100}%`;

          if (data.latest_result) {
            addResult(data);
          }

          if (data.done) {
            statusText.innerText = "AUDIT_COMPLETED";
            showSummary(data);
          }
        } catch (e) {
          console.error("Parse Error:", e);
        }
      }
    }
  } catch (e) {
    statusText.innerText = "ERROR:_LINK_FAILED";
    console.error("Fetch Error:", e);
  } finally {
    startBtn.disabled = false;
    startBtn.innerText = "INITIATE_BENCHMARK";
  }
}

function showSummary(data: any) {
  summaryDashboard.style.display = "block";
  const stats = data.stats;

  // ASR values are attack-only by design.
  statAsr.innerText = `${stats.asr_defense.toFixed(1)}%`;
  statImprove.innerText = `+${stats.improvement.toFixed(1)}%`;
  statAcc.innerText = `${(100 - stats.asr_defense).toFixed(1)}%`;
}

startBtn.addEventListener('click', startBenchmark);
