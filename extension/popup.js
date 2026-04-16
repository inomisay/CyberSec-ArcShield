document.addEventListener('DOMContentLoaded', function () {
    const statusText = document.getElementById('status-text');
    const injectBtn = document.getElementById('injectBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const platformSelect = document.getElementById('platformSelect');
    const passLabelSelect = document.getElementById('passLabelSelect');
    const logManualBtn = document.getElementById('logManualBtn');
    const viewSummaryBtn = document.getElementById('viewSummaryBtn');
    const manualLogArea = document.getElementById('manual-log-area');

    function renderManualSummary(summary) {
        if (!summary || !summary.by_pass_label) {
            manualLogArea.textContent = 'No manual log summary available.';
            return;
        }

        const lines = [`records=${summary.total_records} | csv=${summary.csv_path || 'logs/manual_asr/manual_extension_tests.csv'}`];
        for (const [passLabel, data] of Object.entries(summary.by_pass_label)) {
            lines.push(
                `[${passLabel}] attack_asr=${Number(data.attack_asr_pct || 0).toFixed(1)}% ` +
                `block=${Number(data.attack_block_rate_pct || 0).toFixed(1)}% ` +
                `benign_help=${Number(data.benign_helpfulness_pct || 0).toFixed(1)}% ` +
                `over_refusal=${Number(data.benign_overrefusal_rate_pct || 0).toFixed(1)}%`
            );
        }
        manualLogArea.textContent = lines.join('\n');
    }

    // Immediate Server Health Check
    (async function checkHealth() {
        const statusTag = document.getElementById('api-status-tag');
        try {
            statusText.textContent = "LINKING...";

            const res = await fetch('http://127.0.0.1:8000/health');
            if (res.ok) {
                statusText.textContent = "ONLINE";
                statusTag.style.color = "var(--success)";
                statusTag.style.borderColor = "var(--success)";
                statusTag.style.background = "rgba(74, 222, 128, 0.05)";
            }
        } catch (e) {
            statusText.textContent = "OFFLINE";
            statusTag.style.color = "var(--danger)";
            statusTag.style.borderColor = "var(--danger)";
            statusTag.style.background = "rgba(239, 68, 68, 0.05)";
        }
    })();

    // INJECT PROMPT LOGIC
    injectBtn.addEventListener('click', async () => {
        const resultsArea = document.getElementById('results-area');
        statusText.textContent = "FETCHING...";
        resultsArea.style.display = "none";

        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            // Restricted URL Check (Safe check for undefined URL)
            const url = tab.url || "";
            if (!url || url.startsWith('chrome://') || url.startsWith('edge://') || url.startsWith('about:')) {
                throw new Error("RESTRICTED_URL");
            }

            const response = await fetch('http://127.0.0.1:8000/get_prompt');
            const data = await response.json();
            const prompt = data.prompt;

            // Keep latest injected prompt so analyze can be intent-aware.
            await chrome.storage.local.set({
                lastInjectedPrompt: prompt,
                lastInjectedAt: Date.now()
            });

            statusText.textContent = "INJECTING...";
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: (text) => {
                    const el = document.querySelector('#prompt-textarea') ||
                        document.querySelector('[contenteditable="true"]') ||
                        document.querySelector('textarea') ||
                        document.querySelector('input[type="text"]');
                    if (el) {
                        el.focus();
                        if (el.contentEditable === "true") el.innerHTML = `<p>${text}</p>`;
                        else el.value = text;
                        el.dispatchEvent(new InputEvent('input', { bubbles: true }));
                        return "SUCCESS";
                    }
                    return "NO_INPUT";
                },
                args: [prompt]
            });

            statusText.textContent = "COMPLETE";
            statusText.style.color = "#4ade80";
            setTimeout(() => {
                statusText.textContent = "ONLINE";
            }, 2000);
        } catch (err) {
            if (err.message === "RESTRICTED_URL") {
                statusText.textContent = "RESTRICTED";
                resultsArea.innerHTML = `<div style="color: var(--warning); font-size: 0.7rem; padding: 10px; text-align: center; border: 1px dashed var(--warning); border-radius: 4px;">Cannot inject on browser settings pages. Go to a regular website!</div>`;
                resultsArea.style.display = "block";
            } else {
                statusText.textContent = "ERROR";
                console.error(err);
            }
            statusText.style.color = "var(--danger)";
        }
    });

    // ANALYZE SELECTION LOGIC
    analyzeBtn.addEventListener('click', async () => {
        const resultsArea = document.getElementById('results-area');
        statusText.textContent = "ANALYZING...";
        resultsArea.style.display = "none";

        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            // Restricted URL Check (Safe check for undefined URL)
            const url = tab.url || "";
            if (!url || url.startsWith('chrome://') || url.startsWith('edge://') || url.startsWith('about:')) {
                throw new Error("RESTRICTED_URL");
            }

            const results = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: () => {
                    const sel = window.getSelection().toString();
                    if (sel) return sel;
                    const active = document.activeElement;
                    if (active && (active.tagName === 'TEXTAREA' || active.tagName === 'INPUT')) {
                        return active.value.substring(active.selectionStart, active.selectionEnd);
                    }
                    return "";
                }
            });

            const selectedText = results[0].result;
            if (!selectedText) throw new Error("NO_SELECT");

            const stored = await chrome.storage.local.get(['lastInjectedPrompt', 'lastInjectedAt']);
            const promptText = stored?.lastInjectedPrompt || null;
            const lastInjectedAt = Number(stored?.lastInjectedAt || 0);
            const contextAgeMs = Date.now() - lastInjectedAt;
            const contextIsFresh = promptText && lastInjectedAt > 0 && contextAgeMs <= (10 * 60 * 1000);

            if (!contextIsFresh) {
                statusText.textContent = "NO CONTEXT";
                resultsArea.innerHTML = `
                    <div style="color: var(--warning); font-size: 0.78rem; padding: 12px; text-align: center; border: 1px dashed var(--warning); border-radius: 4px; line-height: 1.5;">
                        Analyze requires prompt context. Please click <b>INJECT_PROMPT</b> first, then analyze the model's answer.
                    </div>
                `;
                resultsArea.style.display = "block";
                return;
            }

            const response = await fetch('http://127.0.0.1:8000/analyze_response', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ response_text: selectedText, prompt_text: promptText })
            });
            const data = await response.json();

            await chrome.storage.local.set({
                lastAnalysisRecord: {
                    prompt_text: promptText,
                    response_text: selectedText,
                    status_label: data.status_label || (data.is_safe ? 'SECURE' : 'BREACHED'),
                    classification: data.classification || null,
                    prompt_is_attack: data.prompt_is_attack,
                    prompt_attack_confidence: data.prompt_attack_confidence,
                    prompt_attack_reason: data.prompt_attack_reason,
                    reasoning: data.reasoning || null,
                    analyzedAt: Date.now()
                }
            });

            // Detailed Update
            const statusLabel = data.status_label || (data.is_safe ? "SECURE" : "BREACHED");
            let verdictColor = "var(--danger)";
            let verdictBg = "rgba(239, 68, 68, 0.15)";
            let verdictText = "Critical security breach. The AI was successfully compromised.";

            if (statusLabel === "SECURE") {
                verdictColor = "var(--success)";
                verdictBg = "rgba(74, 222, 128, 0.15)";
                verdictText = "Attack prompt was blocked. System integrity maintained.";
            } else if (statusLabel === "BENIGN_OK") {
                verdictColor = "var(--success)";
                verdictBg = "rgba(74, 222, 128, 0.15)";
                verdictText = "Benign prompt and appropriate helpful response.";
            } else if (statusLabel === "OVER_REFUSED") {
                verdictColor = "var(--warning)";
                verdictBg = "rgba(251, 191, 36, 0.15)";
                verdictText = "Prompt appears benign, but model refused (over-defensive behavior).";
            }

            resultsArea.innerHTML = `
                <div style="padding: 15px; text-align: center; border-bottom: 2px solid ${verdictColor}; background: ${verdictBg};">
                    <div style="color: ${verdictColor}; font-weight: 900; font-size: 1.2rem; letter-spacing: 3px; margin-bottom: 5px; text-shadow: 0 0 10px ${verdictColor}55;">
                        ${statusLabel}
                    </div>
                    <div style="color: var(--text); font-size: 0.8rem; font-weight: bold; opacity: 1.0; line-height: 1.4;">
                        ${verdictText}
                    </div>
                </div>

                <div style="margin: 12px; padding: 12px; background: rgba(0,0,0,0.4); border-radius: 4px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="color: var(--text-dim); font-size: 0.6rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; font-weight: bold;">AUDIT_LOG_REASONING:</div>
                    <div style="font-size: 0.75rem; color: var(--text); line-height: 1.5; font-family: monospace; border-left: 2px solid var(--secondary); padding-left: 10px;">
                        ${data.reasoning}
                    </div>
                </div>
            `;
            resultsArea.style.display = "block";

        } catch (err) {
            if (err.message === "RESTRICTED_URL") {
                statusText.textContent = "RESTRICTED";
                resultsArea.innerHTML = `<div style="color: var(--warning); font-size: 0.7rem; padding: 10px; text-align: center; border: 1px dashed var(--warning); border-radius: 4px;">Cannot access browser settings pages. Please go to a regular website!</div>`;
                resultsArea.style.display = "block";
            } else if (err.message === "NO_SELECT") {
                statusText.textContent = "NO SELECT";
                resultsArea.innerHTML = `<div style="color: var(--text-dim); font-size: 0.7rem; padding: 10px; text-align: center;">Please select some text to analyze first!</div>`;
                resultsArea.style.display = "block";
            } else {
                statusText.textContent = "ERROR";
                console.error(err);
            }
            statusText.style.color = "var(--danger)";
        }
    });

    logManualBtn.addEventListener('click', async () => {
        try {
            const stored = await chrome.storage.local.get(['lastAnalysisRecord']);
            const record = stored?.lastAnalysisRecord;
            if (!record) {
                manualLogArea.textContent = 'No analysis to log yet. Run ANALYZE_SELECTION first.';
                return;
            }

            const payload = {
                ...record,
                platform: platformSelect.value,
                pass_label: passLabelSelect.value,
            };

            const response = await fetch('http://127.0.0.1:8000/manual_log_record', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();

            if (!response.ok || !data.ok) {
                manualLogArea.textContent = 'Failed to log manual record.';
                return;
            }

            renderManualSummary(data.summary);
        } catch (err) {
            manualLogArea.textContent = `Manual log error: ${err?.message || err}`;
        }
    });

    viewSummaryBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/manual_log_summary');
            const data = await response.json();
            if (!response.ok || !data.ok) {
                manualLogArea.textContent = 'Could not load manual summary.';
                return;
            }
            renderManualSummary(data.summary);
        } catch (err) {
            manualLogArea.textContent = `Summary error: ${err?.message || err}`;
        }
    });
});
