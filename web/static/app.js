"use strict";

const state = { apiKey: "", currentScan: null };
const byId = (id) => document.getElementById(id);

function appendTextList(target, rows) {
  target.replaceChildren();
  for (const row of rows || []) {
    const item = document.createElement("li");
    item.textContent = String(row);
    target.append(item);
  }
}

function renderExplanationSections(sections) {
  const root = byId("explanation-sections");
  root.replaceChildren();
  const labels = [
    ["overall_interpretation", "Overall interpretation"],
    ["observations", "Observed URL characteristics"],
    ["risk_increasing_signals", "Risk-increasing signals"],
    ["risk_reducing_signals", "Risk-reducing signals"],
    ["limitations", "Limitations"],
    ["recommended_actions", "Recommended next steps"],
  ];
  for (const [key, title] of labels) {
    const values = sections?.[key];
    if (!values || (Array.isArray(values) && values.length === 0)) continue;
    const heading = document.createElement("h4");
    heading.textContent = title;
    root.append(heading);
    if (Array.isArray(values)) {
      const list = document.createElement("ul");
      for (const value of values) {
        const item = document.createElement("li");
        item.textContent = String(value);
        list.append(item);
      }
      root.append(list);
    } else {
      const paragraph = document.createElement("p");
      paragraph.textContent = String(values);
      root.append(paragraph);
    }
  }
}

function renderEvidence(evidence) {
  const featureData = evidence?.features || {};
  const modelValues = featureData.model_input_values || {};
  const descriptions = featureData.descriptions || {};
  appendTextList(byId("model-features"), Object.entries(modelValues).map(([name, value]) =>
    `${descriptions[name] || name}: ${value}`));
  const supplemental = featureData.heuristic_only_observations || {};
  appendTextList(byId("supplemental-features"), Object.entries(supplemental).map(([name, value]) =>
    `${name.replaceAll("_", " ")}: ${typeof value === "object" ? JSON.stringify(value) : value}`));
  const signals = evidence?.heuristics?.signals || [];
  appendTextList(byId("evidence-signals"), signals.map((signal) =>
    `${signal.description} Observed value: ${signal.observed_value}; adds ${signal.score_addition} to the separate heuristic score.`));
  const attribution = evidence?.model_attribution || {};
  byId("attribution-note").textContent = attribution.method === "standardized_logistic_log_odds_contribution"
    ? "Per-instance logistic contributions are expressed in log-odds units."
    : "Exact per-instance model contributions are unavailable for this scoring method; global importance is not shown as a per-URL explanation.";
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Content-Type", "application/json");
  if (state.apiKey) headers.set("X-API-Key", state.apiKey);
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    if (response.status === 401) throw new Error("Access key is missing or incorrect.");
    if (response.status === 422) throw new Error("Check the URL and try again.");
    throw new Error("The scanner could not complete this request. Try again shortly.");
  }
  return response.json();
}

function showScan(scan) {
  state.currentScan = scan;
  byId("result-section").hidden = false;
  byId("risk-badge").textContent = scan.level + " RISK";
  byId("risk-badge").className = "risk-badge " + scan.level;
  byId("score-ring").className = "score-ring " + scan.level;
  byId("score-value").textContent = (Number(scan.score) * 100).toFixed(1) + "%";
  byId("result-level").textContent = scan.level + " risk estimate";
  byId("result-url").textContent = scan.url;
  byId("result-model").textContent = "Scoring method: " + scan.model;
  renderEvidence(scan.evidence);
  byId("explanation-sections").replaceChildren();
  byId("result-time").textContent = "Scanned " + new Date(scan.scanned_at).toLocaleString();
  const list = byId("reason-list");
  list.replaceChildren();
  for (const reason of scan.reasons || []) {
    const item = document.createElement("li");
    item.textContent = reason;
    list.append(item);
  }
  byId("no-reasons").hidden = Boolean((scan.reasons || []).length);
  byId("explanation").textContent = (scan.reasons || []).length
    ? "The scanner observed the URL characteristics listed above. They can be associated with phishing, but none proves malicious intent by itself. Verify the domain independently and do not enter sensitive information when uncertain."
    : "No notable lexical signals were returned. This does not guarantee that the destination is safe.";
  byId("explain-provider").textContent = "Deterministic explanation";
  byId("result-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadHistory() {
  const content = byId("history-content");
  content.replaceChildren();
  try {
    const scans = await api("/history?limit=10");
    if (!scans.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No scans yet. Your recent scans will appear here.";
      content.append(empty);
      return;
    }
    for (const scan of scans) {
      const row = document.createElement("div");
      row.className = "history-row";
      const url = document.createElement("a");
      url.className = "history-link";
      url.href = "#scan-title";
      url.textContent = scan.url;
      url.setAttribute("aria-label", "Scan again: " + scan.url);
      url.addEventListener("click", (event) => {
        event.preventDefault();
        byId("url-input").value = scan.url;
        byId("url-input").focus();
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
      const meta = document.createElement("span");
      meta.className = "history-meta";
      meta.textContent = (Number(scan.score) * 100).toFixed(1) + "% · " + new Date(scan.scanned_at).toLocaleDateString();
      const level = document.createElement("span");
      level.className = "history-level " + scan.level;
      level.textContent = scan.level;
      row.append(url, meta, level);
      content.append(row);
    }
  } catch (error) {
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = error.message;
    content.append(message);
  }
}

byId("scan-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = byId("url-input");
  const url = input.value.trim();
  const error = byId("scan-error");
  error.hidden = true;
  if (!url) {
    error.textContent = "Enter a URL to scan.";
    error.hidden = false;
    input.focus();
    return;
  }
  if (url.length > 2048) {
    error.textContent = "The URL must be no longer than 2048 characters.";
    error.hidden = false;
    return;
  }
  const button = byId("scan-button");
  button.disabled = true;
  button.firstElementChild.textContent = "Scanning…";
  try {
    const scan = await api("/scan", { method: "POST", body: JSON.stringify({ url }) });
    showScan(scan);
    await loadHistory();
  } catch (requestError) {
    error.textContent = requestError.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.firstElementChild.textContent = "Scan URL";
  }
});

byId("explain-button").addEventListener("click", async () => {
  if (!state.currentScan) return;
  const button = byId("explain-button");
  button.disabled = true;
  button.textContent = "Preparing…";
  try {
    const result = await api("/explain", {
      method: "POST",
      body: JSON.stringify({ score: state.currentScan.score, level: state.currentScan.level, reasons: state.currentScan.reasons, evidence: state.currentScan.evidence }),
    });
    byId("explanation").textContent = result.explanation;
    renderExplanationSections(result.sections);
    byId("explain-provider").textContent = result.provider === "gemini"
      ? "AI-generated explanation · not a security verdict"
      : "Deterministic explanation · AI unavailable";
  } catch (error) {
    byId("explain-provider").textContent = "Explanation unavailable";
    byId("explanation").textContent = "The explanation service could not respond. The scan score and reasons above are unchanged.";
  } finally {
    button.disabled = false;
    button.textContent = "Explain with AI";
  }
});

byId("refresh-history").addEventListener("click", loadHistory);

async function initialize() {
  try {
    const config = await api("/config");
    byId("api-key-wrap").hidden = !config.api_key_required;
    byId("api-key-input").addEventListener("input", (event) => { state.apiKey = event.target.value; });
  } catch (_) {
    // The scan form reports a useful message if service configuration is unavailable.
  }
  try {
    await api("/health");
    byId("health-dot").classList.add("online");
    byId("health-text").textContent = "Scanner service online";
  } catch (_) {
    byId("health-dot").classList.add("offline");
    byId("health-text").textContent = "Scanner service unavailable";
  }
  await loadHistory();
}

initialize();
