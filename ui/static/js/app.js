const form = document.getElementById("predictionForm");
const urlInput = document.getElementById("urlInput");
const resultBox = document.getElementById("resultBox");
const modelStatus = document.getElementById("modelStatus");
const datasetTable = document.getElementById("datasetTable");
const refreshDataset = document.getElementById("refreshDataset");

const totalSamples = document.getElementById("totalSamples");
const phishingSamples = document.getElementById("phishingSamples");
const legitimateSamples = document.getElementById("legitimateSamples");
const modelName = document.getElementById("modelName");

async function loadSummary() {
  try {
    const response = await fetch("/api/summary");
    if (!response.ok) throw new Error("Summary endpoint unavailable");
    const data = await response.json();
    totalSamples.textContent = data.summary.total;
    phishingSamples.textContent = data.summary.phishing;
    legitimateSamples.textContent = data.summary.legitimate;
    modelName.textContent = data.metadata.model_name || "Trained ML model";
  } catch (error) {
    // Keep default values if the live service is unavailable.
  }
}

async function checkModelStatus() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();

    if (data.model_exists) {
      modelStatus.textContent = "Model ready";
      modelStatus.classList.add("online");
      modelStatus.classList.remove("offline");
    } else {
      modelStatus.textContent = "Train model first";
      modelStatus.classList.add("offline");
      modelStatus.classList.remove("online");
    }
  } catch (error) {
    modelStatus.textContent = "Model unavailable";
    modelStatus.classList.add("offline");
  }
}

function formatValue(value) {
  if (value === 1 || value === true) return "Yes";
  if (value === 0 || value === false) return "No";
  if (value === null || value === undefined || value === "") return "N/A";
  return String(value);
}

function renderPrediction(data) {
  const labelClass = data.label.toLowerCase();
  const confidence = data.confidence === null ? "N/A" : `${data.confidence}%`;
  const features = data.features || {};

  resultBox.className = "result-box";
  resultBox.innerHTML = `
    <div class="prediction-result">
      <div class="result-top">
        <div>
          <span class="prediction-label ${labelClass}">${escapeHtml(data.label)}</span>
          <h2>Model Prediction Result</h2>
          <p class="form-help">${escapeHtml(data.input_url)}</p>
          <p class="form-help">Model used: <strong>${escapeHtml(data.model_name)}</strong></p>
        </div>
        <div class="confidence-box">
          <strong>${confidence}</strong>
          <span>Confidence</span>
        </div>
      </div>

      <div class="feature-grid">
        ${featureItem("HTTPS", formatValue(features.uses_https))}
        ${featureItem("URL Length", features.url_length)}
        ${featureItem("Hostname Length", features.hostname_length)}
        ${featureItem("Path Length", features.path_length)}
        ${featureItem("Suspicious Words", features.suspicious_word_count)}
        ${featureItem("IP Address Used", formatValue(features.has_ip_address))}
        ${featureItem("@ Symbols", features.num_at_symbols)}
        ${featureItem("Dots", features.num_dots)}
        ${featureItem("Hyphens", features.num_hyphens)}
      </div>
    </div>
  `;
}

function featureItem(label, value) {
  return `
    <div class="feature-item">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(formatValue(value))}</strong>
    </div>
  `;
}

function heuristicFallback(url) {
  const lower = url.toLowerCase();
  const suspiciousWords = ["login", "verify", "update", "secure", "account", "bank", "paypal", "password", "confirm", "billing", "gift", "free", "locked", "alert", "reset", "claim"];
  let score = 0;
  const features = {
    uses_https: lower.startsWith("https://") ? 1 : 0,
    url_length: url.length,
    hostname_length: "N/A",
    path_length: "N/A",
    suspicious_word_count: suspiciousWords.filter(word => lower.includes(word)).length,
    has_ip_address: /\b(?:\d{1,3}\.){3}\d{1,3}\b/.test(lower) ? 1 : 0,
    num_at_symbols: (url.match(/@/g) || []).length,
    num_dots: (url.match(/\./g) || []).length,
    num_hyphens: (url.match(/-/g) || []).length,
  };
  if (!features.uses_https) score += 1;
  if (features.has_ip_address) score += 2;
  if (features.suspicious_word_count >= 2) score += 2;
  if (features.num_hyphens >= 2) score += 1;
  if (features.url_length > 75) score += 1;
  if (features.num_dots >= 4) score += 1;
  return {
    input_url: url,
    label: score >= 3 ? "Phishing" : "Legitimate",
    confidence: null,
    model_name: "Rule-based fallback",
    features
  };
}

async function handlePrediction(event) {
  event.preventDefault();

  const url = urlInput.value.trim();
  if (!url) return;

  resultBox.className = "result-box loading";
  resultBox.innerHTML = `
    <div class="empty-state">
      <strong>Analysing URL...</strong>
      <p>Extracting URL features and generating the model prediction.</p>
    </div>
  `;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    renderPrediction(data);
  } catch (error) {
    // Helpful fallback if the live prediction service is unavailable.
    renderPrediction(heuristicFallback(url));
  }
}

async function loadDataset() {
  try {
    const response = await fetch("/api/dataset");
    const data = await response.json();
    if (!Array.isArray(data) || data.length === 0) throw new Error("No dataset records found");
    datasetTable.innerHTML = data.map(row => `
      <tr>
        <td>#${row.id}</td>
        <td class="url-cell" title="${escapeHtml(row.url)}">${escapeHtml(row.url)}</td>
        <td><span class="label-badge ${row.label.toLowerCase()}">${row.label}</span></td>
        <td>${row.length}</td>
        <td>${row.https ? "Yes" : "No"}</td>
      </tr>
    `).join("");
  } catch (error) {
    // Keep the static sample table when backend is not running.
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

form.addEventListener("submit", handlePrediction);
refreshDataset.addEventListener("click", loadDataset);

loadSummary();
checkModelStatus();
loadDataset();
