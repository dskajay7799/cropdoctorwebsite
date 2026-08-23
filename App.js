/* ==========================================================================
   Crop Doctor — Frontend logic
   Talks to the Flask backend (app.py) for real MobileNetV2 inference and
   Neon-backed history. Falls back to browser localStorage for history if
   the backend/database isn't reachable. Never fabricates a diagnosis.
   ========================================================================== */

const DEFAULT_API_URL = "https://cropdoctorwebsite.onrender.com";

const state = {
  apiUrl: localStorage.getItem("cd_api_url") || DEFAULT_API_URL,
  selectedCrop: null,
  selectedFile: null,
  lastResult: null,
  history: [],
};

const ALL_CROPS = [
  { id: "rice", label: "Rice", emoji: "🌾", model_supported: false },
  { id: "wheat", label: "Wheat", emoji: "🌿", model_supported: false },
  { id: "maize", label: "Maize", emoji: "🌽", model_supported: true },
  { id: "tomato", label: "Tomato", emoji: "🍅", model_supported: true },
  { id: "potato", label: "Potato", emoji: "🥔", model_supported: true },
  { id: "apple", label: "Apple", emoji: "🍎", model_supported: true },
  { id: "grape", label: "Grape", emoji: "🍇", model_supported: true },
  { id: "pepper", label: "Pepper", emoji: "🌶️", model_supported: true },
  { id: "soybean", label: "Soybean", emoji: "🫘", model_supported: false },
  { id: "banana", label: "Banana", emoji: "🍌", model_supported: false },
  { id: "mango", label: "Mango", emoji: "🥭", model_supported: false },
  { id: "groundnut", label: "Groundnut", emoji: "🥜", model_supported: false },
  { id: "onion", label: "Onion", emoji: "🧅", model_supported: false },
];

/* ---------------------------- Navigation ---------------------------- */
function goto(tabName) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.getElementById(`tab-${tabName}`).classList.add("active");
  const tabBtn = document.querySelector(`.tab[data-tab="${tabName}"]`);
  if (tabBtn) tabBtn.classList.add("active");
  document.getElementById("nav-tabs").classList.remove("open");
}

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => goto(btn.dataset.tab));
});
document.querySelectorAll("[data-goto]").forEach(btn => {
  btn.addEventListener("click", () => goto(btn.dataset.goto));
});
document.getElementById("menu-toggle").addEventListener("click", () => {
  document.getElementById("nav-tabs").classList.toggle("open");
});

/* ---------------------------- Crop selection ---------------------------- */
function renderCropGrid() {
  const grid = document.getElementById("crop-grid");
  grid.innerHTML = "";
  ALL_CROPS.forEach(crop => {
    const tile = document.createElement("div");
    tile.className = "crop-tile" + (crop.model_supported ? "" : " unsupported");
    tile.innerHTML = `<span class="emoji">${crop.emoji}</span>${crop.label}` +
      (crop.model_supported ? "" : `<span class="badge">Coming soon</span>`);
    tile.addEventListener("click", () => {
      document.querySelectorAll(".crop-tile").forEach(t => t.classList.remove("selected"));
      tile.classList.add("selected");
      state.selectedCrop = crop.id;
      updateAnalyzeButton();
    });
    grid.appendChild(tile);
  });
}
renderCropGrid();

/* ---------------------------- Image upload ---------------------------- */
const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const uploadPlaceholder = document.getElementById("upload-placeholder");

document.getElementById("btn-choose-file").addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});
document.getElementById("btn-camera").addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.setAttribute("capture", "environment");
  fileInput.click();
});
uploadArea.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
});

["dragover", "dragenter"].forEach(evt =>
  uploadArea.addEventListener(evt, e => { e.preventDefault(); uploadArea.classList.add("dragover"); })
);
["dragleave", "drop"].forEach(evt =>
  uploadArea.addEventListener(evt, e => { e.preventDefault(); uploadArea.classList.remove("dragover"); })
);
uploadArea.addEventListener("drop", e => {
  if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please choose an image file.");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    alert("That image is too large. Please choose a photo under 12MB.");
    return;
  }
  state.selectedFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    previewImg.src = reader.result;
    previewImg.hidden = false;
    uploadPlaceholder.hidden = true;
  };
  reader.readAsDataURL(file);
  updateAnalyzeButton();
}

function updateAnalyzeButton() {
  document.getElementById("btn-analyze").disabled = !(state.selectedCrop && state.selectedFile);
}

/* ---------------------------- Analyze ---------------------------- */
document.getElementById("btn-analyze").addEventListener("click", runAnalysis);

async function runAnalysis() {
  const resultStep = document.getElementById("result-step");
  const loading = document.getElementById("analysis-loading");
  const resultBox = document.getElementById("analysis-result");
  resultStep.hidden = false;
  loading.hidden = false;
  resultBox.innerHTML = "";

  const formData = new FormData();
  formData.append("crop", state.selectedCrop);
  formData.append("image", state.selectedFile);

  try {
    const res = await fetch(`${state.apiUrl}/api/analyze`, { method: "POST", body: formData });
    const data = await res.json();
    loading.hidden = true;
    renderResult(data);
    if (data.status === "success") {
      state.lastResult = data;
      await saveHistoryRecord(data);
      updateDashboard();
      updateIntelligence(data);
    }
  } catch (err) {
    loading.hidden = true;
    resultBox.innerHTML = `<div class="error-box"><strong>Couldn't reach the AI service.</strong><br>
      Check your internet connection, or set the correct Backend API URL in Settings.</div>`;
  }
}

function renderResult(data) {
  const box = document.getElementById("analysis-result");

  if (data.status === "crop_mismatch") {
    box.innerHTML = `<div class="notice-box">
      <strong>Image does not appear to match the selected crop.</strong><br>
      Please upload a photo that actually shows the crop you selected.
    </div>`;
    return;
  }
  if (data.status === "crop_not_supported") {
    box.innerHTML = `<div class="notice-box">
      <strong>This crop isn't supported by the trained model yet.</strong><br>
      Currently supported crops: Maize, Tomato, Potato, Apple, Grape, Pepper.
    </div>`;
    return;
  }
  if (data.status === "image_unreadable") {
    box.innerHTML = `<div class="error-box">That image couldn't be read. Please try a clearer photo.</div>`;
    return;
  }
  if (data.status === "model_unavailable") {
    box.innerHTML = `<div class="error-box"><strong>AI model is being connected.</strong><br>Please try again shortly.</div>`;
    return;
  }
  if (data.status !== "success") {
    box.innerHTML = `<div class="error-box">Something went wrong. Please try again.</div>`;
    return;
  }

  const pillClass = data.is_healthy ? "healthy" : (data.confidence_tier === "low" ? "warn" : "diseased");
  const pillText = data.is_healthy ? "Healthy" : data.disease;

  box.innerHTML = `
    <div class="result-card">
      <div class="result-header">
        <div>
          <h3 style="margin:0">${capitalize(data.crop)}</h3>
          <span class="status-pill ${pillClass}">${pillText}</span>
        </div>
        <div style="text-align:right">
          <div><strong>${data.confidence}%</strong> confidence</div>
          <div class="muted">Severity: ${data.severity}</div>
        </div>
      </div>
      <div class="confidence-bar-track"><div class="confidence-bar-fill" style="width:${data.confidence}%"></div></div>

      ${data.warning ? `<div class="notice-box">${data.warning}</div>` : ""}

      <div class="result-section">
        <h4>Main Symptoms</h4>
        <ul>${data.symptoms.map(s => `<li>${s}</li>`).join("")}</ul>
      </div>
      <div class="result-section">
        <h4>Recommended Actions</h4>
        <ul>${data.recommendations.map(s => `<li>${s}</li>`).join("")}</ul>
      </div>
      <div class="result-section">
        <h4>Prevention</h4>
        <ul>${data.prevention.map(s => `<li>${s}</li>`).join("")}</ul>
      </div>
      <div class="result-section learn-links">
        <h4>Learn More</h4>
        ${learnMoreLinks(data.crop, data.disease)}
      </div>
      <div class="result-section">
        <button class="btn btn-secondary" data-goto="assistant" id="btn-ask-ai">Ask AI About This</button>
      </div>
    </div>
  `;
  document.getElementById("btn-ask-ai").addEventListener("click", () => {
    goto("assistant");
    seedAssistantWithDiagnosis(data);
  });
}

function learnMoreLinks(crop, disease) {
  const cropWiki = `https://en.wikipedia.org/wiki/${encodeURIComponent(capitalize(crop))}`;
  const diseaseWiki = `https://en.wikipedia.org/wiki/${encodeURIComponent(disease.replace(/\s+/g, "_"))}`;
  return `<a href="${cropWiki}" target="_blank" rel="noopener">${capitalize(crop)} on Wikipedia ↗</a>
          <a href="${diseaseWiki}" target="_blank" rel="noopener">${disease} on Wikipedia ↗</a>`;
}

function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

/* ---------------------------- History (backend + localStorage fallback) ---------------------------- */
async function saveHistoryRecord(data) {
  const record = {
    crop: data.crop,
    disease: data.disease,
    confidence: data.confidence,
    severity: data.severity,
    status: data.status,
  };
  try {
    const res = await fetch(`${state.apiUrl}/api/history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
    });
    const saved = await res.json();
    state.history.unshift(saved);
  } catch {
    record.id = "local-" + Date.now();
    record.created_at = new Date().toISOString();
    state.history.unshift(record);
    persistLocalHistory();
  }
  renderHistory();
}

function persistLocalHistory() {
  localStorage.setItem("cd_history_fallback", JSON.stringify(state.history.filter(h => String(h.id).startsWith("local-"))));
}

async function loadHistory() {
  try {
    const res = await fetch(`${state.apiUrl}/api/history`);
    state.history = await res.json();
  } catch {
    state.history = JSON.parse(localStorage.getItem("cd_history_fallback") || "[]");
  }
  renderHistory();
  updateDashboard();
}

function renderHistory() {
  const list = document.getElementById("history-list");
  if (!state.history.length) {
    list.innerHTML = `<p class="muted">No analyses yet.</p>`;
  } else {
    list.innerHTML = state.history.map(h => `
      <div class="history-item">
        <div>
          <strong>${capitalize(h.crop)} — ${h.disease}</strong>
          <div class="meta">${new Date(h.created_at).toLocaleString()} · ${h.confidence}% confidence · ${h.severity} severity</div>
        </div>
        <button data-id="${h.id}" class="btn-delete-history">Delete</button>
      </div>
    `).join("");
    list.querySelectorAll(".btn-delete-history").forEach(b => {
      b.addEventListener("click", () => deleteHistoryItem(b.dataset.id));
    });
  }
  renderHistoryChart();
}

async function deleteHistoryItem(id) {
  try {
    await fetch(`${state.apiUrl}/api/history/${id}`, { method: "DELETE" });
  } catch { /* fall through to local removal */ }
  state.history = state.history.filter(h => String(h.id) !== String(id));
  persistLocalHistory();
  renderHistory();
  updateDashboard();
}

document.getElementById("btn-clear-history").addEventListener("click", async () => {
  if (!confirm("Clear all analysis history?")) return;
  try {
    await fetch(`${state.apiUrl}/api/history`, { method: "DELETE" });
  } catch { /* ignore */ }
  state.history = [];
  localStorage.removeItem("cd_history_fallback");
  renderHistory();
  updateDashboard();
});

/* ---------------------------- Dashboard ---------------------------- */
let confidenceChart, historyChart;

function updateDashboard() {
  const latestBox = document.getElementById("dash-latest");
  if (state.history.length) {
    const h = state.history[0];
    latestBox.innerHTML = `<strong>${capitalize(h.crop)} — ${h.disease}</strong><br>
      <span class="muted">${h.confidence}% confidence · ${h.severity} severity</span>`;
  } else {
    latestBox.textContent = "No analysis yet. Analyze a crop to get started.";
  }
  document.getElementById("dash-history-count").textContent = `${state.history.length} analyses saved`;
  renderConfidenceChart();
}

function renderConfidenceChart() {
  const ctx = document.getElementById("chart-confidence");
  const h = state.history[0];
  const healthy = h ? (h.disease.toLowerCase() === "healthy" ? h.confidence : 100 - h.confidence) : 50;
  const disease = 100 - healthy;
  if (confidenceChart) confidenceChart.destroy();
  confidenceChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Healthy probability", "Disease probability"],
      datasets: [{ data: [healthy, disease], backgroundColor: ["#2E7D4F", "#c0392b"] }],
    },
    options: { plugins: { legend: { position: "bottom" } } },
  });
}

function renderHistoryChart() {
  const ctx = document.getElementById("chart-history");
  if (!ctx) return;
  const recent = [...state.history].reverse().slice(-15);
  if (historyChart) historyChart.destroy();
  historyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: recent.map(h => new Date(h.created_at).toLocaleDateString()),
      datasets: [{
        label: "Confidence over time",
        data: recent.map(h => h.confidence),
        borderColor: "#2E7D4F",
        backgroundColor: "rgba(46,125,79,0.15)",
        tension: 0.3,
        fill: true,
      }],
    },
    options: { scales: { y: { min: 0, max: 100 } } },
  });
}

/* ---------------------------- Crop Intelligence ---------------------------- */
function updateIntelligence(data) {
  document.getElementById("intel-diagnosis").innerHTML =
    `<strong>${capitalize(data.crop)} — ${data.disease}</strong><br>
     Confidence: ${data.confidence}% · Severity: ${data.severity}`;
  document.getElementById("intel-links").innerHTML = learnMoreLinks(data.crop, data.disease);
  fetchWeatherRisk();
}

function fetchWeatherRisk() {
  const box = document.getElementById("intel-weather");
  const dashBox = document.getElementById("dash-weather");
  if (!navigator.geolocation) {
    box.textContent = "Weather intelligence unavailable — diagnosis is still available offline.";
    dashBox.textContent = "Location unavailable.";
    return;
  }
  navigator.geolocation.getCurrentPosition(async pos => {
    try {
      const { latitude, longitude } = pos.coords;
      const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,precipitation`);
      const w = await res.json();
      const c = w.current;
      const humidity = c.relative_humidity_2m;
      const risk = humidity > 80 ? "High (humid conditions favor fungal spread)" : humidity > 55 ? "Moderate" : "Low";
      const text = `${c.temperature_2m}°C, ${humidity}% humidity, ${c.precipitation}mm precipitation. Disease risk: <strong>${risk}</strong>`;
      box.innerHTML = text;
      dashBox.innerHTML = text;
    } catch {
      box.textContent = "Weather intelligence unavailable — diagnosis is still available offline.";
      dashBox.textContent = "Weather unavailable.";
    }
  }, () => {
    box.textContent = "Weather intelligence unavailable — diagnosis is still available offline.";
    dashBox.textContent = "Location permission not granted.";
  });
}
fetchWeatherRisk();

/* ---------------------------- AI Assistant (Groq via backend, browser Web Speech) ---------------------------- */
const chatBox = document.getElementById("chat-box");
let diagnosisContext = null;

function addChatMsg(role, text) {
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function seedAssistantWithDiagnosis(data) {
  diagnosisContext = data;
  addChatMsg("bot", `Here's what I found: ${capitalize(data.crop)} — ${data.disease} (${data.confidence}% confidence, ${data.severity} severity). Ask me anything about it.`);
}

document.getElementById("btn-send-chat").addEventListener("click", sendChat);
document.getElementById("chat-input").addEventListener("keydown", e => { if (e.key === "Enter") sendChat(); });

async function sendChat() {
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;
  addChatMsg("user", question);
  input.value = "";

  try {
    const res = await fetch(`${state.apiUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        context: diagnosisContext,
      }),
    });
    const data = await res.json();
    if (data.status === "success") {
      addChatMsg("bot", data.answer);
      speak(data.answer);
    } else {
      addChatMsg("bot", data.message || offlineAssistantAnswer(question));
    }
  } catch {
    addChatMsg("bot", "Couldn't reach the AI Assistant right now. Check your internet connection, or the Backend API URL in Settings.");
  }
}

function offlineAssistantAnswer(question) {
  if (!diagnosisContext) {
    return "Please analyze a crop image first — then I can explain the diagnosis for you.";
  }
  const q = question.toLowerCase();
  if (q.includes("why")) return `${diagnosisContext.disease} typically develops from environmental conditions like humidity and moisture on the leaf surface. See the Prevention section in your results for how to reduce risk.`;
  if (q.includes("prevent")) return `Prevention tips: ${diagnosisContext.prevention.join(" ")}`;
  if (q.includes("symptom")) return `Symptoms to watch for: ${diagnosisContext.symptoms.join(" ")}`;
  if (q.includes("monitor")) return `Keep checking affected leaves every few days, and watch nearby healthy plants for early symptoms.`;
  return `Recommended actions: ${diagnosisContext.recommendations.join(" ")}`;
}

/* ---------- Voice (Web Speech API, with graceful fallback) ---------- */
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.onresult = e => {
    document.getElementById("chat-input").value = e.results[0][0].transcript;
    sendChat();
  };
} else {
  document.getElementById("voice-fallback").hidden = false;
  document.getElementById("btn-mic").disabled = true;
}

document.getElementById("btn-mic").addEventListener("click", () => {
  if (recognition) recognition.start();
});

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const utter = new SpeechSynthesisUtterance(text);
  const stopBtn = document.getElementById("btn-stop-speak");
  stopBtn.hidden = false;
  utter.onend = () => { stopBtn.hidden = true; };
  speechSynthesis.speak(utter);
}
document.getElementById("btn-stop-speak").addEventListener("click", () => {
  speechSynthesis.cancel();
  document.getElementById("btn-stop-speak").hidden = true;
});

/* ---------------------------- Settings ---------------------------- */
document.getElementById("api-url-input").value = state.apiUrl;

document.getElementById("btn-save-api").addEventListener("click", () => {
  const val = document.getElementById("api-url-input").value.trim().replace(/\/$/, "");
  state.apiUrl = val;
  localStorage.setItem("cd_api_url", val);
  document.getElementById("api-status").textContent = "Saved. Checking connection…";
  checkApiHealth();
});

async function checkApiHealth() {
  const statusEl = document.getElementById("api-status");
  try {
    const res = await fetch(`${state.apiUrl}/api/health`);
    const data = await res.json();
    statusEl.textContent = `Connected. Model loaded: ${data.model_loaded ? "yes" : "no"}. Database connected: ${data.database_connected ? "yes" : "no"}.`;
  } catch {
    statusEl.textContent = "Couldn't reach the backend at that URL.";
  }
}

/* ---------------------------- Init ---------------------------- */
checkApiHealth();
loadHistory();
