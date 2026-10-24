/**
 * Face Recognition System (FRS) · Web Demo Controller
 * Connects frontend events to the existing FastAPI /recognize pipeline.
 */

// DOM Elements Registry
const dom = {
  // Navigation & Status
  backendStatus: document.getElementById("backendStatus"),
  galleryCount: document.getElementById("galleryCount"),
  latencyBadge: document.getElementById("latencyBadge"),

  // Input & Dropzone
  dropzone: document.getElementById("dropzone"),
  dropzonePrompt: document.getElementById("dropzonePrompt"),
  fileInput: document.getElementById("fileInput"),
  previewContainer: document.getElementById("previewContainer"),
  previewImg: document.getElementById("previewImg"),
  previewFileName: document.getElementById("previewFileName"),
  btnClearPreview: document.getElementById("btnClearPreview"),
  samplesTray: document.getElementById("samplesTray"),

  // Primary Action
  recognizeBtn: document.getElementById("recognizeBtn"),
  btnLabel: document.getElementById("btnLabel"),
  btnSpinner: document.getElementById("btnSpinner"),

  // States
  idleState: document.getElementById("idleState"),
  loadingState: document.getElementById("loadingState"),
  errorState: document.getElementById("errorState"),
  errorTitle: document.getElementById("errorTitle"),
  errorMessage: document.getElementById("errorMessage"),
  btnRetry: document.getElementById("btnRetry"),
  resultState: document.getElementById("resultState"),

  // Result Elements
  verdictBanner: document.getElementById("verdictBanner"),
  statusPill: document.getElementById("statusPill"),
  identityText: document.getElementById("identityText"),
  verdictSubtext: document.getElementById("verdictSubtext"),
  matchScore: document.getElementById("matchScore"),

  // Verification Side-by-Side
  queryFaceCrop: document.getElementById("queryFaceCrop"),
  queryConfidence: document.getElementById("queryConfidence"),
  connectorScore: document.getElementById("connectorScore"),
  matchedFaceCard: document.getElementById("matchedFaceCard"),
  enrolledFaceCrop: document.getElementById("enrolledFaceCrop"),
  enrolledPersonName: document.getElementById("enrolledPersonName"),
  unknownFaceCard: document.getElementById("unknownFaceCard"),

  // Candidates & Metrics
  matchesTableBody: document.getElementById("matchesTableBody"),
  metricThreshold: document.getElementById("metricThreshold"),
  metricGap: document.getElementById("metricGap"),
  metricConf: document.getElementById("metricConf"),
  metricFaces: document.getElementById("metricFaces"),
};

let currentFile = null;
let currentSampleName = null;

// =============================================================================
// Initialization & Backend Health Check
// =============================================================================
async function init() {
  bindEvents();
  await checkHealth();
  await loadSampleCatalog();
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      dom.backendStatus.textContent = "API Online";
      if (data.enrolled_identities) {
        dom.galleryCount.textContent = `${data.enrolled_identities.length} Identities`;
      }
    } else {
      dom.backendStatus.textContent = "API Degradation";
    }
  } catch (err) {
    dom.backendStatus.textContent = "API Offline";
    dom.backendStatus.parentElement.style.color = "var(--danger-rose)";
    dom.backendStatus.parentElement.style.borderColor = "rgba(239, 68, 68, 0.3)";
  }
}

// =============================================================================
// Event Handlers
// =============================================================================
function bindEvents() {
  // File Picker
  dom.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag & Drop
  ["dragenter", "dragover"].forEach((evt) => {
    dom.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dom.dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dom.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dom.dropzone.classList.remove("dragover");
    });
  });

  dom.dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Clear Preview
  dom.btnClearPreview.addEventListener("click", (e) => {
    e.stopPropagation();
    clearInput();
  });

  // Retry Button
  dom.btnRetry.addEventListener("click", () => {
    clearInput();
  });

  // Recognize Button
  dom.recognizeBtn.addEventListener("click", () => {
    if (currentFile) {
      executeRecognition(currentFile);
    }
  });
}

// =============================================================================
// Quick Demo Samples Catalog
// =============================================================================
async function loadSampleCatalog() {
  try {
    const res = await fetch("/api/samples");
    if (!res.ok) return;
    const samples = await res.json();
    if (!samples || !samples.length) return;

    dom.samplesTray.innerHTML = "";
    samples.forEach((sample) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sample-chip";
      btn.dataset.file = sample.filename;
      btn.dataset.label = sample.label;

      const isHeldBack = sample.tag && sample.tag.includes("Held-back");
      const dotClass = isHeldBack ? "held-back" : "enrolled";

      btn.innerHTML = `
        <span class="chip-dot ${dotClass}"></span>
        <span class="chip-name">${sample.label}</span>
        <span class="chip-badge">${isHeldBack ? "Query" : "Gallery"}</span>
      `;

      btn.addEventListener("click", async () => {
        // Highlight active chip
        document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));
        btn.classList.add("active");
        await loadSampleFile(sample.filename, sample.label);
      });

      dom.samplesTray.appendChild(btn);
    });
  } catch (err) {
    console.warn("Could not load sample catalog:", err);
  }
}

async function loadSampleFile(filename, label) {
  try {
    setLoadingState(true, `Loading sample: ${label}...`);
    const res = await fetch(`/api/samples/${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error("Sample file not found on server");

    const blob = await res.blob();
    const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
    currentSampleName = label;
    handleFileSelected(file, label);

    // Auto-run recognition on sample select for frictionless demo experience!
    await executeRecognition(file);
  } catch (err) {
    showError("Sample Load Error", err.message || "Failed to load sample test photo.");
  }
}

// =============================================================================
// File Selection & Preview
// =============================================================================
function handleFileSelected(file, customLabel = null) {
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    showError("Invalid File Format", "Please upload a valid image file (JPG, PNG, or WebP).");
    return;
  }

  currentFile = file;
  dom.previewFileName.textContent = customLabel || file.name;
  dom.previewImg.src = URL.createObjectURL(file);
  dom.previewContainer.hidden = false;
  dom.dropzonePrompt.hidden = true;
  dom.recognizeBtn.disabled = false;

  showIdleState();
}

function clearInput() {
  currentFile = null;
  currentSampleName = null;
  dom.fileInput.value = "";
  dom.previewImg.src = "";
  dom.previewContainer.hidden = true;
  dom.dropzonePrompt.hidden = false;
  dom.recognizeBtn.disabled = true;
  document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));
  showIdleState();
}

// =============================================================================
// View State Management
// =============================================================================
function showIdleState() {
  dom.idleState.hidden = false;
  dom.loadingState.hidden = true;
  dom.errorState.hidden = true;
  dom.resultState.hidden = true;
  dom.latencyBadge.textContent = "Awaiting Input";
}

function setLoadingState(loading, message = "Analyzing Biometrics...") {
  dom.recognizeBtn.disabled = loading || !currentFile;
  dom.btnSpinner.hidden = !loading;
  dom.btnLabel.textContent = loading ? message : "Run Face Recognition";

  if (loading) {
    dom.idleState.hidden = true;
    dom.loadingState.hidden = false;
    dom.errorState.hidden = true;
    dom.resultState.hidden = true;
    dom.latencyBadge.textContent = "Processing SCRFD + ArcFace...";
  }
}

function showError(title, message) {
  setLoadingState(false);
  dom.errorTitle.textContent = title || "Recognition Failed";
  dom.errorMessage.textContent = message || "An unexpected error occurred.";
  dom.idleState.hidden = true;
  dom.loadingState.hidden = true;
  dom.errorState.hidden = false;
  dom.resultState.hidden = true;
  dom.latencyBadge.textContent = "Execution Error";
}

function showResultState() {
  dom.idleState.hidden = true;
  dom.loadingState.hidden = true;
  dom.errorState.hidden = true;
  dom.resultState.hidden = false;
}

// Number formatting utility
const fmt = (val, dp = 3) => {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return Number(val).toFixed(dp);
};

// =============================================================================
// Execute Recognition API Call
// =============================================================================
async function executeRecognition(file) {
  if (!file) {
    showError("No Image Selected", "Please select or upload a face image first.");
    return;
  }

  const startTime = performance.now();
  setLoadingState(true, "Scanning Face...");

  const formData = new FormData();
  formData.append("file", file);

  let response;
  try {
    response = await fetch("/recognize", {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    showError("Network Error", "Cannot reach the FastAPI server. Please ensure the backend is running at http://127.0.0.1:8000.");
    return;
  }

  let data;
  try {
    data = await response.json();
  } catch (err) {
    showError("Parser Error", "Received an invalid non-JSON response from the server.");
    return;
  }

  const durationMs = Math.round(performance.now() - startTime);
  setLoadingState(false);

  // Check HTTP response status
  if (!response.ok || data.error) {
    const errorMsg = data.message || "Face recognition failed.";
    if (response.status === 422 || data.error === "no_face") {
      showError("No Face Detected", "The SCRFD detector found no clear face in this photo. Please ensure good lighting and a frontal angle.");
    } else {
      showError("Recognition Error", errorMsg);
    }
    return;
  }

  dom.latencyBadge.textContent = `${durationMs} ms`;
  renderRecognitionResult(data);
}

// =============================================================================
// Render Results
// =============================================================================
function renderRecognitionResult(d) {
  const isMatch = d.status === "MATCH";

  // 1. Verdict Banner
  dom.statusPill.textContent = d.status;
  if (isMatch) {
    dom.statusPill.className = "status-pill";
    dom.verdictBanner.className = "verdict-banner";
    dom.identityText.textContent = d.identity;
    dom.verdictSubtext.textContent = "Verified in Gallery (Score exceeds threshold)";
  } else {
    dom.statusPill.className = "status-pill is-unknown";
    dom.verdictBanner.className = "verdict-banner is-unknown";
    dom.identityText.textContent = "UNKNOWN";
    dom.verdictSubtext.textContent = `Score below ${fmt(d.threshold)} threshold · Unrecognized Identity`;
  }

  dom.matchScore.textContent = fmt(d.match_score, 4);

  // 2. Query Face Crop
  if (d.face_crop) {
    dom.queryFaceCrop.src = d.face_crop;
  }
  dom.queryConfidence.textContent = `Det Conf: ${fmt(d.detection_confidence, 3)}`;
  dom.connectorScore.textContent = fmt(d.match_score, 3);

  // 3. Side-by-Side Enrolled Reference Comparison
  if (isMatch && d.identity) {
    dom.enrolledFaceCrop.src = `/gallery/${encodeURIComponent(d.identity)}/crop`;
    dom.enrolledPersonName.textContent = d.identity;
    dom.matchedFaceCard.hidden = false;
    dom.unknownFaceCard.hidden = true;
  } else {
    dom.matchedFaceCard.hidden = true;
    dom.unknownFaceCard.hidden = false;
  }

  // 4. Top-3 Candidates Table with Progress Bars & Threshold Line
  dom.matchesTableBody.innerHTML = "";
  const thresholdVal = d.threshold !== undefined ? d.threshold : 0.4;
  const thresholdPct = Math.max(0, Math.min(100, (thresholdVal / 1.0) * 100));

  (d.matches || []).forEach((m) => {
    const tr = document.createElement("tr");
    const isTop = m.rank === 1 && isMatch;
    if (isTop) tr.className = "top-rank-row";

    // Normalize score 0.0 to 1.0 for the progress bar (clamp 0% to 100%)
    const scorePct = Math.max(0, Math.min(100, (m.score / 1.0) * 100));
    const isLow = m.score < thresholdVal;

    tr.innerHTML = `
      <td><span class="rank-badge">${m.rank}</span></td>
      <td>
        <div class="cell-identity">
          <span>${m.person_id}</span>
          ${isTop ? '<span class="chip-badge" style="color: var(--match-emerald);">Top Match</span>' : ""}
        </div>
      </td>
      <td>
        <div class="similarity-meter">
          <div class="threshold-marker" style="left: ${thresholdPct}%;" title="Provisional Threshold (${fmt(thresholdVal)})"></div>
          <div class="similarity-fill ${isLow ? "low-score" : ""}" style="width: ${scorePct}%;"></div>
        </div>
      </td>
      <td class="cell-score">${fmt(m.score, 4)}</td>
    `;
    dom.matchesTableBody.appendChild(tr);
  });

  // 5. Technical Diagnostics Grid
  dom.metricThreshold.textContent = fmt(d.threshold);
  dom.metricGap.textContent = d.gap !== null ? fmt(d.gap) : "—";
  dom.metricConf.textContent = fmt(d.detection_confidence);
  dom.metricFaces.textContent = d.num_faces || 1;

  showResultState();
}

// Start Application
document.addEventListener("DOMContentLoaded", init);
