// Frontend logic only: pick a file, POST it to the backend, render the JSON.
// There is NO face-recognition logic here — the backend owns all of that.

const els = {
  fileInput: document.getElementById("fileInput"),
  dropzone: document.getElementById("dropzone"),
  dropzoneInner: document.getElementById("dropzoneInner"),
  previewImg: document.getElementById("previewImg"),
  fileName: document.getElementById("fileName"),
  recognizeBtn: document.getElementById("recognizeBtn"),
  btnLabel: document.getElementById("btnLabel"),
  btnSpinner: document.getElementById("btnSpinner"),
  idleState: document.getElementById("idleState"),
  errorState: document.getElementById("errorState"),
  errorMessage: document.getElementById("errorMessage"),
  resultState: document.getElementById("resultState"),
  faceCrop: document.getElementById("faceCrop"),
  matchedItem: document.getElementById("matchedItem"),
  matchedCrop: document.getElementById("matchedCrop"),
  matchedCaption: document.getElementById("matchedCaption"),
  noMatchItem: document.getElementById("noMatchItem"),
  statusPill: document.getElementById("statusPill"),
  identityText: document.getElementById("identityText"),
  matchScore: document.getElementById("matchScore"),
  matchesBody: document.getElementById("matchesBody"),
  mThreshold: document.getElementById("mThreshold"),
  mGap: document.getElementById("mGap"),
  mConf: document.getElementById("mConf"),
  mFaces: document.getElementById("mFaces"),
};

let selectedFile = null;

// ---- File selection + preview -------------------------------------------
function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("Please choose an image file (JPG or PNG).");
    return;
  }
  selectedFile = file;
  els.fileName.textContent = file.name;
  els.previewImg.src = URL.createObjectURL(file);
  els.previewImg.hidden = false;
  els.dropzoneInner.hidden = true;
  els.recognizeBtn.disabled = false;
  showIdle();
}

els.fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

// Drag & drop
["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("dragover");
  })
);
els.dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

// ---- View state helpers -------------------------------------------------
function showIdle() {
  els.idleState.hidden = false;
  els.errorState.hidden = true;
  els.resultState.hidden = true;
}
function showError(msg) {
  els.errorMessage.textContent = msg;
  els.idleState.hidden = true;
  els.errorState.hidden = false;
  els.resultState.hidden = true;
}
function showResult() {
  els.idleState.hidden = true;
  els.errorState.hidden = true;
  els.resultState.hidden = false;
}
function setLoading(loading) {
  els.recognizeBtn.disabled = loading || !selectedFile;
  els.btnSpinner.hidden = !loading;
  els.btnLabel.textContent = loading ? "Recognizing…" : "Recognize Face";
}

const fmt = (v, dp = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(dp));

// ---- Recognize ----------------------------------------------------------
els.recognizeBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    showError("No image selected. Please upload a face image first.");
    return;
  }

  setLoading(true);
  const form = new FormData();
  form.append("file", selectedFile);

  let res, data;
  try {
    res = await fetch("/recognize", { method: "POST", body: form });
  } catch (err) {
    setLoading(false);
    showError("Cannot reach the backend. Is the server running?");
    return;
  }

  try {
    data = await res.json();
  } catch (err) {
    setLoading(false);
    showError("The server returned an unreadable response.");
    return;
  }

  setLoading(false);

  if (!res.ok || data.error) {
    showError(data.message || "Recognition failed. Please try another image.");
    return;
  }

  renderResult(data);
});

// ---- Render -------------------------------------------------------------
function renderResult(d) {
  // Detected face preview from the backend crop.
  if (d.face_crop) {
    els.faceCrop.src = d.face_crop;
    els.faceCrop.hidden = false;
  } else {
    els.faceCrop.hidden = true;
  }

  const isMatch = d.status === "MATCH";
  els.statusPill.textContent = d.status;
  els.statusPill.className = "status-pill " + (isMatch ? "match" : "unknown");
  els.identityText.textContent = isMatch ? d.identity : "UNKNOWN";
  els.matchScore.textContent = fmt(d.match_score);

  // Matched person's enrolled reference photo (only shown on a real MATCH).
  if (isMatch && d.identity) {
    els.matchedCrop.src = `/gallery/${encodeURIComponent(d.identity)}/crop`;
    els.matchedCaption.textContent = `${d.identity} · ${fmt(d.match_score)}`;
    els.matchedItem.hidden = false;
    els.noMatchItem.hidden = true;
  } else {
    els.matchedItem.hidden = true;
    els.noMatchItem.hidden = false;
  }

  // Top-K table
  els.matchesBody.innerHTML = "";
  (d.matches || []).forEach((m) => {
    const tr = document.createElement("tr");
    if (m.rank === 1) tr.className = "top-row";
    tr.innerHTML =
      `<td>${m.rank}</td>` +
      `<td>${m.person_id}</td>` +
      `<td class="score-cell">${fmt(m.score)}</td>`;
    els.matchesBody.appendChild(tr);
  });

  // Technical panel
  els.mThreshold.textContent = fmt(d.threshold);
  els.mGap.textContent = fmt(d.gap);
  els.mConf.textContent = fmt(d.detection_confidence);
  els.mFaces.textContent = d.num_faces ?? "—";

  showResult();
}
