/* ===================================================================
   BIOMEDICAL WASTE AI — CLIENT APPLICATION LOGIC
   MediCore & SmartCBWTF Style Frontend Controller
=================================================================== */

// Global State
let currentConf = 0.25;
let currentIou = 0.45;
let selectedBase64Image = null;
let isAnalyzing = false;
let activeViewMode = 'dual';

// Webcam State
let webcamStream = null;
let webcamTimer = null;
let isWebcamFrozen = false;

// 14 Category Metadata
const WASTE_CLASSES_METADATA = [
  { name: 'gauze', icon: '&#129528;', bin: 'Yellow Bin', cat: 'Soiled / Infectious Cotton', disposal: 'Incineration / Pyrolysis', hazard: 'Blood & Pathogen Contamination' },
  { name: 'unused gauze', icon: '&#129528;', bin: 'Yellow Bin', cat: 'Potentially Contaminated Cotton', disposal: 'Autoclaving / Incineration', hazard: 'Low to Moderate Risk' },
  { name: 'mask', icon: '&#128567;', bin: 'Yellow Bin', cat: 'Infectious PPE (Surgical / N95)', disposal: 'Autoclave followed by Incineration', hazard: 'Respiratory Droplet Hazard' },
  { name: 'scrub', icon: '&#129489;&#8205;&#9877;&#65039;', bin: 'Yellow Bin', cat: 'Infectious Fabric / Gowns', disposal: 'Chemical Disinfection & Autoclave', hazard: 'Body Fluid Contact' },
  { name: 'paper', icon: '&#128196;', bin: 'Yellow Bin', cat: 'Clinical Wrappers & Towels', disposal: 'Incineration', hazard: 'Superficial Fluid Contamination' },
  { name: 'glove', icon: '&#129534;', bin: 'Red Bin', cat: 'Contaminated Gloves (Latex/Nitrile)', disposal: 'Autoclaving & Shredding', hazard: 'Contact Transmission' },
  { name: 'plastic', icon: '&#9851;', bin: 'Red Bin', cat: 'Recyclable Medical Plastic', disposal: 'Autoclave, Shred & Recycle', hazard: 'Recyclable Contaminated Material' },
  { name: 'tube', icon: '&#129514;', bin: 'Red Bin', cat: 'IV Tubes & Catheters', disposal: 'Autoclaving, Shredding & Recycling', hazard: 'Fluid Passage Contamination' },
  { name: 'used syringe', icon: '&#128137;', bin: 'White Translucent', cat: 'Contaminated Sharps with Fixed Needle', disposal: 'Autoclaving & Encapsulation', hazard: 'High Needlestick Biohazard' },
  { name: 'needle', icon: '&#128298;', bin: 'White Translucent', cat: 'Waste Sharps / Needles', disposal: 'Dry Heat Sterilization & Encapsulation', hazard: 'Puncture & Bloodborne Infection' },
  { name: 'needle cap', icon: '&#128138;', bin: 'Red Bin', cat: 'Clean Plastic Needle Caps', disposal: 'Autoclaving & Shredding', hazard: 'Sharps Protective Barrier' },
  { name: 'glass', icon: '&#129519;', bin: 'Blue Cardboard', cat: 'Broken or Intact Glassware', disposal: 'Disinfection & Glass Recycling', hazard: 'Physical Cut & Infection Hazard' },
  { name: 'test tube', icon: '&#129514;', bin: 'Blue Cardboard', cat: 'Laboratory Glass Test Tubes', disposal: 'Chemical Disinfection & Autoclaving', hazard: 'Pathogen & Chemical Hazard' },
  { name: 'fluid bottle', icon: '&#127868;', bin: 'Blue Cardboard', cat: 'Saline / Medicine Glass Bottles', disposal: 'Decontamination & Recycling', hazard: 'Chemical Residue Hazard' }
];

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  renderClassesCatalog();
  setupDragAndDrop();
  setupPasteListener();
  fetchModelInfo();
});

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById('nav-btn-' + tabId);
  if (activeBtn) activeBtn.classList.add('active');

  const titles = {
    studio: 'Detection Studio',
    dashboard: 'Dashboard Overview',
    webcam: 'Live Video Camera Feed',
    classes: 'Waste Category Catalog',
    benchmarks: 'Model Performance & Benchmarks'
  };
  document.getElementById('topbar-breadcrumb').innerText = titles[tabId] || 'Platform';

  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.add('hidden');
  });

  const targetPane = document.getElementById('tab-pane-' + tabId);
  if (targetPane) targetPane.classList.remove('hidden');

  if (tabId !== 'webcam') {
    stopCamera();
  }
}

// View Mode Toggle (Dual, Annotated, Original)
function setViewMode(mode) {
  activeViewMode = mode;
  document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById('view-mode-' + mode);
  if (activeBtn) activeBtn.classList.add('active');

  const grid = document.getElementById('viewports-grid');
  const origCol = document.getElementById('viewport-col-orig');
  const annoCol = document.getElementById('viewport-col-anno');

  if (mode === 'dual') {
    grid.className = 'viewports-grid';
    origCol.classList.remove('hidden');
    annoCol.classList.remove('hidden');
  } else if (mode === 'annotated') {
    grid.className = 'viewports-grid single-view';
    origCol.classList.add('hidden');
    annoCol.classList.remove('hidden');
  } else {
    grid.className = 'viewports-grid single-view';
    origCol.classList.remove('hidden');
    annoCol.classList.add('hidden');
  }
}

// Sliders
function updateParam(param, val) {
  const pct = Math.round(parseFloat(val) * 100) + '%';
  if (param === 'conf') {
    currentConf = parseFloat(val);
    document.getElementById('conf-val').innerText = pct;
  } else {
    currentIou = parseFloat(val);
    document.getElementById('iou-val').innerText = pct;
  }
  if (selectedBase64Image && !isAnalyzing) {
    triggerDetection();
  }
}

// Drag & Drop
function setupDragAndDrop() {
  const dropzone = document.getElementById('dropzone');
  ['dragenter', 'dragover'].forEach(ev => {
    dropzone.addEventListener(ev, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(ev => {
    dropzone.addEventListener(ev, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processSelectedFile(files[0]);
    }
  });
}

// Paste Handler
function setupPasteListener() {
  window.addEventListener('paste', e => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let i in items) {
      const item = items[i];
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        processSelectedFile(item.getAsFile());
        break;
      }
    }
  });
}

function handleFileSelect(e) {
  if (e.target.files && e.target.files[0]) {
    processSelectedFile(e.target.files[0]);
  }
}

function processSelectedFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Invalid file: please select a JPG, PNG, or WEBP image.');
    return;
  }
  const reader = new FileReader();
  reader.onload = evt => {
    setLoadedImage(evt.target.result, file.name);
  };
  reader.readAsDataURL(file);
}

function setLoadedImage(dataUrl, name) {
  selectedBase64Image = dataUrl;

  const origImg = document.getElementById('img-original');
  origImg.src = dataUrl;
  origImg.classList.remove('hidden');
  document.getElementById('orig-empty-state').classList.add('hidden');

  origImg.onload = function() {
    document.getElementById('orig-res-badge').innerText = this.naturalWidth + ' x ' + this.naturalHeight + 'px';
  };

  document.getElementById('btn-detect').disabled = false;
  document.getElementById('btn-download').disabled = true;

  triggerDetection();
}

function clearSelectedImage() {
  selectedBase64Image = null;
  document.getElementById('file-input').value = '';

  const origImg = document.getElementById('img-original');
  origImg.src = '';
  origImg.classList.add('hidden');
  document.getElementById('orig-empty-state').classList.remove('hidden');
  document.getElementById('orig-res-badge').innerText = '—';

  const annoImg = document.getElementById('img-annotated');
  annoImg.src = '';
  annoImg.classList.add('hidden');
  document.getElementById('anno-empty-state').classList.remove('hidden');
  document.getElementById('anno-latency-badge').innerText = '—';

  document.getElementById('btn-detect').disabled = true;
  document.getElementById('btn-download').disabled = true;
  document.getElementById('results-count-pill').innerText = '0 items';
  document.getElementById('segregation-list').innerHTML = `
    <div class="empty-results-msg">
      Load an image to view automated class identification, confidence metrics, and official CPCB bin recommendations.
    </div>
  `;
}

// Load Certified Sample Image
function loadSampleImage(filename, displayName) {
  showToast('Loading sample: ' + displayName + '...');
  fetch('/api/sample/' + filename)
    .then(r => {
      if (!r.ok) throw new Error('Sample not found on server');
      return r.blob();
    })
    .then(blob => {
      const rd = new FileReader();
      rd.onload = e => {
        setLoadedImage(e.target.result, displayName);
        showToast('Sample loaded: ' + displayName);
      };
      rd.readAsDataURL(blob);
    })
    .catch(err => {
      showToast('Error loading sample: ' + err.message);
    });
}

// Detection Trigger
function triggerDetection() {
  if (!selectedBase64Image || isAnalyzing) return;
  isAnalyzing = true;

  const btn = document.getElementById('btn-detect');
  btn.disabled = true;
  document.getElementById('btn-detect-label').innerText = 'Analyzing...';
  document.getElementById('scan-overlay').classList.add('active');

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: selectedBase64Image,
      conf: currentConf,
      iou: currentIou,
      imgsz: 640
    })
  })
  .then(r => {
    if (!r.ok) throw new Error('Prediction API returned HTTP ' + r.status);
    return r.json();
  })
  .then(data => {
    renderDetectionResults(data);
  })
  .catch(err => {
    console.error('Detection error:', err);
    showToast('Inference error: ' + err.message);
  })
  .finally(() => {
    isAnalyzing = false;
    btn.disabled = false;
    document.getElementById('btn-detect-label').innerText = 'Run AI Segregation';
    document.getElementById('scan-overlay').classList.remove('active');
  });
}

// Render Results
function renderDetectionResults(data) {
  document.getElementById('anno-latency-badge').innerText = data.inference_ms + 'ms';

  const annoImg = document.getElementById('img-annotated');
  if (data.annotated_image_base64) {
    annoImg.src = 'data:image/jpeg;base64,' + data.annotated_image_base64;
    annoImg.classList.remove('hidden');
    document.getElementById('anno-empty-state').classList.add('hidden');
    document.getElementById('btn-download').disabled = false;
  }

  const container = document.getElementById('segregation-list');
  const countPill = document.getElementById('results-count-pill');

  if (!data.detections || data.detections.length === 0) {
    countPill.innerText = '0 items';
    container.innerHTML = `
      <div class="empty-results-msg">
        No biomedical waste objects detected above the ${Math.round(currentConf * 100)}% confidence threshold.
      </div>
    `;
    return;
  }

  countPill.innerText = data.detections.length + ' item' + (data.detections.length === 1 ? '' : 's');
  container.innerHTML = '';

  data.detections.forEach((det, idx) => {
    const seg = det.segregation;
    const confPct = Math.round(det.confidence * 100);

    let binClass = 'bin-white';
    if (seg.bin.includes('Yellow')) binClass = 'bin-yellow';
    else if (seg.bin.includes('Red')) binClass = 'bin-red';
    else if (seg.bin.includes('Blue')) binClass = 'bin-blue';

    const card = document.createElement('div');
    card.className = 'item-result-card';
    card.innerHTML = `
      <div class="item-header">
        <span class="item-name">${det.class_name}</span>
        <span class="item-conf">${confPct}% conf</span>
      </div>
      <div>
        <span class="bin-badge ${binClass}">&#128465;&#65039; ${seg.bin}</span>
      </div>
      <div class="item-detail-line"><strong>Category:</strong> ${seg.category}</div>
      <div class="item-detail-line"><strong>Disposal:</strong> ${seg.disposal}</div>
      <div class="item-detail-line" style="color:#b45309;"><strong>Hazard:</strong> ${seg.hazard}</div>
    `;
    container.appendChild(card);
  });
}

function downloadAnnotatedImage() {
  const annoImg = document.getElementById('img-annotated');
  if (!annoImg || !annoImg.src) return;
  const link = document.createElement('a');
  link.href = annoImg.src;
  link.download = 'biomedical_waste_detection_' + Date.now() + '.jpg';
  link.click();
}

// Populate 14 Classes
function renderClassesCatalog() {
  const grid = document.getElementById('classes-grid-container');
  if (!grid) return;
  grid.innerHTML = '';

  WASTE_CLASSES_METADATA.forEach(item => {
    let binClass = 'bin-white';
    if (item.bin.includes('Yellow')) binClass = 'bin-yellow';
    else if (item.bin.includes('Red')) binClass = 'bin-red';
    else if (item.bin.includes('Blue')) binClass = 'bin-blue';

    const card = document.createElement('div');
    card.className = 'class-card';
    card.innerHTML = `
      <div class="class-card-top">
        <span class="class-name-title">${item.name}</span>
        <span class="class-icon">${item.icon}</span>
      </div>
      <div class="class-category-desc">${item.cat}</div>
      <div class="class-category-desc"><strong>Protocol:</strong> ${item.disposal}</div>
      <div class="class-card-footer">
        <span class="bin-badge ${binClass}">&#128465;&#65039; ${item.bin}</span>
        <span style="font-size:11px; color:var(--text-muted);">&#9888; Biohazard</span>
      </div>
    `;
    grid.appendChild(card);
  });
}

function fetchModelInfo() {
  fetch('/api/info')
    .then(r => r.json())
    .then(data => {
      if (data.num_classes) {
        document.getElementById('sidebar-classes-count').innerText = data.num_classes + ' Biomedical Classes';
      }
    })
    .catch(() => {});
}

// Live Camera Functions
function startCamera() {
  const video = document.getElementById('webcam-video');
  const placeholder = document.getElementById('webcam-placeholder');
  const statusPill = document.getElementById('webcam-status-pill');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('Webcam access is not supported in this browser environment.');
    return;
  }

  navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => {
      webcamStream = stream;
      video.srcObject = stream;
      video.classList.remove('hidden');
      placeholder.classList.add('hidden');

      document.getElementById('btn-cam-start').classList.add('hidden');
      document.getElementById('btn-cam-stop').classList.remove('hidden');
      document.getElementById('btn-cam-freeze').classList.remove('hidden');

      statusPill.innerText = 'Live Streaming';
      statusPill.style.background = 'var(--primary-light)';
      statusPill.style.borderColor = 'var(--primary-border)';
      statusPill.style.color = 'var(--primary)';

      isWebcamFrozen = false;
      startWebcamLoop();
    })
    .catch(err => {
      console.error('Camera error:', err);
      showToast('Could not access camera: ' + err.message);
    });
}

function stopCamera() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  if (webcamTimer) {
    clearTimeout(webcamTimer);
    webcamTimer = null;
  }

  const video = document.getElementById('webcam-video');
  video.srcObject = null;
  video.classList.add('hidden');
  document.getElementById('webcam-placeholder').classList.remove('hidden');

  const canvas = document.getElementById('webcam-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  document.getElementById('btn-cam-start').classList.remove('hidden');
  document.getElementById('btn-cam-stop').classList.add('hidden');
  document.getElementById('btn-cam-freeze').classList.add('hidden');

  const statusPill = document.getElementById('webcam-status-pill');
  statusPill.innerText = 'Standby';
  statusPill.style.background = '#f1f5f9';
  statusPill.style.borderColor = '#cbd5e1';
  statusPill.style.color = '#64748b';

  document.getElementById('live-count-badge').innerText = '0 detected';
}

function toggleFreezeCamera() {
  isWebcamFrozen = !isWebcamFrozen;
  const label = document.getElementById('freeze-label');
  const icon = document.getElementById('freeze-icon');
  const statusPill = document.getElementById('webcam-status-pill');

  if (isWebcamFrozen) {
    label.innerText = 'Resume';
    icon.innerHTML = '&#9654;';
    statusPill.innerText = 'Paused';
  } else {
    label.innerText = 'Freeze Frame';
    icon.innerHTML = '&#10074;&#10074;';
    statusPill.innerText = 'Live Streaming';
    startWebcamLoop();
  }
}

function startWebcamLoop() {
  if (!webcamStream || isWebcamFrozen) return;
  const video = document.getElementById('webcam-video');
  if (video.readyState >= 2) {
    captureAndPredictFrame();
  } else {
    webcamTimer = setTimeout(startWebcamLoop, 150);
  }
}

function captureAndPredictFrame() {
  if (!webcamStream || isWebcamFrozen) return;
  const video = document.getElementById('webcam-video');
  const snapshotCanvas = document.getElementById('hidden-webcam-snapshot');

  snapshotCanvas.width = video.videoWidth || 640;
  snapshotCanvas.height = video.videoHeight || 480;
  const ctx = snapshotCanvas.getContext('2d');
  ctx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

  const b64 = snapshotCanvas.toDataURL('image/jpeg', 0.65);

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: b64,
      conf: currentConf,
      iou: currentIou,
      imgsz: 480
    })
  })
  .then(r => r.json())
  .then(data => {
    drawWebcamBoxes(data.detections, video.videoWidth, video.videoHeight);
    updateWebcamLiveList(data.detections);
  })
  .catch(() => {})
  .finally(() => {
    if (webcamStream && !isWebcamFrozen) {
      webcamTimer = setTimeout(startWebcamLoop, 250);
    }
  });
}

function drawWebcamBoxes(detections, vW, vH) {
  const canvas = document.getElementById('webcam-canvas');
  canvas.width = vW;
  canvas.height = vH;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, vW, vH);

  if (!detections || detections.length === 0) return;

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox_xyxy;
    const w = x2 - x1;
    const h = y2 - y1;

    let col = '#059669';
    if (det.segregation.bin.includes('Yellow')) col = '#b45309';
    else if (det.segregation.bin.includes('Red')) col = '#dc2626';
    else if (det.segregation.bin.includes('Blue')) col = '#0284c7';

    ctx.strokeStyle = col;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x1, y1, w, h);

    const label = `${det.class_name} (${Math.round(det.confidence * 100)}%) - ${det.segregation.bin}`;
    ctx.font = 'bold 12px sans-serif';
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = col;
    ctx.fillRect(x1, Math.max(0, y1 - 20), textWidth + 10, 20);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x1 + 5, Math.max(14, y1 - 5));
  });
}

function updateWebcamLiveList(detections) {
  const container = document.getElementById('live-webcam-items');
  const badge = document.getElementById('live-count-badge');

  if (!detections || detections.length === 0) {
    badge.innerText = '0 detected';
    container.innerHTML = '<div class="empty-results-msg">Scanning video frame...</div>';
    return;
  }

  badge.innerText = detections.length + ' detected';
  container.innerHTML = '';

  detections.forEach(det => {
    let binClass = 'bin-white';
    if (det.segregation.bin.includes('Yellow')) binClass = 'bin-yellow';
    else if (det.segregation.bin.includes('Red')) binClass = 'bin-red';
    else if (det.segregation.bin.includes('Blue')) binClass = 'bin-blue';

    const div = document.createElement('div');
    div.className = 'item-result-card';
    div.innerHTML = `
      <div class="item-header">
        <span class="item-name">${det.class_name}</span>
        <span class="item-conf">${Math.round(det.confidence * 100)}%</span>
      </div>
      <span class="bin-badge ${binClass}">&#128465;&#65039; ${det.segregation.bin}</span>
    `;
    container.appendChild(div);
  });
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.innerText = msg;
  t.style.display = 'block';
  setTimeout(() => {
    t.style.display = 'none';
  }, 3000);
}
