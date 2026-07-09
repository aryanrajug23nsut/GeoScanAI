/* ============================================================
   GeoScan.AI — Dashboard Logic (Dark Theme, matches friend's design)
   Vanilla JS + Leaflet 1.9.4
   All coordinates are EPSG:4326 (WGS84 lat/lng).
   ============================================================ */
(function () {
  'use strict';

  // ─── Config ─────────────────────────────────────────────
  const API_BASE = window.GEOSCAN_API_BASE || '/api';
  const ENDPOINTS = {
    upload: `${API_BASE}/upload`,
    results: (tid) => `${API_BASE}/results/${tid}`,
    export: (tid, fmt) => `${API_BASE}/export/${tid}?format=${fmt}`,
    models: `${API_BASE}/models`,
    retrain: `${API_BASE}/retrain`,
    retrainStatus: (jid) => `${API_BASE}/retrain/status/${jid}`,
    ensemble: (tid) => `${API_BASE}/ensemble/${tid}`,
    feedback: `${API_BASE}/feedback`,
    health: `${API_BASE}/health`,
    datasetsUpload: `${API_BASE}/datasets/upload`,
    datasetsList: `${API_BASE}/datasets`,
    datasetsDelete: (id) => `${API_BASE}/datasets/${id}`,
    retrainMerge: `${API_BASE}/retrain/merge`,
    retrainMergeStatus: (jid) => `${API_BASE}/retrain/merge/status/${jid}`,
    detectMap: `${API_BASE}/detect_map`, // NEW: Map view detection
  };
  const ALLOWED_EXT = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.ecw'];
  const MAX_FILE_SIZE = 100 * 1024 * 1024;
  const MIN_DPI = 96;
  const DEFAULT_CENTER = [26.8467, 80.9462]; // Lucknow, UP, India
  const DEFAULT_ZOOM = 7;

  const CLASS_PALETTE = [
    { id: 'rooftop',     label: 'Rooftops',     stroke: '#a855f7', fill: 'rgba(168, 85, 247, 0.4)' },
    { id: 'solar_panel', label: 'Solar Panels', stroke: '#f59e0b', fill: 'rgba(245, 158, 11, 0.5)' },
    { id: 'buildings',   label: 'Buildings',    stroke: '#3b82f6', fill: 'rgba(59, 130, 246, 0.4)' },
    { id: 'vegetation',  label: 'Vegetation',   stroke: '#22c55e', fill: 'rgba(34, 197, 94, 0.4)' },
    { id: 'roads',       label: 'Roads',        stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.4)' },
    { id: 'water',       label: 'Water',        stroke: '#06b6d4', fill: 'rgba(6, 182, 212, 0.4)' },
    { id: 'vehicles',    label: 'Vehicles',     stroke: '#ec4899', fill: 'rgba(236, 72, 153, 0.4)' },
    { id: 'class_h',     label: 'Class H',      stroke: '#8b5cf6', fill: 'rgba(139, 92, 246, 0.4)' },
  ];

  function getClassStyle(classId, fallbackLabel) {
    const found = CLASS_PALETTE.find((c) => c.id === classId);
    if (found) return found;
    let h = 0;
    for (let i = 0; i < (classId || '').length; i++) { h = (h << 5) - h + (classId || '').charCodeAt(i); h |= 0; }
    return fallbackLabel ? { ...CLASS_PALETTE[Math.abs(h) % CLASS_PALETTE.length], label: fallbackLabel } : CLASS_PALETTE[Math.abs(h) % CLASS_PALETTE.length];
  }

  // ─── State ──────────────────────────────────────────────
  const state = {
    file: null, models: ['base-v7.6'], mergeStrategy: 'weighted',
    isProcessing: false, detections: null, taskId: null,
    datasets: [], dsFile: null, mergeJobId: null,
    feedbackCount: 0, activeModelId: null,
    map: null, satLayer: null, polygonLayer: null, fillOpacity: 0.55,
    anchorMode: false, anchorLat: null, anchorLng: null, anchorMarker: null,
  };

  // ─── DOM refs ───────────────────────────────────────────
  const $ = (s) => document.querySelector(s);
  const dom = {
    healthStatus: $('#healthStatus'),
    // Models
    toggleAddModelBtn: $('#toggleAddModelBtn'),
    addModelForm: $('#addModelForm'),
    newModelName: $('#newModelName'),
    newModelType: $('#newModelType'),
    newModelFile: $('#newModelFile'),
    uploadModelBtn: $('#uploadModelBtn'),
    modelList: $('#modelList'),
    mergeStrat: $('#mergeStrat'),
    mergeSelect: $('#mergeSelect'),
    // Upload
    imageZone: $('#imageZone'),
    imageInput: $('#imageInput'),
    imageText: $('#imageText'),
    fileError: $('#fileError'),
    fileMeta: $('#fileMeta'),
    fileName: $('#fileName'),
    fileSize: $('#fileSize'),
    fileRemove: $('#fileRemove'),
    refreshUploadBtn: $('#refreshUploadBtn'),
    submitBtn: $('#submitBtn'),
    setAnchorBtn: $('#setAnchorBtn'),
    anchorCoords: $('#anchorCoords'),
    anchorText: $('#anchorText'),
    // Progress
    progressSection: $('#progressSection'),
    statusBadge: $('#statusBadge'),
    progressBar: $('#progressBar'),
    pctText: $('#pctText'),
    jobIdText: $('#jobIdText'),
    // Downloads
    downloadSection: $('#downloadSection'),
    downloadLinks: $('#downloadLinks'),
    plotMapBtn: $('#plotMapBtn'),
    detectMapBtn: $('#detectMapBtn'), // NEW: Map view detection
    // Dataset pool
    refreshDatasetsBtn: $('#refreshDatasetsBtn'),
    dsZone: $('#dsZone'),
    dsInput: $('#dsInput'),
    dsText: $('#dsText'),
    dsName: $('#dsName'),
    dsAddBtn: $('#dsAddBtn'),
    dsPool: $('#dsPool'),
    mergeFirstCheckbox: $('#mergeFirstCheckbox'),
    mergeRetrainBtn: $('#mergeRetrainBtn'),
    retrainBtnLabel: $('#retrainBtnLabel'),
    mergeProgress: $('#mergeProgress'),
    mergeStage: $('#mergeStage'),
    mergePct: $('#mergePct'),
    mergeFill: $('#mergeFill'),
    mergeStats: $('#mergeStats'),
    // Feedback
    feedbackCount: $('#feedbackCount'),
    // Map
    legend: $('#legend'),
    opacitySlider: $('#opacitySlider'),
    opacityValue: $('#opacityValue'),
    refreshMapBtn: $('#refreshMapBtn'),
    polyCount: $('#polyCount'),
    classCount: $('#classCount'),
    areaCount: $('#areaCount'),
    energyCount: $('#energyCount'),
    // Toast
    toast: $('#toast'),
  };

  // ============================================================
  // 1. MAP INIT
  // ============================================================
  function initMap() {
    state.map = L.map('map', { center: DEFAULT_CENTER, zoom: DEFAULT_ZOOM });
    state.satLayer = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19, attribution: 'Imagery &copy; Esri, Maxar, Earthstar Geographics' }
    ).addTo(state.map);
    state.polygonLayer = L.layerGroup().addTo(state.map);
    setTimeout(() => state.map.invalidateSize(), 200);
    window.addEventListener('resize', () => state.map.invalidateSize());

    // Map click handler for georeference anchoring
    state.map.on('click', (e) => {
      if (state.anchorMode) {
        state.anchorLat = e.latlng.lat;
        state.anchorLng = e.latlng.lng;
        if (state.anchorMarker) state.map.removeLayer(state.anchorMarker);
        state.anchorMarker = L.marker(e.latlng, {
          icon: L.divIcon({
            className: '',
            html: '<div style="width:24px;height:24px;background:var(--green);border:3px solid white;border-radius:50%;box-shadow:0 2px 8px rgba(0,0,0,0.5);"></div>',
            iconSize: [24, 24], iconAnchor: [12, 12]
          })
        }).addTo(state.map);
        state.anchorMarker.bindPopup('Image center location').openPopup();
        dom.anchorText.textContent = `${state.anchorLat.toFixed(5)}, ${state.anchorLng.toFixed(5)}`;
        dom.anchorCoords.style.display = 'flex';
        dom.setAnchorBtn.classList.remove('active');
        dom.setAnchorBtn.innerHTML = '<i class="fas fa-check"></i> Location Set (click again to change)';
        state.anchorMode = false;
        showToast('Location set! Polygons will appear here.', 'ok');
      }
    });
  }

  function clearMap() {
    state.polygonLayer.clearLayers();
    dom.polyCount.textContent = '0';
    dom.classCount.textContent = '0';
    dom.areaCount.textContent = '0 m²';
    dom.energyCount.textContent = '0 kWh/yr';
    dom.legend.innerHTML = '<span class="legend__item legend__item--empty">No detections yet</span>';
  }

  // ============================================================
  // 2. OPACITY SLIDER
  // ============================================================
  function initOpacity() {
    dom.opacitySlider.addEventListener('input', (e) => {
      const pct = parseInt(e.target.value, 10);
      state.fillOpacity = pct / 100;
      dom.opacityValue.textContent = `${pct}%`;
      state.polygonLayer.eachLayer((l) => {
        if (l.setStyle && l.options.fillColor) l.setStyle({ fillOpacity: state.fillOpacity });
      });
    });
  }

  // ============================================================
  // 3. FILE UPLOAD + VALIDATION
  // ============================================================
  function initUpload() {
    dom.imageZone.addEventListener('click', () => dom.imageInput.click());
    dom.imageInput.addEventListener('change', (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); });
    dom.fileRemove.addEventListener('click', clearFile);
    dom.refreshUploadBtn.addEventListener('click', clearFile);
    dom.submitBtn.addEventListener('click', runDetection);

    // Georeference anchor button
    dom.setAnchorBtn.addEventListener('click', () => {
      state.anchorMode = !state.anchorMode;
      if (state.anchorMode) {
        dom.setAnchorBtn.classList.add('active');
        dom.setAnchorBtn.innerHTML = '<i class="fas fa-crosshairs fa-spin"></i> Click map to set center...';
        dom.anchorCoords.style.display = 'flex';
        dom.anchorText.textContent = 'Click the map to set center';
        state.map.getContainer().style.cursor = 'crosshair';
        showToast('Click the map where your image belongs', 'ok');
      } else {
        dom.setAnchorBtn.classList.remove('active');
        dom.setAnchorBtn.innerHTML = '<i class="fas fa-crosshairs"></i> Set Image Location on Map';
        state.map.getContainer().style.cursor = '';
      }
    });

    ['dragenter', 'dragover'].forEach((ev) => dom.imageZone.addEventListener(ev, (e) => { e.preventDefault(); dom.imageZone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach((ev) => dom.imageZone.addEventListener(ev, (e) => { e.preventDefault(); dom.imageZone.classList.remove('dragover'); }));
    dom.imageZone.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
  }

  function getExt(name) { const n = name.toLowerCase(); const d = n.lastIndexOf('.'); return d >= 0 ? n.slice(d) : ''; }
  function formatSize(b) { if (b < 1024) return `${b} B`; if (b < 1048576) return `${(b/1024).toFixed(1)} KB`; if (b < 1073741824) return `${(b/1048576).toFixed(1)} MB`; return `${(b/1073741824).toFixed(2)} GB`; }

  function showFileError(html, warn) {
    dom.fileError.innerHTML = html;
    dom.fileError.style.display = 'flex';
    dom.fileError.classList.toggle('is-warn', !!warn);
    dom.imageZone.classList.add('filled');
  }
  function clearFileError() { dom.fileError.style.display = 'none'; dom.fileError.innerHTML = ''; dom.fileError.classList.remove('is-warn'); }

  async function readPngDpi(file) {
    const buf = await file.slice(0, 4096).arrayBuffer();
    const v = new DataView(buf);
    const sig = [0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A];
    if (buf.byteLength < 8) return null;
    for (let i = 0; i < 8; i++) if (v.getUint8(i) !== sig[i]) return null;
    let off = 8;
    while (off + 8 <= buf.byteLength) {
      const len = v.getUint32(off);
      const type = String.fromCharCode(v.getUint8(off+4),v.getUint8(off+5),v.getUint8(off+6),v.getUint8(off+7));
      if (type === 'pHYs') { if (off+17 > buf.byteLength) return null; const x = v.getUint32(off+8); const u = v.getUint8(off+16); return u === 1 ? Math.round(x * 0.0254) : null; }
      if (type === 'IEND') break;
      off += 8 + len + 4;
    }
    return null;
  }
  async function readJpegDpi(file) {
    const buf = await file.slice(0, 65536).arrayBuffer();
    const v = new DataView(buf);
    if (v.byteLength < 4 || v.getUint16(0) !== 0xFFD8) return null;
    let off = 2;
    while (off + 4 < buf.byteLength) {
      if (v.getUint8(off) !== 0xFF) break;
      const marker = v.getUint8(off + 1);
      if (marker === 0xE0) {
        if (off + 18 <= buf.byteLength) {
          const id = String.fromCharCode(v.getUint8(off+4),v.getUint8(off+5),v.getUint8(off+6),v.getUint8(off+7),v.getUint8(off+8));
          if (id === 'JFIF\0') { if (off+16 > buf.byteLength) return null; const u = v.getUint8(off+13); const x = v.getUint16(off+14); return u === 1 ? x : (u === 2 ? Math.round(x*2.54) : null); }
        }
        off += 2 + v.getUint16(off + 2); continue;
      }
      if (marker === 0xD8 || marker === 0xD9 || (marker >= 0xD0 && marker <= 0xD7)) { off += 2; continue; }
      if (marker === 0xDA) break;
      off += 2 + v.getUint16(off + 2);
    }
    return null;
  }

  async function handleFile(file) {
    clearFileError();
    const ext = getExt(file.name);
    if (!ALLOWED_EXT.includes(ext)) { showFileError(`<i class="fas fa-exclamation-circle"></i> <span><strong>Unsupported format.</strong> Got <code>${ext || '(none)'}</code>. Allowed: ${ALLOWED_EXT.join(', ')}</span>`); showToast(`Rejected: ${ext}`, 'err'); return; }
    if (file.size > MAX_FILE_SIZE) { const mb = (file.size/1048576).toFixed(1); showFileError(`<i class="fas fa-exclamation-circle"></i> <span><strong>File too large.</strong> ${mb} MB. Max: 100 MB.</span>`); showToast(`Rejected: ${mb} MB`, 'err'); return; }
    const dpi = await (ext === '.png' ? readPngDpi(file) : (ext === '.jpg' || ext === '.jpeg') ? readJpegDpi(file) : Promise.resolve(null));
    if (dpi !== null && dpi < MIN_DPI) { showFileError(`<i class="fas fa-exclamation-circle"></i> <span><strong>DPI too low.</strong> Detected ${dpi} DPI. Min: ${MIN_DPI} DPI.</span>`); showToast(`Rejected: ${dpi} DPI`, 'err'); return; }
    if (dpi === null && (ext === '.tif' || ext === '.tiff' || ext === '.ecw')) { showFileError(`<i class="fas fa-exclamation-triangle"></i> <span><strong>DPI not verified client-side.</strong> ${ext.toUpperCase()} DPI will be validated by backend.</span>`, true); }

    state.file = file;
    dom.fileName.textContent = file.name;
    dom.fileSize.textContent = formatSize(file.size) + (dpi ? ` · ${dpi} DPI` : '');
    dom.imageZone.classList.add('filled');
    dom.imageText.textContent = `✓ ${file.name}`;
    dom.fileMeta.style.display = 'flex';
    dom.submitBtn.disabled = false;
    showToast(`Loaded: ${file.name}`, 'ok');
  }

  function clearFile() {
    state.file = null;
    dom.imageInput.value = '';
    dom.imageZone.classList.remove('filled');
    dom.imageText.textContent = 'Click to upload Image (.tif / .jpg / .png / .ecw)';
    dom.fileMeta.style.display = 'none';
    dom.submitBtn.disabled = true;
    clearFileError();
    dom.progressSection.style.display = 'none';
    dom.downloadSection.style.display = 'none';
  }

  // ============================================================
  // 4. MODEL MANAGEMENT
  // ============================================================
  function initModels() {
    dom.toggleAddModelBtn.addEventListener('click', () => dom.addModelForm.classList.toggle('show'));
    dom.uploadModelBtn.addEventListener('click', uploadNewModel);
    dom.mergeSelect.addEventListener('change', (e) => { state.mergeStrategy = e.target.value; });
  }

  async function fetchModels() {
    try {
      const r = await fetch(ENDPOINTS.models);
      if (!r.ok) throw new Error();
      const data = await r.json();
      renderModels(data.base_models || [], data.user_models || []);
      setHealth('ok', `Connected (${(data.base_models||[]).length + (data.user_models||[]).length} models)`);
    } catch { dom.modelList.innerHTML = '<div style="text-align:center;color:var(--red);">Failed to load models</div>'; setHealth('fail', 'Backend Offline'); }
  }

  function renderModels(baseModels, userModels) {
    const icons = { yolo_detection: 'fa-crosshairs', yolo_segmentation: 'fa-draw-polygon', custom: 'fa-cog', rooftop: 'fa-home', solar_panel: 'fa-solar-panel' };
    const colors = { yolo_detection: 'var(--blue)', yolo_segmentation: 'var(--purple)', custom: 'var(--green)', rooftop: 'var(--purple)', solar_panel: 'var(--orange)' };

    let html = '';
    // Base model checkbox (always there)
    html += `<div class="model-item active" data-model-id="base-v7.6">
      <input type="checkbox" class="model-checkbox" value="base-v7.6" checked>
      <div class="model-icon" style="background:${colors.rooftop}20;color:${colors.rooftop}"><i class="fas fa-home"></i></div>
      <div class="model-details"><div class="model-name">Base Model v7.6</div><div class="model-type">rooftop detection</div></div>
      <span class="active-badge">ACTIVE</span>
    </div>`;

    // User models
    userModels.forEach(um => {
      html += `<div class="model-item" data-model-id="${um.id}">
        <input type="checkbox" class="model-checkbox" value="${um.id}">
        <div class="model-icon" style="background:${colors.custom}20;color:${colors.custom}"><i class="fas fa-cog"></i></div>
        <div class="model-details"><div class="model-name">${um.name}</div><div class="model-type">${um.base_model} · ${um.epochs}ep</div></div>
        <button class="model-delete" data-del="${um.id}"><i class="fas fa-trash"></i></button>
      </div>`;
    });

    dom.modelList.innerHTML = html;

    // Wire checkboxes
    dom.modelList.querySelectorAll('.model-checkbox').forEach(cb => {
      cb.addEventListener('change', () => {
        state.models = Array.from(dom.modelList.querySelectorAll('.model-checkbox:checked')).map(i => i.value);
        dom.mergeStrat.style.display = state.models.length > 1 ? 'block' : 'none';
      });
    });

    // Wire delete buttons
    dom.modelList.querySelectorAll('[data-del]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const id = btn.dataset.del;
        if (!confirm('Delete this model?')) return;
        try { const r = await fetch(`${ENDPOINTS.models}/${id}`, { method: 'DELETE' }); if (!r.ok) throw new Error(); showToast('Model deleted', 'ok'); fetchModels(); } catch { showToast('Delete failed', 'err'); }
      });
    });
  }

  async function uploadNewModel() {
    const name = dom.newModelName.value.trim();
    const type = dom.newModelType.value;
    const file = dom.newModelFile.files[0];
    if (!name || !file) return showToast('Name and file required', 'err');
    const fd = new FormData();
    fd.append('name', name); fd.append('model_type', type); fd.append('model_file', file);
    showToast('Model upload feature — use Dataset Pool & Retrain below for training', 'warn');
    dom.addModelForm.classList.remove('show');
    dom.newModelName.value = ''; dom.newModelFile.value = '';
  }

  // ============================================================
  // 5. DETECTION (POST /api/upload → GET /api/results)
  // ============================================================
  function setHealth(kind, text) {
    dom.healthStatus.innerHTML = `<div class="dot ${kind}"></div> ${text}`;
  }

  function updateProgress(status, pct) {
    dom.statusBadge.className = `status status-${status}`;
    dom.statusBadge.innerText = status.charAt(0).toUpperCase() + status.slice(1);
    dom.progressBar.style.width = pct + '%';
    dom.pctText.innerText = pct + '%';
  }

  async function runDetection() {
    if (!state.file || state.isProcessing) return;
    state.isProcessing = true;
    dom.submitBtn.disabled = true;
    dom.submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    dom.progressSection.style.display = 'block';
    dom.downloadSection.style.display = 'none';
    updateProgress('queued', 0);
    dom.jobIdText.textContent = '-';

    const fd = new FormData();
    fd.append('file', state.file);
    fd.append('models', JSON.stringify(state.models));
    const strat = state.mergeStrategy === 'weighted_vote' ? 'weighted' : state.mergeStrategy;
    fd.append('merge_strategy', strat);
    const c = (state.anchorLat !== null) 
      ? { lat: state.anchorLat, lng: state.anchorLng }
      : state.map.getCenter();
    fd.append('center_lat', c.lat); fd.append('center_lon', c.lng);

    try {
      const r = await fetch(ENDPOINTS.upload, { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${r.status}`); }
      const upRes = await r.json();
      state.taskId = upRes.task_id;
      dom.jobIdText.textContent = upRes.task_id;
      updateProgress('running', 30);

      const r2 = await fetch(ENDPOINTS.results(state.taskId));
      if (!r2.ok) throw new Error(`HTTP ${r2.status}`);
      const payload = await r2.json();

      const features = (payload.features || []).map(normalizeFeature);
      const stats = normalizeStats(payload.stats, features);
      state.detections = { features, stats, taskId: state.taskId };

      plotDetections(state.detections);
      updateStats(stats);
      updateLegend(stats.classes || []);
      updateProgress('done', 100);
      showDownloads();
      showToast(`Detection complete: ${features.length} features`, 'ok');
    } catch (err) {
      console.error(err);
      updateProgress('failed', 0);
      showToast(`Detection failed: ${err.message}`, 'err');
    } finally {
      state.isProcessing = false;
      dom.submitBtn.disabled = false;
      dom.submitBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
    }
  }

  // ─── NEW: DETECTION ON MAP VIEW ───────────────────────────
  async function runMapDetection() {
    const btn = dom.detectMapBtn;
    if (!btn) return;
    const originalHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Fetching Map...';
    
    const b = state.map.getBounds();
    const z = state.map.getZoom();
    
    dom.progressSection.style.display = 'block';
    dom.downloadSection.style.display = 'none';
    updateProgress('queued', 0);
    dom.jobIdText.textContent = 'map-view';

    const fd = new FormData();
    fd.append('west', b.getWest());
    fd.append('south', b.getSouth());
    fd.append('east', b.getEast());
    fd.append('north', b.getNorth());
    fd.append('zoom', z);
    fd.append('models', JSON.stringify(state.models));

    try {
      updateProgress('running', 30);
      const r = await fetch(ENDPOINTS.detectMap, { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${r.status}`); }
      
      const payload = await r.json();
      state.taskId = payload.task_id;
      dom.jobIdText.textContent = payload.task_id;

      const features = (payload.features || []).map(normalizeFeature);
      const stats = normalizeStats(payload.stats, features);
      state.detections = { features, stats, taskId: state.taskId };

      plotDetections(state.detections);
      updateStats(stats);
      updateLegend(stats.classes || []);
      updateProgress('done', 100);
      showDownloads();
      showToast(`Map detection complete: ${features.length} features`, 'ok');
    } catch (err) {
      console.error(err);
      updateProgress('failed', 0);
      showToast(`Map detection failed: ${err.message}`, 'err');
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalHtml;
    }
  }

  function normalizeFeature(f) {
    const p = f.properties || {};
    return { class: p.type || p.class || 'unknown', label: p.label || p.type || 'Feature', model: p.model || 'base-v7.6', geometry: f.geometry, properties: { area_m2: p.area_m2||0, confidence: p.confidence||0, usable_area_sqm: p.usable_area_sqm, panel_count: p.panel_count, energy_kwh_yr: p.energy_kwh_yr } };
  }

  function normalizeStats(raw, features) {
    const s = raw || {};
    const cm = new Map(); let ta = 0, te = 0, cs = 0;
    for (const f of features) { const c = f.class||'unknown'; const l = f.label||c; const ex = cm.get(c)||{id:c,label:l,count:0}; ex.count++; cm.set(c,ex); ta += f.properties.area_m2||0; te += f.properties.energy_kwh_yr||0; cs += f.properties.confidence||0; }
    return { total: s.feature_count ?? s.total ?? features.length, classes: Array.from(cm.values()), total_area: s.total_area_m2 ?? s.total_area ?? ta, total_yield: s.total_energy_kwh_yr ?? s.total_yield ?? te, avg_confidence: s.avg_confidence ?? (features.length ? cs/features.length : 0) };
  }

  // ============================================================
  // 6. PLOT POLYGONS ON MAP
  // ============================================================
  function plotDetections(payload) {
    state.polygonLayer.clearLayers();
    if (!payload || !payload.features) return;
    payload.features.forEach((feat, idx) => {
      const classId = feat.class || feat.properties?.class || 'class_h';
      const label = feat.label || feat.properties?.label;
      const style = getClassStyle(classId, label);
      const coords = feat.geometry.coordinates[0].map(([lng, lat]) => [lat, lng]);
      const polygon = L.polygon(coords, { color: style.stroke, weight: 1.5, fillColor: style.fill, fillOpacity: state.fillOpacity });
      const p = feat.properties || {};
      const area = p.area_m2 || 0;
      const conf = p.confidence || 0;
      const model = feat.model || 'ensemble';
      const popup = `<h4>${style.label}</h4>
        <div class="pop-row"><span>Class</span><span>${classId}</span></div>
        <div class="pop-row"><span>Area</span><span>${area.toFixed(1)} m²</span></div>
        <div class="pop-row"><span>Confidence</span><span>${(conf*100).toFixed(1)}%</span></div>
        <div class="pop-row"><span>Model</span><span>${model}</span></div>
        ${p.energy_kwh_yr ? `<div class="pop-row"><span>Energy</span><span>${p.energy_kwh_yr.toLocaleString()} kWh/yr</span></div>` : ''}
        <div class="pop-actions">
          <button class="pop-flag-btn is-missed" data-flag="missed" data-idx="${idx}">Flag: missed</button>
          <button class="pop-flag-btn is-fp" data-flag="false_positive" data-idx="${idx}">Flag: false positive</button>
        </div>`;
      polygon.bindPopup(popup, { maxWidth: 280 });
      polygon.bindTooltip(`${style.label} · ${area.toFixed(0)} m²`, { sticky: true, direction: 'top' });
      polygon.on('popupopen', (e) => {
        const root = e.popup.getElement();
        if (!root) return;
        root.querySelectorAll('.pop-flag-btn').forEach(btn => {
          btn.addEventListener('click', () => { submitFeedback(btn.dataset.flag, btn.dataset.idx); e.popup.close(); });
        });
      });
      state.polygonLayer.addLayer(polygon);
    });
    // Fit bounds
    if (payload.features.length > 0) {
      const ac = [];
      payload.features.forEach(f => f.geometry.coordinates[0].forEach(([lng, lat]) => ac.push([lat, lng])));
      if (ac.length > 0) state.map.fitBounds(L.latLngBounds(ac), { padding: [60, 60], maxZoom: 17 });
    }
  }

  function centroid(latlngs) { const n = latlngs.length - 1; if (n < 3) return [0,0]; const s = latlngs.reduce((a,c)=>[a[0]+c[0],a[1]+c[1]],[0,0]); return [s[0]/n, s[1]/n]; }

  function updateStats(stats) {
    dom.polyCount.textContent = stats.total || 0;
    dom.classCount.textContent = (stats.classes || []).length;
    dom.areaCount.textContent = `${Math.round(stats.total_area || 0).toLocaleString()} m²`;
    dom.energyCount.textContent = `${Math.round(stats.total_yield || 0).toLocaleString()} kWh/yr`;
  }

  function updateLegend(classes) {
    if (!classes || classes.length === 0) { dom.legend.innerHTML = '<span class="legend__item legend__item--empty">No detections yet</span>'; return; }
    dom.legend.innerHTML = classes.map(cls => {
      const s = getClassStyle(cls.id, cls.label);
      return `<span class="legend__item"><span class="legend__swatch" style="background:${s.fill};border:1px solid ${s.stroke};"></span>${s.label} <em style="color:var(--muted);font-style:normal;">(${cls.count})</em></span>`;
    }).join('');
  }

  // ============================================================
  // 7. DOWNLOADS / EXPORT
  // ============================================================
  function showDownloads() { dom.downloadSection.style.display = 'block'; }

  function initDownloads() {
    dom.plotMapBtn.addEventListener('click', () => {
      if (!state.detections) return showToast('No detections to plot', 'warn');
      plotDetections(state.detections);
      showToast('Polygons plotted on map', 'ok');
    });
    dom.refreshMapBtn.addEventListener('click', () => { clearMap(); state.map.setView(DEFAULT_CENTER, DEFAULT_ZOOM); showToast('Map cleared', 'ok'); });

    // Detect on Map View button
    if (dom.detectMapBtn) {
        dom.detectMapBtn.addEventListener('click', runMapDetection);
    }

    // Build download links dynamically
    const formats = [
      { key: 'geojson', icon: 'fa-file-code', name: 'GeoJSON' },
      { key: 'kml', icon: 'fa-map', name: 'KML (Google Earth)' },
      { key: 'csv', icon: 'fa-file-csv', name: 'CSV Report' },
      { key: 'json', icon: 'fa-file-lines', name: 'JSON Metadata' },
      { key: 'shapefile', icon: 'fa-file-zipper', name: 'Shapefile (ZIP)' },
    ];
    // Render export buttons
    let dlHtml = '';
    formats.forEach(f => {
      dlHtml += `<a class="dl-link" data-export="${f.key}"><i class="fas ${f.icon}" style="color:var(--blue)"></i> ${f.name}</a>`;
    });
    dom.downloadLinks.innerHTML = dlHtml;
    dom.downloadLinks.querySelectorAll('[data-export]').forEach(a => {
      a.addEventListener('click', () => handleExport(a.dataset.export));
    });
  }

  async function handleExport(kind) {
    if (!state.detections) return;
    if (state.taskId) {
      try {
        showToast('Fetching export...', 'ok');
        const r = await fetch(ENDPOINTS.export(state.taskId, kind));
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const blob = await r.blob();
        const cd = r.headers.get('Content-Disposition') || '';
        const m = cd.match(/filename="?([^"]+)"?/);
        downloadBlob(blob, m ? m[1] : `detection.${kind}`);
        showToast(`Exported: ${kind}`, 'ok');
        return;
      } catch (err) { showToast(`Backend export failed, using client-side`, 'warn'); }
    }
    // Client-side fallback
    let blob, fn;
    switch (kind) {
      case 'geojson': blob = exportGeoJSON(); fn = 'detections.geojson'; break;
      case 'kml': blob = exportKML(); fn = 'detections.kml'; break;
      case 'csv': blob = exportCSV(); fn = 'report.csv'; break;
      case 'json': blob = exportJSON(); fn = 'metadata.json'; break;
      case 'shapefile': blob = new Blob(['# Use backend for real shapefile'], { type: 'application/zip' }); fn = 'shapefile.zip'; break;
      default: return;
    }
    downloadBlob(blob, fn);
    showToast(`Exported: ${fn}`, 'ok');
  }

  function downloadBlob(blob, fn) { const u = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = u; a.download = fn; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(u); }

  function exportGeoJSON() { const fc = { type: 'FeatureCollection', crs: { type: 'name', properties: { name: 'urn:ogc:def:crs:EPSG::4326' } }, features: state.detections.features.map(f => ({ type: 'Feature', geometry: f.geometry, properties: { type: f.class, model: f.model, ...f.properties } })) }; return new Blob([JSON.stringify(fc, null, 2)], { type: 'application/geo+json' }); }
  function exportKML() { const placemarks = state.detections.features.map(f => { const ring = f.geometry.coordinates[0]; const c = ring.map(([lng,lat]) => `${lng},${lat},0`).join(' '); const p = f.properties; return `<Placemark><name>${f.label}</name><ExtendedData><Data name="area_m2"><value>${p.area_m2||0}</value></Data><Data name="confidence"><value>${p.confidence||0}</value></Data></ExtendedData><Polygon><outerBoundaryIs><LinearRing><coordinates>${c}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>`; }).join('\n'); return new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>GeoScan.AI Detections</name>${placemarks}</Document></kml>`], { type: 'application/vnd.google-earth.kml+xml' }); }
  function exportCSV() { const rows = [['feature_id','type','model','area_m2','confidence','centroid_lat','centroid_lon','polygon_wkt']]; state.detections.features.forEach((f,i) => { const ring = f.geometry.coordinates[0]; const c = centroid(ring.map(([lng,lat])=>[lat,lng])); const wkt = 'POLYGON((' + ring.map(([lng,lat])=>`${lng} ${lat}`).join(', ') + '))'; const p = f.properties; rows.push([i+1, f.class, f.model, (p.area_m2||0).toFixed(2), (p.confidence||0).toFixed(3), c[0].toFixed(6), c[1].toFixed(6), wkt]); }); return new Blob([rows.map(r => r.join(',')).join('\n')], { type: 'text/csv' }); }
  function exportJSON() { return new Blob([JSON.stringify({ task_id: state.taskId, generated_at: new Date().toISOString(), srs: 'EPSG:4326', stats: state.detections.stats, feature_count: state.detections.features.length }, null, 2)], { type: 'application/json' }); }

  // ============================================================
  // 8. DATASET POOL + MERGE RETRAIN
  // ============================================================
  function initDatasetPool() {
    dom.dsZone.addEventListener('click', () => dom.dsInput.click());
    dom.dsInput.addEventListener('change', (e) => { if (e.target.files[0]) handleDsFile(e.target.files[0]); });
    ['dragenter','dragover'].forEach(ev => dom.dsZone.addEventListener(ev, e => { e.preventDefault(); dom.dsZone.classList.add('dragover'); }));
    ['dragleave','drop'].forEach(ev => dom.dsZone.addEventListener(ev, e => { e.preventDefault(); dom.dsZone.classList.remove('dragover'); }));
    dom.dsZone.addEventListener('drop', e => { if (e.dataTransfer.files[0]) handleDsFile(e.dataTransfer.files[0]); });
    dom.dsAddBtn.addEventListener('click', uploadDataset);
    dom.mergeRetrainBtn.addEventListener('click', startMergeRetrain);
    dom.refreshDatasetsBtn.addEventListener('click', loadDatasets);
  }

  function handleDsFile(file) {
    if (!file.name.toLowerCase().endsWith('.zip')) return showToast('Must be .zip', 'err');
    if (file.size > 500*1024*1024) return showToast('Max 500 MB', 'err');
    state.dsFile = file;
    dom.dsText.textContent = file.name;
    dom.dsZone.classList.add('filled');
    dom.dsAddBtn.disabled = false;
  }

  async function uploadDataset() {
    if (!state.dsFile) return;
    const fd = new FormData();
    fd.append('file', state.dsFile);
    const name = dom.dsName.value.trim();
    if (name) fd.append('name', name);
    dom.dsAddBtn.disabled = true;
    dom.dsAddBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    try {
      const r = await fetch(ENDPOINTS.datasetsUpload, { method: 'POST', body: fd });
      if (!r.ok) throw new Error((await r.json().catch({})).detail || `HTTP ${r.status}`);
      const d = await r.json();
      showToast(`Added: ${d.image_count} images, ${d.label_count} labels`, 'ok');
      state.dsFile = null; dom.dsText.textContent = 'Upload dataset .zip'; dom.dsZone.classList.remove('filled'); dom.dsInput.value = ''; dom.dsName.value = '';
      loadDatasets();
    } catch (e) { showToast(`Upload failed: ${e.message}`, 'err'); }
    dom.dsAddBtn.disabled = false;
    dom.dsAddBtn.innerHTML = '<i class="fas fa-plus"></i> Add';
  }

  async function loadDatasets() {
    dom.refreshDatasetsBtn.querySelector('i').classList.add('fa-spin');
    try {
      const r = await fetch(ENDPOINTS.datasetsList);
      if (!r.ok) throw new Error();
      const d = await r.json();
      state.datasets = d.datasets || [];
      renderDsPool();
    } catch { state.datasets = []; renderDsPool(); }
    dom.refreshDatasetsBtn.querySelector('i').classList.remove('fa-spin');
  }

  function renderDsPool() {
    if (state.datasets.length === 0) { dom.dsPool.innerHTML = '<div class="ds-pool__empty">No datasets in pool yet</div>'; dom.mergeRetrainBtn.disabled = true; return; }
    dom.dsPool.innerHTML = state.datasets.map(d => `<div class="ds-item"><i class="fas fa-folder ds-item__icon"></i><div class="ds-item__body"><div class="ds-item__name">${d.name}</div><div class="ds-item__meta">${d.image_count} img · ${d.label_count} lbl</div></div><button class="ds-item__delete" data-del-ds="${d.id}"><i class="fas fa-trash"></i></button></div>`).join('');
    dom.dsPool.querySelectorAll('[data-del-ds]').forEach(b => b.addEventListener('click', async () => { try { await fetch(ENDPOINTS.datasetsDelete(b.dataset.delDs), { method: 'DELETE' }); showToast('Removed', 'ok'); loadDatasets(); } catch { showToast('Delete failed', 'err'); } }));
    dom.mergeRetrainBtn.disabled = false;
  }

  async function startMergeRetrain() {
    if (state.datasets.length === 0 || state.mergeJobId) return;
    const mergeFirst = dom.mergeFirstCheckbox ? dom.mergeFirstCheckbox.checked : false;
    dom.retrainBtnLabel.textContent = 'Retraining...';
    dom.mergeRetrainBtn.disabled = true;
    dom.mergeProgress.style.display = 'block';
    dom.mergeStage.textContent = `Starting (${mergeFirst ? 'merge' : 'no-merge'})...`;
    dom.mergePct.textContent = '0%';
    dom.mergeFill.style.width = '0%';
    dom.mergeStats.style.display = 'none';
    const fd = new FormData();
    fd.append('base_model', 'best_roof.pt');
    fd.append('epochs', '30');
    fd.append('merge_first', mergeFirst ? 'true' : 'false');
    try {
      const r = await fetch(ENDPOINTS.retrainMerge, { method: 'POST', body: fd });
      if (!r.ok) throw new Error((await r.json().catch({})).detail || `HTTP ${r.status}`);
      const d = await r.json();
      state.mergeJobId = d.job_id;
      dom.mergeStats.style.display = 'block';
      dom.mergeStats.textContent = `${d.total_datasets} datasets · ${d.total_images} images · ${d.mode}`;
      dom.mergeStage.textContent = mergeFirst ? 'Merging datasets...' : 'Building multi-path yaml...';
      pollMergeStatus();
    } catch (e) {
      dom.mergeStage.textContent = `Error: ${e.message}`;
      dom.mergePct.textContent = '✗';
      dom.retrainBtnLabel.textContent = 'Retrain on All Datasets';
      dom.mergeRetrainBtn.disabled = false;
      showToast(`Retrain failed: ${e.message}`, 'err');
    }
  }

  function pollMergeStatus() {
    if (!state.mergeJobId) return;
    setTimeout(async () => {
      try {
        const r = await fetch(ENDPOINTS.retrainMergeStatus(state.mergeJobId));
        if (!r.ok) throw new Error();
        const s = await r.json();
        dom.mergeStage.textContent = s.stage.replace(/_/g, ' ');
        dom.mergePct.textContent = `${s.progress}%`;
        dom.mergeFill.style.width = `${s.progress}%`;
        if (s.status === 'done') {
          dom.mergeStage.textContent = 'Done — model hot-swapped ✓';
          dom.mergePct.textContent = '✓';
          showToast('Retraining complete! Model replaced.', 'ok');
          state.mergeJobId = null;
          dom.retrainBtnLabel.textContent = 'Retrain on All Datasets';
          dom.mergeRetrainBtn.disabled = state.datasets.length === 0;
        } else if (s.status === 'error') {
          dom.mergeStage.textContent = `Error: ${s.error}`;
          dom.mergePct.textContent = '✗';
          state.mergeJobId = null;
          dom.retrainBtnLabel.textContent = 'Retrain on All Datasets';
          dom.mergeRetrainBtn.disabled = state.datasets.length === 0;
          showToast(`Failed: ${s.error}`, 'err');
        } else { pollMergeStatus(); }
      } catch { pollMergeStatus(); }
    }, 1500);
  }

  // ============================================================
  // 9. FEEDBACK
  // ============================================================
  async function submitFeedback(type, idx) {
    if (!state.taskId) return showToast('No active detection run', 'warn');
    const feat = state.detections?.features?.[idx];
    const note = feat ? `${type} on ${feat.class} (${feat.properties?.area_m2?.toFixed(1)} m²)` : type;
    const fd = new FormData();
    fd.append('upload_id', state.taskId);
    fd.append('correction_type', type);
    fd.append('note', note);
    try {
      const r = await fetch(ENDPOINTS.feedback, { method: 'POST', body: fd });
      if (!r.ok) throw new Error();
      const d = await r.json();
      state.feedbackCount = d.continuous_learning_pending || 0;
      dom.feedbackCount.textContent = `${state.feedbackCount} / 50 corrections collected`;
      showToast(`Feedback saved (${state.feedbackCount}/50)`, 'ok');
    } catch {
      state.feedbackCount++;
      dom.feedbackCount.textContent = `${state.feedbackCount} / 50 corrections collected (offline)`;
      showToast(`Feedback saved locally (${state.feedbackCount}/50)`, 'warn');
    }
  }

  // ============================================================
  // 10. TOAST + HEALTH
  // ============================================================
  function showToast(msg, kind) {
    dom.toast.textContent = msg;
    dom.toast.className = 'toast';
    if (kind === 'err') dom.toast.classList.add('is-err');
    if (kind === 'warn') dom.toast.classList.add('is-warn');
    dom.toast.style.display = 'flex';
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => { dom.toast.style.display = 'none'; }, 3500);
  }

  async function fetchHealth() {
    try {
      const r = await fetch(ENDPOINTS.health);
      const d = await r.json();
      if (d.status === 'ok') setHealth('ok', `Connected (${(d.base_models_available||[]).length} base models)`);
      else throw new Error();
    } catch { setHealth('fail', 'Backend Offline'); }
  }

  // ============================================================
  // 11. MOCK DETECTION (fallback when backend unreachable)
  // ============================================================
  function mockDetectionResult(fileName, models, mergeStrategy) {
    const center = state.map.getCenter();
    const bounds = state.map.getBounds();
    const latSpan = (bounds.getNorth() - bounds.getSouth()) * 0.6;
    const lngSpan = (bounds.getEast() - bounds.getWest()) * 0.6;
    const cLat = center.lat, cLng = center.lng;
    const classes = [
      { id: 'rooftop', label: 'Rooftops', count: 8 + Math.floor(Math.random()*6) },
      { id: 'solar_panel', label: 'Solar Panels', count: 12 + Math.floor(Math.random()*8) },
    ];
    const features = [];
    function rect(lat, lng, dLat, dLng) { return [[[lng-dLng,lat-dLat],[lng+dLng,lat-dLat],[lng+dLng,lat+dLat],[lng-dLng,lat+dLat],[lng-dLng,lat-dLat]]]; }
    const sizes = { rooftop: { f: [0.010, 0.025], l: [0.015, 0.030] }, solar_panel: { f: [0.002, 0.005], l: [0.003, 0.008] } };
    classes.forEach(cls => {
      const sp = sizes[cls.id] || sizes.rooftop;
      for (let i = 0; i < cls.count; i++) {
        const lat = cLat + (Math.random()-0.5)*latSpan;
        const lng = cLng + (Math.random()-0.5)*lngSpan;
        const dLat = sp.f[0]*latSpan + Math.random()*(sp.f[1]-sp.f[0])*latSpan;
        const dLng = sp.l[0]*lngSpan + Math.random()*(sp.l[1]-sp.l[0])*lngSpan;
        const area = (dLat*111000) * (dLng*111000*Math.cos(cLat*Math.PI/180));
        features.push({ class: cls.id, label: cls.label, model: models[Math.floor(Math.random()*models.length)], geometry: { type: 'Polygon', coordinates: rect(lat,lng,dLat,dLng) }, properties: { area_m2: +area.toFixed(1), confidence: 0.78+Math.random()*0.2, energy_kwh_yr: Math.round(area * 280 * 0.18) } });
      }
    });
    const totalArea = features.reduce((s,f) => s + f.properties.area_m2, 0);
    const totalYield = features.reduce((s,f) => s + (f.properties.energy_kwh_yr||0), 0);
    const avgConf = features.reduce((s,f) => s + f.properties.confidence, 0) / features.length;
    return { features, stats: { total: features.length, classes: classes.map(c => ({ id: c.id, label: c.label, count: features.filter(f => f.class === c.id).length })), total_area: totalArea, total_yield: totalYield, avg_confidence: avgConf } };
  }

  // ============================================================
  // BOOT
  // ============================================================
  function init() {
    initMap();
    initOpacity();
    initUpload();
    initModels();
    initDownloads();
    initDatasetPool();
    fetchHealth();
    fetchModels();
    loadDatasets();
    console.info('[GeoScan.AI] Dashboard initialized.');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();