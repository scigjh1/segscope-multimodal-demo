const state = {
  imageData: null,
  auxiliaryData: null,
  image: null,
  auxiliaryImage: null,
  roi: null,
  dragging: false,
  dragStart: null,
  result: null,
  inputView: "rgb",
  resultView: "overlay",
  history: JSON.parse(localStorage.getItem("segscope-history") || "[]"),
  runNumber: Number(localStorage.getItem("segscope-run-number") || "0"),
};

const elements = {
  canvas: document.querySelector("#inputCanvas"),
  canvasEmpty: document.querySelector("#canvasEmpty"),
  imageInput: document.querySelector("#imageInput"),
  auxInput: document.querySelector("#auxInput"),
  imageFileName: document.querySelector("#imageFileName"),
  auxFileName: document.querySelector("#auxFileName"),
  sampleButton: document.querySelector("#sampleButton"),
  runButton: document.querySelector("#runButton"),
  resetButton: document.querySelector("#resetButton"),
  clearRoiButton: document.querySelector("#clearRoiButton"),
  clearHistoryButton: document.querySelector("#clearHistoryButton"),
  fusionWeight: document.querySelector("#fusionWeight"),
  fusionWeightValue: document.querySelector("#fusionWeightValue"),
  iterations: document.querySelector("#iterations"),
  mode: document.querySelector("#mode"),
  resultImage: document.querySelector("#resultImage"),
  resultEmpty: document.querySelector("#resultEmpty"),
  resultState: document.querySelector("#resultState"),
  pipelineLabel: document.querySelector("#pipelineLabel"),
  imageDimensions: document.querySelector("#imageDimensions"),
  runId: document.querySelector("#runId"),
  runState: document.querySelector("#runState"),
  downloadButton: document.querySelector("#downloadButton"),
  engineStatus: document.querySelector("#engineStatus"),
  historyBody: document.querySelector("#historyBody"),
  toast: document.querySelector("#toast"),
};

const context = elements.canvas.getContext("2d");

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.className = `toast show${isError ? " error" : ""}`;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { elements.toast.className = "toast"; }, 2600);
}

function setRunState(label, type) {
  elements.runState.textContent = label;
  elements.runState.className = `status ${type}`;
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function urlToDataUrl(url) {
  const response = await fetch(url);
  const blob = await response.blob();
  return readFile(blob);
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = dataUrl;
  });
}

function fitCanvasToImage(image) {
  const maxWidth = 960;
  const maxHeight = 620;
  const scale = Math.min(maxWidth / image.naturalWidth, maxHeight / image.naturalHeight, 1);
  elements.canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  elements.canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
  elements.imageDimensions.textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
}

function drawCanvas() {
  const image = state.inputView === "aux" ? state.auxiliaryImage : state.image;
  context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  if (!image) {
    elements.canvasEmpty.style.display = "flex";
    return;
  }
  elements.canvasEmpty.style.display = "none";
  context.drawImage(image, 0, 0, elements.canvas.width, elements.canvas.height);
  if (state.roi) {
    context.save();
    context.strokeStyle = "#ff6827";
    context.fillStyle = "rgba(255, 104, 39, 0.09)";
    context.lineWidth = 3;
    context.setLineDash([9, 6]);
    context.fillRect(state.roi.x, state.roi.y, state.roi.width, state.roi.height);
    context.strokeRect(state.roi.x, state.roi.y, state.roi.width, state.roi.height);
    context.restore();
  }
}

function updateRoiValues() {
  for (const key of ["X", "Y", "W", "H"]) {
    const prop = { X: "x", Y: "y", W: "width", H: "height" }[key];
    document.querySelector(`#roi${key}`).textContent = state.roi ? Math.round(state.roi[prop]) : "-";
  }
}

async function setPrimaryImage(dataUrl, fileName) {
  state.imageData = dataUrl;
  state.image = await loadImage(dataUrl);
  fitCanvasToImage(state.image);
  state.roi = null;
  elements.imageFileName.textContent = fileName;
  drawCanvas();
  updateRoiValues();
}

async function setAuxiliaryImage(dataUrl, fileName) {
  state.auxiliaryData = dataUrl;
  state.auxiliaryImage = await loadImage(dataUrl);
  elements.auxFileName.textContent = fileName;
  if (state.inputView === "aux") drawCanvas();
}

async function loadSample(prefix = "sample") {
  try {
    const [imageData, auxiliaryData] = await Promise.all([
      urlToDataUrl(`/assets/${prefix}_rgb.png`),
      urlToDataUrl(`/assets/${prefix}_aux.png`),
    ]);
    await setPrimaryImage(imageData, `${prefix}_rgb.png`);
    await setAuxiliaryImage(auxiliaryData, `${prefix}_aux.png`);
    state.roi = {
      x: Math.round(elements.canvas.width * 0.14),
      y: Math.round(elements.canvas.height * 0.16),
      width: Math.round(elements.canvas.width * 0.68),
      height: Math.round(elements.canvas.height * 0.68),
    };
    drawCanvas();
    updateRoiValues();
    showToast("样例图像已载入");
  } catch {
    showToast("样例载入失败", true);
  }
}

function pointerPosition(event) {
  const rect = elements.canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(elements.canvas.width, (event.clientX - rect.left) * elements.canvas.width / rect.width)),
    y: Math.max(0, Math.min(elements.canvas.height, (event.clientY - rect.top) * elements.canvas.height / rect.height)),
  };
}

elements.canvas.addEventListener("pointerdown", (event) => {
  if (!state.image) return;
  state.dragging = true;
  state.dragStart = pointerPosition(event);
  elements.canvas.setPointerCapture(event.pointerId);
});

elements.canvas.addEventListener("pointermove", (event) => {
  if (!state.dragging) return;
  const current = pointerPosition(event);
  state.roi = {
    x: Math.min(state.dragStart.x, current.x),
    y: Math.min(state.dragStart.y, current.y),
    width: Math.abs(current.x - state.dragStart.x),
    height: Math.abs(current.y - state.dragStart.y),
  };
  drawCanvas();
  updateRoiValues();
});

elements.canvas.addEventListener("pointerup", (event) => {
  state.dragging = false;
  elements.canvas.releasePointerCapture(event.pointerId);
  if (state.roi && (state.roi.width < 8 || state.roi.height < 8)) state.roi = null;
  drawCanvas();
  updateRoiValues();
});

function roiForOriginalImage() {
  if (!state.roi || !state.image) return null;
  const scaleX = state.image.naturalWidth / elements.canvas.width;
  const scaleY = state.image.naturalHeight / elements.canvas.height;
  return [
    state.roi.x * scaleX,
    state.roi.y * scaleY,
    state.roi.width * scaleX,
    state.roi.height * scaleY,
  ];
}

function activeResultData() {
  return state.result?.[state.resultView] || null;
}

function updateResultView() {
  const data = activeResultData();
  if (!data) {
    elements.resultImage.style.display = "none";
    elements.resultEmpty.style.display = "flex";
    return;
  }
  elements.resultImage.src = data;
  elements.resultImage.style.display = "block";
  elements.resultEmpty.style.display = "none";
}

function updateMetrics(metrics) {
  document.querySelector("#metricLatency").textContent = metrics.processingMs;
  document.querySelector("#metricFps").textContent = metrics.processingMs ? (1000 / metrics.processingMs).toFixed(1) : "-";
  document.querySelector("#metricArea").textContent = metrics.areaRatio;
  document.querySelector("#metricComponents").textContent = metrics.components;
  document.querySelector("#metricAgreement").textContent = metrics.agreement;
}

function renderHistory() {
  if (!state.history.length) {
    elements.historyBody.innerHTML = '<tr><td colspan="7" class="empty-row">暂无实验记录</td></tr>';
    return;
  }
  elements.historyBody.innerHTML = state.history.map((item) => `
    <tr>
      <td>${item.id}</td><td>${item.dimensions}</td><td>${item.modality}</td><td>${item.iterations}</td>
      <td>${item.latency} ms</td><td>${item.fps}</td><td>${item.area}%</td>
    </tr>`).join("");
}

async function runSegmentation() {
  if (!state.imageData) {
    showToast("请先选择主图像", true);
    return;
  }
  setRunState("运行中", "running");
  elements.runButton.disabled = true;
  elements.runButton.innerHTML = '<i data-lucide="loader-circle"></i>正在分割';
  if (window.lucide) window.lucide.createIcons();
  try {
    const response = await fetch("/api/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: state.imageData,
        auxiliary: state.auxiliaryData,
        roi: roiForOriginalImage(),
        iterations: Number(elements.iterations.value),
        mode: elements.mode.value,
        fusionWeight: Number(elements.fusionWeight.value) / 100,
      }),
    });
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || "分割失败");
    state.result = payload.result;
    state.runNumber += 1;
    localStorage.setItem("segscope-run-number", String(state.runNumber));
    const id = `RUN-${String(state.runNumber).padStart(4, "0")}`;
    elements.runId.textContent = id;
    elements.resultState.textContent = `${payload.result.width} × ${payload.result.height}`;
    elements.pipelineLabel.textContent = payload.result.pipeline;
    elements.downloadButton.disabled = false;
    updateResultView();
    updateMetrics(payload.result.metrics);
    const fps = payload.result.metrics.processingMs ? (1000 / payload.result.metrics.processingMs).toFixed(1) : "-";
    state.history.unshift({
      id,
      dimensions: `${payload.result.width}×${payload.result.height}`,
      modality: state.auxiliaryData ? "双模态" : "单模态",
      iterations: elements.mode.value === "fast" ? "快速" : elements.iterations.value,
      latency: payload.result.metrics.processingMs,
      fps,
      area: payload.result.metrics.areaRatio,
    });
    state.history = state.history.slice(0, 8);
    localStorage.setItem("segscope-history", JSON.stringify(state.history));
    renderHistory();
    setRunState("已完成", "success");
    showToast(`分割完成，端到端 ${payload.result.metrics.processingMs} ms`);
  } catch (error) {
    setRunState("失败", "error");
    showToast(error.message, true);
  } finally {
    elements.runButton.disabled = false;
    elements.runButton.innerHTML = '<i data-lucide="play"></i>运行分割';
    if (window.lucide) window.lucide.createIcons();
  }
}

elements.imageInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (file) await setPrimaryImage(await readFile(file), file.name);
});
elements.auxInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (file) await setAuxiliaryImage(await readFile(file), file.name);
});
elements.sampleButton.addEventListener("click", () => loadSample());
document.getElementById("phantomButton").addEventListener("click", () => loadSample("phantom"));
elements.runButton.addEventListener("click", runSegmentation);
elements.fusionWeight.addEventListener("input", () => { elements.fusionWeightValue.textContent = `${elements.fusionWeight.value}%`; });
elements.clearRoiButton.addEventListener("click", () => { state.roi = null; updateRoiValues(); drawCanvas(); });
elements.clearHistoryButton.addEventListener("click", () => { state.history = []; localStorage.removeItem("segscope-history"); renderHistory(); });
elements.resetButton.addEventListener("click", async () => {
  state.result = null;
  state.roi = null;
  elements.resultImage.removeAttribute("src");
  elements.downloadButton.disabled = true;
  document.querySelectorAll(".metric-card strong").forEach((element) => { element.textContent = "-"; });
  setRunState("待运行", "neutral");
  updateResultView();
  await loadSample();
});
elements.downloadButton.addEventListener("click", () => {
  const data = activeResultData();
  if (!data) return;
  const link = document.createElement("a");
  link.href = data;
  link.download = `${elements.runId.textContent}-${state.resultView}.png`;
  link.click();
});

document.querySelectorAll("[data-input-view]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-input-view]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.inputView = button.dataset.inputView;
    drawCanvas();
  });
});
document.querySelectorAll("[data-result-view]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-result-view]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.resultView = button.dataset.resultView;
    updateResultView();
  });
});

async function initialize() {
  if (window.lucide) window.lucide.createIcons();
  renderHistory();
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    elements.engineStatus.classList.add("online");
    elements.engineStatus.lastChild.textContent = health.engine;
    document.querySelector('option[value="private"]').disabled = !health.privatePredictorConfigured;
  } catch {
    elements.engineStatus.lastChild.textContent = "引擎离线";
  }
  await loadSample();
}

initialize();
