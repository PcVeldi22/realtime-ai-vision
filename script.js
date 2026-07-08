// NeuroVision - Real-time AI object detection in the browser
// Uses TensorFlow.js + COCO-SSD for object detection, and Tesseract.js
// for on-device OCR so the app can read text printed on packaging.
// Everything runs 100% client-side.

const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const bgCanvas = document.getElementById("bg-canvas");
const bgCtx = bgCanvas.getContext("2d");

const statusEl = document.getElementById("status");
const fpsEl = document.getElementById("fps");
const countEl = document.getElementById("count");
const latencyEl = document.getElementById("latency");
const listEl = document.getElementById("detection-list");
const startBtn = document.getElementById("start-btn");
const snapshotBtn = document.getElementById("snapshot-btn");
const ocrTextEl = document.getElementById("ocr-text");

let model = null;
let running = false;
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// Minimum confidence required before a detection is shown.
const MIN_CONFIDENCE = 0.6;
const MAX_BOXES = 15;

const COLORS = ["#00e5ff", "#7c4dff", "#34d399", "#ffb020", "#ff4d6d", "#f472b6"];
const colorForClass = (() => {
      const map = new Map();
      let i = 0;
      return (cls) => {
              if (!map.has(cls)) {
                        map.set(cls, COLORS[i % COLORS.length]);
                        i++;
              }
              return map.get(cls);
      };
})();

function resizeBgCanvas() {
      bgCanvas.width = window.innerWidth;
      bgCanvas.height = window.innerHeight;
}
window.addEventListener("resize", resizeBgCanvas);
resizeBgCanvas();

const particles = Array.from({ length: 60 }, () => ({
      x: Math.random(),
      y: Math.random(),
      vx: (Math.random() - 0.5) * 0.0006,
      vy: (Math.random() - 0.5) * 0.0006,
}));

function drawBackground() {
      bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
      bgCtx.fillStyle = "#00e5ff";
      particles.forEach((p) => {
              p.x += p.vx;
              p.y += p.vy;
              if (p.x < 0 || p.x > 1) p.vx *= -1;
              if (p.y < 0 || p.y > 1) p.vy *= -1;
              const x = p.x * bgCanvas.width;
              const y = p.y * bgCanvas.height;
              bgCtx.globalAlpha = 0.4;
              bgCtx.beginPath();
              bgCtx.arc(x, y, 1.6, 0, Math.PI * 2);
              bgCtx.fill();
      });
      requestAnimationFrame(drawBackground);
}
drawBackground();

// ---------- OCR (reads printed text on packaging/labels) ----------
let ocrWorker = null;
let ocrReady = false;
let ocrBusy = false;
let lastOcrAttempt = 0;
const OCR_INTERVAL_MS = 1800;
const OCR_MIN_WIDTH = 50;
const OCR_MIN_HEIGHT = 40;

let lastOcrBox = null;
let lastOcrText = "";
let lastOcrTime = 0;
const OCR_LABEL_TTL = 5000;

const ocrCanvas = document.createElement("canvas");
const ocrCtx = ocrCanvas.getContext("2d");

async function initOcr() {
      try {
              ocrWorker = await Tesseract.createWorker("eng");
              ocrReady = true;
      } catch (err) {
              console.error("OCR init failed:", err);
              ocrTextEl.textContent = "OCR unavailable in this browser";
      }
}

function pickOcrTarget(predictions) {
      let best = null;
      let bestArea = 0;
      predictions.forEach((p) => {
              const [, , w, h] = p.bbox;
              if (w < OCR_MIN_WIDTH || h < OCR_MIN_HEIGHT) return;
              const area = w * h;
              if (area > bestArea) {
                        bestArea = area;
                        best = p;
              }
      });
      return best;
}

async function runOcr(pred) {
      ocrBusy = true;
      try {
              const [x, y, w, h] = pred.bbox;
              const scale = 2;
              ocrCanvas.width = Math.max(1, Math.round(w * scale));
              ocrCanvas.height = Math.max(1, Math.round(h * scale));
              ocrCtx.drawImage(video, x, y, w, h, 0, 0, ocrCanvas.width, ocrCanvas.height);

        const { data } = await ocrWorker.recognize(ocrCanvas);
              const text = (data.text || "").replace(/\s+/g, " ").trim();

        if (text.length >= 2) {
                  lastOcrBox = pred.bbox;
                  lastOcrText = text;
                  lastOcrTime = performance.now();
                  ocrTextEl.textContent = text.length > 90 ? text.slice(0, 90) + "…" : text;
        } else {
                  ocrTextEl.textContent = "No readable text found in view";
        }
      } catch (err) {
              console.error("OCR recognize failed:", err);
      } finally {
              ocrBusy = false;
      }
}

// ---------- Camera + object detection ----------
async function setup() {
      startBtn.disabled = true;
      startBtn.textContent = "Starting…";
      statusEl.textContent = "Requesting camera…";
      statusEl.className = "hud-value status-loading";

  try {
          const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false,
          });
          video.srcObject = stream;
          await new Promise((resolve) => (video.onloadedmetadata = resolve));
          video.play();

        overlay.width = video.videoWidth;
          overlay.height = video.videoHeight;

        statusEl.textContent = "Loading AI models…";

        const [loadedModel] = await Promise.all([
                  cocoSsd.load({ base: "mobilenet_v2" }),
                  initOcr(),
                ]);
          model = loadedModel;

        statusEl.textContent = "Live";
          statusEl.className = "hud-value status-ready";
          startBtn.textContent = "Camera Running";
          snapshotBtn.disabled = false;
          running = true;
          detectFrame();
  } catch (err) {
          console.error(err);
          statusEl.textContent = "Camera/model error";
          statusEl.className = "hud-value status-error";
          startBtn.disabled = false;
          startBtn.textContent = "Start Camera";
  }
}

async function detectFrame() {
      if (!running || !model) return;

  const t0 = performance.now();
      const rawPredictions = await model.detect(video, MAX_BOXES, MIN_CONFIDENCE);
      const t1 = performance.now();

  const predictions = filterPredictions(rawPredictions);
      render(predictions);
      latencyEl.textContent = `${(t1 - t0).toFixed(0)} ms`;

  if (ocrReady && !ocrBusy && performance.now() - lastOcrAttempt > OCR_INTERVAL_MS) {
          const target = pickOcrTarget(predictions);
          if (target) {
                    lastOcrAttempt = performance.now();
                    runOcr(target);
          }
  }

  frameCount++;
      const now = performance.now();
      if (now - lastFrameTime >= 1000) {
              fps = frameCount;
              frameCount = 0;
              lastFrameTime = now;
              fpsEl.textContent = fps;
      }

  requestAnimationFrame(detectFrame);
}

function boxArea(box) {
      return Math.max(0, box[2]) * Math.max(0, box[3]);
}

function iou(a, b) {
      const [ax, ay, aw, ah] = a;
      const [bx, by, bw, bh] = b;
      const x1 = Math.max(ax, bx);
      const y1 = Math.max(ay, by);
      const x2 = Math.min(ax + aw, bx + bw);
      const y2 = Math.min(ay + ah, by + bh);
      const interW = Math.max(0, x2 - x1);
      const interH = Math.max(0, y2 - y1);
      const interArea = interW * interH;
      const unionArea = boxArea(a) + boxArea(b) - interArea;
      return unionArea <= 0 ? 0 : interArea / unionArea;
}

function filterPredictions(predictions) {
      const sorted = [...predictions]
        .filter((p) => p.score >= MIN_CONFIDENCE)
        .sort((a, b) => b.score - a.score);

  const kept = [];
      for (const pred of sorted) {
              const overlapsKept = kept.some((k) => iou(k.bbox, pred.bbox) > 0.6);
              if (!overlapsKept) kept.push(pred);
      }
      return kept;
}

function render(predictions) {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      countEl.textContent = predictions.length;

  if (predictions.length === 0) {
          listEl.innerHTML = `<li class="empty">No objects detected yet</li>`;
          return;
  }

  listEl.innerHTML = "";

  const ocrStillFresh = performance.now() - lastOcrTime < OCR_LABEL_TTL;

  predictions
        .sort((a, b) => b.score - a.score)
        .forEach((pred) => {
                  const [x, y, w, h] = pred.bbox;
                  const color = colorForClass(pred.class);

                       ctx.strokeStyle = color;
                  ctx.lineWidth = 2.5;
                  ctx.strokeRect(x, y, w, h);

                       let label = `${pred.class} ${(pred.score * 100).toFixed(0)}%`;
                  let readText = null;
                  if (ocrStillFresh && lastOcrBox && iou(lastOcrBox, pred.bbox) > 0.3) {
                              readText = lastOcrText.length > 24 ? lastOcrText.slice(0, 24) + "…" : lastOcrText;
                              label = `${pred.class} \u2192 "${readText}"`;
                  }

                       ctx.font = "600 15px Segoe UI, sans-serif";
                  const textWidth = ctx.measureText(label).width;

                       ctx.fillStyle = color;
                  ctx.fillRect(x - 1, y - 22, textWidth + 12, 22);
                  ctx.fillStyle = "#04101a";
                  ctx.fillText(label, x + 5, y - 6);

                       const li = document.createElement("li");
                  const readHtml = readText ? ` &rarr; <em>"${readText}"</em>` : "";
                  li.innerHTML = `<span>${pred.class}${readHtml}</span><span class="conf">${(pred.score * 100).toFixed(1)}%</span>`;
                  li.style.borderLeftColor = color;
                  listEl.appendChild(li);
        });
}

function takeSnapshot() {
      const composite = document.createElement("canvas");
      composite.width = overlay.width;
      composite.height = overlay.height;
      const cctx = composite.getContext("2d");
      cctx.drawImage(video, 0, 0, composite.width, composite.height);
      cctx.drawImage(overlay, 0, 0);

  const link = document.createElement("a");
      link.download = `neurovision-snapshot-${Date.now()}.png`;
      link.href = composite.toDataURL("image/png");
      link.click();
}

startBtn.addEventListener("click", setup);
snapshotBtn.addEventListener("click", takeSnapshot);
