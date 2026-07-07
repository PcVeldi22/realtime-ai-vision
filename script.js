// NeuroVision - Real-time AI object detection in the browser
// Uses TensorFlow.js + COCO-SSD, 100% client-side.

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

let model = null;
let running = false;
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

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

                                    // subtle animated particle grid in the background
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

                                                                                              async function setup() {
                                                                                                startBtn.disabled = true;
                                                                                                  startBtn.textContent = "Starting…";
                                                                                                    statusEl.textContent = "Requesting camera…";
                                                                                                      statusEl.className = "hud-value status-loading";
                                                                                                      
                                                                                                        try {
                                                                                                            const stream = await navigator.mediaDevices.getUserMedia({
                                                                                                                  video: { facingMode: "user" },
                                                                                                                        audio: false,
                                                                                                                            });
                                                                                                                                video.srcObject = stream;
                                                                                                                                    await new Promise((resolve) => (video.onloadedmetadata = resolve));
                                                                                                                                        video.play();
                                                                                                                                        
                                                                                                                                            overlay.width = video.videoWidth;
                                                                                                                                                overlay.height = video.videoHeight;
                                                                                                                                                
                                                                                                                                                    statusEl.textContent = "Loading AI model…";
                                                                                                                                                    
                                                                                                                                                        model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
                                                                                                                                                        
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
                                                                                                                                                                                                              const predictions = await model.detect(video);
                                                                                                                                                                                                                const t1 = performance.now();
                                                                                                                                                                                                                
                                                                                                                                                                                                                  render(predictions);
                                                                                                                                                                                                                    latencyEl.textContent = `${(t1 - t0).toFixed(0)} ms`;
                                                                                                                                                                                                                    
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
                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                              function render(predictions) {
                                                                                                                                                                                                                                                ctx.clearRect(0, 0, overlay.width, overlay.height);
                                                                                                                                                                                                                                                  countEl.textContent = predictions.length;
                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                    if (predictions.length === 0) {
                                                                                                                                                                                                                                                        listEl.innerHTML = `<li class="empty">No objects detected yet</li>`;
                                                                                                                                                                                                                                                            return;
                                                                                                                                                                                                                                                              }
                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                listEl.innerHTML = "";
                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                  predictions
                                                                                                                                                                                                                                                                      .sort((a, b) => b.score - a.score)
                                                                                                                                                                                                                                                                          .forEach((pred) => {
                                                                                                                                                                                                                                                                                const [x, y, w, h] = pred.bbox;
                                                                                                                                                                                                                                                                                      const color = colorForClass(pred.class);
                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                            ctx.strokeStyle = color;
                                                                                                                                                                                                                                                                                                  ctx.lineWidth = 2.5;
                                                                                                                                                                                                                                                                                                        ctx.strokeRect(x, y, w, h);
                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                              const label = `${pred.class} ${(pred.score * 100).toFixed(0)}%`;
                                                                                                                                                                                                                                                                                                                    ctx.font = "600 15px Segoe UI, sans-serif";
                                                                                                                                                                                                                                                                                                                          const textWidth = ctx.measureText(label).width;
                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                ctx.fillStyle = color;
                                                                                                                                                                                                                                                                                                                                      ctx.fillRect(x - 1, y - 22, textWidth + 12, 22);
                                                                                                                                                                                                                                                                                                                                            ctx.fillStyle = "#04101a";
                                                                                                                                                                                                                                                                                                                                                  ctx.fillText(label, x + 5, y - 6);
                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                        const li = document.createElement("li");
                                                                                                                                                                                                                                                                                                                                                              li.innerHTML = `<span>${pred.class}</span><span class="conf">${(pred.score * 100).toFixed(1)}%</span>`;
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
