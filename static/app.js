const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"];
const RDBU = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7",
              "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"];
const COLORS = { m1: "#2a78d6", m2: "#eb6834", ink: "#172033", ink2: "#5a6578", base: "#9aa3b2" };
const CELL = 12, PAD_L = 42, PAD_B = 32, PAD_T = 6, PAD_R = 8;
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (id) => document.getElementById(id);
const state = { lead: 1, scale: "sp", testMetrics: [], last: null };

const RDBU_RGB = RDBU.map((h) => {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
});

function spColor(v) {
  const t = Math.min(1, Math.max(0, (v + 3) / 6)) * (RDBU_RGB.length - 1);
  const i = Math.min(RDBU_RGB.length - 2, Math.floor(t));
  const f = t - i;
  const a = RDBU_RGB[i], b = RDBU_RGB[i + 1];
  return `rgb(${a.map((c, k) => Math.round(c + (b[k] - c) * f)).join(",")})`;
}

const easeOut = (t) => 1 - Math.pow(1 - t, 3);

function animate(duration, step) {
  if (REDUCED) { step(1); return; }
  const start = performance.now();
  const frame = (now) => {
    const t = Math.min(1, (now - start) / duration);
    step(easeOut(t));
    if (t < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function setupSegmented(id, key, onChange) {
  const group = $(id);
  const buttons = [...group.querySelectorAll("button")];
  const slider = group.querySelector(".slider");
  const place = () => {
    const active = group.querySelector("button.active");
    slider.style.width = `${active.offsetWidth}px`;
    slider.style.transform = `translateX(${active.offsetLeft - 3}px)`;
  };
  buttons.forEach((btn) => btn.addEventListener("click", () => {
    buttons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state[key] = key === "lead" ? Number(btn.dataset.value) : btn.dataset.value;
    place();
    onChange();
  }));
  window.addEventListener("resize", place);
  requestAnimationFrame(place);
}

function setupControls(info) {
  const year = $("year");
  for (let y = info.last_year; y >= info.first_year; y--) year.add(new Option(y, y));
  const month = $("month");
  MONTHS.forEach((m, i) => month.add(new Option(m, i + 1)));
  month.value = 7;
  state.testMetrics = info.test_metrics;
  setupSegmented("lead", "lead", updateTestR2);
  setupSegmented("scale", "scale", updateTestR2);
  $("run").addEventListener("click", runForecast);
  updateTestR2();
}

function updateTestR2() {
  const pick = (m) => state.testMetrics.find((r) => r.model === m && r.lead === state.lead && r.target === state.scale);
  const a = pick("model1"), b = pick("model2");
  if (!a || !b) return;
  $("test-r2").innerHTML = `Test-period R² (2013–2025)<br>${state.scale}, ` +
    `${state.lead}-month lead: Model 1 <b>${a.R2.toFixed(3)}</b> · Model 2 <b>${b.R2.toFixed(3)}</b>`;
}

async function runForecast() {
  const btn = $("run");
  btn.classList.add("busy");
  btn.querySelector(".run-text").textContent = "Running…";
  $("loading").classList.add("show");
  $("error").classList.add("hidden");
  const params = new URLSearchParams({
    year: $("year").value, month: $("month").value, lead: state.lead, scale: state.scale,
  });
  try {
    const res = await fetch(`/api/forecast?${params}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    render(data);
  } catch (err) {
    $("error").textContent = err.message;
    $("error").classList.remove("hidden");
  } finally {
    btn.classList.remove("busy");
    btn.querySelector(".run-text").textContent = "Run Forecast";
    $("loading").classList.remove("show");
  }
}

function replayReveal() {
  document.querySelectorAll(".reveal").forEach((el) => {
    el.classList.add("replay");
    void el.offsetWidth;
    el.classList.remove("replay");
  });
}

function render(d) {
  state.last = d;
  document.body.classList.remove("empty");
  const label = d.scale;
  replayReveal();

  const s1 = d.scores.model1, s2 = d.scores.model2;
  countUp("m1-mse", s1.MSE); countUp("m1-mae", s1.MAE);
  countUp("m2-mse", s2.MSE); countUp("m2-mae", s2.MAE);
  setDelta("d-mse", s2.MSE - s1.MSE, false);
  setDelta("d-mae", s2.MAE - s1.MAE, false);

  $("maps-title").textContent = `${label} maps for ${d.target}`;
  animate(900, (p) => {
    drawMap($("map-obs"), d.observed, d.lats, d.lons, p);
    drawMap($("map-m1"), d.model1, d.lats, d.lons, p);
    drawMap($("map-m2"), d.model2, d.lats, d.lons, p);
  });
  drawBars(s1, s2);
}

function countUp(id, target) {
  const el = $(id);
  if (!Number.isFinite(target)) { el.textContent = "–"; return; }
  animate(800, (p) => { el.textContent = (target * p).toFixed(3); });
}

function setDelta(id, diff, higherIsBetter) {
  const el = $(id);
  const better = higherIsBetter ? diff > 0 : diff < 0;
  el.textContent = `${diff >= 0 ? "▲ +" : "▼ −"}${Math.abs(diff).toFixed(3)} vs Model 1`;
  el.className = Math.abs(diff) < 5e-4 ? "" : better ? "good" : "bad";
  void el.offsetWidth;
  el.classList.add("show");
}

function drawMap(canvas, grid, lats, lons, progress = 1) {
  const H = grid.length, W = grid[0].length;
  const cssW = PAD_L + W * CELL + PAD_R, cssH = PAD_T + H * CELL + PAD_B;
  const dpr = Math.max(2, window.devicePixelRatio || 1);
  if (canvas.width !== cssW * dpr) { canvas.width = cssW * dpr; canvas.height = cssH * dpr; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = "#eef1f5";
  ctx.fillRect(PAD_L, PAD_T, W * CELL, H * CELL);

  const front = progress * (H + W);
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      const v = grid[i][j];
      if (v === null) continue;
      const local = Math.min(1, Math.max(0, front - (i + (W - 1 - j) * 0.6)) / 4);
      if (local <= 0) continue;
      ctx.globalAlpha = local;
      ctx.fillStyle = spColor(v);
      ctx.fillRect(PAD_L + j * CELL, PAD_T + i * CELL, CELL + 0.5, CELL + 0.5);
    }
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#a9b2c0";
  ctx.lineWidth = 1;
  ctx.strokeRect(PAD_L + 0.5, PAD_T + 0.5, W * CELL - 1, H * CELL - 1);

  ctx.fillStyle = COLORS.ink2;
  ctx.font = "14px Segoe UI, system-ui, sans-serif";
  ctx.textAlign = "center";
  lons.forEach((lon, j) => {
    if (Math.abs(lon - Math.round(lon)) < 1e-6) ctx.fillText(`${lon}°`, PAD_L + j * CELL + CELL / 2, PAD_T + H * CELL + 20);
  });
  ctx.textAlign = "right";
  lats.forEach((lat, i) => {
    if (Math.abs(lat - Math.round(lat)) < 1e-6) ctx.fillText(`${lat}°`, PAD_L - 5, PAD_T + i * CELL + CELL / 2 + 5);
  });
  canvas._map = { grid, lats, lons, cssW, cssH };
}

function attachTooltip(canvas) {
  const tip = $("tooltip");
  canvas.addEventListener("mousemove", (e) => {
    const m = canvas._map;
    if (!m) return;
    const r = canvas.getBoundingClientRect();
    const scale = Math.min(r.width / m.cssW, r.height / m.cssH);
    const offX = (r.width - m.cssW * scale) / 2, offY = (r.height - m.cssH * scale) / 2;
    const x = (e.clientX - r.left - offX) / scale - PAD_L;
    const y = (e.clientY - r.top - offY) / scale - PAD_T;
    const j = Math.floor(x / CELL), i = Math.floor(y / CELL);
    const v = m.grid[i] && m.grid[i][j];
    if (x < 0 || y < 0 || v === undefined || v === null) { tip.classList.add("hidden"); return; }
    tip.innerHTML = `${m.lats[i].toFixed(2)}°N, ${m.lons[j].toFixed(2)}°E<br>sp <b>${v.toFixed(2)}</b>`;
    tip.style.left = `${e.clientX + 14}px`;
    tip.style.top = `${e.clientY + 14}px`;
    tip.classList.remove("hidden");
  });
  canvas.addEventListener("mouseleave", () => tip.classList.add("hidden"));
}

function svg(tag, attrs, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  if (text !== undefined) el.textContent = text;
  return el;
}

function drawBars(s1, s2) {
  const box = $("bars");
  box.innerHTML = "";
  const items = [["MSE", "MSE", "lower is better"], ["MAE", "MAE", "lower is better"]];
  items.forEach(([key, name, hint], n) => {
    const wrap = document.createElement("div");
    box.appendChild(wrap);
    const W = Math.max(220, wrap.clientWidth - 16), H = Math.max(140, wrap.clientHeight - 12);
    const top = 30, bottom = 22;
    const vals = [s1[key], s2[key]];
    const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    const span = (hi - lo) || 1;
    const yMin = lo - (lo < 0 ? 0.22 * span : 0), yMax = hi + 0.22 * span;
    const y = (v) => top + (yMax - v) / (yMax - yMin) * (H - top - bottom);
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}` });
    const title = svg("text", { x: W / 2, y: 18, "text-anchor": "middle", "font-size": 13, fill: COLORS.ink });
    title.appendChild(svg("tspan", { "font-weight": 700 }, name));
    title.appendChild(svg("tspan", { fill: COLORS.ink2, "font-size": 11.5 }, `  (${hint})`));
    s.appendChild(title);
    s.appendChild(svg("line", { x1: 16, x2: W - 16, y1: y(0), y2: y(0), stroke: COLORS.base }));
    const bw = Math.min(90, W / 4), gap = (W - 2 * bw) / 3;
    vals.forEach((v, idx) => {
      const x = gap + idx * (bw + gap);
      const y0 = y(0), y1 = y(v);
      const rect = svg("rect", { x, y: Math.min(y0, y1), width: bw, height: Math.max(1, Math.abs(y1 - y0)), rx: 6,
        fill: idx ? COLORS.m2 : COLORS.m1, class: `bar-rect ${v >= 0 ? "up" : "down"}` });
      rect.style.animationDelay = `${n * 90 + idx * 120}ms`;
      s.appendChild(rect);
      s.appendChild(svg("text", { x: x + bw / 2, y: v >= 0 ? y1 - 6 : y1 + 15, "text-anchor": "middle",
        "font-size": 12.5, "font-weight": 600, fill: COLORS.ink, class: "bar-label" }, v.toFixed(3)));
      s.appendChild(svg("text", { x: x + bw / 2, y: H - 6, "text-anchor": "middle", "font-size": 12,
        fill: COLORS.ink2 }, idx ? "Model 2" : "Model 1"));
    });
    wrap.appendChild(s);
  });
}

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (state.last) drawBars(state.last.scores.model1, state.last.scores.model2); }, 200);
});

["map-obs", "map-m1", "map-m2"].forEach((id) => attachTooltip($(id)));

fetch("/api/info")
  .then((r) => r.json())
  .then(setupControls)
  .catch(() => {
    $("error").textContent = "Could not reach the server. Is web.py running?";
    $("error").classList.remove("hidden");
  });
