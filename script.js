/* Mathematics Behind Physics-Informed Neural Networks: interactive demos.
   Part 1: numerics. Part 2: helpers and demos. Part 3: page flow, guide character, intro story. */

/* =========================================================
   PART 1 - NUMERICS (no DOM access)
   ========================================================= */

/* Seeded random numbers so that demos are reproducible. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

/* Activation functions used in the network demos. */
const ACT = {
  tanh: { fn: Math.tanh },
  relu: { fn: (z) => Math.max(0, z) },
  sigmoid: { fn: (z) => 1 / (1 + Math.exp(-z)) }
};

/* ---- Shallow network: u(x) = sum_j v_j tanh(a_j x + c_j) + d ----
   Parameter vector p = [a_1..a_H, c_1..c_H, v_1..v_H, d]  (3H + 1 numbers).
   The derivatives u_x and u_xx follow from the chain rule (what automatic
   differentiation computes), so they are exact. */
function netU(p, H, x) {
  let s = p[3 * H];
  for (let j = 0; j < H; j++) s += p[2 * H + j] * Math.tanh(p[j] * x + p[H + j]);
  return s;
}

function netD(p, H, x) {
  let u = p[3 * H], ux = 0, uxx = 0;
  for (let j = 0; j < H; j++) {
    const a = p[j], v = p[2 * H + j];
    const t = Math.tanh(a * x + p[H + j]);
    const s = 1 - t * t;                 // tanh'(z)
    u += v * t;
    ux += v * a * s;                     // d/dx
    uxx += v * a * a * (-2 * t * s);     // d^2/dx^2, tanh''(z) = -2 tanh(z)(1 - tanh^2(z))
  }
  return { u, ux, uxx };
}

function initNet(H, seed, aScale, vScale) {
  const rng = mulberry32(seed);
  const p = new Array(3 * H + 1).fill(0);
  for (let j = 0; j < H; j++) {
    const a = randn(rng) * aScale;
    p[j] = a;
    p[H + j] = -a * rng();               // kink located inside [0,1]
    p[2 * H + j] = randn(rng) * vScale;
  }
  return p;
}

/* ---- Levenberg-Marquardt for least squares: minimise ||r(p)||^2 ----
   Jacobian by forward differences, damping mu adapted after each step. */
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[piv][i])) piv = k;
    if (Math.abs(M[piv][i]) < 1e-300) return null;
    [M[i], M[piv]] = [M[piv], M[i]];
    for (let k = i + 1; k < n; k++) {
      const f = M[k][i] / M[i][i];
      if (f === 0) continue;
      for (let c = i; c <= n; c++) M[k][c] -= f * M[i][c];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let c = i + 1; c < n; c++) s -= M[i][c] * x[c];
    x[i] = s / M[i][i];
  }
  return x;
}

function sumSq(r) {
  let s = 0;
  for (let i = 0; i < r.length; i++) s += r[i] * r[i];
  return s;
}

/* One LM iteration. lm = { p, mu, cost }; returns the updated lm object. */
function lmStep(lm, resFn) {
  const p = lm.p, n = p.length;
  const r = resFn(p), m = r.length;
  const J = [];
  for (let k = 0; k < n; k++) {
    const eps = 1e-7 * Math.max(1, Math.abs(p[k]));
    const q = p.slice();
    q[k] += eps;
    const rk = resFn(q);
    const col = new Array(m);
    for (let i = 0; i < m; i++) col[i] = (rk[i] - r[i]) / eps;
    J.push(col);                          // J[k][i] = d r_i / d p_k
  }
  const A = [], g = new Array(n);
  for (let a = 0; a < n; a++) {
    A.push(new Array(n));
    let ga = 0;
    for (let i = 0; i < m; i++) ga += J[a][i] * r[i];
    g[a] = ga;
    for (let b = 0; b <= a; b++) {
      let s = 0;
      for (let i = 0; i < m; i++) s += J[a][i] * J[b][i];
      A[a][b] = s;
      if (b < a) A[b][a] = s;
    }
  }
  const cost0 = sumSq(r);
  let mu = lm.mu;
  for (let tries = 0; tries < 12; tries++) {
    const B = A.map((row, i) => row.map((v, j) => (i === j ? v + mu * (A[i][i] + 1e-12) : v)));
    const d = solveLinear(B, g.map((v) => -v));
    if (d) {
      const q = p.map((v, i) => v + d[i]);
      const c1 = sumSq(resFn(q));
      if (isFinite(c1) && c1 < cost0) {
        return { p: q, mu: Math.max(mu / 3, 1e-12), cost: c1 };
      }
    }
    mu *= 4;
  }
  return { p, mu, cost: cost0, stalled: true };
}

/* ---- Bratu equation  u_xx + C e^u = 0,  u(0) = u(1) = 0 ---- */

/* Analytical solution u(x) = -2 ln( cosh(theta(2x-1)) / cosh(theta) ),
   where theta = sqrt(C/8) cosh(theta). This returns the smaller root;
   null if no solution exists (C larger than the critical value). */
function bratuTheta(C) {
  const s = Math.sqrt(C / 8);
  const g = (t) => s * Math.cosh(t) - t;
  const tc = Math.asinh(1 / s);          // minimiser of g
  if (g(tc) > 0) return null;
  let lo = 0, hi = tc;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (g(mid) > 0) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

function bratuExact(C, x) {
  const th = bratuTheta(C);
  if (th === null) return null;
  return -2 * Math.log(Math.cosh(th * (2 * x - 1)) / Math.cosh(th));
}

/* Critical value C* above which the Bratu problem has no solution. */
function bratuCritical() {
  let lo = 0.1, hi = 10;
  for (let i = 0; i < 100; i++) {
    const mid = 0.5 * (lo + hi);
    if (bratuTheta(mid) === null) hi = mid; else lo = mid;
  }
  return lo;
}

/* Finite difference method: unknowns u_1..u_{N-1}, u_0 = u_N = 0.
   Newton iteration; the Jacobian is tridiagonal (Thomas algorithm). */
function bratuFDM(C, N) {
  const h = 1 / N, n = N - 1, h2 = h * h;
  const u = new Array(n).fill(0);
  let converged = false;
  for (let it = 0; it < 60; it++) {
    const F = new Array(n), diag = new Array(n);
    let maxF = 0;
    for (let i = 0; i < n; i++) {
      const um = i > 0 ? u[i - 1] : 0, up = i < n - 1 ? u[i + 1] : 0;
      F[i] = (up - 2 * u[i] + um) / h2 + C * Math.exp(u[i]);
      diag[i] = -2 / h2 + C * Math.exp(u[i]);
      maxF = Math.max(maxF, Math.abs(F[i]));
    }
    if (!isFinite(maxF)) break;
    // Solve J d = -F, J = tridiag(1/h2, diag, 1/h2)
    const off = 1 / h2, cp = new Array(n), dp = new Array(n);
    cp[0] = off / diag[0];
    dp[0] = -F[0] / diag[0];
    for (let i = 1; i < n; i++) {
      const den = diag[i] - off * cp[i - 1];
      cp[i] = off / den;
      dp[i] = (-F[i] - off * dp[i - 1]) / den;
    }
    const d = new Array(n);
    d[n - 1] = dp[n - 1];
    for (let i = n - 2; i >= 0; i--) d[i] = dp[i] - cp[i] * d[i + 1];
    let maxd = 0;
    for (let i = 0; i < n; i++) { u[i] += d[i]; maxd = Math.max(maxd, Math.abs(d[i])); }
    if (!isFinite(maxd) || Math.abs(u[Math.floor(n / 2)]) > 50) break;
    if (maxd < 1e-12) { converged = true; break; }
  }
  const x = [0], uu = [0];
  for (let i = 1; i < N; i++) { x.push(i * h); uu.push(u[i - 1]); }
  x.push(1); uu.push(0);
  return { x, u: uu, converged };
}

/* Inverse-problem family  u_xx + l1 exp(l2 u / 2) = 0.
   With v = l2 u / 2 this becomes the Bratu equation with C = l1 l2 / 2,
   so u = (2 / l2) v. */
function inverseExact(l1, l2, x) {
  const v = bratuExact(0.5 * l1 * l2, x);
  return v === null ? null : (2 / l2) * v;
}

/* =========================================================
   PART 2 - SMALL HELPERS (formatting, controls, charts, trainer)
   ========================================================= */

const $ = (id) => document.getElementById(id);
const COLOR = { accent: '#1f4e79', dark: '#444444', warm: '#b4532a', grid: '#e4e4e0' };
const SUB = { 1: '₁', 2: '₂', 3: '₃', 4: '₄', 5: '₅', 6: '₆' };

/* Render a LaTeX string into an element (plain text if KaTeX is missing). */
function tex(el, s, display) {
  if (typeof katex !== 'undefined') {
    katex.render(s, el, { throwOnError: false, displayMode: !!display });
  } else {
    el.textContent = s;
  }
}

function renderAuto(el) {
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(el, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }],
      throwOnError: false
    });
  }
}

/* Number to short string without trailing zeros. */
function num(v, d) {
  if (d === undefined) d = 4;
  if (!isFinite(v)) return String(v);
  const s = Number(v.toFixed(d));
  return Object.is(s, -0) ? '0' : String(s);
}

/* Readout format: plain for ordinary values, exponent for very small/large ones. */
function sig(v, n) {
  if (!isFinite(v)) return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e5) return String(Number(v.toPrecision(n || 4)));
  return v.toExponential(3);
}

const par = (v) => '(' + num(v) + ')';   // value in parentheses for LaTeX products

function debounce(fn, ms) {
  let t = null;
  return function () {
    clearTimeout(t);
    t = setTimeout(fn, ms);
  };
}

/* Run fn the first time an element scrolls near the viewport. */
function onceVisible(el, fn) {
  if (typeof IntersectionObserver === 'undefined') { fn(); return; }
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
  }, { rootMargin: '200px' });
  io.observe(el);
}

function libWarning(text) {
  const d = document.createElement('div');
  d.className = 'takeaway';
  d.textContent = text;
  document.querySelector('main').prepend(d);
}

/* ---- control factory ----
   o.type: 'slider' (range + number), 'number', 'check', 'select'
   returns { get, set(v, silent) } */
function addControl(parentId, o) {
  const wrap = document.createElement('div');
  wrap.className = 'ctl' + (o.type === 'check' ? ' check' : '');
  $(parentId).appendChild(wrap);
  const fire = (v) => { if (o.onChange) o.onChange(v); };

  if (o.type === 'check') {
    wrap.innerHTML = '<label><input type="checkbox" id="' + o.id + '"><span>' + o.label + '</span></label>';
    const box = $(o.id);
    box.checked = !!o.value;
    box.addEventListener('input', () => fire(box.checked));
    return { get: () => box.checked, set: (v, silent) => { box.checked = !!v; if (!silent) fire(box.checked); } };
  }

  if (o.type === 'select') {
    wrap.innerHTML = '<label for="' + o.id + '">' + o.label + '</label><select id="' + o.id + '">' +
      o.options.map((op) => '<option value="' + op.value + '">' + op.label + '</option>').join('') + '</select>';
    const sel = $(o.id);
    sel.value = o.value;
    sel.addEventListener('input', () => fire(sel.value));
    return { get: () => sel.value, set: (v, silent) => { sel.value = v; if (!silent) fire(sel.value); } };
  }

  const hasRange = o.type === 'slider';
  const hardMin = o.hardMin !== undefined ? o.hardMin : o.min;
  const hardMax = o.hardMax !== undefined ? o.hardMax : o.max;
  wrap.innerHTML = '<label for="' + o.id + '">' + o.label + '</label><div class="ctl-row">' +
    (hasRange ? '<input type="range" id="' + o.id + '-r">' : '') +
    '<input type="number" id="' + o.id + '" step="any"></div>';
  const numEl = $(o.id), rangeEl = hasRange ? $(o.id + '-r') : null;
  let val = o.value;

  if (hasRange) {
    if (o.log) {
      rangeEl.min = Math.log10(o.min); rangeEl.max = Math.log10(o.max); rangeEl.step = 0.01;
    } else {
      rangeEl.min = o.min; rangeEl.max = o.max; rangeEl.step = o.step;
    }
  }
  const show = () => {
    numEl.value = String(Number(val.toPrecision(6)));
    if (hasRange) rangeEl.value = o.log ? Math.log10(Math.max(val, o.min)) : val;
  };
  show();

  if (hasRange) {
    rangeEl.addEventListener('input', () => {
      val = o.log ? Math.pow(10, parseFloat(rangeEl.value)) : parseFloat(rangeEl.value);
      if (o.log) val = Number(val.toPrecision(3));
      numEl.value = String(Number(val.toPrecision(6)));
      fire(val);
    });
  }
  numEl.addEventListener('input', () => {
    const v = parseFloat(numEl.value);
    if (!isFinite(v) || v < hardMin || v > hardMax) return;
    val = v;
    if (hasRange) rangeEl.value = o.log ? Math.log10(Math.max(val, o.min)) : val;
    fire(val);
  });
  numEl.addEventListener('change', () => {          // leaving the field: clamp and tidy
    let v = parseFloat(numEl.value);
    if (!isFinite(v)) v = val;
    val = Math.min(hardMax, Math.max(hardMin, v));
    show();
    fire(val);
  });
  return { get: () => val, set: (v, silent) => { val = v; show(); if (!silent) fire(val); } };
}

/* ---- Chart.js helpers ---- */
function newChart(canvasId, xTitle, yTitle, scales) {
  if (typeof Chart === 'undefined') return null;
  scales = scales || {};
  const axis = (title, extra) => Object.assign({
    type: 'linear',
    title: { display: true, text: title, color: '#444' },
    grid: { color: COLOR.grid },
    ticks: { color: '#555', maxTicksLimit: 8 }
  }, extra || {});
  return new Chart($(canvasId), {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 22, boxHeight: 2, font: { size: 12 } } },
        tooltip: {
          callbacks: {
            label: (c) => c.dataset.label + ': (' + sig(c.parsed.x, 4) + ', ' + sig(c.parsed.y, 4) + ')'
          }
        }
      },
      scales: { x: axis(xTitle, scales.x), y: axis(yTitle, scales.y) }
    }
  });
}

function setChart(chart, datasets) {
  if (!chart) return;
  chart.data.datasets = datasets;
  chart.update('none');
}

const lineSet = (label, data, color, o) => Object.assign({
  label, data, showLine: true, borderColor: color, backgroundColor: color,
  borderWidth: 2, pointRadius: 0, pointHoverRadius: 3, tension: 0, fill: false
}, o || {});

const dotSet = (label, data, color, o) => Object.assign({
  label, data, showLine: false, borderColor: color, backgroundColor: color,
  pointRadius: 3, pointHoverRadius: 5
}, o || {});

/* Sample f on [a,b] with n+1 points. */
function sampleFn(f, a, b, n) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    pts.push({ x, y: f(x) });
  }
  return pts;
}

/* ---- animated Levenberg-Marquardt runner ----
   Runs LM steps inside requestAnimationFrame so the page stays responsive.
   onFrame(lm, iterationsDone, finished) is called after every frame. */
function startTrainer(lm, resFn, maxIter, onFrame) {
  let cancelled = false, iter = 0;
  function frame() {
    if (cancelled) return;
    const t0 = performance.now();
    do {
      lm = lmStep(lm, resFn);
      iter++;
      if (lm.stalled || lm.cost < 1e-16 || iter >= maxIter) {
        onFrame(lm, iter, true);
        return;
      }
    } while (performance.now() - t0 < 14);
    onFrame(lm, iter, false);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  return { cancel: () => { cancelled = true; } };
}

/* =========================================================
   PART 3 - PAGE BEHAVIOUR
   ========================================================= */

/* ---------- navigation ---------- */
function setupNav() {
  const btn = $('menu-btn'), list = $('nav-links');
  btn.addEventListener('click', () => {
    const open = list.classList.toggle('open');
    btn.setAttribute('aria-expanded', String(open));
  });
  list.addEventListener('click', (e) => {
    if (e.target.tagName === 'A') { list.classList.remove('open'); btn.setAttribute('aria-expanded', 'false'); }
  });

  const links = Array.from(list.querySelectorAll('a'));
  const sections = links.map((a) => $(a.getAttribute('href').slice(1)));
  let ticking = false;
  function mark() {
    ticking = false;
    let current = 0;
    sections.forEach((s, i) => { if (s && s.getBoundingClientRect().top <= 90) current = i; });
    if (window.scrollY < 120) current = -1;
    links.forEach((a, i) => a.classList.toggle('active', i === current));
  }
  window.addEventListener('scroll', () => {
    if (!ticking) { ticking = true; requestAnimationFrame(mark); }
  }, { passive: true });
  mark();
}

/* ---------- 2.1 composite function ---------- */
const isPlain = (a) => /^[a-zA-Z]$/.test(a) || /^\d+(\.\d+)?$/.test(a);
const wrapArg = (a) => (isPlain(a) ? a : '(' + a + ')');

const COMPOSITE = {
  add1: { name: 'x + 1', fn: (x) => x + 1, t: (a) => a + '+1' },
  sq:   { name: 'x²', fn: (x) => x * x, t: (a) => wrapArg(a) + '^2' },
  dbl:  { name: '2x', fn: (x) => 2 * x, t: (a) => (/^[a-zA-Z]$/.test(a) ? '2' + a : (/^-?\d/.test(a) ? '2\\cdot ' + wrapArg(a) : '2' + wrapArg(a))) },
  sin:  { name: 'sin(x)', fn: Math.sin, t: (a) => '\\sin(' + a + ')' }
};

function initComposite() {
  const opts = Object.keys(COMPOSITE).map((k) => ({ value: k, label: COMPOSITE[k].name }));
  const chart = newChart('cf-chart', 'x', 'value');
  let cg, cf, cx;

  function update() {
    const g = COMPOSITE[cg.get()], f = COMPOSITE[cf.get()], x = cx.get();
    const gx = g.fn(x), y = f.fn(gx);

    $('cf-flow').innerHTML =
      '<span class="fbox">x = ' + num(x) + '</span><span class="arrow">&rarr;</span>' +
      '<span class="fbox">g(x) = ' + num(gx) + '</span><span class="arrow">&rarr;</span>' +
      '<span class="fbox">f(g(x)) = ' + num(y) + '</span><span class="arrow">=</span>' +
      '<span class="fbox">y = ' + num(y) + '</span>';

    const steps = $('cf-steps');
    steps.innerHTML = '<div class="step"></div><div class="step"></div><div class="step"></div>';
    const rows = steps.children;
    tex(rows[0], 'f(x)=' + f.t('x') + ',\\quad g(x)=' + g.t('x') + ',\\quad y=f\\circ g(x)=' + f.t(g.t('x')));
    tex(rows[1], 'g(' + num(x) + ')=' + g.t(num(x)) + '=' + num(gx));
    tex(rows[2], 'f(g(' + num(x) + '))=f(' + num(gx) + ')=' + f.t(num(gx)) + '=' + num(y));

    const gCurve = sampleFn(g.fn, -3, 3, 120);
    const fgCurve = sampleFn((t) => f.fn(g.fn(t)), -3, 3, 120);
    setChart(chart, [
      lineSet('g(x) = ' + g.name, gCurve, COLOR.dark, { borderDash: [6, 4], borderWidth: 1.5 }),
      lineSet('f(g(x))', fgCurve, COLOR.accent),
      dotSet('g(x) at chosen x', [{ x, y: gx }], COLOR.dark, { pointStyle: 'rectRot', pointRadius: 5, backgroundColor: '#fff' }),
      dotSet('y = f(g(x)) at chosen x', [{ x, y }], COLOR.warm, { pointRadius: 5 })
    ]);
  }

  cg = addControl('cf-controls', { type: 'select', id: 'cf-g', label: 'Inner function g(x)', options: opts, value: 'add1', onChange: update });
  cf = addControl('cf-controls', { type: 'select', id: 'cf-f', label: 'Outer function f(x)', options: opts, value: 'sq', onChange: update });
  cx = addControl('cf-controls', { type: 'slider', id: 'cf-x', label: 'x', min: -3, max: 3, step: 0.1, hardMin: -100, hardMax: 100, value: 2, onChange: update });
  update();
}

/* ---------- 2.3 network + layer-wise calculator, 2.4 matrix form ---------- */
const NN_KEYS = ['x1', 'x2', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'b1', 'b2', 'b3'];
const NN_DEFAULT = { x1: 0.5, x2: -1, w1: 0.8, w2: -0.5, w3: 0.3, w4: 0.9, w5: 1.2, w6: -0.7, b1: 0.1, b2: -0.2, b3: 0.05 };
const NN = Object.assign({}, NN_DEFAULT);
let nnAct = null, nnOutAct = null;

const ACT_TEX = {
  tanh: (a) => '\\tanh(' + a + ')',
  relu: (a) => '\\max(0,\\,' + a + ')',
  sigmoid: (a) => '\\sigma(' + a + ')'
};
const ACT_DEF = {
  tanh: 'f(z)=\\tanh(z)=\\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}',
  relu: 'f(z)=\\mathrm{ReLU}(z)=\\max(0,z)',
  sigmoid: 'f(z)=\\sigma(z)=\\dfrac{1}{1+e^{-z}}'
};
const subscript = (k) => k[0] + (SUB[k[1]] || k[1]);

/* Scalar forward pass: h1, h2, y exactly as in the slides. */
function nnForward(s, actName, applyOut) {
  const f = ACT[actName].fn;
  const z1 = s.w1 * s.x1 + s.w2 * s.x2 + s.b1;
  const z2 = s.w3 * s.x1 + s.w4 * s.x2 + s.b2;
  const h1 = f(z1), h2 = f(z2);
  const z3 = s.w5 * h1 + s.w6 * h2 + s.b3;
  return { z1, z2, z3, h1, h2, y: applyOut ? f(z3) : z3 };
}

function matMul(A, B) {
  return A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
}
function matAdd(A, B) { return A.map((row, i) => row.map((v, j) => v + B[i][j])); }
function matTex(M) {
  return '\\begin{bmatrix}' + M.map((r) => r.map((v) => num(v)).join('&')).join('\\\\') + '\\end{bmatrix}';
}

function drawNetwork(r) {
  const s = NN, P = { x1: [70, 90], x2: [70, 210], h1: [300, 90], h2: [300, 210], y: [540, 150] };
  const edge = (a, b, w, label, t, dy) => {
    const [x0, y0] = P[a], [x1, y1] = P[b];
    const col = w >= 0 ? COLOR.accent : COLOR.warm;
    const width = 0.8 + 1.6 * Math.min(Math.abs(w), 2);
    const lx = x0 + (x1 - x0) * t, ly = y0 + (y1 - y0) * t + (dy || 0);
    return '<line x1="' + x0 + '" y1="' + y0 + '" x2="' + x1 + '" y2="' + y1 + '" stroke="' + col + '" stroke-width="' + width.toFixed(2) + '" stroke-opacity="0.8"/>' +
      '<text class="halo" x="' + lx + '" y="' + ly + '" text-anchor="middle" style="font-size:12px">' + label + ' = ' + num(w, 3) + '</text>';
  };
  const node = (k, label, below, above) => {
    const [x, y] = P[k];
    return '<circle id="nn-n-' + k + '" cx="' + x + '" cy="' + y + '" r="24" fill="#fff" stroke="#333" stroke-width="1.3"/>' +
      '<text x="' + x + '" y="' + (y + 5) + '" text-anchor="middle">' + label + '</text>' +
      (below ? '<text x="' + x + '" y="' + (y + 44) + '" text-anchor="middle" style="font-size:12px">' + below + '</text>' : '') +
      (above ? '<text x="' + x + '" y="' + (y - 34) + '" text-anchor="middle" style="font-size:12px;fill:#5f6368">' + above + '</text>' : '');
  };
  $('nn-svg').innerHTML =
    edge('x1', 'h1', s.w1, 'w₁', 0.3, -6) + edge('x1', 'h2', s.w3, 'w₃', 0.3, 0) +
    edge('x2', 'h1', s.w2, 'w₂', 0.3, 0) + edge('x2', 'h2', s.w4, 'w₄', 0.3, 12) +
    edge('h1', 'y', s.w5, 'w₅', 0.5, -6) + edge('h2', 'y', s.w6, 'w₆', 0.5, 14) +
    node('x1', 'x₁', '= ' + num(s.x1, 3)) + node('x2', 'x₂', '= ' + num(s.x2, 3)) +
    node('h1', 'h₁', '= ' + num(r.h1, 4), 'b₁ = ' + num(s.b1, 3)) +
    node('h2', 'h₂', '= ' + num(r.h2, 4), 'b₂ = ' + num(s.b2, 3)) +
    node('y', 'y', '= ' + num(r.y, 4), 'b₃ = ' + num(s.b3, 3)) +
    '<text x="70" y="30" text-anchor="middle" style="fill:#5f6368">input</text>' +
    '<text x="300" y="30" text-anchor="middle" style="fill:#5f6368">hidden layer</text>' +
    '<text x="540" y="30" text-anchor="middle" style="fill:#5f6368">output</text>';
}

function updateNetwork() {
  const act = nnAct.get(), applyOut = nnOutAct.get(), s = NN;
  const r = nnForward(s, act, applyOut);
  const A = ACT_TEX[act];

  drawNetwork(r);

  const lines = [
    'z_1=w_1x_1+w_2x_2+b_1=' + par(s.w1) + par(s.x1) + '+' + par(s.w2) + par(s.x2) + '+' + par(s.b1) + '=' + num(r.z1),
    'h_1=f(z_1)=' + A(num(r.z1)) + '=' + num(r.h1),
    'z_2=w_3x_1+w_4x_2+b_2=' + par(s.w3) + par(s.x1) + '+' + par(s.w4) + par(s.x2) + '+' + par(s.b2) + '=' + num(r.z2),
    'h_2=f(z_2)=' + A(num(r.z2)) + '=' + num(r.h2),
    'z_3=w_5h_1+w_6h_2+b_3=' + par(s.w5) + par(r.h1) + '+' + par(s.w6) + par(r.h2) + '+' + par(s.b3) + '=' + num(r.z3),
    applyOut ? 'y=f(z_3)=' + A(num(r.z3)) + '=' + num(r.y) : 'y=z_3=' + num(r.y)
  ];
  const box = $('nn-steps');
  box.innerHTML = lines.map(() => '<div class="step"></div>').join('');
  lines.forEach((l, i) => tex(box.children[i], l));
  box.lastElementChild.classList.add('step-final');
  box.insertAdjacentHTML('afterbegin', '<div class="step note">Step by step (values rounded to 4 decimals):</div>');

  const note = $('nn-act-note');
  note.innerHTML = 'Activation: <span></span>' + (applyOut ? '' : ' (not applied at the output)');
  tex(note.querySelector('span'), ACT_DEF[act]);

  updateMatrix(r, act, applyOut);
}

/* Matrix form H1 = f(X W1 + B1), y = f(H1 W2 + B2), computed with real matrix products. */
function updateMatrix(rScalar, act, applyOut) {
  const s = NN, f = ACT[act].fn;
  const X = [[s.x1, s.x2]];
  const W1 = [[s.w1, s.w3], [s.w2, s.w4]];
  const B1 = [[s.b1, s.b2]];
  const W2 = [[s.w5], [s.w6]];
  const B2 = [[s.b3]];

  const XW1 = matMul(X, W1);
  const Z1 = matAdd(XW1, B1);
  const H1 = Z1.map((row) => row.map(f));
  const H1W2 = matMul(H1, W2);
  const Z2 = matAdd(H1W2, B2);
  const Y = applyOut ? Z2.map((row) => row.map(f)) : Z2;

  const stages = [
    ['Input matrix \\(X\\)', 'X=' + matTex(X)],
    ['\\(\\times\\, W_1\\)', 'XW_1=' + matTex(X) + matTex(W1) + '=' + matTex(XW1)],
    ['\\(+\\, B_1\\)', 'Z_1=XW_1+B_1=' + matTex(XW1) + '+' + matTex(B1) + '=' + matTex(Z1)],
    ['Activation \\(f\\)', 'H_1=f(Z_1)=' + matTex(H1)],
    ['Hidden layer \\(H_1\\), \\(\\times\\, W_2\\)', 'H_1W_2=' + matTex(H1) + matTex(W2) + '=' + matTex(H1W2)],
    ['\\(+\\, B_2\\)', 'Z_2=H_1W_2+B_2=' + matTex(H1W2) + '+' + matTex(B2) + '=' + matTex(Z2)],
    [applyOut ? 'Activation \\(f\\), output' : 'Output', 'y=' + (applyOut ? 'f(Z_2)=' : 'Z_2=') + matTex(Y)]
  ];
  const box = $('mx-stages');
  box.innerHTML = stages.map((st) => '<div class="stage-row"><div class="stage-name">' + st[0] + '</div><div class="stage-tex"></div></div>').join('');
  stages.forEach((st, i) => {
    renderAuto(box.children[i].firstElementChild);
    tex(box.children[i].lastElementChild, st[1]);
  });

  const diff = Math.abs(Y[0][0] - rScalar.y);
  $('mx-check').textContent = 'Check: matrix form gives y = ' + num(Y[0][0], 6) + ', the scalar form of section 2.3 gives y = ' +
    num(rScalar.y, 6) + (diff < 1e-12 ? ' (identical).' : ' (difference ' + diff.toExponential(2) + ').');
}

function initNetwork() {
  nnAct = addControl('nn-act-controls', {
    type: 'select', id: 'nn-act', label: 'Activation function f',
    options: [{ value: 'tanh', label: 'tanh' }, { value: 'relu', label: 'ReLU' }, { value: 'sigmoid', label: 'sigmoid' }],
    value: 'tanh', onChange: updateNetwork
  });
  nnOutAct = addControl('nn-act-controls', {
    type: 'check', id: 'nn-outact', label: 'Apply f at the output (as in the slides)', value: true, onChange: updateNetwork
  });

  const grid = $('nn-inputs');
  grid.innerHTML = NN_KEYS.map((k) =>
    '<label><span>' + subscript(k) + '</span><input type="number" step="0.1" id="nn-in-' + k + '" value="' + NN[k] + '"></label>').join('');
  NN_KEYS.forEach((k) => {
    $('nn-in-' + k).addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      if (isFinite(v)) { NN[k] = v; updateNetwork(); }
    });
  });

  const setAll = (vals) => {
    NN_KEYS.forEach((k) => { NN[k] = vals[k]; $('nn-in-' + k).value = vals[k]; });
    updateNetwork();
  };
  $('nn-reset').addEventListener('click', () => setAll(NN_DEFAULT));
  $('nn-random').addEventListener('click', () => {
    const v = {};
    NN_KEYS.forEach((k) => { v[k] = Math.round((Math.random() * 3 - 1.5) * 100) / 100; });
    setAll(v);
  });
  updateNetwork();
}

/* ---------- 2.5 more hidden layers: sizes of W_k and B_k ---------- */
function initDeepSizes() {
  let cin, cl, ch, cout;
  function update() {
    const nin = cin.get(), L = cl.get(), H = ch.get(), nout = cout.get();
    let total = 0, rows = '';
    for (let k = 1; k <= L + 1; k++) {
      const a = k === 1 ? nin : H, b = k === L + 1 ? nout : H;
      const p = a * b + b;
      total += p;
      rows += '<tr><td>' + k + (k === L + 1 ? ' (output)' : '') + '</td><td>' + a + ' × ' + b + '</td><td>1 × ' + b + '</td><td>' + p + '</td></tr>';
    }
    document.querySelector('#ds-table tbody').innerHTML = rows;
    $('ds-total').textContent = 'Total: ' + total + ' trainable parameters (weights and biases).';
  }
  const mk = (id, label, min, max, v) => addControl('ds-controls', { type: 'slider', id, label, min, max, step: 1, value: v, onChange: update });
  cin = mk('ds-in', 'Inputs', 1, 10, 2);
  cl = mk('ds-L', 'Hidden layers', 1, 8, 4);
  ch = mk('ds-H', 'Nodes per hidden layer', 1, 100, 20);
  cout = mk('ds-out', 'Outputs', 1, 5, 1);
  update();
}

/* ---------- 2.6 function approximation ---------- */
function initApprox() {
  const FA = { seed: 1, lm: null, trainer: null, base: 0, xs: [], res: null };
  const chart = newChart('fa-chart', 'x', 'u(x)');
  let ck, cA, cN, cH, cTrue, cNet;

  function target(x) { return cA.get() * Math.sin(ck.get() * Math.PI * x); }

  function draw() {
    const H = cH.get(), p = FA.lm.p;
    const curveTrue = sampleFn(target, 0, 1, 200);
    const curveNet = sampleFn((x) => netU(p, H, x), 0, 1, 200);
    const pts = FA.xs.map((x) => ({ x, y: target(x) }));
    const sets = [];
    if (cTrue.get()) sets.push(lineSet('True function', curveTrue, COLOR.dark, { borderWidth: 2.5 }));
    if (cNet.get()) sets.push(lineSet('Network approximation', curveNet, COLOR.accent));
    sets.push(dotSet('Collocation points', pts, COLOR.warm, { pointRadius: pts.length > 80 ? 2 : 3.5 }));
    setChart(chart, sets);
  }

  function stat(iters) {
    const cost = sumSq(FA.res(FA.lm.p));
    $('fa-stat').textContent = 'Iterations: ' + iters + '  |  loss (MSE at the ' + FA.xs.length + ' points): ' + cost.toExponential(3) +
      '  |  slides (BFGS, 101 points): 3.80874e-06';
  }

  function setup(reinit) {
    if (FA.trainer) FA.trainer.cancel();
    const N = Math.round(cN.get()), H = Math.round(cH.get());
    FA.xs = Array.from({ length: N }, (_, i) => i / (N - 1));          // x_i = i/(N-1); N = 101 gives 0.01 i
    FA.res = (p) => FA.xs.map((x) => (netU(p, H, x) - target(x)) / Math.sqrt(N));
    if (reinit || !FA.lm || FA.lm.p.length !== 3 * H + 1) {
      FA.lm = { p: initNet(H, FA.seed, 4, 0.5), mu: 1e-2, cost: NaN };
      FA.base = 0;
    }
    draw();
    stat(FA.base);
  }

  function train(iters) {
    if (FA.trainer) FA.trainer.cancel();
    const start = FA.base;
    FA.trainer = startTrainer(FA.lm, FA.res, iters, (lm, it) => {
      FA.lm = lm; FA.base = start + it;
      draw(); stat(FA.base);
    });
  }

  const retrain = debounce(() => train(150), 250);
  const onParam = () => { setup(true); retrain(); };

  ck = addControl('fa-controls', { type: 'slider', id: 'fa-k', label: 'Frequency k, in sin(kπx)', min: 1, max: 6, step: 0.5, value: 4, onChange: onParam });
  cA = addControl('fa-controls', { type: 'slider', id: 'fa-A', label: 'Amplitude A', min: 0.2, max: 2, step: 0.1, value: 1, onChange: onParam });
  cN = addControl('fa-controls', { type: 'slider', id: 'fa-N', label: 'Number of collocation points', min: 5, max: 200, step: 1, value: 101, onChange: onParam });
  cH = addControl('fa-controls', { type: 'slider', id: 'fa-H', label: 'Hidden neurons', min: 3, max: 40, step: 1, value: 20, onChange: onParam });
  cTrue = addControl('fa-controls', { type: 'check', id: 'fa-show-true', label: 'Show true function', value: true, onChange: draw });
  cNet = addControl('fa-controls', { type: 'check', id: 'fa-show-net', label: 'Show network approximation', value: true, onChange: draw });

  $('fa-train').addEventListener('click', () => train(150));
  $('fa-reset').addEventListener('click', () => { FA.seed += 1; setup(true); train(150); });

  setup(true);
  onceVisible($('fa-chart'), () => train(150));
}

/* ---------- 3.3 central difference demo ---------- */
function initFiniteDifference() {
  const FD = { preset: null };
  let cm, c0, cp, ch;

  const PRESETS = {
    bratu: { xk: 0.5, f: (x) => bratuExact(2, x), exact: (x, u) => -2 * Math.exp(u), label: 'Exact u_xx(x_k) = -C e^{u(x_k)} with C = 2' },
    sine: { xk: 0.1, f: (x) => Math.sin(4 * Math.PI * x), exact: (x) => -16 * Math.PI * Math.PI * Math.sin(4 * Math.PI * x), label: 'Exact u_xx(x_k) = -(4π)² sin(4π x_k)' }
  };

  function applyPreset(name) {
    FD.preset = name;
    const P = PRESETS[name], h = ch.get();
    cm.set(P.f(P.xk - h), true); c0.set(P.f(P.xk), true); cp.set(P.f(P.xk + h), true);
    update();
  }

  function drawSvg(um, u0, up, h) {
    const X = [150, 320, 490], base = 215;
    const lo0 = Math.min(um, u0, up), hi0 = Math.max(um, u0, up);
    const pad = Math.max(0.15 * (hi0 - lo0), 0.05);
    const lo = lo0 - pad, hi = hi0 + pad;
    const Y = (u) => base - 20 - (150 * (u - lo)) / (hi - lo);
    const ys = [Y(um), Y(u0), Y(up)];
    // parabola through the three points, parameter s in [-1, 1]
    const a2 = (up - 2 * u0 + um) / 2, a1 = (up - um) / 2;
    let path = '';
    for (let i = 0; i <= 40; i++) {
      const s = -1 + (2 * i) / 40;
      path += (i ? 'L' : 'M') + (320 + 170 * s).toFixed(1) + ' ' + Y(u0 + a1 * s + a2 * s * s).toFixed(1) + ' ';
    }
    const names = ['xₖ₋₁', 'xₖ', 'xₖ₊₁'];
    const vals = [um, u0, up], labs = ['uₖ₋₁', 'uₖ', 'uₖ₊₁'];
    let s = '<line x1="60" y1="' + base + '" x2="580" y2="' + base + '" stroke="#999"/>';
    s += '<path d="' + path + '" fill="none" stroke="' + COLOR.accent + '" stroke-width="1.5" stroke-dasharray="5 4"/>';
    for (let i = 0; i < 3; i++) {
      s += '<line x1="' + X[i] + '" y1="' + ys[i] + '" x2="' + X[i] + '" y2="' + base + '" stroke="#bbb" stroke-dasharray="3 3"/>' +
        '<circle cx="' + X[i] + '" cy="' + ys[i] + '" r="6" fill="' + COLOR.accent + '"/>' +
        '<text class="halo" x="' + X[i] + '" y="' + (ys[i] - 12) + '" text-anchor="middle">' + labs[i] + ' = ' + num(vals[i], 4) + '</text>' +
        '<text x="' + X[i] + '" y="' + (base + 18) + '" text-anchor="middle">' + names[i] + '</text>';
    }
    s += '<line x1="150" y1="' + (base + 34) + '" x2="320" y2="' + (base + 34) + '" stroke="#333" marker-start="url(#ah)" marker-end="url(#ah)"/>' +
      '<line x1="320" y1="' + (base + 34) + '" x2="490" y2="' + (base + 34) + '" stroke="#333" marker-start="url(#ah)" marker-end="url(#ah)"/>' +
      '<text x="235" y="' + (base + 31) + '" text-anchor="middle" class="halo">h = ' + num(h, 4) + '</text>' +
      '<text x="405" y="' + (base + 31) + '" text-anchor="middle" class="halo">h = ' + num(h, 4) + '</text>';
    $('fd-svg').innerHTML =
      '<defs><marker id="ah" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L6 3L0 6z" fill="#333"/></marker></defs>' + s;
  }

  function update() {
    const um = cm.get(), u0 = c0.get(), up = cp.get(), h = ch.get();
    drawSvg(um, u0, up, h);
    const val = (up - 2 * u0 + um) / (h * h);
    tex($('fd-result'), 'u_{xx}(x_k)\\approx\\dfrac{u_{k+1}-2u_k+u_{k-1}}{h^2}=\\dfrac{' + num(up, 6) + '-2(' + num(u0, 6) + ')+(' + num(um, 6) + ')}{(' + num(h, 6) + ')^2}=' + sig(val, 6), true);
    const ex = $('fd-exact');
    if (FD.preset) {
      const P = PRESETS[FD.preset];
      const e = P.exact(P.xk, u0);
      ex.textContent = P.label + ' = ' + sig(e, 6) + ';  error of the finite difference = ' + sig(val - e, 3) + '.';
    } else {
      ex.textContent = 'Values entered by hand; choose a preset to compare with an exact second derivative.';
    }
  }

  const manual = () => { FD.preset = null; update(); };
  const mk = (id, label, v) => addControl('fd-controls', { type: 'slider', id, label, min: -2, max: 2, step: 0.01, hardMin: -1000, hardMax: 1000, value: v, onChange: manual });
  cm = mk('fd-um', 'uₖ₋₁', 0);
  c0 = mk('fd-u0', 'uₖ', 0);
  cp = mk('fd-up', 'uₖ₊₁', 0);
  ch = addControl('fd-controls', {
    type: 'slider', id: 'fd-h', label: 'h', min: 0.005, max: 0.25, step: 0.005, hardMin: 0.0001, hardMax: 1, value: 0.1,
    onChange: () => { if (FD.preset) applyPreset(FD.preset); else update(); }
  });
  $('fd-bratu').addEventListener('click', () => applyPreset('bratu'));
  $('fd-sine').addEventListener('click', () => applyPreset('sine'));
  applyPreset('bratu');
}

/* ---------- 3.4 interactive Bratu equation ---------- */
function initBratu() {
  const chart = newChart('br-chart', 'x', 'u(x)', { y: { suggestedMin: -0.05 } });
  let cC, cN, cAna, cNum;
  const CSTAR = bratuCritical();

  function update() {
    const C = cC.get(), N = Math.round(cN.get());
    const th = bratuTheta(C);
    const fdm = bratuFDM(C, N);
    const sets = [];
    const info = [];

    if (th !== null && cAna.get()) {
      sets.push(lineSet('Analytical', sampleFn((x) => bratuExact(C, x), 0, 1, 200), COLOR.dark, { borderWidth: 3, borderColor: '#9aa0a6', backgroundColor: '#9aa0a6' }));
    }
    if (fdm.converged && cNum.get()) {
      sets.push(lineSet('Numerical', fdm.x.map((x, i) => ({ x, y: fdm.u[i] })), COLOR.accent, { borderWidth: 1.5, pointRadius: N <= 50 ? 3 : 0 }));
    }
    sets.push(dotSet('Boundary conditions u(0) = u(1) = 0', [{ x: 0, y: 0 }, { x: 1, y: 0 }], '#222', { pointStyle: 'rectRot', pointRadius: 6, backgroundColor: '#fff' }));
    setChart(chart, sets);

    info.push('h = 1/N = ' + sig(1 / N, 4) + ', C* ≈ ' + num(CSTAR, 4));
    if (th === null) {
      info.push('No solution exists for C > C*: the analytical formula has no real θ and Newton\'s method does not converge');
    } else {
      info.push('θ = ' + num(th, 12) + ', u(1/2) = ' + num(bratuExact(C, 0.5), 6));
      if (fdm.converged) {
        let err = 0;
        fdm.x.forEach((x, i) => { err = Math.max(err, Math.abs(fdm.u[i] - bratuExact(C, x))); });
        info.push('max |numerical − analytical| at the grid points = ' + err.toExponential(2));
      }
    }
    $('br-info').textContent = info.join('. ') + '.';
  }

  cC = addControl('br-controls', { type: 'slider', id: 'br-C', label: 'C', min: 0.1, max: 3.5, step: 0.01, hardMin: 0.01, hardMax: 6, value: 2, onChange: update });
  cN = addControl('br-controls', { type: 'slider', id: 'br-N', label: 'Number of subintervals N (grid points N − 1)', min: 4, max: 200, step: 1, value: 100, onChange: update });
  cAna = addControl('br-controls', { type: 'check', id: 'br-ana', label: 'Show analytical solution', value: true, onChange: update });
  cNum = addControl('br-controls', { type: 'check', id: 'br-num', label: 'Show numerical solution', value: true, onChange: update });
  update();
}

/* ---------- 4.1 PINN pipeline ---------- */
function initPipeline() {
  const DESC = [
    'Input: the coordinates \\((x,t)\\) of the collocation points, both interior points and boundary/initial points.',
    'Neural network: a feedforward network with weights \\(W\\). Its output \\(u(x,t;W)\\) is a continuous function of \\(x\\) and \\(t\\).',
    'The network output \\(u(x,t)\\) is the approximation of the solution of the differential equation.',
    'Derivatives: \\(u_t\\), \\(u_x\\), \\(u_{xx}\\) are obtained from the network by the chain rule (automatic differentiation).',
    'PDE residual: for the Burgers equation, \\(u_t+u\\,u_x-\\frac{0.01}{\\pi}u_{xx}\\) at the interior points. It is zero for an exact solution.',
    'Loss: \\(L(W)=\\mathrm{MSE}_i+\\mathrm{MSE}_b\\), the mean squared PDE residual plus the mean squared boundary/initial residual. Training minimises it over \\(W\\).'
  ];
  const buttons = Array.from(document.querySelectorAll('#pf-flow .stage'));
  function select(i) {
    buttons.forEach((b, j) => b.classList.toggle('active', i === j));
    $('pf-desc').textContent = DESC[i];
    renderAuto($('pf-desc'));
  }
  buttons.forEach((b) => b.addEventListener('click', () => select(parseInt(b.dataset.i, 10))));
  select(0);
}

/* ---------- 4.2 collocation points ---------- */
function initCollocation() {
  const rng = mulberry32(5);
  const RI = Array.from({ length: 1000 }, () => [rng(), rng()]);   // fixed random sequences: points stay put as N grows
  const RB = Array.from({ length: 300 }, () => [rng(), rng()]);
  const chart = newChart('cp-chart', 'x', 't', { x: { min: -1.05, max: 1.05 }, y: { min: -0.05, max: 1.05 } });
  let cMode, cNi, cNb;

  function interior(n, mode) {
    const pts = [];
    if (mode === 'random') {
      for (let i = 0; i < n; i++) pts.push({ x: -1 + 2 * RI[i][0], y: RI[i][1] });
    } else {
      const nx = Math.max(1, Math.round(Math.sqrt(2 * n))), nt = Math.max(1, Math.round(n / nx));
      for (let i = 0; i < nx; i++) for (let j = 0; j < nt; j++) pts.push({ x: -1 + (2 * (i + 0.5)) / nx, y: (j + 0.5) / nt });
    }
    return pts;
  }

  function boundary(n, mode) {
    const n0 = Math.ceil(n / 2), rest = n - n0, nl = Math.ceil(rest / 2), nr = rest - nl;
    const init = [], left = [], right = [];
    for (let i = 0; i < n0; i++) {
      const x = mode === 'random' ? -1 + 2 * RB[i][0] : (n0 === 1 ? 0 : -1 + (2 * i) / (n0 - 1));
      init.push({ x, y: 0 });
    }
    for (let j = 0; j < nl; j++) left.push({ x: -1, y: mode === 'random' ? RB[n0 + j][1] : (j + 1) / nl });
    for (let j = 0; j < nr; j++) right.push({ x: 1, y: mode === 'random' ? RB[n0 + nl + j][1] : (j + 1) / nr });
    return { init, left, right };
  }

  function update() {
    const mode = cMode.get(), I = interior(Math.round(cNi.get()), mode), B = boundary(Math.round(cNb.get()), mode);
    setChart(chart, [
      dotSet('Interior (collocation) points', I, COLOR.accent, { pointRadius: I.length > 400 ? 1.8 : 2.6 }),
      dotSet('Boundary / initial points', B.init.concat(B.left, B.right), COLOR.warm, { pointStyle: 'rect', pointRadius: 3.5 })
    ]);
    $('cp-info').textContent = 'Interior points N_i = ' + I.length + (mode === 'grid' ? ' (rounded to a regular grid)' : '') +
      '; boundary/initial points N_b = ' + (B.init.length + B.left.length + B.right.length) +
      ' (' + B.init.length + ' on t = 0, ' + B.left.length + ' on x = −1, ' + B.right.length + ' on x = 1).';
  }

  cMode = addControl('cp-controls', {
    type: 'select', id: 'cp-mode', label: 'Distribution of points',
    options: [{ value: 'random', label: 'Random' }, { value: 'grid', label: 'Grid' }], value: 'random', onChange: update
  });
  cNi = addControl('cp-controls', { type: 'slider', id: 'cp-ni', label: 'Number of interior points', min: 10, max: 1000, step: 10, value: 300, onChange: update });
  cNb = addControl('cp-controls', { type: 'slider', id: 'cp-nb', label: 'Number of boundary/initial points', min: 4, max: 200, step: 2, value: 60, onChange: update });
  update();
}

/* ---------- 4.3 loss function ---------- */
function initLoss() {
  let ci, cb, ca;
  function update() {
    const mi = ci.get(), mb = cb.get(), a = ca.get(), total = mi + a * mb;
    const pad = (s) => s.padEnd(14);
    $('ls-result').textContent =
      pad('Interior Loss') + ': ' + sig(mi) + '\n' +
      pad('Boundary Loss') + ': ' + sig(mb) + '\n' +
      pad('α') + ': ' + a.toFixed(2) + '\n\n' +
      pad('Total Loss') + ': ' + sig(total) + '\n' +
      pad('') + '  = ' + sig(mi) + ' + ' + a.toFixed(2) + ' × ' + sig(mb);
    const wi = total > 0 ? (100 * mi) / total : 50;
    const bar = $('ls-bar');
    bar.children[0].style.width = wi + '%';
    bar.children[1].style.width = (100 - wi) + '%';
  }
  ci = addControl('ls-controls', { type: 'number', id: 'ls-mi', label: 'MSE interior (PDE residual)', min: 0, max: 1e6, value: 0.024, onChange: update });
  cb = addControl('ls-controls', { type: 'number', id: 'ls-mb', label: 'MSE boundary (boundary/initial residual)', min: 0, max: 1e6, value: 0.011, onChange: update });
  ca = addControl('ls-controls', { type: 'slider', id: 'ls-a', label: 'α (weight of the boundary term)', min: 0.01, max: 100, log: true, value: 1, onChange: update });
  update();
}

/* ---------- 4.4 derivatives / automatic differentiation ---------- */
function initDerivatives() {
  // fixed network, H = 4: p = [a, c, v, d]
  const H = 4, P = [2, -1.5, 3, 1, -0.5, 0.8, -1.2, 0.3, 1, 0.7, -0.8, 0.5, 0.1];
  const chart = newChart('ad-chart', 'x', 'value');
  let cx, cu, cux, cuxx;
  const dx = 0.01;

  function update() {
    const x = cx.get(), d = netD(P, H, x);
    const sets = [];
    if (cu.get()) {
      sets.push(lineSet('u(x)', sampleFn((t) => netU(P, H, t), -2, 2, 200), COLOR.accent));
      sets.push(dotSet('u at x', [{ x, y: d.u }], COLOR.accent, { pointRadius: 5 }));
    }
    if (cux.get()) {
      sets.push(lineSet('uₓ(x)', sampleFn((t) => netD(P, H, t).ux, -2, 2, 200), COLOR.dark, { borderDash: [6, 4] }));
      sets.push(dotSet('uₓ at x', [{ x, y: d.ux }], COLOR.dark, { pointRadius: 5 }));
    }
    if (cuxx.get()) {
      sets.push(lineSet('uₓₓ(x)', sampleFn((t) => netD(P, H, t).uxx, -2, 2, 200), COLOR.warm, { borderDash: [2, 3] }));
      sets.push(dotSet('uₓₓ at x', [{ x, y: d.uxx }], COLOR.warm, { pointRadius: 5 }));
    }
    setChart(chart, sets);

    const u = (t) => netU(P, H, t);
    const fdUx = (u(x + dx) - u(x - dx)) / (2 * dx);
    const fdUxx = (u(x + dx) - 2 * u(x) + u(x - dx)) / (dx * dx);
    const pad = (s) => s.padEnd(8);
    $('ad-result').textContent =
      pad('x = ' + num(x, 3)) + '  chain rule      finite difference (h = 0.01)\n' +
      pad('u') + '= ' + sig(d.u, 6).padEnd(14) + '\n' +
      pad('u_x') + '= ' + sig(d.ux, 6).padEnd(14) + '  ' + sig(fdUx, 6) + '\n' +
      pad('u_xx') + '= ' + sig(d.uxx, 6).padEnd(14) + '  ' + sig(fdUxx, 6);
  }
  cx = addControl('ad-controls', { type: 'slider', id: 'ad-x', label: 'x', min: -2, max: 2, step: 0.01, hardMin: -2, hardMax: 2, value: 0.5, onChange: update });
  cu = addControl('ad-controls', { type: 'check', id: 'ad-su', label: 'Show u', value: true, onChange: update });
  cux = addControl('ad-controls', { type: 'check', id: 'ad-sux', label: 'Show ∂u/∂x', value: true, onChange: update });
  cuxx = addControl('ad-controls', { type: 'check', id: 'ad-suxx', label: 'Show ∂²u/∂x²', value: true, onChange: update });
  update();
}

/* ---------- 4.5 architecture ---------- */
function initArchitecture() {
  let cL, cH;
  function update() {
    const L = Math.round(cL.get()), H = Math.round(cH.get()), shown = Math.min(H, 6);
    const cols = [{ n: 2, label: 'Input', names: ['x', 't'] }];
    for (let i = 1; i <= L; i++) cols.push({ n: shown, label: 'Hidden ' + i });
    cols.push({ n: 1, label: 'Output', names: ['u'] });

    const gap = 130, W = 120 + gap * (cols.length - 1), cy = 120, dy = 34;
    const pos = cols.map((c, i) => Array.from({ length: c.n }, (_, j) => [60 + gap * i, cy + (j - (c.n - 1) / 2) * dy]));
    let svg = '';
    for (let i = 0; i < cols.length - 1; i++) {
      pos[i].forEach((a) => pos[i + 1].forEach((b) => {
        svg += '<line x1="' + a[0] + '" y1="' + a[1] + '" x2="' + b[0] + '" y2="' + b[1] + '" stroke="#c4c4bd" stroke-width="1"/>';
      }));
    }
    cols.forEach((c, i) => {
      pos[i].forEach((p, j) => {
        svg += '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="12" fill="#fff" stroke="' + COLOR.accent + '" stroke-width="1.3"/>';
        if (c.names) svg += '<text x="' + p[0] + '" y="' + (p[1] + 4.5) + '" text-anchor="middle" style="font-style:italic">' + c.names[j] + '</text>';
      });
      svg += '<text x="' + (60 + gap * i) + '" y="262" text-anchor="middle" style="fill:#5f6368">' + c.label + '</text>';
      if (i > 0 && i < cols.length - 1 && H > 6) svg += '<text x="' + (60 + gap * i) + '" y="' + (cy + 2.5 * dy + 28) + '" text-anchor="middle">⋮</text>';
    });
    svg += '<text x="' + (60 + gap * (cols.length - 1)) + '" y="' + (cy - 34) + '" text-anchor="middle" style="font-style:italic">u(x,t)</text>';
    const svgEl = $('ar-svg');
    svgEl.setAttribute('viewBox', '0 0 ' + W + ' 272');
    svgEl.innerHTML = svg;

    const params = 3 * H + (L - 1) * (H * H + H) + (H + 1);
    $('ar-info').textContent = L + ' hidden layer' + (L > 1 ? 's' : '') + ' with ' + H + ' neurons each' +
      (H > 6 ? ' (only 6 neurons are drawn per layer)' : '') + '; ' + params + ' trainable parameters for 2 inputs and 1 output.';
  }
  cL = addControl('ar-controls', { type: 'slider', id: 'ar-L', label: 'Hidden layers', min: 1, max: 6, step: 1, value: 2, onChange: update });
  cH = addControl('ar-controls', { type: 'slider', id: 'ar-H', label: 'Neurons per hidden layer', min: 1, max: 50, step: 1, value: 10, onChange: update });
  update();
}

/* ---------- 5 forward problem: shallow PINN for the Bratu equation ---------- */
function initForward() {
  const rng = mulberry32(7);
  const XI = Array.from({ length: 200 }, () => rng());      // random interior points in (0,1)
  const FW = { seed: 1, lm: null, trainer: null, base: 0 };
  const chart = newChart('fw-chart', 'x', 'u(x)', { y: { suggestedMin: -0.05 } });
  let cC, cA, cN, cH, cAna, cPinn;

  const params = () => ({ C: cC.get(), alpha: cA.get(), Ni: Math.round(cN.get()), H: Math.round(cH.get()) });

  /* Residual vector r such that ||r||^2 = MSE_i + alpha MSE_b. */
  function residual(p, q) {
    const r = [];
    for (let i = 0; i < q.Ni; i++) {
      const d = netD(p, q.H, XI[i]);
      r.push((d.uxx + q.C * Math.exp(d.u)) / Math.sqrt(q.Ni));
    }
    [0, 1].forEach((xb) => r.push(Math.sqrt(q.alpha) * netU(p, q.H, xb) / Math.sqrt(2)));
    return r;
  }

  function parts(p, q) {
    let mi = 0;
    for (let i = 0; i < q.Ni; i++) {
      const d = netD(p, q.H, XI[i]);
      mi += Math.pow(d.uxx + q.C * Math.exp(d.u), 2) / q.Ni;
    }
    const mb = (Math.pow(netU(p, q.H, 0), 2) + Math.pow(netU(p, q.H, 1), 2)) / 2;
    return { mi, mb };
  }

  function draw(iters) {
    const q = params(), p = FW.lm.p, sets = [];
    if (cAna.get()) sets.push(lineSet('Analytical', sampleFn((x) => bratuExact(q.C, x), 0, 1, 200), COLOR.dark, { borderWidth: 3, borderColor: '#9aa0a6', backgroundColor: '#9aa0a6' }));
    if (cPinn.get()) {
      sets.push(lineSet('PINN u(x; W)', sampleFn((x) => netU(p, q.H, x), 0, 1, 200), COLOR.accent, { borderWidth: 1.8 }));
      const pts = [];
      for (let i = 0; i < q.Ni; i++) pts.push({ x: XI[i], y: netU(p, q.H, XI[i]) });
      sets.push(dotSet('Interior points (N_i)', pts, COLOR.warm, { pointRadius: 1.8 }));
    }
    sets.push(dotSet('Boundary points', [{ x: 0, y: 0 }, { x: 1, y: 0 }], '#222', { pointStyle: 'rectRot', pointRadius: 6, backgroundColor: '#fff' }));
    setChart(chart, sets);

    const pt = parts(p, q);
    let err = 0;
    for (let i = 0; i <= 100; i++) err = Math.max(err, Math.abs(netU(p, q.H, i / 100) - bratuExact(q.C, i / 100)));
    $('fw-stat').textContent =
      'Iterations     : ' + iters + '\n' +
      'MSE interior   : ' + pt.mi.toExponential(3) + '\n' +
      'MSE boundary   : ' + pt.mb.toExponential(3) + '\n' +
      'Total loss     : ' + (pt.mi + q.alpha * pt.mb).toExponential(3) + '   (MSE_i + α MSE_b, α = ' + sig(q.alpha, 3) + ')\n' +
      'max |u - exact|: ' + err.toExponential(2) + '   (slides: loss 2.47038e-07 with BFGS)';
  }

  function train(iters) {
    if (FW.trainer) FW.trainer.cancel();
    const q = params(), start = FW.base;
    FW.lm.mu = 1e-2;
    FW.trainer = startTrainer(FW.lm, (p) => residual(p, q), iters, (lm, it) => {
      FW.lm = lm; FW.base = start + it; draw(FW.base);
    });
  }

  function reinit() {
    const q = params();
    if (FW.trainer) FW.trainer.cancel();
    FW.lm = { p: initNet(q.H, FW.seed, 2, 0.05), mu: 1e-2, cost: NaN };   // small output weights: start near u = 0
    FW.base = 0;
    draw(0);
  }

  const retrain = debounce(() => train(250), 250);
  const onParam = () => { if (FW.trainer) FW.trainer.cancel(); draw(FW.base); retrain(); };

  cC = addControl('fw-controls', { type: 'slider', id: 'fw-C', label: 'C', min: 0.1, max: 3.5, step: 0.01, hardMin: 0.01, hardMax: 3.5, value: 2, onChange: onParam });
  cA = addControl('fw-controls', { type: 'slider', id: 'fw-a', label: 'α (weight of the boundary term)', min: 0.01, max: 100, log: true, value: 1, onChange: onParam });
  cN = addControl('fw-controls', { type: 'slider', id: 'fw-N', label: 'Interior points N_i', min: 5, max: 200, step: 1, value: 99, onChange: onParam });
  cH = addControl('fw-controls', { type: 'slider', id: 'fw-H', label: 'Hidden neurons', min: 3, max: 30, step: 1, value: 20, onChange: () => { reinit(); retrain(); } });
  cAna = addControl('fw-controls', { type: 'check', id: 'fw-ana', label: 'Show analytical solution', value: true, onChange: () => draw(FW.base) });
  cPinn = addControl('fw-controls', { type: 'check', id: 'fw-pinn', label: 'Show PINN solution', value: true, onChange: () => draw(FW.base) });

  $('fw-train').addEventListener('click', () => train(250));
  $('fw-reset').addEventListener('click', () => { FW.seed += 1; reinit(); train(250); });

  reinit();
  onceVisible($('fw-chart'), () => train(250));
}

/* ---------- 6 inverse problem: estimate lambda_1, lambda_2 ---------- */
function initInverse() {
  const H = 15;
  const rngN = mulberry32(11);
  const NOISE = Array.from({ length: 100 }, () => randn(rngN));
  const IV = { lm: null, trainer: null, base: 0, xs: [], obs: [], q: null, cstar: bratuCritical() };
  const chart = newChart('iv-chart', 'x', 'u(x)', { y: { suggestedMin: -0.01 } });
  let ct1, ct2, cM, cNoise, cg1, cg2, ok = true;

  function setupData() {
    if (IV.trainer) IV.trainer.cancel();
    const t1 = ct1.get(), t2 = ct2.get(), M = Math.round(cM.get());
    ok = (0.5 * t1 * t2) <= IV.cstar * 0.999;
    IV.xs = Array.from({ length: M }, (_, i) => (i + 1) / (M + 1));
    IV.obs = [];
    if (ok) {
      const exact = IV.xs.map((x) => inverseExact(t1, t2, x));
      const umax = Math.max.apply(null, exact.map(Math.abs));
      IV.obs = exact.map((u, i) => u + (cNoise.get() / 100) * umax * NOISE[i]);
    }
    IV.lm = null; IV.base = 0;
    $('iv-run').disabled = !ok;
    draw();
  }

  /* Residuals F_DE and F_data at the data points; q = [network weights..., lambda1, lambda2]. */
  function residual(q) {
    const P = 3 * H + 1, p = q.slice(0, P), l1 = q[P], l2 = q[P + 1], M = IV.xs.length, r = [];
    for (let i = 0; i < M; i++) {
      const d = netD(p, H, IV.xs[i]);
      r.push((d.uxx + l1 * Math.exp((l2 * d.u) / 2)) / Math.sqrt(M));   // F_DE
      r.push((d.u - IV.obs[i]) / Math.sqrt(M));                           // F_data
    }
    return r;
  }

  function draw() {
    const t1 = ct1.get(), t2 = ct2.get(), P = 3 * H + 1, sets = [];
    let est1 = cg1.get(), est2 = cg2.get(), estLabel = ' (initial guess)', stat = '';
    if (ok) {
      sets.push(lineSet('True solution', sampleFn((x) => inverseExact(t1, t2, x), 0, 1, 200), COLOR.dark, { borderWidth: 2.5, borderColor: '#9aa0a6', backgroundColor: '#9aa0a6' }));
      sets.push(dotSet('Observed data', IV.xs.map((x, i) => ({ x, y: IV.obs[i] })), COLOR.warm, { pointRadius: 3.5 }));
    }
    if (IV.lm) {
      const q = IV.lm.p;
      est1 = q[P]; est2 = q[P + 1]; estLabel = '';
      sets.push(lineSet('PINN estimate u(x; W)', sampleFn((x) => netU(q, H, x), 0, 1, 200), COLOR.accent));
      const r = residual(q);
      let mde = 0, mdata = 0;
      r.forEach((v, i) => { if (i % 2 === 0) mde += v * v; else mdata += v * v; });
      stat = 'Iterations: ' + IV.base + '  |  MSE_DE = ' + mde.toExponential(3) + '  |  MSE_data = ' + mdata.toExponential(3) + '  |  slides: loss 9.1483e-08 (Levenberg–Marquardt)';
    }
    setChart(chart, sets);

    const rel = (e, t) => (IV.lm ? (100 * Math.abs(e - t) / Math.abs(t)).toFixed(2) + ' %' : '—');
    document.querySelector('#iv-table tbody').innerHTML =
      '<tr><td>λ₁</td><td>' + num(t1, 4) + '</td><td>' + num(est1, 4) + estLabel + '</td><td>' + rel(est1, t1) + '</td></tr>' +
      '<tr><td>λ₂</td><td>' + num(t2, 4) + '</td><td>' + num(est2, 4) + estLabel + '</td><td>' + rel(est2, t2) + '</td></tr>';
    $('iv-stat').textContent = ok ? (stat || 'Press Estimate to start from the initial guess.') :
      'No solution for these values: λ₁λ₂/2 = ' + num(0.5 * t1 * t2, 3) + ' exceeds C* ≈ ' + num(IV.cstar, 3) + '.';
    $('iv-run').textContent = IV.lm ? 'Continue estimating' : 'Estimate';
  }

  function run() {
    if (!ok) return;
    if (IV.trainer) IV.trainer.cancel();
    if (!IV.lm) {
      IV.lm = { p: initNet(H, 1, 2, 0.05).concat([cg1.get(), cg2.get()]), mu: 1e-2, cost: NaN };
      IV.base = 0;
    }
    const start = IV.base;
    IV.trainer = startTrainer(IV.lm, residual, 400, (lm, it) => {
      IV.lm = lm; IV.base = start + it; draw();
    });
  }

  const mk = (id, label, min, max, step, v) => addControl('iv-controls', { type: 'slider', id, label, min, max, step, value: v, onChange: setupData });
  ct1 = mk('iv-t1', 'True λ₁', 0.2, 3, 0.05, 1);
  ct2 = mk('iv-t2', 'True λ₂', 0.5, 4, 0.05, 2);
  cM = mk('iv-M', 'Number of observations', 5, 99, 1, 30);
  cNoise = mk('iv-noise', 'Noise (% of max |u|)', 0, 5, 0.5, 0);
  cg1 = mk('iv-g1', 'Initial guess λ₁', 0.1, 3, 0.05, 0.5);
  cg2 = mk('iv-g2', 'Initial guess λ₂', 0.5, 4, 0.05, 1);
  $('iv-run').addEventListener('click', run);
  $('iv-reset').addEventListener('click', setupData);
  setupData();
}


/* =========================================================
   PART 3 - PAGE FLOW, GUIDE CHARACTER, INTRO STORY
   ========================================================= */

const REDUCED = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
const SVGNS = 'http://www.w3.org/2000/svg';

const mascotSvg = (id, cls) =>
  '<svg class="' + cls + '" viewBox="0 0 64 64" aria-hidden="true" focusable="false"><use href="#' + id + '"></use></svg>';
const slug = (t) => t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

/* ---------- page flow: chapter mascots, next links, reveal, progress bar, side contents ---------- */
function initFlow() {
  const sections = Array.from(document.querySelectorAll('main > section'));
  const navLinks = Array.from(document.querySelectorAll('#nav-links a'));
  const MOOD = {
    'big-idea': 'neuro', 'introduction': 'neuro', 'neural-networks': 'neuro-think', 'differential-equations': 'neuro-think',
    'pinns': 'neuro-ooh', 'forward': 'neuro', 'inverse': 'neuro-think', 'advantages': 'neuro-ooh', 'references': 'neuro'
  };

  sections.forEach((sec, i) => {
    const h2 = sec.querySelector('h2');
    if (h2) h2.insertAdjacentHTML('afterbegin', mascotSvg(MOOD[sec.id] || 'neuro', 'chapter-mascot'));
    sec.querySelectorAll('h3').forEach((h) => { if (!h.id) h.id = slug(h.textContent); });
    if (i < sections.length - 1) {          // guided "next" link at the end of every part
      const next = sections[i + 1];
      const label = navLinks.find((a) => a.getAttribute('href') === '#' + next.id);
      sec.insertAdjacentHTML('beforeend',
        '<div class="next-link"><a href="#' + next.id + '">Next: ' + (label ? label.textContent : '') + ' &rarr;</a></div>');
    }
  });

  document.querySelectorAll('section .takeaway').forEach((el) => {
    el.innerHTML = mascotSvg('neuro', 'mini-mascot') + '<span>' + el.innerHTML + '</span>';
  });

  // fade-in when a block first scrolls into view (skipped for reduced motion)
  if ('IntersectionObserver' in window && !REDUCED) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting || e.boundingClientRect.top < 0) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { threshold: 0.06, rootMargin: '0px 0px -30px 0px' });
    sections.forEach((sec) => Array.from(sec.children).forEach((ch) => {
      if (ch.tagName === 'H2') return;
      ch.classList.add('reveal');
      io.observe(ch);
    }));
  }

  // reading progress and "on this page" list for the current part (wide screens only)
  const bar = $('progress');
  const toc = document.createElement('aside');
  toc.id = 'toc';
  toc.setAttribute('aria-label', 'On this page');
  document.body.appendChild(toc);
  let tocFor = null, ticking = false;

  function buildToc(sec) {
    if (tocFor === sec) return;
    tocFor = sec;
    const hs = sec ? Array.from(sec.querySelectorAll('h3')) : [];
    if (!hs.length) { toc.classList.remove('show'); toc.innerHTML = ''; return; }
    toc.innerHTML = '<div class="toc-title">On this page</div><ul>' +
      hs.map((h) => '<li><a href="#' + h.id + '">' + h.textContent + '</a></li>').join('') + '</ul>';
    toc.classList.add('show');
  }

  function onScroll() {
    ticking = false;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.transform = 'scaleX(' + (max > 0 ? Math.min(1, window.scrollY / max) : 0) + ')';
    let cur = null;
    sections.forEach((s) => { if (s.getBoundingClientRect().top <= 110) cur = s; });
    buildToc(cur);
    if (cur) {
      let on = -1;
      Array.from(cur.querySelectorAll('h3')).forEach((h, i) => { if (h.getBoundingClientRect().top <= 140) on = i; });
      toc.querySelectorAll('a').forEach((a, i) => a.classList.toggle('on', i === on));
    }
  }
  window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
  window.addEventListener('resize', onScroll);
  onScroll();
}

/* ---------- hero guide: tips on click ---------- */
function initHero() {
  const btn = $('hero-mascot'), bubble = $('hero-bubble');
  if (!btn) return;
  const tips = [
    'Hi, I am Neuro. I will show you how a curve can learn to obey the laws of physics.',
    'Tip: every slider and number box on this page redraws a graph.',
    'A neural network is just functions inside functions. Part 2 starts there.',
    'The loss is a score for how badly the curve breaks the rule. Smaller is better.',
    'The Train and Estimate buttons run real, tiny PINNs inside your browser.',
    'Not sure where to begin? Open the Big Idea part: no mathematics needed.'
  ];
  let i = 0;
  btn.addEventListener('click', () => {
    i = (i + 1) % tips.length;
    btn.classList.remove('hop');
    void btn.offsetWidth;
    btn.classList.add('hop');
    bubble.classList.add('swap');
    setTimeout(() => { bubble.textContent = tips[i]; bubble.classList.remove('swap'); }, 180);
  });
  btn.addEventListener('animationend', (e) => { if (e.animationName === 'hop') btn.classList.remove('hop'); });
}

/* ---------- 2.3: signal travelling through the network diagram ---------- */
function initPulse() {
  const btn = $('nn-animate');
  if (!btn) return;
  const P = { x1: [70, 90], x2: [70, 210], h1: [300, 90], h2: [300, 210], y: [540, 150] };

  function send(svg, legs) {
    legs.forEach((leg) => {
      const c = document.createElementNS(SVGNS, 'circle');
      c.setAttribute('class', 'pulse');
      c.setAttribute('r', '5');
      c.setAttribute('fill', COLOR.warm);
      const m = document.createElementNS(SVGNS, 'animateMotion');
      m.setAttribute('dur', '0.8s');
      m.setAttribute('begin', 'indefinite');
      m.setAttribute('fill', 'remove');
      m.setAttribute('path', 'M' + P[leg[0]].join(' ') + ' L' + P[leg[1]].join(' '));
      c.appendChild(m);
      svg.appendChild(c);
      m.beginElement();
      setTimeout(() => c.remove(), 820);
    });
  }

  btn.addEventListener('click', () => {
    const svg = $('nn-svg');
    svg.querySelectorAll('.pulse').forEach((n) => n.remove());
    svg.querySelectorAll('.lit').forEach((n) => n.classList.remove('lit'));
    btn.disabled = true;
    send(svg, [['x1', 'h1'], ['x1', 'h2'], ['x2', 'h1'], ['x2', 'h2']]);
    setTimeout(() => { ['h1', 'h2'].forEach((k) => { const n = $('nn-n-' + k); if (n) n.classList.add('lit'); }); send(svg, [['h1', 'y'], ['h2', 'y']]); }, 850);
    setTimeout(() => { const n = $('nn-n-y'); if (n) n.classList.add('lit'); btn.disabled = false; }, 1700);
  });
}

/* ---------- 4.1: walk through the pipeline once, until the reader takes over ---------- */
function initPipelineAuto() {
  const flow = $('pf-flow');
  if (!flow) return;
  const stages = Array.from(flow.querySelectorAll('.stage'));
  let i = 0, timer = null;
  const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
  stages.forEach((b) => b.addEventListener('click', (e) => { if (e.isTrusted) stop(); }));
  if (REDUCED) return;
  onceVisible(flow, () => {
    timer = setInterval(() => {
      i += 1;
      stages[i].click();
      if (i >= stages.length - 1) stop();
    }, 2300);
  });
}

/* ---------- Big idea: a real small PINN run, told in four steps ----------
   Step 1 slides the weights between three random networks, step 2 shows the rule,
   step 3 measures how badly the untrained curve breaks it, step 4 replays the
   Levenberg-Marquardt iterations of a real training run (Bratu, C = 2). */
function initStory() {
  const svg = $('story-svg');
  if (!svg) return;
  const H = 12, C = 2, NBAR = 14, NSMP = 40;
  const B = { x0: 50, x1: 610, top: 24, bot: 200, u0: -0.1, u1: 0.45, base: 306, k: 20, cap: 60 };
  const X = (x) => B.x0 + (B.x1 - B.x0) * x;
  const Y = (u) => B.bot - ((u - B.u0) / (B.u1 - B.u0)) * (B.bot - B.top);
  const nets = [3, 4, 5].map((s) => initNet(H, s, 2.2, 0.45));
  const exact = sampleFn((x) => bratuExact(C, x), 0, 1, 80);
  /* Vertical range that fits the curves being shown (always includes u = 0, at least 0.45 tall). */
  function fitRange(curves) {
    let lo = 0, hi = 0;
    curves.forEach((c) => c.forEach((p) => { lo = Math.min(lo, p.y); hi = Math.max(hi, p.y); }));
    const span = Math.max(hi - lo, 0.45), mid = (hi + lo) / 2, pad = span * 0.12;
    B.u0 = mid - span / 2 - pad;
    B.u1 = mid + span / 2 + pad;
  }
  const pathOf = (pts) => pts.map((p, i) => (i ? 'L' : 'M') + X(p.x).toFixed(1) + ' ' + Y(p.y).toFixed(1)).join(' ');
  const S = { stage: 1, t: 0, fi: 0, playing: false, visible: false, raf: null, last: 0, timers: [] };
  let frames = null;
  const stage1Curves = [];
  nets.forEach((p, i) => {
    const q = nets[(i + 1) % 3];
    stage1Curves.push(sampleFn((x) => netU(p, H, x), 0, 1, 80));
    stage1Curves.push(sampleFn((x) => netU(p.map((v, k) => 0.5 * (v + q[k])), H, x), 0, 1, 80));
  });

  const tabs = Array.from(document.querySelectorAll('#story-tabs .stage'));
  const prev = $('story-prev'), next = $('story-next'), replay = $('story-replay'), caption = $('story-caption');

  const CAPTIONS = {
    1: 'A neural network is a <strong>flexible curve</strong>. Its shape is set by numbers called weights \\(W\\). Slide the weights and the curve changes shape.',
    2: 'Physics gives a <strong>rule</strong>. Here it is the Bratu equation \\(u_{xx}+2e^{u}=0\\) with both ends pinned, \\(u(0)=u(1)=0\\). The dashed curve is one that obeys it.',
    3: '<strong>Measure the mistakes.</strong> At sample points we check how badly the curve breaks the rule. The orange bars show \\(|u_{xx}+2e^{u}|\\); the loss is built from them (mean of squares, plus the error at the two pins).',
    4: '<strong>Nudge and repeat.</strong> The optimiser changes the weights to shrink the bars, again and again, and the curve settles onto the dashed one. This is a real run: 12 tanh neurons, Levenberg–Marquardt.'
  };

  /* Pre-compute the training run once: curve, rule violation at sample points, loss. */
  function ensureFrames() {
    if (frames) return;
    const xi = Array.from({ length: NSMP }, (_, i) => (i + 0.5) / NSMP);
    const res = (p) => {
      const r = [];
      xi.forEach((x) => { const d = netD(p, H, x); r.push((d.uxx + C * Math.exp(d.u)) / Math.sqrt(NSMP)); });
      [0, 1].forEach((xb) => r.push(netU(p, H, xb) / Math.sqrt(2)));
      return r;
    };
    const xs = Array.from({ length: NBAR }, (_, j) => (j + 1) / (NBAR + 1));
    const snap = (p, cost, it) => ({
      it, cost,
      curve: sampleFn((x) => netU(p, H, x), 0, 1, 80),
      bars: xs.map((x) => { const d = netD(p, H, x); return { x, r: Math.abs(d.uxx + C * Math.exp(d.u)) }; })
    });
    let lm = { p: nets[0].slice(), mu: 1e-2, cost: 0 };
    frames = [snap(lm.p, sumSq(res(lm.p)), 0)];
    for (let i = 1; i <= 60; i++) {
      lm = lmStep(lm, res);
      frames.push(snap(lm.p, lm.cost, i));
      if (lm.stalled || lm.cost < 1e-6) break;
    }
  }

  function render() {
    const st = S.stage;
    let curve = null, weights = '', frame = null;
    if (st === 1) {
      let p = nets[0];
      if (!REDUCED) {
        const ph = S.t / 2600, i = Math.floor(ph) % 3, f = ph - Math.floor(ph), e = f * f * (3 - 2 * f);
        p = nets[i].map((v, k) => v + (nets[(i + 1) % 3][k] - v) * e);
      }
      curve = sampleFn((x) => netU(p, H, x), 0, 1, 80);
      weights = 'weights W:  a₁ = ' + num(p[0], 2) + ',  c₁ = ' + num(p[H], 2) + ',  v₁ = ' + num(p[2 * H], 2) + ',  …  (' + (3 * H + 1) + ' in total)';
    } else if (st >= 3) {
      frame = frames[Math.min(frames.length - 1, Math.floor(S.fi))];
      curve = frame.curve;
    }

    if (st === 1) fitRange(stage1Curves);
    else if (st === 2) fitRange([exact]);
    else fitRange([curve, exact]);

    let s = '<defs><clipPath id="story-clip"><rect x="' + (B.x0 - 6) + '" y="' + (B.top - 6) + '" width="' + (B.x1 - B.x0 + 12) + '" height="' + (B.bot - B.top + 12) + '"/></clipPath></defs>';
    s += '<rect x="' + B.x0 + '" y="' + B.top + '" width="' + (B.x1 - B.x0) + '" height="' + (B.bot - B.top) + '" fill="none" stroke="#e4e4e0"/>';
    s += '<line x1="' + B.x0 + '" y1="' + Y(0) + '" x2="' + B.x1 + '" y2="' + Y(0) + '" stroke="#bbb"/>';
    s += '<text x="' + B.x0 + '" y="' + (B.bot + 18) + '" text-anchor="middle" style="fill:#5f6368">x = 0</text>';
    s += '<text x="' + B.x1 + '" y="' + (B.bot + 18) + '" text-anchor="middle" style="fill:#5f6368">x = 1</text>';
    s += '<text x="' + (B.x0 - 8) + '" y="' + (Y(0) + 4) + '" text-anchor="end" style="fill:#5f6368">0</text>';

    if (st >= 2) s += '<path d="' + pathOf(exact) + '" fill="none" stroke="#9aa0a6" stroke-width="2.2" stroke-dasharray="6 5" clip-path="url(#story-clip)"/>';
    if (curve) s += '<path d="' + pathOf(curve) + '" fill="none" stroke="' + COLOR.accent + '" stroke-width="2.8" stroke-linejoin="round" clip-path="url(#story-clip)"/>';
    if (st >= 2) {
      [0, 1].forEach((x) => {
        s += '<rect x="' + (X(x) - 5) + '" y="' + (Y(0) - 5) + '" width="10" height="10" transform="rotate(45 ' + X(x) + ' ' + Y(0) + ')" fill="#fff" stroke="#222" stroke-width="1.6"/>';
      });
      s += '<text class="halo" x="' + (B.x0 + 12) + '" y="' + (Y(0) - 10) + '">u(0) = 0</text>';
      s += '<text class="halo" x="' + (B.x1 - 12) + '" y="' + (Y(0) - 10) + '" text-anchor="end">u(1) = 0</text>';
    }
    if (st === 1) s += '<text x="' + (B.x0 + 8) + '" y="' + (B.top + 16) + '" style="fill:#5f6368">' + weights + '</text>';
    if (st === 2) s += '<text x="' + ((B.x0 + B.x1) / 2) + '" y="' + (B.top + 22) + '" text-anchor="middle" class="big">rule:  uₓₓ + 2e^u = 0</text>';

    if (st < 3) {
      const hint = st === 1 ? 'Nothing forces this curve to obey any rule yet.' : 'In the next step we measure how far a curve is from obeying the rule.';
      s += '<text x="' + ((B.x0 + B.x1) / 2) + '" y="' + (B.base - 34) + '" text-anchor="middle" style="fill:#5f6368;font-style:italic">' + hint + '</text>';
    }
    if (st >= 3 && frame) {
      s += '<text x="' + B.x0 + '" y="' + (B.base - B.cap - 12) + '" style="fill:#5f6368">rule violation at sample points: |uₓₓ + 2e^u|</text>';
      s += '<line x1="' + B.x0 + '" y1="' + B.base + '" x2="' + B.x1 + '" y2="' + B.base + '" stroke="#bbb"/>';
      frame.bars.forEach((b) => {
        const h = Math.max(1.5, Math.min(B.cap, b.r * B.k));
        s += '<rect x="' + (X(b.x) - 8) + '" y="' + (B.base - h) + '" width="16" height="' + h.toFixed(1) + '" fill="' + COLOR.warm + '" opacity="0.85"/>';
      });
      s += '<text class="big" x="' + (B.x1 - 6) + '" y="' + (B.top + 20) + '" text-anchor="end">loss = ' + frame.cost.toExponential(1) + '</text>';
      if (st === 4) s += '<text x="' + (B.x1 - 6) + '" y="' + (B.top + 38) + '" text-anchor="end" style="fill:#5f6368">iteration ' + frame.it + '</text>';
    }
    svg.innerHTML = s;
  }

  const animating = () => (S.stage === 1 && !REDUCED) || (S.stage === 4 && S.playing);

  function tick(now) {
    S.raf = null;
    if (!S.visible) return;
    const dt = Math.min(now - S.last, 100);
    S.last = now;
    S.t += dt;
    if (S.stage === 4 && S.playing) {
      S.fi += dt / 90;
      if (S.fi >= frames.length - 1) { S.fi = frames.length - 1; S.playing = false; }
    }
    render();
    if (animating()) S.raf = requestAnimationFrame(tick);
  }

  function kick() {
    render();
    if (S.visible && !S.raf && animating()) { S.last = performance.now(); S.raf = requestAnimationFrame(tick); }
  }

  function stopAuto() { S.timers.forEach(clearTimeout); S.timers = []; }

  function setStage(n, byUser) {
    if (byUser) stopAuto();
    S.stage = n; S.fi = 0; S.playing = n === 4;
    if (n >= 3) ensureFrames();
    tabs.forEach((b) => b.classList.toggle('active', parseInt(b.dataset.s, 10) === n));
    caption.innerHTML = CAPTIONS[n];
    renderAuto(caption);
    prev.disabled = n === 1;
    next.disabled = n === 4;
    replay.hidden = n !== 4;
    kick();
  }

  tabs.forEach((b) => b.addEventListener('click', () => setStage(parseInt(b.dataset.s, 10), true)));
  prev.addEventListener('click', () => setStage(Math.max(1, S.stage - 1), true));
  next.addEventListener('click', () => setStage(Math.min(4, S.stage + 1), true));
  replay.addEventListener('click', () => setStage(4, true));

  if ('IntersectionObserver' in window) {
    new IntersectionObserver((entries) => {
      S.visible = entries.some((e) => e.isIntersecting);
      if (S.visible) kick();
    }, { threshold: 0.2 }).observe(svg);
  } else {
    S.visible = true;
  }

  setStage(1, false);
  if (!REDUCED) {
    onceVisible(svg, () => {      // play the story once, then leave it to the reader
      S.timers = [
        setTimeout(() => setStage(2, false), 3800),
        setTimeout(() => setStage(3, false), 7400),
        setTimeout(() => setStage(4, false), 11200)
      ];
    });
  }
}
/* ---------- start-up ---------- */
function init() {
  if (typeof katex === 'undefined' || typeof renderMathInElement !== 'function') {
    libWarning('The KaTeX library could not be loaded (no connection to cdn.jsdelivr.net?). Equations will appear as plain LaTeX.');
  } else {
    renderAuto(document.body);
  }
  if (typeof Chart === 'undefined') {
    libWarning('The Chart.js library could not be loaded (no connection to cdn.jsdelivr.net?). Graphs are not available; calculators still work.');
  }
  setupNav();
  initComposite();
  initNetwork();
  initDeepSizes();
  initApprox();
  initFiniteDifference();
  initBratu();
  initPipeline();
  initCollocation();
  initLoss();
  initDerivatives();
  initArchitecture();
  initForward();
  initInverse();
  initFlow();
  initHero();
  initStory();
  initPulse();
  initPipelineAuto();
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { bratuTheta, bratuExact, bratuCritical, bratuFDM, inverseExact, netU, netD, initNet, lmStep, mulberry32, randn };
}
