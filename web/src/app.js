"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const catalog = JSON.parse($("catalog").textContent);
  const sections = Object.fromEntries(["static", "continuous", "renderer"].map(key => [key, catalog.filter(c => c.section === key)]));
  const colors = ["#d97706", "#0ea5e9", "#16a34a", "#dc2626"];
  const phone = matchMedia("(max-width:900px)");
  const reduced = matchMedia("(prefers-reduced-motion:reduce)");
  const cache = new Map();
  const remembered = {static: 0, continuous: 0, renderer: 0};
  const speeds = {};
  let continuousTrail = 1200;
  const st = {section: "continuous", index: 0, t: 0, playing: !reduced.matches,
    speed: 1, trail: 1200, follow: true, guide: true, marker: true,
    zoom: 1.10, pan: [0, 0], draw: 0, mode: "reports", axis: 0, filter: "all"};
  let data = null, panes = [], cameraFrames = [], viewport = null;
  let requestId = 0, raf = 0, last = null, ending = 0, dirty = true, sizesDirty = true;
  const current = () => sections[st.section][st.index];
  const defaultFollow = () => st.section === "continuous" && (phone.matches || current().follow);
  const isReports = () => st.section === "renderer" && st.mode === "reports";
  const dialogOpen = () => Boolean(document.querySelector("dialog[open]"));
  const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));

  function integrate(reports, start, factor = 1) {
    const path = new Float64Array(reports.length + 2);
    path[0] = start[0]; path[1] = start[1];
    for (let i = 0; i < reports.length; i += 2) {
      path[i + 2] = path[i] + reports[i] * factor;
      path[i + 3] = path[i + 1] + reports[i + 1] * factor;
    }
    return path;
  }

  async function load(meta) {
    if (cache.has(meta.id)) {
      const result = cache.get(meta.id); cache.delete(meta.id); cache.set(meta.id, result); return result;
    }
    if (!window.DecompressionStream) throw new Error("This browser cannot open the motion data. Please use a current browser.");
    const payload = JSON.parse($("motion-" + meta.id).textContent);
    const bytes = Uint8Array.from(atob(payload.blob), c => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    const buffer = await new Response(stream).arrayBuffer();
    if (buffer.byteLength !== payload.bytes) throw new Error("The motion data is incomplete.");
    if (window.crypto?.subtle) {
      const digest = await crypto.subtle.digest("SHA-256", buffer);
      const hex = [...new Uint8Array(digest)].map(v => v.toString(16).padStart(2, "0")).join("");
      if (hex !== payload.sha256) throw new Error("The motion data could not be verified.");
    }
    const arrays = {};
    const types = {"<f8": Float64Array, "<f4": Float32Array, "<i2": Int16Array};
    for (const [key, spec] of Object.entries(payload.layout)) {
      arrays[key] = new types[spec.type](buffer, spec.offset, spec.shape.reduce((a, b) => a * b, 1));
    }
    const result = {...arrays, paths: []};
    if (arrays.humanReports) result.human = integrate(arrays.humanReports, meta.start, meta.humanFactor);
    for (let i = 0; i < 4; i++) {
      const reports = meta.section === "static" ? arrays["draw" + i]
        : arrays.reports.subarray(i * meta.duration * 2, (i + 1) * meta.duration * 2);
      result.paths.push(integrate(reports, meta.start, meta.factor));
    }
    if (arrays.smooth) result.smoothPath = integrate(arrays.smooth, meta.start);
    cache.set(meta.id, result);
    while (cache.size > 3) cache.delete(cache.keys().next().value);
    return result;
  }

  function markDirty() {
    dirty = true;
    if (!raf && !document.hidden) raf = requestAnimationFrame(frame);
  }
  function play(on) {
    st.playing = on; last = null;
    $("play").setAttribute("aria-label", on ? "Pause" : "Play");
    $("play").setAttribute("aria-pressed", String(on));
    $("play").classList.toggle("on", on);
    $("play").textContent = on ? "❚❚ pause" : "▶ play";
    if (on && data && st.t >= current().duration) { st.t = 0; ending = 0; }
    markDirty();
  }
  function frame(now) {
    raf = 0;
    const dt = last === null ? 0 : Math.max(0, now - last); last = now;
    if (st.playing && data && !dialogOpen()) {
      if (st.t >= current().duration) {
        ending += dt;
        if (ending >= 650) { st.t = 0; ending = 0; }
      } else st.t = Math.min(current().duration, st.t + dt * st.speed);
      dirty = true;
    }
    if (dirty && data) { dirty = false; draw(); }
    if (st.playing && data && !document.hidden && !dialogOpen()) raf = requestAnimationFrame(frame);
    else last = null;
  }

  function fullCamera() {
    const [x0, y0, x1, y1] = current().bounds;
    return {cx: (x0 + x1) / 2, cy: (y0 + y1) / 2,
      sx: Math.max(1, x1 - x0) * 1.16, sy: Math.max(1, y1 - y0) * 1.16};
  }
  function buildCameras() {
    cameraFrames = [];
    if (st.section !== "continuous") return;
    const n = current().duration + 1, base = fullCamera();
    const tracks = [data.target, data.human, ...data.paths].filter(Boolean);
    let previous = null;
    for (let t = 0; t < n; t += 40) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const path of tracks) for (let j = Math.max(0, t - 1000); j <= t; j += 12) {
        x0 = Math.min(x0, path[j * 2]); x1 = Math.max(x1, path[j * 2]);
        y0 = Math.min(y0, path[j * 2 + 1]); y1 = Math.max(y1, path[j * 2 + 1]);
      }
      const r = Math.max(data.radius[t * 2], data.radius[t * 2 + 1]);
      const c = {cx: (x0 + x1) / 2, cy: (y0 + y1) / 2,
        sx: Math.max((x1 - x0) * 1.25 + r * 2, r * 7, base.sx * .30),
        sy: Math.max((y1 - y0) * 1.25 + r * 2, r * 7, base.sy * .30)};
      if (previous) for (const k of ["cx", "cy", "sx", "sy"]) {
        c[k] = previous[k] + (c[k] - previous[k]) * (k[0] === "s" && c[k] > previous[k] ? .48 : .2);
      }
      // Neighboring frames share a padded envelope, including the whole target.
      for (let j = Math.max(0, t - 40); j <= Math.min(n - 1, t + 40); j++) for (const path of tracks) {
        const rx = path === data.target ? data.radius[j * 2] : 0;
        const ry = path === data.target ? data.radius[j * 2 + 1] : 0;
        c.sx = Math.max(c.sx, 2.36 * (Math.abs(path[j * 2] - c.cx) + rx));
        c.sy = Math.max(c.sy, 2.36 * (Math.abs(path[j * 2 + 1] - c.cy) + ry));
      }
      cameraFrames.push(c); previous = c;
    }
  }
  function resize() {
    sizesDirty = false;
    const dpr = Math.max(2, Math.min(window.devicePixelRatio || 1, 3));
    for (const p of panes) {
      p.visible = !p.el.classList.contains("mobile-hidden") || !phone.matches;
      if (!p.visible) continue;
      const rect = p.canvas.getBoundingClientRect();
      p.w = rect.width; p.h = rect.height; p.dpr = dpr;
      p.canvas.width = Math.max(1, Math.round(p.w * dpr));
      p.canvas.height = Math.max(1, Math.round(p.h * dpr));
    }
  }
  function computeView() {
    if (sizesDirty) resize();
    const visible = panes.filter(p => p.visible);
    if (isReports()) {
      const meta = current();
      viewport = {cx: meta.reportCenter + st.pan[0], cy: st.pan[1],
        sx: Math.min(...visible.map(p => Math.max(1, p.w - 60) / meta.reportSpan)) * st.zoom,
        sy: Math.min(...visible.map(p => Math.max(1, p.h - 44) / (2.5 * meta.maxReport[st.axis]))) * st.zoom};
      return viewport;
    }
    let camera = fullCamera();
    if (st.follow && cameraFrames.length) {
      const at = st.t / 40, index = Math.min(cameraFrames.length - 1, Math.floor(at));
      const a = cameraFrames[index], b = cameraFrames[Math.min(index + 1, cameraFrames.length - 1)];
      camera = Object.fromEntries(["cx", "cy", "sx", "sy"].map(k => [k, a[k] + (b[k] - a[k]) * Math.min(1, at - index)]));
    }
    const scale = Math.min(...visible.map(p => Math.min(p.w / camera.sx, p.h / camera.sy))) * st.zoom;
    viewport = {cx: camera.cx + st.pan[0], cy: camera.cy + st.pan[1], sx: scale, sy: scale};
    return viewport;
  }
  function origin(p) { return isReports() ? [42 + (p.w - 60) / 2, 9 + (p.h - 44) / 2] : [p.w / 2, p.h / 2]; }
  function point(p, x, y) {
    const [ox, oy] = origin(p);
    const direction = st.section === "static" || (st.section === "renderer" && !isReports()) ? 1 : -1;
    return [ox + (x - viewport.cx) * viewport.sx, oy + direction * (y - viewport.cy) * viewport.sy];
  }
  function world(p, x, y) {
    const [ox, oy] = origin(p);
    const direction = st.section === "static" || (st.section === "renderer" && !isReports()) ? 1 : -1;
    return [viewport.cx + (x - ox) / viewport.sx, viewport.cy + direction * (y - oy) / viewport.sy];
  }
  function fit() {
    st.follow = defaultFollow(); st.zoom = isReports() ? 1 : 1.10; st.pan = [0, 0];
    updateControls(); markDirty();
  }
  function freezeFollow() {
    if (!st.follow) return;
    const old = {...computeView()}, base = fullCamera();
    const scale = Math.min(...panes.filter(p => p.visible).map(p => Math.min(p.w / base.sx, p.h / base.sy)));
    st.follow = false; st.zoom = old.sx / scale; st.pan = [old.cx - base.cx, old.cy - base.cy];
    updateControls();
  }
  function zoom(factor, pane = null, anchor = null) {
    if (!data) return;
    computeView();
    const before = pane && anchor ? world(pane, ...anchor) : null;
    st.zoom = clamp(st.zoom * factor, .2, isReports() ? 60 : 80); computeView();
    if (before) { const after = world(pane, ...anchor); st.pan[0] += before[0] - after[0]; st.pan[1] += before[1] - after[1]; }
    markDirty();
  }

  function stroke(p, path, lo, hi, color, width, alpha = 1, step = 1) {
    hi = Math.min(hi, path.length / 2 - 1); lo = Math.max(0, lo);
    if (hi <= lo) return;
    const g = p.ctx;
    g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.beginPath();
    g.moveTo(...point(p, path[lo * 2], path[lo * 2 + 1]));
    for (let i = lo + step; i <= hi; i += step) g.lineTo(...point(p, path[i * 2], path[i * 2 + 1]));
    if ((hi - lo) % step) g.lineTo(...point(p, path[hi * 2], path[hi * 2 + 1]));
    g.stroke(); g.globalAlpha = 1;
  }
  function dot(p, xy, color, radius = 3.3) {
    let [x, y] = point(p, ...xy); const g = p.ctx;
    if (x < 5 || y < 5 || x > p.w - 5 || y > p.h - 5) {
      const dx = x - p.w / 2, dy = y - p.h / 2;
      const q = Math.min((p.w / 2 - 8) / Math.max(Math.abs(dx), 1e-9), (p.h / 2 - 8) / Math.max(Math.abs(dy), 1e-9));
      g.save(); g.translate(p.w / 2 + dx * q, p.h / 2 + dy * q); g.rotate(Math.atan2(dy, dx));
      g.fillStyle = color; g.beginPath(); g.moveTo(4, 0); g.lineTo(-3, -3); g.lineTo(-3, 3); g.closePath(); g.fill(); g.restore(); return;
    }
    g.fillStyle = color; g.strokeStyle = "#fff"; g.lineWidth = 1.3;
    g.beginPath(); g.arc(x, y, radius, 0, Math.PI * 2); g.fill(); g.stroke();
  }
  function drawPath(p, tick) {
    const meta = current(), prefixTime = meta.prefixDuration;
    const path = p.reference ? (data.human || data.target) : data.paths[p.draw];
    const color = p.reference ? (meta.human ? "#2563eb" : "#059669") : colors[p.draw];
    if (data.target) {
      if (st.section === "continuous" && st.guide) stroke(p, data.target, 0, meta.duration, "#dce8e3", 1, .75, Math.max(1, Math.floor(meta.duration / 1000)));
      const i = st.section === "continuous" ? tick * 2 : 0;
      const [x, y] = point(p, data.target[i], data.target[i + 1]);
      p.ctx.fillStyle = "#ecfdf5"; p.ctx.strokeStyle = "#059669"; p.ctx.lineWidth = 2;
      p.ctx.beginPath(); p.ctx.ellipse(x, y, Math.max(1, data.radius[i] * viewport.sx), Math.max(1, data.radius[i + 1] * viewport.sy), 0, 0, Math.PI * 2); p.ctx.fill(); p.ctx.stroke();
    }
    if (data.smoothPath) stroke(p, data.smoothPath, 0, meta.duration, "#9aa5b1", 1.6, .65);
    if (data.prefix?.length > 2) {
      if (prefixTime) {
        stroke(p, data.prefix, 0, Math.min(tick, prefixTime), "#9aa5b1", 2.2);
        dot(p, [data.prefix[0], data.prefix[1]], "#1f2933", 3.7);
        if (st.marker) dot(p, meta.start, "#7c3aed", 4.5);
      } else if (st.trail > 0 && tick < st.trail) {
        stroke(p, data.prefix, 0, data.prefix.length / 2 - 1, "#9aa5b1", 1.9, .65 * (1 - tick / st.trail));
      }
    }
    if (tick < prefixTime) return;
    const end = Math.min(tick - prefixTime, path.length / 2 - 1);
    if (!st.trail) stroke(p, path, 0, end, color, 2.15);
    else {
      const start = Math.max(0, end - st.trail);
      for (let part = 0; part < 18; part++) {
        const lo = Math.round(start + (end - start) * part / 18), hi = Math.round(start + (end - start) * (part + 1) / 18);
        const alpha = .08 + .92 * Math.pow(Math.max(0, 1 - (end - (lo + hi) / 2) / st.trail), 1.2);
        stroke(p, path, lo, hi, color, 2.15, alpha);
      }
    }
    dot(p, [path[end * 2], path[end * 2 + 1]], color);
  }

  function niceStep(span, count) {
    const rough = Math.max(1e-9, span / count), power = 10 ** Math.floor(Math.log10(rough));
    const ratio = rough / power;
    return (ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10) * power;
  }
  function drawReports(p, tick) {
    const g = p.ctx, meta = current(), left = 42, right = p.w - 18, top = 9, bottom = p.h - 35;
    const lo = world(p, left, bottom), hi = world(p, right, top);
    const xStep = Math.max(1, niceStep(hi[0] - lo[0], Math.max(2, (right - left) / 100)));
    const yStep = Math.max(1, niceStep(hi[1] - lo[1], 5));
    g.font = "11px Segoe UI, system-ui, sans-serif"; g.fillStyle = "#657384";
    g.lineWidth = 1; g.strokeStyle = "#edf0f3";
    for (let x = Math.ceil(lo[0] / xStep) * xStep; x <= hi[0]; x += xStep) {
      const sx = point(p, x, 0)[0];
      g.beginPath(); g.moveTo(sx, top); g.lineTo(sx, bottom); g.stroke();
      g.textAlign = "center"; g.fillText(String(Math.round(x)), sx, bottom + 16);
    }
    for (let y = Math.ceil(lo[1] / yStep) * yStep; y <= hi[1]; y += yStep) {
      const sy = point(p, 0, y)[1];
      g.strokeStyle = y === 0 ? "#dce1e7" : "#edf0f3";
      g.beginPath(); g.moveTo(left, sy); g.lineTo(right, sy); g.stroke();
      g.textAlign = "right"; g.fillText(String(Math.round(y)), left - 7, sy + 3);
    }
    g.textAlign = "center"; g.fillText("ms", (left + right) / 2, p.h - 3);
    g.save(); g.translate(10, (top + bottom) / 2); g.rotate(-Math.PI / 2); g.fillText("counts", 0, 0); g.restore();
    g.save(); g.beginPath(); g.rect(left, top, Math.max(1, right - left), Math.max(1, bottom - top)); g.clip();
    const start = clamp(Math.floor(lo[0]) - 1, 0, meta.duration - 1), end = clamp(Math.ceil(hi[0]) + 1, 0, meta.duration);
    g.strokeStyle = "#9aa5b1"; g.lineWidth = 1.7; g.globalAlpha = .75; g.beginPath();
    for (let i = start; i < end; i++) {
      const xy = point(p, i + .5, data.smooth[i * 2 + st.axis]);
      if (i === start) g.moveTo(...xy); else g.lineTo(...xy);
    }
    g.stroke(); g.globalAlpha = 1;
    const raw = p.reference ? data.humanReports : data.reports.subarray(p.draw * meta.duration * 2, (p.draw + 1) * meta.duration * 2);
    g.strokeStyle = p.reference ? "#2563eb" : colors[p.draw]; g.lineWidth = 1.35; g.beginPath();
    for (let i = start; i < end; i++) {
      const a = point(p, i, raw[i * 2 + st.axis]), b = point(p, i + 1, raw[i * 2 + st.axis]);
      if (i === start) g.moveTo(...a); else g.lineTo(...a);
      g.lineTo(...b);
    }
    g.stroke();
    const cursor = point(p, tick, 0)[0];
    g.strokeStyle = "#1f293338"; g.lineWidth = 1; g.beginPath(); g.moveTo(cursor, top); g.lineTo(cursor, bottom); g.stroke(); g.restore();
  }
  function draw() {
    computeView();
    const tick = clamp(Math.floor(st.t), 0, current().duration);
    for (const p of panes) {
      if (!p.visible) continue;
      const g = p.ctx; g.setTransform(p.dpr, 0, 0, p.dpr, 0, 0); g.clearRect(0, 0, p.w, p.h);
      g.lineJoin = "round"; g.lineCap = "round";
      if (isReports()) drawReports(p, tick); else drawPath(p, tick);
    }
    const label = st.section === "continuous" ? `${(tick / 1000).toFixed(1)} / ${(current().duration / 1000).toFixed(1)} s` : `${tick} / ${current().duration} ms`;
    $("clock").textContent = label;
  }

  function installNavigation(p) {
    const active = new Map();
    let touchGesture = null;
    p.canvas.addEventListener("wheel", event => {
      event.preventDefault(); const r = p.canvas.getBoundingClientRect();
      zoom(Math.exp(-event.deltaY * .0015), p, [event.clientX - r.left, event.clientY - r.top]);
    }, {passive: false});
    p.canvas.addEventListener("pointerdown", event => {
      if (!data) return;
      event.preventDefault(); computeView();
      p.canvas.focus({preventScroll: true});
      if (!active.size) touchGesture = event.pointerType === "touch"
        ? {start: [event.clientX, event.clientY], dragged: false, pinched: false} : null;
      if (event.pointerType !== "touch") { freezeFollow(); p.canvas.classList.add("drag"); }
      p.canvas.setPointerCapture(event.pointerId); active.set(event.pointerId, [event.clientX, event.clientY]);
      if (active.size > 1 && touchGesture) touchGesture.pinched = true;
      markDirty();
    });
    p.canvas.addEventListener("pointermove", event => {
      if (!active.has(event.pointerId) || !data) return;
      const before = [...active.values()]; active.set(event.pointerId, [event.clientX, event.clientY]); const after = [...active.values()];
      if (before.length === 1) {
        if (touchGesture?.pinched) return;
        if (touchGesture && phone.matches && st.section === "continuous" && st.follow) return;
        if (touchGesture && !touchGesture.dragged) {
          if (Math.hypot(event.clientX - touchGesture.start[0], event.clientY - touchGesture.start[1]) < 5) return;
          freezeFollow(); touchGesture.dragged = true; p.canvas.classList.add("drag");
        }
        st.pan[0] -= (after[0][0] - before[0][0]) / viewport.sx;
        st.pan[1] += (st.section === "continuous" || isReports() ? 1 : -1) * (after[0][1] - before[0][1]) / viewport.sy;
      } else {
        const middle = points => [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2];
        const distance = points => Math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]);
        const a = middle(before), b = middle(after), rect = p.canvas.getBoundingClientRect();
        zoom(distance(after) / Math.max(1, distance(before)), p, [a[0] - rect.left, a[1] - rect.top]); computeView();
        st.pan[0] -= (b[0] - a[0]) / viewport.sx;
        st.pan[1] += (st.section === "continuous" || isReports() ? 1 : -1) * (b[1] - a[1]) / viewport.sy;
      }
      computeView(); markDirty();
    });
    const release = event => {
      if (!active.has(event.pointerId)) return;
      active.delete(event.pointerId);
      if (!active.size) {
        if (event.type === "pointerup" && touchGesture && !touchGesture.pinched && !touchGesture.dragged
          && !(phone.matches && st.section === "continuous" && st.follow)) freezeFollow();
        touchGesture = null; p.canvas.classList.remove("drag"); markDirty();
      }
    };
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) p.canvas.addEventListener(name, release);
    p.canvas.addEventListener("dblclick", fit);
  }
  function buildPanes() {
    const host = $("panes"); host.replaceChildren(); panes = [];
    host.className = "panes" + (st.section === "renderer" ? (isReports() ? " reports" : " paired") : "");
    function add(reference, index) {
      const el = document.createElement("article"); el.className = "pane " + (reference ? "reference" : "model");
      const heading = document.createElement("h2"), chip = document.createElement("i"), title = document.createElement("span");
      chip.style.background = reference ? (current().human ? "#2563eb" : "#059669") : colors[index];
      title.className = "pane-title";
      const duration = document.createElement("span"); duration.className = "ms";
      heading.append(chip, title, duration);
      const canvas = document.createElement("canvas"); canvas.tabIndex = 0;
      canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", `${title.textContent}. Drag to pan; scroll or pinch to zoom.`);
      el.append(heading, canvas); host.append(el);
      const pane = {el, canvas, title, duration, ctx: canvas.getContext("2d"), reference, draw: index, w: 1, h: 1, visible: true};
      panes.push(pane); installNavigation(pane);
    }
    add(true, -1);
    if (st.section === "renderer") add(false, st.draw);
    else for (let i = 0; i < 4; i++) add(false, i);
    responsive();
  }
  function responsive() {
    if (!data) { sizesDirty = true; markDirty(); return; }
    arrangeControls();
    for (const p of panes) {
      p.el.classList.toggle("mobile-hidden", st.section !== "renderer" && !p.reference && p.draw !== st.draw);
      p.title.textContent = st.section === "renderer"
        ? p.reference ? (isReports() ? "real hardware" : "Human path") : "ABCurves renderer"
        : p.reference ? (current().human ? (phone.matches ? "Human" : "A real human") : "Moving target")
        : phone.matches ? `ABCurves ${p.draw + 1}` : `ABCurves · ${st.section === "static" ? "finish" : "draw"} ${p.draw + 1}`;
      const path = p.reference ? data.human || data.target : data.paths[p.draw];
      p.duration.textContent = isReports() ? (st.axis === 0 ? "dx · counts/ms" : "dy · counts/ms")
        : st.section === "continuous" ? "" : `${path.length / 2 - 1} ms`;
    }
    updateCanvasLabels();
    updateDescription(); updateLegend();
    sizesDirty = true; markDirty();
  }
  function updateDescription() {
    const side = phone.matches ? "The human recording and one selected draw share the same input." : "The human recording is on the left; four independent draws are on the right.";
    $("description").innerHTML = st.section === "static"
      ? `<b>Static Planner.</b> One real A→B start, one fixed target, four possible finishes. ${side}`
      : st.section === "continuous"
        ? `<b>Continuous Planner.</b> Follows a moving target, adjusting the motion along the way. ${current().human ? side : "Four independently generated paths follow the same designed target."}`
        : "<b>Renderer.</b> A path becomes integer mouse reports, one per millisecond. Here the input is a real human path: compare its recorded texture with the renderer’s output.";
  }
  function updateCanvasLabels() {
    const help = phone.matches && st.section === "continuous" && st.follow
      ? "Pinch to zoom. Camera settings are in View."
      : "Drag to pan; scroll or pinch to zoom.";
    for (const p of panes) p.canvas.setAttribute("aria-label", `${p.title.textContent}. ${help}`);
  }
  function updateLegend() {
    const line = (color, label) => `<span><i class="sw" style="background:${color}"></i>${label}</span>`;
    const dotKey = (color, label) => `<span><i class="dotk" style="background:${color}"></i>${label}</span>`;
    const generated = phone.matches || st.section === "renderer" ? colors[st.draw] : `linear-gradient(90deg,${colors.join(",")})`;
    $("legend").innerHTML = st.section === "static"
      ? line("#9aa5b1", "A→B · human input") + dotKey("#1f2933", "A · start") + (st.marker ? dotKey("#7c3aed", "B · cut") : "") + dotKey("#059669", "target C") + line("#2563eb", "real finish") + line(generated, phone.matches ? "sampled finish" : "four sampled finishes")
      : st.section === "continuous"
        ? dotKey("#059669", "moving target") + (current().human ? line("#2563eb", "human recording") : "") + line(generated, phone.matches ? "sampled path" : "four sampled paths")
        : line("#9aa5b1", "same smooth human input") + line("#2563eb", isReports() ? "real reports" : "recorded path") + line(generated, isReports() ? "rendered reports" : "rendered path");
  }
  let controlsOnPhone = null;
  function arrangeControls() {
    if (controlsOnPhone === phone.matches) return;
    controlsOnPhone = phone.matches;
    if (phone.matches) {
      $("playbackSettings").append($("speedControl"));
      $("motionSettings").append($("trailControl"), $("followControl"), $("guideControl"));
      $("comparisonSettings").append($("rendererControls"), $("drawPicker"));
      $("viewActions").append($("replay"), $("marker"), $("fit"), $("browse"));
    } else {
      if ($("settings").open) $("settings").close();
      const controls = document.querySelector(".controls");
      controls.insertBefore($("browse"), controls.querySelector(".spacer"));
      controls.insertBefore($("speedControl"), $("play"));
      for (const id of ["replay", "marker", "fit"]) controls.insertBefore($(id), $("viewButton"));
      $("motionOptions").append($("trailControl"), $("followControl"), $("guideControl"));
      $("comparisonOptions").append($("rendererControls"), $("drawPicker"));
    }
  }
  function updateControls() {
    arrangeControls();
    document.body.dataset.section = st.section;
    $("showcase").setAttribute("aria-labelledby", "tab-" + st.section);
    for (const button of document.querySelectorAll("#tabs button")) {
      const on = button.dataset.section === st.section;
      button.classList.toggle("on", on); button.setAttribute("aria-selected", String(on)); button.tabIndex = on ? 0 : -1;
    }
    updateDescription(); updateLegend();
    $("rendererControls").hidden = st.section !== "renderer";
    $("axes").hidden = !isReports();
    $("filters").hidden = st.section !== "continuous";
    $("speed").value = String(st.speed); $("spdl").textContent = st.speed.toFixed(2) + "×"; $("trail").value = String(st.trail);
    $("follow").checked = st.follow; $("guide").checked = st.guide;
    $("marker").hidden = st.section !== "static";
    $("marker").classList.toggle("on", st.marker); $("marker").setAttribute("aria-pressed", String(st.marker));
    $("motionOptions").hidden = st.section !== "continuous" || phone.matches;
    $("motionSettings").hidden = st.section !== "continuous";
    for (const button of document.querySelectorAll("#drawPicker button")) {
      const on = Number(button.dataset.draw) === st.draw; button.classList.toggle("on", on); button.setAttribute("aria-pressed", String(on));
      button.textContent = (st.section === "static" ? "finish " : "draw ") + (Number(button.dataset.draw) + 1);
    }
    $("footExplanation").textContent = st.section === "static"
      ? "Every card shares the same real A→B input and target; only the finish differs."
      : st.section === "continuous"
        ? current().human ? "The human recording is untouched. Each Continuous Planner draw includes the 1 kHz Renderer." : "The target is designed. Each Continuous Planner draw includes the 1 kHz Renderer."
        : "The renderer follows the same smoothed human recording in both views. No planner is used in this comparison.";
    $("sourceNote").textContent = st.section === "static"
      ? "24 recorded examples at 1 kHz. Missing history before A is treated as 96 stationary reports."
      : st.section === "continuous" ? "11 recorded pursuits, followed by 7 designed targets."
      : "Human and generated reports use the same time and count scales; four draws show the variation.";
    $("mobileHelp").textContent = st.section === "continuous" && st.follow
      ? "Pinch to zoom; the camera follows the motion." : "Drag to pan, pinch to zoom. The views stay in sync.";
    if (data) updateCanvasLabels();
  }
  function fillPicker() {
    const picker = $("pick"); picker.replaceChildren();
    let group = null, lastKind = null;
    for (const meta of sections[st.section]) {
      const kind = meta.human ? "Human" : "Targets";
      if (st.section === "continuous" && kind !== lastKind) { group = document.createElement("optgroup"); group.label = kind; picker.append(group); lastKind = kind; }
      const option = document.createElement("option"); option.value = meta.id; option.textContent = meta.title;
      (group || picker).append(option);
    }
    picker.value = current().id;
  }
  async function select(section, index, updateHash = true) {
    const token = ++requestId, list = sections[section];
    st.section = section; st.index = (index + list.length) % list.length; remembered[section] = st.index;
    const meta = current(); data = null; last = null; ending = 0; st.t = 0;
    st.speed = speeds[section] ?? meta.defaultSpeed; st.trail = section === "continuous" ? continuousTrail : meta.trail;
    st.follow = defaultFollow(); st.zoom = 1.10; st.pan = [0, 0];
    if (isReports()) st.zoom = 1;
    $("loader").classList.remove("hidden"); $("loader").removeAttribute("aria-hidden"); $("loadText").textContent = "Drawing A → B → C …"; $("retry").hidden = true;
    fillPicker(); updateControls();
    try {
      const loaded = await load(meta); if (token !== requestId) return;
      data = loaded; buildCameras(); buildPanes();
      if (updateHash) history.replaceState(null, "", "#" + meta.id);
      $("loader").classList.add("hidden"); $("loader").setAttribute("aria-hidden", "true");
      $("status").textContent = meta.title; markDirty();
    } catch (error) {
      if (token !== requestId) return;
      $("loadText").textContent = error.message || "This example could not be loaded."; $("retry").hidden = false; play(false);
    }
  }

  function thumbnail(meta) {
    const points = meta.thumbnail, xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const lo = [Math.min(...xs), Math.min(...ys)], hi = [Math.max(...xs), Math.max(...ys)];
    const scale = Math.min(236 / Math.max(hi[0] - lo[0], 1), 65 / Math.max(hi[1] - lo[1], 1));
    const direction = meta.section === "continuous" ? -1 : 1;
    const path = points.map((p, i) => `${i ? "L" : "M"}${(128 + (p[0] - (lo[0] + hi[0]) / 2) * scale).toFixed(2)},${(38 + direction * (p[1] - (lo[1] + hi[1]) / 2) * scale).toFixed(2)}`).join(" ");
    return `<svg viewBox="0 0 256 76" aria-hidden="true"><path d="${path}" fill="none" stroke="${meta.human ? "#7da5ef" : "#67b99c"}" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  }
  function gallery() {
    const host = $("cards"); host.replaceChildren();
    for (const meta of sections[st.section]) {
      if (st.section === "continuous" && st.filter !== "all" && (st.filter === "human") !== meta.human) continue;
      const button = document.createElement("button"); button.className = "card" + (meta.id === current().id ? " current" : "");
      button.innerHTML = thumbnail(meta);
      const title = document.createElement("strong"); title.textContent = meta.title; button.append(title);
      if (meta.id === current().id) button.setAttribute("aria-current", "true");
      button.onclick = () => { $("gallery").close(); select(st.section, sections[st.section].indexOf(meta)); };
      host.append(button);
    }
  }
  function openDialog(id) { $(id).showModal(); last = null; }
  for (const dialog of document.querySelectorAll("dialog")) {
    dialog.querySelector(".close").onclick = () => dialog.close();
    dialog.addEventListener("close", () => { last = null; markDirty(); });
    dialog.addEventListener("click", event => {
      if (event.target !== dialog) return;
      const r = dialog.getBoundingClientRect();
      if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
    });
  }
  $("prev").onclick = () => select(st.section, st.index - 1);
  $("next").onclick = () => select(st.section, st.index + 1);
  $("pick").onchange = () => select(st.section, sections[st.section].findIndex(c => c.id === $("pick").value));
  $("play").onclick = () => play(!st.playing);
  $("replay").onclick = () => { $("settings").close(); st.t = 0; ending = 0; play(true); };
  $("retry").onclick = () => { play(!reduced.matches); select(st.section, st.index); };
  $("fit").onclick = () => { fit(); $("settings").close(); };
  $("viewButton").onclick = () => openDialog("settings");
  $("browse").onclick = () => { $("settings").close(); gallery(); openDialog("gallery"); };
  $("speed").oninput = () => { st.speed = Number($("speed").value); speeds[st.section] = st.speed; $("spdl").textContent = st.speed.toFixed(2) + "×"; last = null; };
  $("trail").onchange = () => { st.trail = Number($("trail").value); continuousTrail = st.trail; markDirty(); };
  $("follow").onchange = () => { st.follow = $("follow").checked; st.zoom = 1.10; st.pan = [0, 0]; updateControls(); markDirty(); };
  $("guide").onchange = () => { st.guide = $("guide").checked; markDirty(); };
  $("marker").onclick = () => { st.marker = !st.marker; updateControls(); markDirty(); };
  for (const button of document.querySelectorAll("#tabs button")) {
    button.onclick = () => select(button.dataset.section, remembered[button.dataset.section]);
    button.addEventListener("keydown", event => {
      const buttons = [...document.querySelectorAll("#tabs button")], index = buttons.indexOf(button);
      const next = event.key === "ArrowRight" ? (index + 1) % 3 : event.key === "ArrowLeft" ? (index + 2) % 3 : event.key === "Home" ? 0 : event.key === "End" ? 2 : -1;
      if (next < 0) return;
      event.preventDefault(); buttons[next].focus(); buttons[next].click();
    });
  }
  for (const button of document.querySelectorAll("#drawPicker button")) button.onclick = () => {
    st.draw = Number(button.dataset.draw); updateControls();
    if (st.section === "renderer") buildPanes(); else responsive();
  };
  for (const button of document.querySelectorAll("[data-mode]")) button.onclick = () => {
    st.mode = button.dataset.mode; st.pan = [0, 0]; st.zoom = isReports() ? 1 : 1.10;
    for (const b of document.querySelectorAll("[data-mode]")) { b.classList.toggle("on", b === button); b.setAttribute("aria-pressed", String(b === button)); }
    updateControls(); buildPanes();
  };
  for (const button of document.querySelectorAll("[data-axis]")) button.onclick = () => {
    st.axis = Number(button.dataset.axis); st.pan[1] = 0;
    for (const b of document.querySelectorAll("[data-axis]")) { b.classList.toggle("on", b === button); b.setAttribute("aria-pressed", String(b === button)); }
    responsive();
  };
  for (const button of document.querySelectorAll("[data-filter]")) button.onclick = () => {
    st.filter = button.dataset.filter;
    for (const b of document.querySelectorAll("[data-filter]")) { b.classList.toggle("on", b === button); b.setAttribute("aria-pressed", String(b === button)); }
    gallery();
  };
  document.addEventListener("keydown", event => {
    if (dialogOpen() || /^(INPUT|SELECT|TEXTAREA|BUTTON|A)$/.test(event.target.tagName)) return;
    if (event.code === "Space") { event.preventDefault(); play(!st.playing); }
    else if (event.key === "ArrowRight") select(st.section, st.index + 1);
    else if (event.key === "ArrowLeft") select(st.section, st.index - 1);
    else if (event.key === "+" || event.key === "=") zoom(1.35);
    else if (event.key === "-") zoom(1 / 1.35);
    else if (event.key === "0") fit();
  });
  new ResizeObserver(() => { sizesDirty = true; markDirty(); }).observe($("panes"));
  phone.addEventListener("change", () => {
    if (phone.matches && st.section === "continuous") { st.follow = true; st.zoom = 1.10; st.pan = [0, 0]; }
    updateControls(); responsive();
  });
  document.addEventListener("visibilitychange", () => {
    last = null;
    if (document.hidden) { cancelAnimationFrame(raf); raf = 0; } else markDirty();
  });
  function route() {
    let id = location.hash.slice(1);
    if (/^[HS]\d{2}$/.test(id)) id = "continuous/" + id;
    const found = catalog.find(c => c.id === id);
    if (found) select(found.section, sections[found.section].indexOf(found), false);
    else select("continuous", 0);
  }
  window.addEventListener("hashchange", route);
  if (reduced.matches) document.querySelectorAll(".loadtrace animate").forEach(node => node.remove());
  play(st.playing); route();
})();
