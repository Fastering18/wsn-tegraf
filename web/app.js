import {
  CANVAS_H,
  CANVAS_W,
  INIT_ENERGY_J,
  buildPrimTrace,
  canvasToMeters,
  computeCompositeWeight,
  createPresetGraph,
  createRandomConnectedGraph,
  euclideanMeters,
  formatWeight,
  getEdgeBaseWeight,
} from "./prim-engine.js";

const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  presetKey: "wsn10",
  weightMode: "euclidean", // "euclidean" | "energy"
  interactionMode: "select", // "select" | "add-node" | "add-edge" | "delete"
  graph: null,
  startId: 0,
  selectedNodeId: null,
  pendingEdgeFrom: null,
  pendingEdgePair: null,
  frames: [],
  stepIdx: 0,
  playing: false,
  timerId: null,
  dragState: null,
};

const els = {
  presetSelect: document.getElementById("presetSelect"),
  rootSelect: document.getElementById("rootSelect"),
  weightHint: document.getElementById("weightHint"),
  modeHint: document.getElementById("modeHint"),
  weightButtons: [...document.querySelectorAll("[data-weight-mode]")],
  modeButtons: [...document.querySelectorAll("[data-mode]")],
  selectedNodeTag: document.getElementById("selectedNodeTag"),
  nodeInspectorBody: document.getElementById("nodeInspectorBody"),
  nodeEnergyControl: document.getElementById("nodeEnergyControl"),
  nodeEnergyRange: document.getElementById("nodeEnergyRange"),
  nodeEnergyVal: document.getElementById("nodeEnergyVal"),
  randomGraphBtn: document.getElementById("randomGraphBtn"),
  resetGraphBtn: document.getElementById("resetGraphBtn"),
  graphCanvas: document.getElementById("graphCanvas"),
  edgeDialog: document.getElementById("edgeDialog"),
  edgeForm: document.getElementById("edgeForm"),
  edgeDialogLabel: document.getElementById("edgeDialogLabel"),
  edgeWeightInput: document.getElementById("edgeWeightInput"),
  edgeCancelBtn: document.getElementById("edgeCancelBtn"),
  statVisited: document.getElementById("statVisited"),
  statWeight: document.getElementById("statWeight"),
  statStep: document.getElementById("statStep"),
  btnFirst: document.getElementById("btnFirst"),
  btnPrev: document.getElementById("btnPrev"),
  btnPlay: document.getElementById("btnPlay"),
  btnNext: document.getElementById("btnNext"),
  btnLast: document.getElementById("btnLast"),
  iconPlay: document.getElementById("iconPlay"),
  iconPause: document.getElementById("iconPause"),
  stepScrubber: document.getElementById("stepScrubber"),
  speedSelect: document.getElementById("speedSelect"),
  stepNarration: document.getElementById("stepNarration"),
  pseudocodeItems: [...document.querySelectorAll("#pseudocodeList li")],
  pqCount: document.getElementById("pqCount"),
  pqTableBody: document.getElementById("pqTableBody"),
  mstEdgeCount: document.getElementById("mstEdgeCount"),
  mstEdgeList: document.getElementById("mstEdgeList"),
};

function createSvgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

function stopPlayback() {
  state.playing = false;
  if (state.timerId !== null) {
    clearTimeout(state.timerId);
    state.timerId = null;
  }
  els.iconPlay.hidden = false;
  els.iconPause.hidden = true;
}

function scheduleNextTick() {
  if (!state.playing) return;
  const delay = Number(els.speedSelect.value) || 650;
  state.timerId = setTimeout(() => {
    if (!state.playing) return;
    if (state.stepIdx < state.frames.length - 1) {
      state.stepIdx += 1;
      renderFrame();
      scheduleNextTick();
    } else {
      stopPlayback();
    }
  }, delay);
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (state.stepIdx >= state.frames.length - 1) {
    state.stepIdx = 0;
  }
  state.playing = true;
  els.iconPlay.hidden = true;
  els.iconPause.hidden = false;
  renderFrame();
  scheduleNextTick();
}

function rebuildTrace(preserveStep = false, defaultStep = 0) {
  const prevIdx = state.stepIdx;
  state.frames = buildPrimTrace(state.graph, state.startId, state.weightMode);
  const maxStep = Math.max(0, state.frames.length - 1);
  els.stepScrubber.max = String(maxStep);
  if (preserveStep) {
    state.stepIdx = Math.min(prevIdx, maxStep);
  } else {
    state.stepIdx = Math.min(defaultStep, maxStep);
  }
  renderRootSelect();
  renderNodeInspector();
  renderFrame();
}

function loadPreset(key, initialStep = 0) {
  stopPlayback();
  closeEdgeDialog();
  state.presetKey = key;
  state.graph = createPresetGraph(key);
  state.startId = state.graph.defaultRoot;
  state.selectedNodeId = key === "wsn10" ? 2 : state.graph.defaultRoot;
  state.pendingEdgeFrom = null;
  rebuildTrace(false, initialStep);
}

function renderRootSelect() {
  const nodes = state.graph.nodes;
  els.rootSelect.innerHTML = "";
  for (const n of nodes) {
    const opt = document.createElement("option");
    opt.value = String(n.id);
    opt.textContent = `${n.label} (id ${n.id})`;
    if (n.id === state.startId) opt.selected = true;
    els.rootSelect.appendChild(opt);
  }
}

function renderNodeInspector() {
  const node = state.graph.nodes.find((n) => n.id === state.selectedNodeId);
  if (!node) {
    els.selectedNodeTag.textContent = "Belum ada";
    els.nodeInspectorBody.innerHTML =
      '<p class="field-hint">Klik salah satu node di kanvas untuk melihat koordinat atau mengatur sisa baterainya (J).</p>';
    els.nodeEnergyControl.hidden = true;
    return;
  }

  els.selectedNodeTag.textContent = node.label;
  els.nodeInspectorBody.innerHTML = `
    <div class="node-meta-grid">
      <div><span>Posisi (m)</span><strong>(${node.xm.toFixed(1)}, ${node.ym.toFixed(1)})</strong></div>
      <div><span>Peran</span><strong>${node.id === state.startId ? "Simpul awal" : "Node sensor"}</strong></div>
    </div>
  `;
  els.nodeEnergyControl.hidden = false;
  els.nodeEnergyRange.value = String(node.e.toFixed(2));
  els.nodeEnergyVal.textContent = `${node.e.toFixed(2)} J`;
}

function edgeMatches(frameEdge, graphEdge) {
  if (!frameEdge) return false;
  return (
    (frameEdge.from === graphEdge.u && frameEdge.to === graphEdge.v) ||
    (frameEdge.from === graphEdge.v && frameEdge.to === graphEdge.u)
  );
}

function getEdgeDisplayWeight(edge, nodeMap, frame) {
  if (state.weightMode === "euclidean") {
    return getEdgeBaseWeight(edge, nodeMap, state.graph.isDynamicDistance);
  }

  // If this edge is already in MST or PQ, show its evaluated weight from the frame
  const inTree = frame.takenEdges.find((e) => edgeMatches(e, edge));
  if (inTree) return inTree.weight;
  if (edgeMatches(frame.candidateEdge, edge)) return frame.candidateEdge.weight;
  if (edgeMatches(frame.rejectedEdge, edge)) return frame.rejectedEdge.weight;
  const inPq = frame.pq.find((e) => edgeMatches(e, edge));
  if (inPq) return inPq.weight;

  // Otherwise show baseline composite weight at degree 0
  let maxBase = 1;
  for (const e of state.graph.edges) {
    const w = getEdgeBaseWeight(e, nodeMap, state.graph.isDynamicDistance);
    if (w > maxBase) maxBase = w;
  }
  return computeCompositeWeight(
    edge,
    edge.u,
    edge.v,
    nodeMap,
    new Map(),
    state.graph.isDynamicDistance,
    maxBase
  );
}

function renderGraphSvg(frame) {
  const svg = els.graphCanvas;
  svg.innerHTML = "";

  // Subtle coordinate grid
  const gridLayer = createSvgEl("g");
  for (let x = 95; x < CANVAS_W - 40; x += 112) {
    gridLayer.appendChild(createSvgEl("line", { x1: x, y1: 24, x2: x, y2: CANVAS_H - 24, class: "svg-grid-line" }));
  }
  for (let y = 78; y < CANVAS_H - 30; y += 100) {
    gridLayer.appendChild(createSvgEl("line", { x1: 36, y1: y, x2: CANVAS_W - 36, y2: y, class: "svg-grid-line" }));
  }
  svg.appendChild(gridLayer);

  const nodeMap = new Map(state.graph.nodes.map((n) => [n.id, n]));
  const isDense = state.graph.nodes.length > 16;
  const radius = isDense ? 11.5 : 16;

  const edgeLayer = createSvgEl("g");
  const weightLayer = createSvgEl("g");

  for (const edge of state.graph.edges) {
    const uNode = nodeMap.get(edge.u);
    const vNode = nodeMap.get(edge.v);
    if (!uNode || !vNode) continue;

    const isTree = frame.takenEdges.some((e) => edgeMatches(e, edge));
    const isCand = edgeMatches(frame.candidateEdge, edge);
    const isRej = edgeMatches(frame.rejectedEdge, edge);
    const isFront = !isTree && frame.pq.some((e) => edgeMatches(e, edge) && !frame.takenNodes.has(e.to));

    let edgeClass = "svg-edge";
    if (isTree) edgeClass += " is-tree";
    else if (isCand) edgeClass += " is-candidate";
    else if (isRej) edgeClass += " is-rejected";
    else if (isFront) edgeClass += " is-frontier";

    const line = createSvgEl("line", {
      x1: uNode.cx,
      y1: uNode.cy,
      x2: vNode.cx,
      y2: vNode.cy,
      class: edgeClass,
    });
    edgeLayer.appendChild(line);

    const hitLine = createSvgEl("line", {
      x1: uNode.cx,
      y1: uNode.cy,
      x2: vNode.cx,
      y2: vNode.cy,
      class: "svg-edge-hit",
    });
    hitLine.addEventListener("click", (evt) => {
      evt.stopPropagation();
      handleEdgeClick(edge, evt);
    });
    edgeLayer.appendChild(hitLine);

    const showLabel = !isDense || isTree || isCand || isFront;
    if (showLabel) {
      const mx = (uNode.cx + vNode.cx) / 2;
      const my = (uNode.cy + vNode.cy) / 2;
      const wText = formatWeight(getEdgeDisplayWeight(edge, nodeMap, frame), state.weightMode);
      const pillW = Math.max(24, wText.length * 7 + 8);
      const pillH = 16;
      let bgClass = "svg-weight-bg";
      if (isTree) bgClass += " is-tree";
      else if (isCand) bgClass += " is-candidate";

      weightLayer.appendChild(
        createSvgEl("rect", {
          x: mx - pillW / 2,
          y: my - pillH / 2,
          width: pillW,
          height: pillH,
          rx: 4,
          class: bgClass,
        })
      );
      const txt = createSvgEl("text", { x: mx, y: my, class: "svg-weight-text" });
      txt.textContent = wText;
      weightLayer.appendChild(txt);
    }
  }

  svg.appendChild(edgeLayer);
  svg.appendChild(weightLayer);

  const nodeLayer = createSvgEl("g");
  for (const node of state.graph.nodes) {
    const g = createSvgEl("g", { "data-node-id": node.id });
    const inTree = frame.takenNodes.has(node.id);
    const isActive = frame.activeNode === node.id;
    const isRoot = node.id === state.startId;
    const isSelected = node.id === state.selectedNodeId || node.id === state.pendingEdgeFrom;

    if (isRoot) {
      g.appendChild(
        createSvgEl("circle", {
          cx: node.cx,
          cy: node.cy,
          r: radius + 4.5,
          class: "svg-node-ring",
        })
      );
    }

    let cls = "svg-node-circle";
    if (isActive) cls += " is-active";
    else if (inTree) cls += " is-tree";
    if (isRoot) cls += " is-root";
    if (isSelected) cls += " is-selected";

    const circle = createSvgEl("circle", {
      cx: node.cx,
      cy: node.cy,
      r: radius,
      class: cls,
    });
    g.appendChild(circle);

    const lbl = createSvgEl("text", {
      x: node.cx,
      y: node.cy,
      class: "svg-node-label",
      style: isDense ? "font-size: 9.5px" : "",
    });
    lbl.textContent = node.label;
    g.appendChild(lbl);

    if (state.weightMode === "energy") {
      const barW = isDense ? 20 : 26;
      const barH = 3.5;
      const ratio = Math.max(0.05, Math.min(1, node.e / (INIT_ENERGY_J * 1.1)));
      const bx = node.cx - barW / 2;
      const by = node.cy + radius + 5;
      let fillColor = "#12805c";
      if (node.e < 0.22) fillColor = "#c53030";
      else if (node.e < 0.36) fillColor = "#b45309";

      g.appendChild(createSvgEl("rect", { x: bx, y: by, width: barW, height: barH, rx: 1.5, class: "svg-energy-bar-bg" }));
      g.appendChild(
        createSvgEl("rect", {
          x: bx,
          y: by,
          width: (barW * ratio).toFixed(1),
          height: barH,
          rx: 1.5,
          fill: fillColor,
          class: "svg-energy-bar-fill",
        })
      );
    }

    g.addEventListener("pointerdown", (evt) => handleNodePointerDown(node, evt));
    nodeLayer.appendChild(g);
  }
  svg.appendChild(nodeLayer);
}

function renderFrame() {
  const frame = state.frames[state.stepIdx];
  if (!frame) return;

  const nodeMap = new Map(state.graph.nodes.map((n) => [n.id, n]));
  const labelOf = (id) => nodeMap.get(id)?.label ?? String(id);

  renderGraphSvg(frame);

  els.statVisited.textContent = `${frame.takenNodes.size} / ${state.graph.nodes.length}`;
  els.statWeight.textContent = formatWeight(frame.totalWeight, state.weightMode);
  els.statStep.textContent = `${state.stepIdx} / ${Math.max(0, state.frames.length - 1)}`;
  els.stepScrubber.value = String(state.stepIdx);
  els.stepNarration.textContent = frame.narration;

  els.btnFirst.disabled = state.stepIdx === 0;
  els.btnPrev.disabled = state.stepIdx === 0;
  els.btnNext.disabled = state.stepIdx >= state.frames.length - 1;
  els.btnLast.disabled = state.stepIdx >= state.frames.length - 1;

  for (const li of els.pseudocodeItems) {
    li.classList.toggle("is-current", Number(li.dataset.line) === frame.line);
  }

  // Priority Queue Table
  els.pqCount.textContent = `${frame.pq.length} item`;
  els.pqTableBody.innerHTML = "";
  if (frame.pq.length === 0) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="3" class="pq-empty">Antrean prioritas masih kosong pada langkah ini.</td>`;
    els.pqTableBody.appendChild(tr);
  } else {
    frame.pq.forEach((item, idx) => {
      const tr = document.createElement("tr");
      const stale = frame.takenNodes.has(item.to);
      if (idx === 0 && !stale) tr.classList.add("pq-top");
      if (stale) tr.classList.add("pq-stale");

      const statusLabel = stale ? "Di T (usang)" : idx === 0 ? "Terpilih" : "Menunggu";
      tr.innerHTML = `
        <td>${labelOf(item.from)} - ${labelOf(item.to)}</td>
        <td>${statusLabel}</td>
        <td>${formatWeight(item.weight, state.weightMode)}</td>
      `;
      els.pqTableBody.appendChild(tr);
    });
  }

  // Committed MST Edges
  els.mstEdgeCount.textContent = `${frame.takenEdges.length} sisi`;
  els.mstEdgeList.innerHTML = "";
  if (frame.takenEdges.length === 0) {
    const empty = document.createElement("span");
    empty.className = "mst-empty-inline";
    empty.textContent = "Belum ada sisi yang masuk ke MST.";
    els.mstEdgeList.appendChild(empty);
  } else {
    for (const e of frame.takenEdges) {
      const pill = document.createElement("span");
      pill.className = "mst-pill";
      pill.textContent = `${labelOf(e.from)}-${labelOf(e.to)} (${formatWeight(e.weight, state.weightMode)})`;
      els.mstEdgeList.appendChild(pill);
    }
  }
}

function svgPointFromEvent(evt) {
  const rect = els.graphCanvas.getBoundingClientRect();
  return {
    cx: ((evt.clientX - rect.left) / rect.width) * CANVAS_W,
    cy: ((evt.clientY - rect.top) / rect.height) * CANVAS_H,
  };
}

function handleNodePointerDown(node, evt) {
  evt.stopPropagation();
  stopPlayback();

  if (state.interactionMode === "delete") {
    if (state.graph.nodes.length <= 1) return;
    state.graph.nodes = state.graph.nodes.filter((n) => n.id !== node.id);
    state.graph.edges = state.graph.edges.filter((e) => e.u !== node.id && e.v !== node.id);
    if (state.startId === node.id) {
      state.startId = state.graph.nodes[0].id;
    }
    if (state.selectedNodeId === node.id) {
      state.selectedNodeId = null;
    }
    rebuildTrace(false);
    return;
  }

  if (state.interactionMode === "add-edge") {
    if (state.pendingEdgeFrom === null) {
      state.pendingEdgeFrom = node.id;
      state.selectedNodeId = node.id;
      renderNodeInspector();
      renderFrame();
    } else if (state.pendingEdgeFrom === node.id) {
      state.pendingEdgeFrom = null;
      renderFrame();
    } else {
      const u = state.pendingEdgeFrom;
      const v = node.id;
      state.pendingEdgeFrom = null;
      const exists = state.graph.edges.some((e) => (e.u === u && e.v === v) || (e.u === v && e.v === u));
      if (!exists) {
        const uNode = state.graph.nodes.find((n) => n.id === u);
        const vNode = state.graph.nodes.find((n) => n.id === v);
        const dist = Number(euclideanMeters(uNode, vNode).toFixed(1));
        if (state.graph.isDynamicDistance) {
          state.graph.edges.push({ id: `${u}-${v}`, u, v, manualWeight: dist });
          rebuildTrace(false);
        } else {
          openEdgeDialog({ isNew: true, u, v, defaultWeight: Math.max(1, Math.round(dist / 10)) });
        }
      } else {
        renderFrame();
      }
    }
    return;
  }

  // Select & drag
  state.selectedNodeId = node.id;
  renderNodeInspector();
  renderFrame();
  state.dragState = { nodeId: node.id, moved: false };
  els.graphCanvas.setPointerCapture?.(evt.pointerId);
}

function handleEdgeClick(edge) {
  stopPlayback();
  if (state.interactionMode === "delete") {
    state.graph.edges = state.graph.edges.filter((e) => e.id !== edge.id);
    rebuildTrace(false);
    return;
  }
  if (!state.graph.isDynamicDistance) {
    openEdgeDialog({ isNew: false, edge, defaultWeight: edge.manualWeight });
  }
}

function openEdgeDialog(info) {
  state.pendingEdgePair = info;
  const nodeMap = new Map(state.graph.nodes.map((n) => [n.id, n]));
  const uId = info.isNew ? info.u : info.edge.u;
  const vId = info.isNew ? info.v : info.edge.v;
  const uLbl = nodeMap.get(uId)?.label ?? uId;
  const vLbl = nodeMap.get(vId)?.label ?? vId;
  els.edgeDialogLabel.textContent = `Bobot sisi (${uLbl}, ${vLbl})`;
  els.edgeWeightInput.value = String(info.defaultWeight);
  els.edgeDialog.hidden = false;
  els.edgeWeightInput.focus();
  els.edgeWeightInput.select();
}

function closeEdgeDialog() {
  els.edgeDialog.hidden = true;
  state.pendingEdgePair = null;
}

function bindEvents() {
  els.presetSelect.addEventListener("change", (e) => loadPreset(e.target.value, 0));

  els.weightButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      stopPlayback();
      state.weightMode = btn.dataset.weightMode;
      els.weightButtons.forEach((b) => {
        const active = b === btn;
        b.classList.toggle("is-active", active);
        b.setAttribute("aria-pressed", String(active));
      });
      els.weightHint.textContent =
        state.weightMode === "energy"
          ? "Bobot gabungan: 0.50*jarak + 0.35*penalti_energi + 0.15*beban_derajat."
          : "Bobot w(u, v) memakai jarak lurus antar-node atau angka manual.";
      rebuildTrace(false, state.stepIdx > 0 ? state.stepIdx : 0);
    });
  });

  els.modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.interactionMode = btn.dataset.mode;
      state.pendingEdgeFrom = null;
      closeEdgeDialog();
      els.modeButtons.forEach((b) => b.classList.toggle("is-active", b === btn));
      const hints = {
        select: "Klik node untuk cek baterai, atau geser posisi node di kanvas.",
        "add-node": "Klik ruang kosong di kanvas untuk menambah node baru.",
        "add-edge": "Klik dua node berurutan untuk menghubungkan sisi.",
        delete: "Klik node atau sisi yang ingin dihapus.",
      };
      els.modeHint.textContent = hints[state.interactionMode];
      renderFrame();
    });
  });

  els.rootSelect.addEventListener("change", (e) => {
    stopPlayback();
    state.startId = Number(e.target.value);
    rebuildTrace(false);
  });

  els.nodeEnergyRange.addEventListener("input", (e) => {
    const node = state.graph.nodes.find((n) => n.id === state.selectedNodeId);
    if (!node) return;
    node.e = Number(e.target.value);
    els.nodeEnergyVal.textContent = `${node.e.toFixed(2)} J`;
    if (state.weightMode === "energy") {
      rebuildTrace(true);
    } else {
      renderFrame();
    }
  });

  els.randomGraphBtn.addEventListener("click", () => {
    stopPlayback();
    closeEdgeDialog();
    state.graph = createRandomConnectedGraph(9);
    state.startId = state.graph.defaultRoot;
    state.selectedNodeId = 0;
    state.pendingEdgeFrom = null;
    rebuildTrace(false);
  });

  els.resetGraphBtn.addEventListener("click", () => loadPreset(state.presetKey));

  // Canvas background click (add node) & pointer drag
  els.graphCanvas.addEventListener("click", (evt) => {
    if (state.interactionMode !== "add-node") return;
    stopPlayback();
    const { cx, cy } = svgPointFromEvent(evt);
    const { xm, ym } = canvasToMeters(cx, cy);
    const maxId = state.graph.nodes.reduce((m, n) => Math.max(m, n.id), -1);
    const newId = maxId + 1;
    const prefix = state.presetKey === "classic7" || state.presetKey === "ref4" ? "" : "N";
    const newNode = {
      id: newId,
      label: `${prefix}${newId}`,
      xm,
      ym,
      e: 0.5,
      cx,
      cy,
    };
    state.graph.nodes.push(newNode);
    state.selectedNodeId = newId;
    rebuildTrace(false);
  });

  els.graphCanvas.addEventListener("pointermove", (evt) => {
    if (!state.dragState) return;
    const node = state.graph.nodes.find((n) => n.id === state.dragState.nodeId);
    if (!node) return;
    const { cx, cy } = svgPointFromEvent(evt);
    const clampedX = Math.max(24, Math.min(CANVAS_W - 24, cx));
    const clampedY = Math.max(24, Math.min(CANVAS_H - 24, cy));
    node.cx = clampedX;
    node.cy = clampedY;
    const { xm, ym } = canvasToMeters(clampedX, clampedY);
    node.xm = xm;
    node.ym = ym;
    state.dragState.moved = true;
    renderNodeInspector();
    if (state.graph.isDynamicDistance) {
      rebuildTrace(true);
    } else {
      renderFrame();
    }
  });

  window.addEventListener("pointerup", () => {
    state.dragState = null;
  });

  els.edgeForm.addEventListener("submit", (evt) => {
    evt.preventDefault();
    const val = Number(els.edgeWeightInput.value);
    if (!Number.isFinite(val) || val <= 0 || !state.pendingEdgePair) return;
    if (state.pendingEdgePair.isNew) {
      const { u, v } = state.pendingEdgePair;
      state.graph.edges.push({ id: `${u}-${v}`, u, v, manualWeight: val });
    } else {
      state.pendingEdgePair.edge.manualWeight = val;
    }
    closeEdgeDialog();
    rebuildTrace(false);
  });

  els.edgeCancelBtn.addEventListener("click", closeEdgeDialog);

  // Playback controls
  els.btnPlay.addEventListener("click", togglePlayback);
  els.btnFirst.addEventListener("click", () => {
    stopPlayback();
    state.stepIdx = 0;
    renderFrame();
  });
  els.btnPrev.addEventListener("click", () => {
    stopPlayback();
    if (state.stepIdx > 0) {
      state.stepIdx -= 1;
      renderFrame();
    }
  });
  els.btnNext.addEventListener("click", () => {
    stopPlayback();
    if (state.stepIdx < state.frames.length - 1) {
      state.stepIdx += 1;
      renderFrame();
    }
  });
  els.btnLast.addEventListener("click", () => {
    stopPlayback();
    state.stepIdx = Math.max(0, state.frames.length - 1);
    renderFrame();
  });
  els.stepScrubber.addEventListener("input", (e) => {
    stopPlayback();
    state.stepIdx = Number(e.target.value);
    renderFrame();
  });

  window.addEventListener("keydown", (evt) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (evt.code === "Space") {
      evt.preventDefault();
      togglePlayback();
    } else if (evt.code === "ArrowRight") {
      evt.preventDefault();
      stopPlayback();
      if (state.stepIdx < state.frames.length - 1) {
        state.stepIdx += 1;
        renderFrame();
      }
    } else if (evt.code === "ArrowLeft") {
      evt.preventDefault();
      stopPlayback();
      if (state.stepIdx > 0) {
        state.stepIdx -= 1;
        renderFrame();
      }
    } else if (evt.code === "Home") {
      stopPlayback();
      state.stepIdx = 0;
      renderFrame();
    } else if (evt.code === "End") {
      stopPlayback();
      state.stepIdx = Math.max(0, state.frames.length - 1);
      renderFrame();
    }
  });
}

bindEvents();
loadPreset("wsn10", 5);

