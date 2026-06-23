const state = {
  config: null,
  runs: [],
  selectedProject: null,
  selectedRun: null,
  runDetail: null,
  configModel: null,
  repository: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}

function showToast(message, isError = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 3200);
}

function setView(name) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `${name}View`));
  if (name === "runs" && state.runDetail) requestAnimationFrame(drawCharts);
}

function formatNumber(value, digits = 4) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(2)}%` : "—";
}

function fillSelect(select, values, selected) {
  select.replaceChildren(...values.map((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === selected;
    return option;
  }));
}

async function loadConfig(model = null) {
  const suffix = model ? `?model=${encodeURIComponent(model)}` : "";
  state.config = await api(`/api/config${suffix}`);
  state.configModel = state.config.model.name;
  const activeProject = state.selectedProject || state.config.meta.values.PROJECT || state.config.projects[0];
  state.selectedProject = activeProject;
  fillSelect($("#projectSelect"), state.config.projects, activeProject);
  fillSelect($("#modelConfigSelect"), state.config.models, state.configModel);
  $("#activeModel").textContent = state.config.meta.values.MODEL || "—";
  $("#modelConfigTitle").textContent = state.configModel;
  renderFields($("#metaFields"), state.config.meta.values, state.config.models, state.config.projects);
  renderFields($("#modelFields"), state.config.model.values);
}

function renderFields(container, values, modelOptions = [], projectOptions = []) {
  container.replaceChildren(...Object.entries(values).map(([name, value]) => {
    const wrapper = document.createElement("div");
    wrapper.className = "field";
    const label = document.createElement("label");
    label.htmlFor = `field-${name}`;
    label.innerHTML = `<span>${name}</span><small>${typeof value}</small>`;
    wrapper.append(label);

    if (typeof value === "boolean") {
      const booleanWrapper = document.createElement("div");
      booleanWrapper.className = "boolean-field";
      const status = document.createElement("span");
      status.textContent = value ? "Enabled" : "Disabled";
      const switchLabel = document.createElement("label");
      switchLabel.className = "switch";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = `field-${name}`;
      input.name = name;
      input.checked = value;
      input.dataset.type = "boolean";
      input.addEventListener("change", () => { status.textContent = input.checked ? "Enabled" : "Disabled"; });
      const indicator = document.createElement("i");
      switchLabel.append(input, indicator);
      booleanWrapper.append(status, switchLabel);
      wrapper.append(booleanWrapper);
      return wrapper;
    }

    const options = name === "MODEL" ? modelOptions : name === "PROJECT" ? projectOptions : null;
    let input;
    if (options) {
      input = document.createElement("select");
      fillSelect(input, options, value);
    } else {
      input = document.createElement("input");
      input.type = typeof value === "number" ? "number" : "text";
      if (typeof value === "number") input.step = Number.isInteger(value) ? "1" : "any";
      input.value = value;
    }
    input.id = `field-${name}`;
    input.name = name;
    input.dataset.type = typeof value;
    wrapper.append(input);
    return wrapper;
  }));
}

function formValues(container) {
  return Object.fromEntries($$("input, select").filter((input) => container.contains(input) && input.name).map((input) => {
    let value = input.type === "checkbox" ? input.checked : input.value;
    if (input.dataset.type === "number") value = input.step === "1" ? Number.parseInt(value, 10) : Number(value);
    return [input.name, value];
  }));
}

async function saveConfig(scope, form, button) {
  button.disabled = true;
  try {
    const configSection = scope === "meta" ? state.config.meta : state.config.model;
    const currentValues = configSection.values;
    const submittedValues = formValues(form);
    const changedValues = Object.fromEntries(
      Object.entries(submittedValues).filter(([name, value]) => value !== currentValues[name])
    );
    if (!Object.keys(changedValues).length) {
      showToast("変更された項目はありません。");
      return;
    }
    const body = {
      scope,
      version: configSection.version,
      values: changedValues,
    };
    if (scope === "model") body.model = state.configModel;
    const result = await api("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await loadConfig(scope === "model" ? state.configModel : null);
    const target = result.scope === "meta" ? "src/CONSTANTS.py" : `src/model/${result.model}/constants.py`;
    showToast(`${target} を保存しました。`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function loadRepository() {
  state.repository = await api("/api/repository");
  $("#repoBranch").textContent = state.repository.branch;
  $("#repoCommit").textContent = state.repository.commit;
  $("#repoSubject").textContent = state.repository.subject;
  $("#repoChanges").textContent = state.repository.clean ? "CLEAN" : `${state.repository.change_count} FILES`;
  $("#repoChanges").classList.toggle("warning-text", !state.repository.clean);
  $("#repoSync").textContent = state.repository.upstream
    ? `${state.repository.behind}↓ ${state.repository.ahead}↑`
    : "NOT SET";
}

async function pullRepository() {
  const button = $("#pullButton");
  const output = $("#gitOutput");
  button.disabled = true;
  output.hidden = false;
  output.textContent = "git pull --ff-only を実行しています…";
  try {
    const result = await api("/api/repository/pull", { method: "POST" });
    state.repository = result.repository;
    output.textContent = result.message;
    try {
      await loadRepository();
      await loadConfig(state.configModel);
      await loadRuns(false);
    } catch (refreshError) {
      showToast("pullは完了しましたが、画面を再読み込みできませんでした。", true);
      return;
    }
    if (result.restart_required) {
      showToast("更新しました。Webアプリを再起動してください。");
    } else {
      showToast(result.updated ? "GitHubの変更を取得しました。" : "すでに最新です。");
    }
  } catch (error) {
    output.textContent = error.message;
    showToast("pullできませんでした。詳細を確認してください。", true);
  } finally {
    button.disabled = false;
  }
}

async function loadRuns(selectFirst = true) {
  const payload = await api(`/api/runs?project=${encodeURIComponent(state.selectedProject)}`);
  state.runs = payload.runs;
  $("#runCount").textContent = String(state.runs.length);
  renderRunList();
  if (selectFirst && state.runs.length) await selectRun(state.runs[0].timestamp);
  if (!state.runs.length) clearRunView();
}

function renderRunList() {
  const list = $("#runList");
  if (!state.runs.length) {
    list.innerHTML = '<p class="empty-list">このプロジェクトの実験ログはまだありません。</p>';
    return;
  }
  list.replaceChildren(...state.runs.map((run) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-item";
    button.classList.toggle("active", run.timestamp === state.selectedRun);
    button.innerHTML = `<strong>${escapeHtml(run.timestamp)}</strong><small>${escapeHtml(run.model || "unknown")} · <b>${formatPercent(run.valid_accuracy)}</b></small>`;
    button.addEventListener("click", () => selectRun(run.timestamp));
    return button;
  }));
}

async function selectRun(timestamp) {
  state.selectedRun = timestamp;
  renderRunList();
  try {
    state.runDetail = await api(`/api/run?project=${encodeURIComponent(state.selectedProject)}&run=${encodeURIComponent(timestamp)}`);
    renderRunDetail();
  } catch (error) {
    state.runDetail = null;
    clearRunView(error.message);
  }
}

function clearRunView(message = "左の一覧からepochログのある実験を選択してください。") {
  $("#runTitle").textContent = "ログがありません";
  $("#runSubtitle").textContent = message;
  ["#validAccuracy", "#validLoss", "#epochCount", "#epochTime"].forEach((id) => $(id).textContent = "—");
  $("#logTableBody").replaceChildren();
  drawLineChart($("#accuracyChart"), [], []);
  drawLineChart($("#lossChart"), [], []);
}

function renderRunDetail() {
  const rows = state.runDetail.epochs;
  const latest = rows.at(-1) || {};
  const run = state.runs.find((item) => item.timestamp === state.selectedRun) || {};
  $("#runTitle").textContent = state.selectedRun;
  $("#runSubtitle").textContent = `${state.selectedProject} / ${run.model || "model unknown"} / ${rows.length} epochs recorded`;
  $("#validAccuracy").textContent = formatPercent(latest.valid_accuracy);
  $("#validLoss").textContent = formatNumber(latest.valid_loss);
  $("#epochCount").textContent = latest.epoch || rows.length;
  $("#epochTarget").textContent = run.epochs ? `TARGET ${run.epochs}` : "RECORDED";
  $("#epochTime").textContent = formatNumber(latest.ep_time, 2);
  $("#savedGraphImage").src = state.runDetail.graph_url;
  $("#savedGraph").hidden = true;
  $("#logTableWrap").hidden = false;
  renderLogTable(rows);
  requestAnimationFrame(drawCharts);
}

function renderLogTable(rows) {
  $("#logTableBody").replaceChildren(...[...rows].reverse().map((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(row.epoch)}</td><td>${formatNumber(row.ep_time, 2)}s</td><td>${formatNumber(row.train_loss)}</td><td>${formatPercent(row.train_accuracy)}</td><td>${formatNumber(row.valid_loss)}</td><td>${formatPercent(row.valid_accuracy)}</td>`;
    return tr;
  }));
}

function drawCharts() {
  if (!state.runDetail) return;
  const rows = state.runDetail.epochs;
  drawLineChart(
    $("#accuracyChart"),
    rows.map((row) => Number(row.train_accuracy)),
    rows.map((row) => Number(row.valid_accuracy)),
    { min: 0, max: 1 }
  );
  drawLineChart(
    $("#lossChart"),
    rows.map((row) => Number(row.train_loss)),
    rows.map((row) => Number(row.valid_loss))
  );
}

function drawLineChart(canvas, trainValues, validValues, fixedRange = null) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 500;
  const height = canvas.clientHeight || 280;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 48, right: 18, top: 24, bottom: 35 };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const values = [...trainValues, ...validValues].filter(Number.isFinite);
  if (!values.length) {
    ctx.fillStyle = "#85909c";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("NO DATA", pad.left, pad.top + 10);
    return;
  }
  let min = fixedRange?.min ?? Math.min(...values);
  let max = fixedRange?.max ?? Math.max(...values);
  if (max === min) max = min + 1;
  const rangePad = fixedRange ? 0 : (max - min) * 0.08;
  min -= rangePad;
  max += rangePad;

  ctx.strokeStyle = "#27303a";
  ctx.fillStyle = "#73808c";
  ctx.lineWidth = 1;
  ctx.font = "9px ui-monospace, monospace";
  ctx.textAlign = "right";
  for (let index = 0; index <= 4; index += 1) {
    const y = pad.top + chartHeight * index / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillText((max - (max - min) * index / 4).toFixed(2), pad.left - 8, y + 3);
  }
  const drawSeries = (series, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((value, index) => {
      const x = pad.left + (series.length === 1 ? 0 : chartWidth * index / (series.length - 1));
      const y = pad.top + chartHeight * (max - value) / (max - min);
      index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  drawSeries(trainValues, "#44e0d0");
  drawSeries(validValues, "#f1b45a");
  ctx.fillStyle = "#73808c";
  ctx.textAlign = "center";
  ctx.fillText("1", pad.left, height - 12);
  ctx.fillText(String(Math.max(trainValues.length, validValues.length)), width - pad.right, height - 12);
}

function bindEvents() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));
  $("#projectSelect").addEventListener("change", async (event) => {
    state.selectedProject = event.target.value;
    state.selectedRun = null;
    await loadRuns();
  });
  $("#modelConfigSelect").addEventListener("change", (event) => loadConfig(event.target.value).catch(handleError));
  $("#metaForm").addEventListener("submit", (event) => {
    event.preventDefault();
    saveConfig("meta", event.currentTarget, event.submitter);
  });
  $("#modelForm").addEventListener("submit", (event) => {
    event.preventDefault();
    saveConfig("model", event.currentTarget, event.submitter);
  });
  $("#refreshButton").addEventListener("click", async () => {
    try {
      await loadConfig(state.configModel);
      await loadRuns(false);
      await loadRepository();
      if (state.selectedRun) await selectRun(state.selectedRun);
      showToast("ログと設定を再読み込みしました。");
    } catch (error) { handleError(error); }
  });
  $("#pullButton").addEventListener("click", pullRepository);
  $("#toggleGraphButton").addEventListener("click", () => {
    const graph = $("#savedGraph");
    graph.hidden = !graph.hidden;
    $("#logTableWrap").hidden = !graph.hidden;
    $("#toggleGraphButton").textContent = graph.hidden ? "Saved graph" : "Epoch table";
  });
  window.addEventListener("resize", () => requestAnimationFrame(drawCharts));
}

function handleError(error) {
  console.error(error);
  showToast(error.message || "読み込みに失敗しました。", true);
}

async function start() {
  bindEvents();
  try {
    await loadConfig();
    await Promise.all([loadRuns(), loadRepository()]);
  } catch (error) {
    handleError(error);
    clearRunView(error.message);
  }
}

start();
