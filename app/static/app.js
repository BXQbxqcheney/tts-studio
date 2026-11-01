"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  enginePill: $("engine-pill"),
  concurrencyPill: $("concurrency-pill"),
  version: $("version"),
  tokenToggle: $("token-toggle"),
  tokenPanel: $("token-panel"),
  tokenInput: $("token-input"),
  tokenSave: $("token-save"),
  tokenClear: $("token-clear"),
  voiceCount: $("voice-count"),
  voiceList: $("voice-list"),
  voiceForm: $("voice-form"),
  voiceName: $("voice-name"),
  voiceFile: $("voice-file"),
  voiceTranscript: $("voice-transcript"),
  voiceDesc: $("voice-desc"),
  uploadStatus: $("upload-status"),
  text: $("text"),
  charCount: $("char-count"),
  model: $("model"),
  modelHint: $("model-hint"),
  mode: $("mode"),
  modeHint: $("mode-hint"),
  quality: $("quality"),
  postprocess: $("postprocess"),
  speed: $("speed"),
  pitch: $("pitch"),
  volume: $("volume"),
  emotion: $("emotion"),
  generate: $("generate"),
  cancel: $("cancel"),
  runStatus: $("run-status"),
  metrics: $("metrics"),
  player: $("player"),
  download: $("download"),
  longProgress: $("long-progress"),
  longBar: $("long-bar"),
  longStatus: $("long-status"),
  resumeLong: $("resume-long"),
  historyList: $("history-list"),
  toast: $("toast"),
};

const state = {
  token: sessionStorage.getItem("tts_token") || "",
  voices: [],
  selectedVoiceId: null,
  generating: false,
  engineReady: false,
  controller: null,
  requestId: null,
  timer: null,
  startedAt: 0,
  playerUrl: null,
  toastTimer: null,
  longJobId: localStorage.getItem("tts_active_long_job") || null,
  longPoll: null,
};

const QUALITY_LABELS = { quality: "最高质量", balanced: "均衡", fast: "快速" };
const MAX_SYNC = 500;
const MAX_LONG = 10000;

function makeId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function showToast(message, kind = "") {
  els.toast.textContent = message;
  els.toast.className = `toast ${kind}`.trim();
  els.toast.hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => {
    els.toast.hidden = true;
  }, 8000);
}

function showTokenPanel(show) {
  els.tokenPanel.hidden = !show;
  if (show) {
    els.tokenInput.focus();
  }
}

function setPill(text, kind) {
  els.enginePill.textContent = text;
  els.enginePill.className = `pill pill-${kind}`;
}

function handleError(error) {
  const code = error.code || "ERROR";
  const requestId = error.request_id ? ` · ${error.request_id}` : "";
  showToast(`${code} · ${error.message}${requestId}`, "error");
  if (error.status === 401) {
    showTokenPanel(true);
  }
  console.error(error);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) {
    headers.set("Authorization", `Bearer ${state.token}`);
  }
  const response = await fetch(path, { ...options, headers });
  const type = response.headers.get("content-type") || "";
  let data = null;
  if (type.includes("application/json")) {
    data = await response.json().catch(() => null);
  }
  if (!response.ok) {
    const info = (data && data.error) || {
      code: `HTTP_${response.status}`,
      message: response.statusText || "请求失败",
    };
    const error = new Error(info.message || "请求失败");
    error.code = info.code;
    error.request_id = info.request_id;
    error.status = response.status;
    if (response.status === 401) {
      showTokenPanel(true);
    }
    throw error;
  }
  return data;
}

async function authBlob(path) {
  const headers = state.token ? { Authorization: `Bearer ${state.token}` } : {};
  const response = await fetch(path, { headers });
  if (!response.ok) {
    let info = null;
    try {
      info = (await response.json()).error;
    } catch (error) {
      info = null;
    }
    const failure = new Error((info && info.message) || "音频读取失败");
    failure.code = (info && info.code) || `HTTP_${response.status}`;
    failure.status = response.status;
    if (response.status === 401) {
      showTokenPanel(true);
    }
    throw failure;
  }
  return response.blob();
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    els.version.textContent = `v${health.version}`;
    if (els.concurrencyPill) {
      const active = health.active_jobs;
      els.concurrencyPill.textContent =
        active === undefined
          ? `并发 ${health.concurrency ?? "--"}`
          : `并发 ${active}/${health.concurrency}${health.parallelism > 1 ? ` · 段内 ${health.parallelism}` : ""}`;
      els.concurrencyPill.className = `pill ${active ? "pill-ok" : "pill-idle"}`;
      els.concurrencyPill.title =
        health.engine === "gptsovits"
          ? "GPT-SoVITS 单模型串行推理；长文按段推进，完成一段更新一次"
          : "Ming 多线程并行推理";
    }
    state.engineReady = health.status === "ok";
    if (health.status === "ok") {
      setPill("就绪", "ok");
    } else if (health.status === "loading") {
      setPill("模型加载中", "loading");
    } else {
      setPill("模型异常", "error");
    }
    els.enginePill.title = health.error || "";
    updateGenerateState();
    return health;
  } catch (error) {
    state.engineReady = false;
    setPill("服务未连接", "error");
    updateGenerateState();
    return null;
  }
}

async function loadVoices() {
  try {
    const data = await api("/api/voices");
    state.voices = data.voices || [];
    if (!state.selectedVoiceId || !state.voices.some((voice) => voice.voice_id === state.selectedVoiceId)) {
      state.selectedVoiceId = state.voices[0] ? state.voices[0].voice_id : null;
    }
    renderVoices();
    updateGenerateState();
  } catch (error) {
    handleError(error);
  }
}

function selectVoice(voiceId) {
  state.selectedVoiceId = voiceId;
  renderVoices();
  updateGenerateState();
}

function renderVoices() {
  els.voiceCount.textContent = String(state.voices.length);
  els.voiceList.replaceChildren();
  if (!state.voices.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "还没有音色。先上传一段 10–30 秒参考音频。";
    els.voiceList.append(empty);
    return;
  }
  for (const voice of state.voices) {
    const card = document.createElement("div");
    card.className = `voice-card${voice.voice_id === state.selectedVoiceId ? " selected" : ""}`;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.addEventListener("click", () => selectVoice(voice.voice_id));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectVoice(voice.voice_id);
      }
    });

    const main = document.createElement("div");
    main.className = "voice-main";
    const name = document.createElement("div");
    name.className = "voice-name";
    name.textContent = voice.name;
    const meta = document.createElement("div");
    meta.className = "voice-meta";
    meta.textContent = `${voice.voice_id} · ${voice.duration_sec.toFixed(1)}s`;
    main.append(name, meta);

    const actions = document.createElement("div");
    actions.className = "voice-actions";
    const play = document.createElement("button");
    play.type = "button";
    play.className = "mini-btn";
    play.textContent = "试听";
    play.addEventListener("click", (event) => {
      event.stopPropagation();
      playVoice(voice);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "mini-btn danger";
    remove.textContent = "删除";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteVoice(voice);
    });
    actions.append(play, remove);
    card.append(main, actions);
    els.voiceList.append(card);
  }
}

async function playVoice(voice) {
  try {
    const blob = await authBlob(voice.audio_url);
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    audio.onerror = () => URL.revokeObjectURL(url);
    await audio.play();
  } catch (error) {
    handleError(error);
  }
}

async function deleteVoice(voice) {
  if (!window.confirm(`删除音色「${voice.name}」？`)) {
    return;
  }
  try {
    await api(`/api/voices/${encodeURIComponent(voice.voice_id)}`, { method: "DELETE" });
    showToast("已删除", "ok");
    await loadVoices();
  } catch (error) {
    handleError(error);
  }
}

async function uploadVoice(event) {
  event.preventDefault();
  const file = els.voiceFile.files && els.voiceFile.files[0];
  if (!file) {
    showToast("请选择参考音频", "error");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showToast("文件超过 20MB", "error");
    return;
  }
  const form = new FormData();
  form.append("name", els.voiceName.value.trim());
  form.append("transcript", els.voiceTranscript.value.trim());
  form.append("description", els.voiceDesc.value.trim());
  form.append("reference_audio", file);
  els.uploadStatus.textContent = "上传并处理中…";
  els.uploadStatus.className = "status-text";
  try {
    const voice = await api("/api/voices", { method: "POST", body: form });
    els.uploadStatus.textContent = "已注册，参考音频已归一化。";
    els.uploadStatus.className = "status-text ok";
    els.voiceForm.reset();
    await loadVoices();
    selectVoice(voice.voice_id);
    showToast("音色已注册", "ok");
  } catch (error) {
    els.uploadStatus.textContent = error.message;
    els.uploadStatus.className = "status-text error";
    handleError(error);
  }
}

function updateCharCount() {
  const length = els.text.value.length;
  els.charCount.textContent = `${length} / ${MAX_LONG}`;
}

function updateGenerateState() {
  els.generate.disabled =
    state.generating || !state.engineReady || !state.selectedVoiceId || !els.text.value.trim();
}

function syncControls() {
  const hasBasic = Boolean(els.speed.value || els.pitch.value || els.volume.value);
  els.emotion.disabled = hasBasic;
  const hasEmotion = Boolean(els.emotion.value);
  [els.speed, els.pitch, els.volume].forEach((element) => {
    element.disabled = hasEmotion;
  });
}

function buildPayload() {
  const text = els.text.value.trim();
  if (!text) {
    throw new Error("请输入文本");
  }
  if (text.length > MAX_LONG) {
    throw new Error(`最多支持 ${MAX_LONG} 字`);
  }
  if (!state.selectedVoiceId) {
    throw new Error("请先选择音色");
  }
  const payload = {
    text,
    voice_id: state.selectedVoiceId,
    model: els.model.value,
    mode: els.mode.value,
    client_request_id: makeId(),
    quality: els.quality.value,
    postprocess: els.postprocess.value,
  };
  const basic = { speed: els.speed.value, pitch: els.pitch.value, volume: els.volume.value };
  for (const [key, value] of Object.entries(basic)) {
    if (value) {
      payload[key] = value;
    }
  }
  if (els.emotion.value) {
    if (payload.speed || payload.pitch || payload.volume) {
      throw new Error("情绪不能与语速/音调/音量同时使用");
    }
    payload.emotion = els.emotion.value;
  }
  return payload;
}

function renderMetrics(data) {
  els.metrics.replaceChildren();
  const fields = [
    ["时长", `${data.duration_sec.toFixed(2)} s`],
    ["排队", `${data.queue_sec.toFixed(2)} s`],
    ["推理 + 后处理", `${data.infer_sec.toFixed(2)} s`],
    ["RTF", data.rtf.toFixed(2)],
    ["档位", `${QUALITY_LABELS[data.flow_steps === 10 ? "quality" : data.flow_steps === 6 ? "balanced" : "fast"]}`],
    ["后处理", data.postprocess],
    ["显存峰值", `${data.vram_peak_mb.toFixed(0)} MB`],
  ];
  for (const [label, value] of fields) {
    const cell = document.createElement("div");
    const key = document.createElement("span");
    key.textContent = label;
    const strong = document.createElement("b");
    strong.textContent = value;
    cell.append(key, strong);
    els.metrics.append(cell);
  }
  els.metrics.hidden = false;
}

async function showResult(data) {
  renderMetrics(data);
  try {
    const blob = await authBlob(data.audio_url);
    if (state.playerUrl) {
      URL.revokeObjectURL(state.playerUrl);
    }
    state.playerUrl = URL.createObjectURL(blob);
    els.player.src = state.playerUrl;
    els.player.hidden = false;
    els.download.href = state.playerUrl;
    els.download.download = `tts_${data.request_id}.wav`;
    els.download.hidden = false;
    els.player.play().catch(() => {});
  } catch (error) {
    handleError(error);
  }
}

async function generate() {
  if (state.generating) {
    return;
  }
  if (els.text.value.trim().length > MAX_SYNC) {
    await generateLong();
    return;
  }
  let payload;
  try {
    payload = buildPayload();
  } catch (error) {
    showToast(error.message, "error");
    return;
  }
  state.generating = true;
  state.longJobId = null;
  localStorage.removeItem("tts_active_long_job");
  els.longProgress.hidden = true;
  els.resumeLong.hidden = true;
  state.requestId = payload.client_request_id;
  state.controller = new AbortController();
  state.startedAt = performance.now();
  els.runStatus.textContent = "生成中…";
  els.runStatus.className = "status-text";
  els.metrics.hidden = true;
  els.player.hidden = true;
  els.download.hidden = true;
  els.cancel.hidden = false;
  updateGenerateState();
  state.timer = setInterval(() => {
    const seconds = (performance.now() - state.startedAt) / 1000;
    els.runStatus.textContent = `生成中 · ${seconds.toFixed(1)} s`;
  }, 200);
  try {
    const data = await api("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: state.controller.signal,
    });
    els.runStatus.textContent = `已生成 · ${data.elapsed_sec.toFixed(2)} s`;
    els.runStatus.className = "status-text ok";
    await showResult(data);
    await loadHistory();
  } catch (error) {
    if (error.name === "AbortError") {
      els.runStatus.textContent = "已停止前端等待";
      els.runStatus.className = "status-text";
    } else {
      els.runStatus.textContent = "生成失败";
      els.runStatus.className = "status-text error";
      handleError(error);
    }
  } finally {
    clearInterval(state.timer);
    state.timer = null;
    state.generating = false;
    state.controller = null;
    state.requestId = null;
    els.cancel.hidden = true;
    updateGenerateState();
  }
}

async function generateLong() {
  let payload;
  try {
    payload = buildPayload();
  } catch (error) {
    showToast(error.message, "error");
    return;
  }
  state.generating = true;
  state.longJobId = payload.client_request_id;
  localStorage.setItem("tts_active_long_job", state.longJobId);
  state.requestId = null;
  state.controller = null;
  els.runStatus.textContent = "长任务提交中…";
  els.runStatus.className = "status-text";
  els.metrics.hidden = true;
  els.player.hidden = true;
  els.download.hidden = true;
  els.cancel.hidden = false;
  els.longProgress.hidden = false;
  els.longBar.style.width = "0%";
  els.longStatus.textContent = "正在创建后台任务…";
  els.resumeLong.hidden = true;
  updateGenerateState();
  try {
    const data = await api("/api/tts/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.longJobId = data.job_id;
    localStorage.setItem("tts_active_long_job", data.job_id);
    els.runStatus.textContent = "长任务已提交";
    els.longStatus.textContent = `共 ${data.segments_total} 段，等待 GPU…`;
    startLongPolling(data.job_id);
  } catch (error) {
    state.generating = false;
    els.cancel.hidden = true;
    els.longProgress.hidden = true;
    updateGenerateState();
    handleError(error);
  }
}

function stopLongPolling() {
  if (state.longPoll) {
    clearInterval(state.longPoll);
    state.longPoll = null;
  }
}

function startLongPolling(jobId) {
  stopLongPolling();
  state.longPoll = setInterval(() => pollLong(jobId), 1000);
  pollLong(jobId);
}

async function pollLong(jobId) {
  if (!state.generating || state.longJobId !== jobId) {
    return;
  }
  try {
    const data = await api(`/api/tts/jobs/${encodeURIComponent(jobId)}`);
    const total = data.segments_total || 0;
    const done = data.segments_done || 0;
    const percent = total ? Math.max(0, Math.min(100, (done / total) * 100)) : 0;
    els.longProgress.hidden = false;
    els.longBar.style.width = `${percent.toFixed(1)}%`;
    els.longStatus.textContent = `已完成 ${done}/${total} 段 · ${percent.toFixed(1)}%`;
    if (data.status === "queued") {
      els.runStatus.textContent = "长任务排队中…";
      els.runStatus.className = "status-text";
      return;
    }
    if (data.status === "running" || data.status === "cancelling") {
      els.runStatus.textContent =
        data.status === "cancelling" ? "取消中，当前段完成后停止" : "长任务生成中…";
      els.runStatus.className = "status-text";
      return;
    }
    if (data.status === "done") {
      stopLongPolling();
      state.generating = false;
      els.cancel.hidden = true;
      els.resumeLong.hidden = true;
      els.longBar.style.width = "100.0%";
      els.runStatus.textContent = `已生成 · ${data.duration_sec.toFixed(1)} s`;
      els.runStatus.className = "status-text ok";
      await showResult(data);
      await loadHistory();
      state.longJobId = null;
      localStorage.removeItem("tts_active_long_job");
      updateGenerateState();
      return;
    }
    stopLongPolling();
    state.generating = false;
    els.cancel.hidden = true;
    els.resumeLong.hidden = false;
    els.runStatus.textContent = data.status === "cancelled" ? "已取消" : "长任务未完成";
    els.runStatus.className = "status-text error";
    els.longStatus.textContent = data.error || `状态：${data.status}`;
    updateGenerateState();
  } catch (error) {
    if (error.status === 404) {
      stopLongPolling();
      state.generating = false;
      els.cancel.hidden = true;
      state.longJobId = null;
      localStorage.removeItem("tts_active_long_job");
      updateGenerateState();
      handleError(error);
      return;
    }
    els.longStatus.textContent = `状态查询失败：${error.message}`;
  }
}

async function resumeLongJob() {
  const jobId = state.longJobId || localStorage.getItem("tts_active_long_job");
  if (!jobId) {
    return;
  }
  try {
    const data = await api(`/api/tts/jobs/${encodeURIComponent(jobId)}/resume`, { method: "POST" });
    state.longJobId = data.job_id || jobId;
    localStorage.setItem("tts_active_long_job", state.longJobId);
    state.generating = true;
    els.cancel.hidden = false;
    els.resumeLong.hidden = true;
    els.longProgress.hidden = false;
    els.runStatus.textContent = "已恢复长任务";
    els.runStatus.className = "status-text";
    startLongPolling(state.longJobId);
    updateGenerateState();
    showToast("长任务已恢复", "ok");
  } catch (error) {
    handleError(error);
  }
}

async function restoreLongJob() {
  const saved = localStorage.getItem("tts_active_long_job");
  if (!saved) {
    return;
  }
  state.longJobId = saved;
  try {
    const data = await api(`/api/tts/jobs/${encodeURIComponent(saved)}`);
    if (data.mode && ["extreme", "high", "higher", "medium", "low", "minimal"].includes(data.mode)) {
      els.mode.value = data.mode;
      updateModeHint();
    }
    if (data.model && ["gptsovits", "gptsovits_ft", "ming"].includes(data.model)) {
      els.model.value = data.model;
      updateModelHint();
    }
    if (["queued", "running", "cancelling"].includes(data.status)) {
      state.generating = true;
      els.cancel.hidden = false;
      els.longProgress.hidden = false;
      els.runStatus.textContent = "已恢复长任务";
      els.runStatus.className = "status-text";
      startLongPolling(saved);
      updateGenerateState();
      return;
    }
    if (data.status === "done") {
      els.longProgress.hidden = false;
      els.longBar.style.width = "100.0%";
      els.longStatus.textContent = `已完成 ${data.segments_total}/${data.segments_total} 段 · 100.0%`;
      els.runStatus.textContent = `已生成 · ${data.duration_sec.toFixed(1)} s`;
      els.runStatus.className = "status-text ok";
      await showResult(data);
      await loadHistory();
      state.longJobId = null;
      localStorage.removeItem("tts_active_long_job");
      return;
    }
    els.longProgress.hidden = false;
    els.longStatus.textContent = data.error || `状态：${data.status}`;
    els.resumeLong.hidden = false;
  } catch (error) {
    if (error.status === 404) {
      state.longJobId = null;
      localStorage.removeItem("tts_active_long_job");
    }
  }
}

async function cancelGeneration() {
  if (state.longJobId && state.generating) {
    try {
      const data = await api(`/api/tts/jobs/${encodeURIComponent(state.longJobId)}/cancel`, {
        method: "POST",
      });
      const labels = {
        queued: "已取消排队",
        between_segments: "已请求取消，当前段完成后停止",
        not_cancellable: "当前任务无法取消",
        completed: "任务已完成",
      };
      showToast(labels[data.cancel_scope] || data.cancel_scope, "ok");
      els.runStatus.textContent = "取消中…";
      els.runStatus.className = "status-text";
    } catch (error) {
      handleError(error);
    }
    return;
  }
  const requestId = state.requestId;
  if (!requestId) {
    return;
  }
  try {
    const data = await api(`/api/tts/${encodeURIComponent(requestId)}/cancel`, { method: "POST" });
    const labels = {
      queued: "已取消排队",
      between_segments: "已请求取消，当前段完成后停止",
      not_cancellable: "当前请求无法取消",
      completed: "请求已完成",
    };
    showToast(labels[data.cancel_scope] || data.cancel_scope, "ok");
  } catch (error) {
    handleError(error);
  }
  if (state.controller) {
    state.controller.abort();
  }
}

function formatTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("zh-CN", { hour12: false });
}

async function downloadHistory(item) {
  try {
    const blob = await authBlob(item.audio_url);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `tts_${item.request_id}.wav`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    handleError(error);
  }
}

async function loadHistory() {
  try {
    const data = await api("/api/history?limit=5");
    const items = data.items || [];
    els.historyList.replaceChildren();
    if (!items.length) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "还没有生成记录。";
      els.historyList.append(empty);
      return;
    }
    for (const item of items) {
      const row = document.createElement("div");
      row.className = "history-row";
      const text = document.createElement("div");
      text.className = "history-text";
      text.textContent = item.text_preview;
      text.title = item.text_preview;
      const meta = document.createElement("div");
      meta.className = "history-meta";
      meta.textContent = `${formatTime(item.created_at)} · ${item.duration_sec.toFixed(1)}s · RTF ${item.rtf.toFixed(2)} · ${item.postprocess}`;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mini-btn";
      button.textContent = "下载";
      button.addEventListener("click", () => downloadHistory(item));
      row.append(text, meta, button);
      els.historyList.append(row);
    }
  } catch (error) {
    handleError(error);
  }
}

function saveToken() {
  state.token = els.tokenInput.value.trim();
  if (state.token) {
    sessionStorage.setItem("tts_token", state.token);
  } else {
    sessionStorage.removeItem("tts_token");
  }
  showTokenPanel(false);
  showToast(state.token ? "已保存访问令牌" : "已清除访问令牌", "ok");
  loadHealth();
  loadVoices();
  loadHistory();
}

function clearToken() {
  els.tokenInput.value = "";
  state.token = "";
  sessionStorage.removeItem("tts_token");
  showToast("已清除访问令牌", "ok");
  loadHealth();
  loadVoices();
  loadHistory();
}

function updateModeHint() {
  const labels = {
    extreme: "极限：单实例 + batch4，最快",
    high: "高：双实例 batch1，并行",
    higher: "较高：单实例 batch2",
    medium: "中：单实例 batch1",
    low: "低：单实例 batch1，120 字块",
    minimal: "极低：单实例 batch1，120 字块，最省显存",
  };
  els.modeHint.textContent = labels[els.mode.value] || "";
}

function updateModelHint() {
  const limits = { gptsovits: 1000, gptsovits_ft: 160, ming: 80 };
  const max = limits[els.model.value] || 1000;
  const labels = {
    gptsovits: "长段：底模 GPT + default SoVITS，约 1000 字/段",
    gptsovits_ft: "微调：default GPT + default SoVITS，约 160 字/段",
    ming: "Ming 零样本，约 80 字/段",
  };
  els.modelHint.textContent = `${labels[els.model.value] || ""}；短句可回退 Ming（需安装 Ming 运行时）。`;
}

function init() {
  els.tokenInput.value = state.token;
  els.tokenToggle.addEventListener("click", () => showTokenPanel(els.tokenPanel.hidden));
  els.tokenSave.addEventListener("click", saveToken);
  els.tokenClear.addEventListener("click", clearToken);
  els.voiceForm.addEventListener("submit", uploadVoice);
  els.text.addEventListener("input", () => {
    updateCharCount();
    updateGenerateState();
  });
  els.generate.addEventListener("click", generate);
  els.model.addEventListener("change", updateModelHint);
  els.mode.addEventListener("change", updateModeHint);
  els.cancel.addEventListener("click", cancelGeneration);
  els.resumeLong.addEventListener("click", resumeLongJob);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state.longJobId && state.generating) {
      pollLong(state.longJobId);
    }
  });
  els.emotion.addEventListener("change", () => {
    if (els.emotion.value) {
      [els.speed, els.pitch, els.volume].forEach((element) => {
        element.value = "";
      });
    }
    syncControls();
  });
  [els.speed, els.pitch, els.volume].forEach((element) => {
    element.addEventListener("change", () => {
      if (element.value) {
        els.emotion.value = "";
      }
      syncControls();
    });
  });
  syncControls();
  updateModelHint();
  updateModeHint();
  updateCharCount();
  updateGenerateState();
  loadHealth();
  loadVoices();
  loadHistory();
  restoreLongJob();
  setInterval(loadHealth, 5000);
}

init();
