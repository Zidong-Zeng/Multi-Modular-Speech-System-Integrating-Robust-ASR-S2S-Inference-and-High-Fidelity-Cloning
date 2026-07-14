(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { samples: [], selected: null, jobId: null, timer: null, config: null };
  const stages = ["c1", "c2", "c3", "c5", "c4"];

  async function api(path, options) {
    const response = await fetch(path, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `请求失败 (${response.status})`);
    return body;
  }

  function formatDuration(seconds) {
    const value = Number(seconds || 0);
    if (value >= 60) return `${Math.floor(value / 60)} 分 ${Math.round(value % 60)} 秒`;
    return `${value.toFixed(1)} 秒`;
  }

  function selectedOptions() {
    return {
      asr_model: $("asr-model-select").value,
      vad_enabled: $("vad-enabled").checked,
      pyannote_enabled: $("pyannote-enabled").checked,
      correction_enabled: $("correction-enabled").checked,
      correction_backend: $("correction-backend-select").value,
      c4_enabled: $("c4-enabled").checked,
    };
  }

  function validate() {
    const warning = $("validation-message");
    const blocked = state.selected && Number(state.selected.duration_sec) > 30 && !$("vad-enabled").checked;
    const apiBlocked = $("correction-enabled").checked && $("correction-backend-select").value === "openai_compatible" && state.config && !state.config.correction_api_configured;
    $("correction-backend-select").disabled = !$("correction-enabled").checked;
    warning.hidden = !blocked && !apiBlocked;
    warning.textContent = blocked ? "当前样例超过 30 秒，必须开启 VAD 才能开始。" : apiBlocked ? "外部 API 模型未在 config.json 中配置完整。" : "";
    $("start-button").disabled = !state.selected || blocked || apiBlocked || Boolean(state.jobId);
  }

  function showSample(sample) {
    state.selected = sample || null;
    $("sample-duration").textContent = sample ? formatDuration(sample.duration_sec) : "--";
    $("sample-rate").textContent = sample ? `${sample.sample_rate || "--"} Hz` : "--";
    $("sample-mode").textContent = sample ? (sample.long_audio ? "长音频" : "短音频") : "--";
    $("sample-reference").textContent = sample && sample.reference ? "有" : "无";
    validate();
  }

  function renderStages(status, events) {
    const current = status.stage || "";
    let completed = 0;
    stages.forEach((name) => {
      const element = $(`stage-${name}`);
      element.classList.remove("stage-running", "stage-completed", "stage-failed", "stage-skipped");
      if (events.some((event) => event.status === "running" && event.stage === name)) element.classList.add("stage-running");
      if (events.some((event) => event.status === "completed" && event.stage === name)) { element.classList.add("stage-completed"); completed += 1; }
      if (status.status === "failed" && current === name) element.classList.add("stage-failed");
      if (name === "c4" && !$("c4-enabled").checked) element.classList.add("stage-skipped");
    });
    $("progress-bar").style.width = `${Math.round((completed / ( $("c4-enabled").checked ? 5 : 4)) * 100)}%`;
    $("pipeline-status").textContent = status.status === "completed" ? "已完成" : status.status === "failed" ? "运行失败" : status.status === "cancelled" ? "已停止" : current ? `正在处理 ${current.toUpperCase()}` : "队列中";
  }

  function renderEvents(events) {
    $("log-count").textContent = `${events.length} 条`;
    const log = $("event-log");
    log.replaceChildren();
    if (!events.length) { log.textContent = "任务日志将在这里出现"; return; }
    events.slice(-80).forEach((event) => {
      const row = document.createElement("div");
      row.className = "event-row";
      const time = document.createElement("time");
      time.textContent = new Date(Number(event.timestamp || 0) * 1000).toLocaleTimeString();
      const text = document.createElement("span");
      text.textContent = event.message || [event.stage, event.status, event.error].filter(Boolean).join(" · ");
      row.append(time, text);
      log.append(row);
    });
    log.scrollTop = log.scrollHeight;
  }

  function renderResult(result) {
    if (!result) return;
    const empty = "等待识别结果";
    [["asr-result", result.asr || result.hypothesis || empty], ["correction-result", result.corrected || result.hypothesis || empty], ["translation-result", result.translation || result.translation_zh || empty]].forEach(([id, value]) => {
      const node = $(id); node.textContent = value || empty; node.classList.toggle("empty-state", !value);
    });
    if (result.c5_audio_url) { $("c5-audio").src = result.c5_audio_url; $("audio-file-label").textContent = result.c5_audio_name || "已生成音频"; }
    if (Array.isArray(result.chunks)) {
      $("chunk-summary").textContent = `${result.chunks.length} 个分块`;
      const table = $("chunk-table"); table.replaceChildren();
      result.chunks.forEach((chunk, index) => {
        const row = document.createElement("tr");
        [index + 1, `${formatDuration((chunk.start_ms || 0) / 1000)} - ${formatDuration((chunk.end_ms || 0) / 1000)}`, chunk.speaker || "未识别", chunk.text || "", Array.isArray(chunk.nbest) ? chunk.nbest.length : 0, chunk.status || "已完成"].forEach((value) => { const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell); });
        table.append(row);
      });
    }
    if (result.c4_translation || result.c4_error) {
      $("c4-result").hidden = false;
      $("c4-text").textContent = result.c4_translation || `C4 对照未完成：${result.c4_error}`;
    }
  }

  async function poll() {
    if (!state.jobId) return;
    try {
      const [status, eventPayload] = await Promise.all([api(`/api/jobs/${state.jobId}`), api(`/api/jobs/${state.jobId}/events`)]);
      renderStages(status, eventPayload.events || []); renderEvents(eventPayload.events || []);
      $("job-badge").textContent = state.jobId;
      if (["completed", "failed", "cancelled", "interrupted"].includes(status.status)) {
        clearInterval(state.timer); state.timer = null; $("stop-button").disabled = true; localStorage.removeItem("activeSpeechJob"); state.jobId = null; validate();
        if (status.result) renderResult(status.result);
      }
    } catch (error) { $("server-status").textContent = error.message; }
  }

  async function start() {
    if (!state.selected) return;
    try {
      const payload = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample_id: state.selected.sample_id, ...selectedOptions() }) });
      state.jobId = payload.job_id; localStorage.setItem("activeSpeechJob", state.jobId); $("stop-button").disabled = false; validate();
      state.timer = setInterval(poll, 800); await poll();
    } catch (error) { $("validation-message").hidden = false; $("validation-message").textContent = error.message; }
  }

  async function stop() { if (state.jobId) await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); }

  async function loadSamples() {
    try {
      const payload = await api("/api/samples"); state.samples = payload.samples || [];
      const select = $("sample-select"); select.replaceChildren();
      state.samples.forEach((sample) => { const option = document.createElement("option"); option.value = sample.sample_id; option.textContent = `${sample.name} · ${formatDuration(sample.duration_sec)}`; select.append(option); });
      showSample(state.samples[0]);
    } catch (error) { $("server-status").textContent = error.message; }
  }

  async function loadConfig() {
    try {
      const payload = await api("/api/config");
      state.config = payload;
      if (payload.default_correction_backend) $("correction-backend-select").value = payload.default_correction_backend;
      validate();
    } catch (error) {
      state.config = { correction_api_configured: true };
    }
  }

  $("sample-select").addEventListener("change", (event) => showSample(state.samples.find((sample) => sample.sample_id === event.target.value)));
  ["asr-model-select", "vad-enabled", "pyannote-enabled", "correction-enabled", "correction-backend-select", "c4-enabled"].forEach((id) => $(id).addEventListener("change", validate));
  $("start-button").addEventListener("click", start); $("stop-button").addEventListener("click", stop);
  loadConfig();
  loadSamples();
  state.jobId = localStorage.getItem("activeSpeechJob");
  if (state.jobId) {
    $("stop-button").disabled = false;
    state.timer = setInterval(poll, 800);
    poll();
  }
}());
