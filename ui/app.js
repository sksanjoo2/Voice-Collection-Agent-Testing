const $ = id => document.getElementById(id);
let sessionId = null;
let recorder = null;
let busy = false;

const SILENCE_THRESHOLD = 0.018;
const SILENCE_SECONDS = 0.90;
const MIN_SPEECH_SECONDS = 0.28;

function setActive(on) {
  ["llm", "stt", "tts", "risk", "language", "mode", "apiKey", "ollamaEndpoint", "ollamaModel"].forEach(id => $(id).disabled = on);
  $("start").disabled = on;
  $("end").disabled = !on;
  $("message").disabled = !on;
  document.querySelector(".send").disabled = !on;
  $("mic").disabled = !on || $("stt").value === "text";
  $("callTitle").textContent = on ? "Conversation in progress" : "No active call";
  $("statusText").textContent = on ? "Connected · GPU models" : "Ready";
}

function addMessage(role, text) {
  document.querySelector(".empty")?.remove();
  const item = document.createElement("div");
  const label = document.createElement("small");
  const body = document.createElement("div");
  item.className = `message ${role}`;
  label.textContent = role;
  body.textContent = text;
  item.append(label, body);
  $("transcript").append(item);
  item.scrollIntoView({behavior: "smooth"});
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function wavBase64(chunks, rate) {
  const length = chunks.reduce((n, chunk) => n + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const put = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  put(0, "RIFF"); view.setUint32(4, 36 + length * 2, true); put(8, "WAVEfmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); put(36, "data");
  view.setUint32(40, length * 2, true);
  let offset = 44;
  for (const chunk of chunks) for (const sample of chunk) {
    view.setInt16(offset, Math.max(-1, Math.min(1, sample)) * 32767, true);
    offset += 2;
  }
  let binary = "";
  for (const byte of new Uint8Array(buffer)) binary += String.fromCharCode(byte);
  return btoa(binary);
}

async function stopRecorder() {
  if (!recorder) return null;
  const current = recorder;
  recorder = null;
  current.node.disconnect();
  current.source.disconnect();
  current.stream.getTracks().forEach(track => track.stop());
  await current.context.close();
  $("mic").textContent = "Start mic";
  $("mic").classList.remove("recording");
  return current;
}

async function startListening() {
  if (!sessionId || recorder || busy || $("stt").value === "text") return;
  if (!window.isSecureContext) throw new Error("Open the UI at http://127.0.0.1:8080 for microphone access.");
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not provide microphone access.");
  $("statusText").textContent = "Waiting for microphone permission…";
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true}, video: false,
  });
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const context = new AudioContextClass();
  await context.resume();
  const source = context.createMediaStreamSource(stream);
  const node = context.createScriptProcessor(4096, 1, 1);
  const state = {stream, context, source, node, chunks: [], preRoll: [], speech: false, speechSamples: 0, silenceSamples: 0, stopping: false};
  node.onaudioprocess = event => {
    const chunk = new Float32Array(event.inputBuffer.getChannelData(0));
    let startedThisChunk = false;
    let energy = 0;
    for (const value of chunk) energy += value * value;
    const rms = Math.sqrt(energy / chunk.length);
    if (!state.speech) {
      state.preRoll.push(chunk);
      while (state.preRoll.reduce((n, part) => n + part.length, 0) > context.sampleRate * 0.35) state.preRoll.shift();
    }
    if (rms > SILENCE_THRESHOLD) {
      if (!state.speech) { state.speech = true; state.chunks.push(...state.preRoll); state.preRoll = []; startedThisChunk = true; }
      state.speechSamples += chunk.length;
      state.silenceSamples = 0;
      $("statusText").textContent = "Voice detected · listening";
    } else if (state.speech) state.silenceSamples += chunk.length;
    if (state.speech && !startedThisChunk) state.chunks.push(chunk);
    const tooLong = state.chunks.reduce((n, part) => n + part.length, 0) > context.sampleRate * 25;
    const speechLongEnough = state.speechSamples >= context.sampleRate * MIN_SPEECH_SECONDS;
    if (!state.stopping && state.speech && speechLongEnough && (state.silenceSamples > context.sampleRate * SILENCE_SECONDS || tooLong)) {
      state.stopping = true;
      finishSpeech().catch(showMicError);
    }
  };
  source.connect(node);
  node.connect(context.destination);
  recorder = state;
  $("mic").textContent = "Stop mic";
  $("mic").classList.add("recording");
  $("statusText").textContent = "Ready · speak naturally";
}

async function finishSpeech() {
  const capture = await stopRecorder();
  if (!capture?.speech || !capture.chunks.length) return;
  busy = true;
  $("statusText").textContent = "Transcribing with Indic Conformer…";
  const result = await api("/api/stt", {
    audio: wavBase64(capture.chunks, capture.context.sampleRate),
    language: $("language").value,
  });
  const text = result.text.trim();
  if (text) await sendTurn(text);
  busy = false;
  if (sessionId && $("mode").value === "handsfree") await startListening();
}

async function speak(text) {
  if ($("tts").value === "silent") return;
  if ($("tts").value === "browser") {
    await new Promise(resolve => {
      const utterance = new SpeechSynthesisUtterance(text);
      const voices = speechSynthesis.getVoices();
      utterance.voice = voices[0] || null;
      utterance.onend = utterance.onerror = resolve;
      speechSynthesis.cancel(); speechSynthesis.speak(utterance);
    });
    return;
  }
  $("statusText").textContent = "Generating speech with Indic Parler-TTS…";
  const result = await api("/api/tts", {text});
  await new Promise((resolve, reject) => {
    const audio = new Audio(`data:audio/wav;base64,${result.audio}`);
    audio.onended = resolve; audio.onerror = reject; audio.play().catch(reject);
  });
}

async function sendTurn(text) {
  if (!sessionId) return;
  addMessage("debtor", text);
  $("message").value = "";
  $("statusText").textContent = "Bot is thinking…";
  const turn = await api("/api/call/turn", {session_id: sessionId, text});
  addMessage("agent", turn.text);
  await speak(turn.text);
  if (turn.escalate) addMessage("agent", "[Escalated to a human support specialist]");
  if (turn.ended) await endCall();
}

function showMicError(error) {
  busy = false;
  stopRecorder().catch(() => {});
  $("statusText").textContent = "Microphone unavailable";
  alert(`Microphone error: ${error.message}\n\nAllow microphone access for 127.0.0.1, then refresh.`);
}

function showServiceError(error) {
  busy = false;
  stopRecorder().catch(() => {});
  $("statusText").textContent = "Service unavailable";
  alert(`Conversation service error: ${error.message}\n\nCheck that the selected LLM, STT, and TTS models are ready, then try again.`);
}

async function endCall() {
  await stopRecorder();
  if (sessionId) await api("/api/call/end", {session_id: sessionId}).catch(() => {});
  sessionId = null; busy = false; speechSynthesis.cancel(); setActive(false);
}

$("start").onclick = async () => {
  sessionId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  try {
    await api("/api/call/start", {session_id: sessionId, llm: $("llm").value, api_key: $("apiKey").value, ollama_endpoint: $("ollamaEndpoint").value, ollama_model: $("ollamaModel").value, risk_tier: $("risk").value, stt: $("stt").value, tts: $("tts").value, language: $("language").value, mode: $("mode").value});
    $("transcript").innerHTML = "";
    setActive(true);
    if ($("mode").value === "handsfree") await startListening();
    else $("message").focus();
  } catch (error) { sessionId = null; setActive(false); showServiceError(error); }
};
$("end").onclick = endCall;
$("composer").onsubmit = async event => {
  event.preventDefault();
  const text = $("message").value.trim();
  if (!text || busy) return;
  busy = true;
  try { await stopRecorder(); await sendTurn(text); }
  catch (error) { addMessage("agent", `Error: ${error.message}`); }
  finally { busy = false; if (sessionId && $("mode").value === "handsfree") startListening().catch(showMicError); }
};
$("mic").onclick = async () => {
  try {
    if (recorder) await finishSpeech();
    else await startListening();
  } catch (error) { showMicError(error); }
};
