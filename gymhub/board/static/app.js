"use strict";

const MONACO_BASE = "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min";
const LEVEL_ORDER = { easy: 0, medium: 1, hard: 2 };
const view = document.getElementById("view");
const state = { editor: null, pollTimer: null, refreshTimer: null, monaco: null, disposers: [] };

// ---------- helpers ------------------------------------------------------------------------------

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// A gym's sources and further reading (gym.json "sources"), folded under its description.
function sourcesList(sources) {
  const items = (Array.isArray(sources) ? sources : []).filter((item) => /^https?:\/\//.test(item?.url || ""));
  if (!items.length) return "";
  const rows = items.map((item) => `<li><a href="${esc(item.url)}" target="_blank" rel="noopener noreferrer">${esc(item.title)}</a>${item.note ? ` <span class="muted">${esc(item.note)}</span>` : ""}</li>`);
  return `<details class="gym-sources"><summary>Sources and further reading (${items.length})</summary><ul>${rows.join("")}</ul></details>`;
}

// Markdown for statements, hints, choices and lessons, with two additions:
// math ($…$ inline, $$…$$ on its own) rendered by KaTeX, and > [!NOTE] / [!TIP] / [!WARNING] callouts.
const MATH = /(```[\s\S]*?```|`[^`\n]*`)|\$\$([\s\S]+?)\$\$|\$(?![\s\d])([^$\n]+?)(?<!\s)\$/g;
const CALLOUT = /<blockquote>\s*<p>\[!(NOTE|TIP|WARNING)\]\s*(?:<br>)?/g;

function md(text) {
  const formulas = [];
  // Math is cut out before Markdown sees it (so _ and * stay math) and put back after sanitizing,
  // as KaTeX's own escaped output. Code spans and blocks keep their dollars.
  const source = String(text || "").replace(MATH, (whole, code, block, inline) => {
    if (code) return code;
    formulas.push({ tex: block ?? inline, display: block != null });
    return `@@MATH${formulas.length - 1}@@`;
  });
  let html = DOMPurify.sanitize(marked.parse(source));
  html = html.replace(CALLOUT, (_, type) => `<blockquote class="callout ${type.toLowerCase()}"><p><b class="callout-title">${{ NOTE: "Note", TIP: "Tip", WARNING: "Warning" }[type]}</b> `);
  return html.replace(/@@MATH(\d+)@@/g, (_, index) => {
    const { tex, display } = formulas[Number(index)];
    if (!window.katex) return `<code>${esc(tex)}</code>`; // KaTeX did not load: show the source
    return katex.renderToString(tex, { displayMode: display, throwOnError: false, output: "htmlAndMathml" });
  });
}

// Diagrams. ```mermaid blocks are drawn by Mermaid, loaded the first time a page has one (strict mode:
// no scripts or links from the diagram). Inline <svg> passes through md() as it is; style.css themes the
// d-* classes. A MutationObserver catches every place md() output lands, so callers need do nothing.
const MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js";
let mermaidReady = null;
let diagramCount = 0;

function loadMermaid() {
  if (window.mermaid) return Promise.resolve(window.mermaid);
  mermaidReady ||= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = MERMAID_URL;
    script.onload = () => (window.mermaid ? resolve(window.mermaid) : reject(new Error("no mermaid")));
    script.onerror = () => { mermaidReady = null; reject(new Error("Mermaid did not load")); };
    document.head.appendChild(script);
  });
  return mermaidReady;
}

function darkTheme() {
  const set = document.documentElement.dataset.theme;
  return set ? set === "dark" : window.matchMedia?.("(prefers-color-scheme: dark)").matches;
}

async function drawDiagrams() {
  const blocks = [...document.querySelectorAll("pre > code.language-mermaid:not([data-drawn])")];
  if (!blocks.length) return;
  blocks.forEach((code) => (code.dataset.drawn = "pending"));
  let mermaid;
  try {
    mermaid = await loadMermaid();
  } catch {
    blocks.forEach((code) => (code.dataset.drawn = "failed")); // offline: the source stays readable
    return;
  }
  mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: darkTheme() ? "dark" : "neutral",
                       fontFamily: "inherit", themeVariables: { fontSize: "14px" }, flowchart: { htmlLabels: false } });
  for (const code of blocks) {
    const pre = code.parentElement;
    if (!pre?.isConnected) continue;
    try {
      const { svg } = await mermaid.render(`diagram-${++diagramCount}`, code.textContent);
      const figure = document.createElement("div");
      figure.className = "diagram mermaid";
      figure.innerHTML = svg;
      pre.replaceWith(figure);
    } catch (error) {
      code.dataset.drawn = "failed";
      document.querySelectorAll(`#ddiagram-${diagramCount}, #diagram-${diagramCount}`).forEach((node) => node.remove());
      pre.insertAdjacentHTML("beforebegin", `<p class="muted small">This diagram could not be drawn: ${esc(error.message || error)}</p>`);
    }
  }
}

let diagramsQueued = false;
new MutationObserver(() => {
  if (diagramsQueued) return;
  diagramsQueued = true;
  requestAnimationFrame(() => {
    diagramsQueued = false;
    if (document.querySelector("pre > code.language-mermaid:not([data-drawn])")) drawDiagrams();
  });
}).observe(document.body, { childList: true, subtree: true });

// Editorial: the worked solution of a code exercise. The board sends it once the learner has submitted
// (or chose to open it); before that the problem says only that there is one.
function renderEditorial(host, gymId, slug, problem) {
  if (!host) return;
  if (problem.editorial) {
    host.innerHTML = `<div class="markdown editorial">${md(problem.editorial)}</div>`;
    return;
  }
  host.innerHTML = `<div class="editorial-locked">
      <p><b>The editorial opens after your first submission.</b></p>
      <p class="muted">It walks through a solution: the idea, what it costs, and the mistakes people make. Read after a real attempt,
        even a failed one, it sticks far better.</p>
      <button class="ed-open">Show it now</button></div>`;
  host.querySelector(".ed-open").addEventListener("click", async (event) => {
    event.target.disabled = true;
    try {
      const reply = await api(`/api/gyms/${encodeURIComponent(gymId)}/problems/${encodeURIComponent(slug)}/editorial`, { method: "POST", body: "{}" });
      problem.editorial = reply.editorial;
      problem.editorial_locked = false;
      renderEditorial(host, gymId, slug, problem);
    } catch (error) {
      event.target.disabled = false;
      host.insertAdjacentHTML("beforeend", `<p class="bad small">${esc(error.message)}</p>`);
    }
  });
}

// A locked editorial may have opened since the page loaded (a submission): ask again before showing it.
async function refreshEditorial(host, gymId, slug, problem) {
  if (problem.editorial_locked) {
    try {
      const fresh = await api(`/api/gyms/${encodeURIComponent(gymId)}/problems/${encodeURIComponent(slug)}`);
      if (fresh.problem.editorial) Object.assign(problem, { editorial: fresh.problem.editorial, editorial_locked: false });
    } catch {}
  }
  renderEditorial(host, gymId, slug, problem);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", "X-Gymhub": "1", ...(options.headers || {}) },
    credentials: "same-origin",
  });
  if (response.status === 401 && state.me && !path.startsWith("/api/auth/")) {
    // The session ended while the page was open: back to the sign-in page.
    state.me = null;
    boot();
    throw new Error("Your session ended. Sign in again.");
  }
  if (response.status === 401 || response.status === 403) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    if (detail && typeof detail === "object") {
      const error = new Error(detail.message || "Access denied");
      Object.assign(error, detail);
      throw error;
    }
    throw new Error(detail || "Not signed in");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail && typeof body.detail === "object" ? body.detail : { message: body.detail, errors: body.errors };
    const error = new Error(detail.message || `${response.status} ${response.statusText}`);
    error.errors = detail.errors || [];
    if (detail.saved) error.saved = true;
    throw error;
  }
  return response.json();
}

function ago(seconds) {
  if (!seconds) return "never";
  const delta = Date.now() / 1000 - seconds;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)} h ago`;
  return `${Math.floor(delta / 86400)} d ago`;
}

function actionLabel(action) {
  return action === "test" ? "submit" : action === "answer" ? "answer" : action;
}

function statusBadge(job) {
  if (job.status === "queued") return `<span class="badge wait">queued</span>`;
  if (job.status === "running") return `<span class="badge run">running</span>`;
  if (job.status === "cancelled") return `<span class="badge">cancelled</span>`;
  if (job.status === "error") return `<span class="badge bad">error</span>`;
  return job.ok ? `<span class="badge good">pass</span>` : `<span class="badge bad">fail</span>`;
}

function langState(progress, slug, lang) {
  const entry = progress?.exercises?.[slug]?.langs?.[lang];
  if (!entry) return { mark: "·", cls: "none", title: "not started" };
  if (entry.first_solved) {
    const best = entry.best_ms != null ? ` · best ${formatMs(entry.best_ms)}` : "";
    return { mark: "✓", cls: "good", title: `solved${best}`, best: entry.best_ms };
  }
  return { mark: "✗", cls: "bad", title: `${entry.attempts || 0} tries, not passing yet` };
}

function setCrumbs(parts) {
  const crumbs = document.getElementById("crumbs");
  crumbs.classList.toggle("hidden", !parts.length);
  crumbs.innerHTML = parts
    .map(([label, href]) => (href ? `<a href="${href}">${esc(label)}</a>` : `<span>${esc(label)}</span>`))
    .join(`<span class="sep">/</span>`);
}

// A machine's GPU for display. Runners report "unknown" when nvidia-smi is missing; their
// capabilities say whether there is a GPU at all.
function gpuLabel(machine) {
  const gpu = machine?.gpu;
  if (gpu && gpu !== "unknown") return gpu;
  const gpus = machine?.capabilities?.gpus;
  if (gpus?.length) return gpus[0].name;
  return machine?.capabilities ? "no GPU" : "";
}

function setRunnerPill(runners) {
  state.runners = runners || [];
  noteRunners(runners);
  const pill = document.getElementById("runner-pill");
  const online = (runners || []).filter((runner) => runner.online);
  if (online.length) {
    const runner = online[0];
    // A dispatcher has no GPU of its own: it connects the board to the pool (#162).
    const gpu = runner.capabilities?.dispatch_only || runner.capabilities?.pool?.dispatch_only ? "pool bridge" : gpuLabel(runner);
    pill.className = "pill on";
    pill.textContent = `${runner.name}${gpu ? ` · ${gpu}` : ""}${runner.busy ? " · busy" : ""}`;
    pill.title = `Runner online, last heartbeat ${ago(runner.heartbeat_at)}`;
  } else {
    pill.className = "pill off";
    pill.textContent = "runner offline";
    pill.title = "Start it on your machine: python -m gymhub.runner";
  }
}

function clearTimers() {
  for (const dispose of state.disposers.splice(0)) {
    try {
      dispose();
    } catch {}
  }
  if (state.cleanup) {
    state.cleanup();
    state.cleanup = null;
  }
  state.eventSource?.close();
  state.eventSource = null;
  clearInterval(state.refreshTimer);
  clearInterval(state.registrationTimer);
  clearTimeout(state.pollTimer);
  state.refreshTimer = null;
  state.pollTimer = null;
  if (state.editor) {
    state.editor.dispose();
    state.editor = null;
  }
}

function fail(error) {
  view.innerHTML = `<section class="panel error"><h2>Something went wrong</h2><p>${esc(error.message || error)}</p></section>`;
}

// ---------- gym -----------------------------------------------------------------------------------

// "~5 min" and the concepts an exercise practises, when its author said so (problem.json).
function exerciseMeta(problem) {
  const parts = [];
  if (problem.minutes) parts.push(`<span class="muted small">~${Number(problem.minutes)} min</span>`);
  for (const concept of problem.concepts || []) parts.push(`<span class="concept">${esc(concept)}</span>`);
  return parts.length ? `<span class="ex-meta">${parts.join("")}</span>` : "";
}

// One status per exercise: solved in any language (or its question kind), tried, or not started.
function exerciseState(progress, testsChanged, problem) {
  const keys = problem.kind && problem.kind !== "code" ? [problem.kind] : problem.langs || [];
  const entries = keys.map((key) => [key, progress?.exercises?.[problem.slug]?.langs?.[key]]).filter(([, entry]) => entry);
  const solvedIn = entries.filter(([, entry]) => entry.first_solved).map(([key]) => key);
  if (solvedIn.length) {
    const changed = solvedIn.some((key) => (testsChanged?.[problem.slug] || []).includes(key));
    return { mark: "✓", cls: "good", solved: true, title: changed ? "solved; the tests changed since, submit again to confirm" : `solved in ${solvedIn.join(", ")}`, changed };
  }
  if (entries.length) return { mark: "✗", cls: "bad", solved: false, title: "tried, not passing yet" };
  return { mark: "·", cls: "none", solved: false, title: "not started" };
}

// Exercises that suit a phone: answering, choosing, ordering, matching, filling blanks, reading, and
// code exercises short enough to type on one (minutes from problem.json). Essays and long code do not.
const PHONE_KINDS = new Set(["answer", "choice", "order", "match", "fill", "checklist", "lesson"]);
const PHONE_CODE_MINUTES = 10;

function goodOnPhone(problem) {
  if (problem.kind && problem.kind !== "code") return PHONE_KINDS.has(problem.kind);
  return Number(problem.minutes) > 0 && Number(problem.minutes) <= PHONE_CODE_MINUTES;
}

// The "Good on a phone" filter is one choice per viewer, for every gym.
function readPhoneFilter() {
  try {
    return localStorage.getItem("gymhub:phone-filter") === "1";
  } catch {
    return false;
  }
}

function writePhoneFilter(on) {
  try {
    localStorage.setItem("gymhub:phone-filter", on ? "1" : "0");
  } catch {}
}

function readGymFilters(gymId) {
  try {
    return JSON.parse(localStorage.getItem(`gymhub:filters:${gymId}`) || "{}");
  } catch {
    return {};
  }
}

// A gym is a path: a header with progress and Continue, filters, then one row per exercise.
async function renderGym(gymId) {
  const data = await api(`/api/gyms/${encodeURIComponent(gymId)}`);
  const gym = data.gym;
  setCrumbs([["Gyms", "#/gyms"], [gym.title, null]]);
  const problems = data.problems.map((problem) => ({ ...problem, status: exerciseState(data.progress, data.tests_changed, problem) }));
  // Lessons are on the path (Continue may open one) but are reading, not exercises to count.
  const path = problems.filter((problem) => !problem.warmup);
  const practice = path.filter((problem) => problem.kind !== "lesson");
  const solved = practice.filter((problem) => problem.status.solved).length;
  const next = path.find((problem) => !problem.status.solved);
  const summaries = Object.fromEntries((gym.tracks || []).map((track) => [track.name, track.summary]));
  const tracks = [...new Set(problems.map((problem) => problem.track || ""))];
  const levels = ["easy", "medium", "hard"].filter((level) => problems.some((problem) => problem.difficulty === level));
  const filters = { track: "", level: "", unsolved: false, ...readGymFilters(gym.id) };
  if (!tracks.includes(filters.track)) filters.track = "";
  // Offered on small screens, when it would leave some exercises out but not all of them.
  const phoneCount = problems.filter(goodOnPhone).length;
  const offerPhone = smallScreen() && phoneCount > 0 && phoneCount < problems.length;
  filters.phone = offerPhone && readPhoneFilter();
  const href = (problem) => `#/gym/${encodeURIComponent(gym.id)}/${encodeURIComponent(problem.slug)}`;

  // Owners and editors fork from the Manage panel; the header offers it to learners only.
  const manages = gym.role === "owner" || gym.role === "editor";
  const forkButton = gym.can_fork && state.me?.can_run && !manages ? `<button class="small-button" id="fork-gym">Fork</button>` : "";
  // Shared and unlisted gyms stay on your Gyms page until you leave; public ones are listed for everyone.
  const leaveButton = gym.role !== "owner" && gym.visibility !== "public" ? `<button class="small-button ghost" id="leave-gym">Leave</button>` : "";
  const reportButton = !manages ? `<button class="small-button ghost" id="report-gym" title="Tell this board's admins something is wrong with this gym">Report</button>` : "";
  const percent = practice.length ? Math.round((100 * solved) / practice.length) : 0;
  view.innerHTML = `
    <section class="panel gym-intro">
      <div class="gym-intro-text"><h2>${esc(gym.title)} ${visibilityBadge(gym)}</h2><p>${esc(gym.description)}</p>
        ${sourcesList(gym.sources)}
        ${gym.forked_from ? `<p class="muted small">Forked from ${esc(gym.forked_from.gym)} v${esc(gym.forked_from.version)}</p>` : ""}
        ${gym.machine_note ? `<div class="notice" id="machine-note">${esc(gym.machine_note)}</div>` : ""}</div>
      <div class="gym-intro-side">
        <div class="gym-row-progress"><div class="bar"><span style="width:${percent}%"></span></div><span class="small"><b>${solved}</b> of ${practice.length} solved</span></div>
        <div class="gym-intro-actions">${reportButton}${leaveButton}${forkButton}${next ? `<a class="button primary" href="${href(next)}" title="${esc(next.title)}">${solved ? "Continue" : "Start"} →</a>` : `<span class="badge good">all solved</span>`}</div>
      </div>
    </section>
    <div class="gym-filters" role="group" aria-label="Filter exercises">
      ${tracks.length > 1 ? `<select id="f-track" aria-label="Track"><option value="">All tracks</option>${tracks.map((track) => `<option value="${esc(track)}">${esc(track)}</option>`).join("")}</select>` : ""}
      <div class="chips">${["", ...levels].map((level) => `<button class="tag" data-level="${level}">${level || "any level"}</button>`).join("")}</div>
      <label class="check"><input type="checkbox" id="f-unsolved"> Unsolved only</label>
      ${offerPhone ? `<label class="check" title="Questions to answer, choose, order, match or fill in, and short code exercises: little typing"><input type="checkbox" id="f-phone"> Good on a phone</label>` : ""}
      <span class="spacer"></span><span id="f-count" class="muted small"></span>
    </div>
    <section class="panel ex-list" id="ex-list"></section>
    <div id="manage-host"></div>`;

  const $ = (id) => document.getElementById(id);
  function showList() {
    try {
      const { phone, ...perGym } = filters;
      localStorage.setItem(`gymhub:filters:${gym.id}`, JSON.stringify(perGym));
    } catch {}
    if ($("f-phone")) $("f-phone").checked = filters.phone;
    if ($("f-track")) $("f-track").value = filters.track;
    $("f-unsolved").checked = filters.unsolved;
    view.querySelectorAll("[data-level]").forEach((chip) => chip.classList.toggle("on", chip.dataset.level === filters.level));
    const shown = problems.filter(
      (problem) =>
        (!filters.track || (problem.track || "") === filters.track) &&
        (!filters.level || problem.difficulty === filters.level) &&
        (!filters.unsolved || !problem.status.solved) &&
        (!filters.phone || goodOnPhone(problem))
    );
    const row = (problem) => `<a class="ex-row${problem === next ? " next" : ""}" href="${href(problem)}">
        <span class="mark ${problem.status.cls}${problem.status.changed ? " tests-changed" : ""}" title="${esc(problem.status.title)}">${problem.status.mark}${problem.status.changed ? "*" : ""}</span>
        <span class="ex-title">${esc(problem.title)}${problem.warmup ? ` <span class="badge">example</span>` : ""}${problem === next ? ` <span class="badge run">next</span>` : ""}</span>
        <span class="ex-kind muted small">${[problem.kind && problem.kind !== "code" ? esc(KIND_LABEL[problem.kind] || problem.kind) : "", problem.minutes ? `~${Number(problem.minutes)} min` : ""].filter(Boolean).join(" · ")}</span>
        <span class="level ${esc(problem.difficulty)}">${esc(problem.difficulty)}</span></a>`;
    const groups = filters.track ? [[filters.track, shown]] : tracks.map((track) => [track, shown.filter((problem) => (problem.track || "") === track)]);
    const trackHead = (track) => {
      const counted = practice.filter((problem) => (problem.track || "") === track);
      const done = counted.filter((problem) => problem.status.solved).length;
      return `<div class="track-head"><h3 class="track">${esc(track)}</h3><span class="muted small">${done} / ${counted.length}</span>
        ${summaries[track] ? `<p class="track-summary">${esc(summaries[track])}</p>` : ""}</div>`;
    };
    $("ex-list").innerHTML =
      groups
        .filter(([, items]) => items.length)
        .map(([track, items]) => `${track && (tracks.length > 1 || summaries[track]) ? trackHead(track) : ""}${items.map(row).join("")}`)
        .join("") || `<p class="muted">${problems.length ? "No exercise matches these filters." : "No exercises."}</p>`;
    $("f-count").textContent = shown.length === problems.length ? `${problems.length} exercises` : `${shown.length} of ${problems.length} exercises`;
  }
  showList();
  $("f-track")?.addEventListener("change", (event) => {
    filters.track = event.target.value;
    showList();
  });
  $("f-unsolved").addEventListener("change", (event) => {
    filters.unsolved = event.target.checked;
    showList();
  });
  $("f-phone")?.addEventListener("change", (event) => {
    filters.phone = event.target.checked;
    writePhoneFilter(filters.phone);
    showList();
  });
  view.querySelectorAll("[data-level]").forEach((chip) =>
    chip.addEventListener("click", () => {
      filters.level = chip.dataset.level;
      showList();
    })
  );
  $("fork-gym")?.addEventListener("click", () => forkGym(gym.id, gym.title));
  $("report-gym")?.addEventListener("click", () => reportGym(gym, $("report-gym")));
  if (manages) renderManage(gym, $("manage-host")).catch(() => {});
  $("leave-gym")?.addEventListener("click", async () => {
    if (!confirm(`Leave "${gym.title}"? It disappears from your gyms. Your progress is kept if you join again.`)) return;
    await api(`/api/gyms/${encodeURIComponent(gym.id)}/leave`, { method: "POST" });
    location.hash = "#/gyms";
  });
}

// ---------- exercise ------------------------------------------------------------------------------

function loadMonaco() {
  if (state.monaco) return Promise.resolve(state.monaco);
  return new Promise((resolve, reject) => {
    window.MonacoEnvironment = {
      getWorkerUrl() {
        const source = `self.MonacoEnvironment={baseUrl:'${MONACO_BASE}/'};importScripts('${MONACO_BASE}/vs/base/worker/workerMain.js');`;
        return `data:text/javascript;charset=utf-8,${encodeURIComponent(source)}`;
      },
    };
    require.config({ paths: { vs: `${MONACO_BASE}/vs` } });
    require(["vs/editor/editor.main"], () => {
      state.monaco = window.monaco;
      resolve(window.monaco);
    }, reject);
  });
}

// ---------- phones and touch screens (issue #112) ------------------------------------------------

// A phone-sized or touch-first screen: Monaco types badly there, and the IDE view does not fit.
function smallScreen() {
  return window.matchMedia("(max-width: 640px), (pointer: coarse)").matches;
}

// "plain" (a textarea) or "full" (Monaco). The viewer's own choice wins; otherwise plain on small screens.
function editorKind() {
  try {
    const chosen = localStorage.getItem("gymhub:editor");
    if (chosen === "plain" || chosen === "full") return chosen;
  } catch {}
  return smallScreen() ? "plain" : "full";
}

function setEditorKind(kind) {
  try {
    localStorage.setItem("gymhub:editor", kind);
  } catch {}
}

// A plain code editor: a monospace textarea with no autocorrect, Tab and Shift+Tab indenting, and
// Enter keeping the indent. It offers the part of Monaco's editor API that the exercise page and the
// custom cases use, so either can sit behind the same code.
//   commands: {run, test, save} for Ctrl/Cmd+Enter, Ctrl/Cmd+Shift+Enter and Ctrl/Cmd+S.
function createPlainEditor(host, { value = "", label = "Code", minHeight = 200, commands = {} } = {}) {
  host.classList.add("plain-host");
  host.innerHTML = `<textarea class="plain-editor" aria-label="${esc(label)}" spellcheck="false" autocomplete="off"
    autocorrect="off" autocapitalize="off" wrap="off" enterkeyhint="enter"></textarea>`;
  const area = host.querySelector("textarea");
  const listeners = [];
  let escaped = false; // Escape, then Tab: leave the editor instead of indenting (no keyboard trap)
  let before = null; // the text before the last executeEdits, for trigger("…", "undo")
  const INDENT = "    ";

  const fit = () => {
    area.style.height = "auto";
    area.style.height = `${Math.max(area.scrollHeight + 2, minHeight)}px`;
  };
  const changed = () => {
    fit();
    listeners.forEach((listener) => listener());
  };
  // Through the browser's own editing, so its undo (Ctrl+Z, shake to undo) keeps working.
  const insert = (text) => {
    if (!document.execCommand?.("insertText", false, text)) {
      area.setRangeText(text, area.selectionStart, area.selectionEnd, "end");
      changed();
    }
  };
  const lineStart = (position) => area.value.lastIndexOf("\n", position - 1) + 1;

  area.addEventListener("input", changed);
  area.addEventListener("keydown", (event) => {
    const mod = event.ctrlKey || event.metaKey;
    if (event.key === "Escape") {
      escaped = true;
      return;
    }
    if (mod && event.key === "Enter") {
      event.preventDefault();
      (event.shiftKey ? commands.test : commands.run)?.();
    } else if (mod && event.key.toLowerCase() === "s" && commands.save) {
      event.preventDefault();
      commands.save();
    } else if (event.key === "Tab" && !mod && !event.altKey && !escaped) {
      event.preventDefault();
      const start = lineStart(area.selectionStart);
      if (event.shiftKey) {
        const remove = area.value.slice(start, start + INDENT.length).match(/^ {0,4}/)[0].length;
        if (!remove) return;
        const caret = area.selectionStart;
        area.setSelectionRange(start, start + remove);
        insert("");
        area.setSelectionRange(Math.max(start, caret - remove), Math.max(start, caret - remove));
      } else insert(INDENT);
    } else if (event.key === "Enter" && !mod && !event.shiftKey && !event.altKey && !event.isComposing && event.keyCode !== 229) {
      // Keep the indent, one level more after a line that opens a block (Python ":" or a "{").
      const start = lineStart(area.selectionStart);
      const line = area.value.slice(start, area.selectionStart);
      const indent = line.match(/^[ \t]*/)[0] + (/[:{\[(]\s*$/.test(line) ? INDENT : "");
      event.preventDefault();
      insert(`\n${indent}`);
    }
    if (event.key !== "Tab") escaped = false;
  });

  const api = {
    textarea: area,
    getValue: () => area.value,
    setValue(text) {
      area.value = text ?? "";
      changed();
    },
    onDidChangeModelContent(listener) {
      listeners.push(listener);
      return { dispose: () => listeners.splice(listeners.indexOf(listener), 1) };
    },
    focus: () => area.focus(),
    getModel: () => ({ getFullModelRange: () => null }),
    executeEdits(_source, edits) {
      before = area.value;
      api.setValue(edits.map((edit) => edit.text).join(""));
    },
    pushUndoStop() {},
    trigger(_source, command) {
      if (command === "undo" && before !== null) {
        api.setValue(before);
        before = null;
      }
    },
    layout: fit,
    dispose: () => (host.innerHTML = ""),
  };
  area.value = value;
  requestAnimationFrame(fit);
  return api;
}

function draftKey(gym, slug, lang) {
  return `gymhub:draft:${gym}:${slug}:${lang}`;
}

function readDraft(gym, slug, lang) {
  try {
    return JSON.parse(localStorage.getItem(draftKey(gym, slug, lang)) || "null");
  } catch {
    return null;
  }
}

function writeDraft(gym, slug, lang, draft) {
  try {
    if (draft) localStorage.setItem(draftKey(gym, slug, lang), JSON.stringify(draft));
    else localStorage.removeItem(draftKey(gym, slug, lang));
  } catch {
    /* storage unavailable: drafts just are not kept */
  }
}

function preferredLang(gym, problem) {
  try {
    const saved = localStorage.getItem(`gymhub:lang:${gym}`);
    if (saved && problem.langs.includes(saved)) return saved;
  } catch {}
  return problem.langs[0];
}

async function renderProblem(gymId, slug, review = false) {
  const data = await api(`/api/gyms/${encodeURIComponent(gymId)}/problems/${encodeURIComponent(slug)}`);
  const { problem, gym } = data;
  if (problem.kind && problem.kind !== "code") return renderQuestion(gymId, slug, data);
  // The IDE view needs a wide screen; phones get the simple page whatever was chosen on a desktop.
  if (viewMode() === "ide" && !review && !smallScreen()) return renderIde(gymId, slug);
  setCrumbs([["Gyms", "#/gyms"], [gym.title || gymId, `#/gym/${gymId}`], [problem.title, null]]);
  const langMeta = Object.fromEntries((gym.langs || []).map((lang) => [lang.id, lang]));
  let lang = preferredLang(gymId, problem);
  let current = problem.current || {};
  const plain = editorKind() === "plain";
  // Review: solve again from the starter. Drafts are kept apart and the saved solution is never touched.
  if (review) current = { ...(problem.starters || {}) };
  const draftGym = review ? `review:${gymId}` : gymId;
  let hintsShown = Math.max(0, data.progress?.hints_seen || 0);

  view.innerHTML = `
    <div class="workspace${plain ? " plain" : ""}">
      <section class="left panel">
        <div class="tabs" id="tabs">
          <button data-tab="statement" class="active">Problem</button>
          <button data-tab="hints">Hints <span class="muted">${problem.hints?.length || 0}</span></button>
          ${problem.editorial || problem.editorial_locked ? `<button data-tab="editorial">Editorial</button>` : ""}
          <button data-tab="history">Runs</button>
          ${gym.tutor ? `<button data-tab="tutor">Tutor</button>` : ""}
          ${gym.leaderboard && problem.benchable ? `<button data-tab="leaderboard">Leaderboard</button>` : ""}
        </div>
        <div id="tab-statement" class="tab markdown">${exerciseMeta(problem) ? `<p class="ex-meta-line">${exerciseMeta(problem)}</p>` : ""}${md(problem.statement)}</div>
        <div id="tab-hints" class="tab hidden"></div>
        <div id="tab-editorial" class="tab hidden"></div>
        <div id="tab-history" class="tab hidden"></div>
        <div id="tab-tutor" class="tab hidden markdown"></div>
        <div id="tab-leaderboard" class="tab hidden"></div>
      </section>
      <section class="right">
        <div class="toolbar">
          <select id="lang">${problem.langs.map((id) => `<option value="${esc(id)}">${esc(langMeta[id]?.label || id)}</option>`).join("")}</select>
          <span id="lang-status" class="muted small"></span>
          <div class="spacer"></div>
          <select id="run-on" class="run-on hidden" aria-label="Run on"></select>
          <button id="btn-run" title="Ctrl/Cmd+Enter: run every case on the code in the editor and show each output. Nothing is saved or recorded.">Run</button>
          <button id="btn-try" class="hidden">${smallScreen() ? "Try" : "Try in browser"}</button>
          <button id="btn-test" class="primary" title="Ctrl/Cmd+Shift+Enter: save the file, run the full tests, record progress">Submit</button>
          <details class="menu" id="more-menu"><summary title="More actions" aria-label="More actions">⋯</summary><div>
            <button id="btn-bench" ${problem.benchable ? "" : "disabled"} title="Time the solution against the reference">Bench</button>
            ${gym.runtime === "gpu_gym" && problem.benchable ? `<label class="profile-toggle menu-check" title="After the benchmark, profile one call with Nsight Compute: occupancy, memory and SM load, the main warp stall. Slow."><input type="checkbox" id="bench-profile"> Profile the bench with Nsight Compute</label>` : ""}
            <button id="btn-save" title="Ctrl/Cmd+S: write the file on your machine without running it">Save <span class="muted small">Ctrl/Cmd+S</span></button>
            <button id="btn-cases" title="Your own test cases for Run">Custom cases <span id="cases-count" class="muted"></span></button>
            <button id="btn-editor-kind" title="Remembered in this browser">${plain ? "Use the full editor" : "Use the plain editor"}</button>
            ${smallScreen() ? "" : `<button id="btn-ide" title="Switch to the VS Code-style workspace (remembered in this browser)">Open in IDE view</button>`}
            <hr>
            <button id="btn-machine">Load the file from my machine</button>
            <button id="btn-starter">Reset to starter</button>
          </div></details>
        </div>
        <div id="banner" class="banner hidden"></div>
        <div id="editor" class="editor" aria-label="Your solution"></div>
        <div id="cases-host" class="cases hidden"></div>
        <div class="output-head"><span id="job-status" class="muted">Output</span><div class="spacer"></div>
          <button id="btn-cancel" class="ghost hidden">Cancel</button></div>
        <pre id="output" class="output"></pre>
      </section>
    </div>`;

  const $ = (id) => document.getElementById(id);
  if (review) {
    setCrumbs([["Home", "#/"], [`Review: ${problem.title}`, null]]);
    $("editor").insertAdjacentHTML("beforebegin", `<div class="review-note">Review: solve it again from the starter. Submit checks it in a scratch copy; your saved solution stays as it is.
      <a href="#/gym/${encodeURIComponent(gymId)}/${encodeURIComponent(slug)}">Open my solution</a></div>`);
    for (const id of ["btn-save", "btn-bench", "btn-machine", "btn-starter", "bench-profile"]) $(id)?.closest("label, button")?.classList.add("hidden");
  }
  $("lang").value = lang;
  if (!data.owner) $("btn-machine").textContent = "Reload my saved code";
  if (!state.me?.can_run) {
    for (const id of ["btn-run", "btn-test", "btn-bench"]) {
      $(id).disabled = true;
      $(id).title = "Running code is not enabled for members yet.";
    }
    $("output").textContent =
      "You can read, edit and save your code. Running it is not enabled for members on this board yet; ask an admin.";
  }

  // Tabs
  $("tabs").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    $("tabs").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === button));
    ["statement", "hints", "editorial", "history", "tutor", "leaderboard"].forEach((tab) => $(`tab-${tab}`)?.classList.toggle("hidden", tab !== button.dataset.tab));
    if (button.dataset.tab === "leaderboard") renderLeaderboard();
    if (button.dataset.tab === "editorial") refreshEditorial($("tab-editorial"), gymId, slug, problem);
  });

  function renderHints() {
    const hints = problem.hints || [];
    if (!hints.length) {
      $("tab-hints").innerHTML = `<p class="muted">This exercise has no hints.</p>`;
      return;
    }
    const shown = hints
      .slice(0, hintsShown)
      .map((hint, index) => `<div class="hint"><h4>Hint ${index + 1} of ${hints.length}</h4>${md(hint)}</div>`)
      .join("");
    const more =
      hintsShown < hints.length
        ? `<button id="btn-hint">Reveal hint ${hintsShown + 1}</button>`
        : `<p class="muted">That was the last hint.</p>${gym.tutor ? `<button id="btn-tutor-hint">Ask the tutor for another hint</button>` : ""}`;
    $("tab-hints").innerHTML = shown + more;
    $("btn-tutor-hint")?.addEventListener("click", () => askTutor("hint"));
    const button = $("btn-hint");
    if (button)
      button.addEventListener("click", () => {
        hintsShown++;
        renderHints();
        api("/api/jobs", { method: "POST", body: JSON.stringify({ gym: gymId, slug, action: "hint", options: { count: hintsShown } }) }).catch(() => {});
      });
  }
  renderHints();

  let submissions = data.jobs;

  // Leaderboard (gymhub/leaderboard.py): best bench per person, for one language and GPU.
  async function renderLeaderboard(choice = {}) {
    const host = $("tab-leaderboard");
    if (!host) return;
    const params = new URLSearchParams({ lang: choice.lang ?? lang, ...(choice.hardware ? { hardware: choice.hardware } : {}) });
    try {
      const board = await api(`/api/gyms/${encodeURIComponent(gymId)}/problems/${encodeURIComponent(slug)}/leaderboard?${params}`);
      const select = (name, values, current) =>
        `<select data-board="${name}">${values.map((v) => `<option ${v === current ? "selected" : ""}>${esc(v)}</option>`).join("")}</select>`;
      host.innerHTML = board.total
        ? `<div class="board-filters">${select("lang", board.langs, board.lang)} ${select("hardware", board.hardwares, board.hardware)}
            <span class="muted small">${board.total} ${board.total === 1 ? "person" : "people"}${board.me ? ` · you are #${board.me.rank}` : ""}</span></div>
          <table class="table leaderboard"><thead><tr><th>#</th><th>who</th><th>median</th><th>when</th></tr></thead><tbody>${board.entries
            .map((e) => `<tr class="${e.me ? "me" : ""}"><td>${e.rank}</td><td>${esc(e.name)}${e.me ? " (you)" : ""}</td><td class="mono">${formatMs(e.ms, 4)}</td><td class="muted small">${ago(e.at)}</td></tr>`)
            .join("")}</tbody></table>
          <p class="muted small">Only benchmarks run on trusted machines count, and times are compared on the same GPU only.</p>`
        : `<p class="muted">No results yet. Bench a passing solution on one of the board's machines to get on the board.</p>`;
      host.querySelectorAll("[data-board]").forEach((s) =>
        s.addEventListener("change", () => renderLeaderboard({ lang: host.querySelector('[data-board="lang"]').value, hardware: host.querySelector('[data-board="hardware"]').value }))
      );
    } catch (error) {
      host.innerHTML = `<p class="error">${esc(error.message)}</p>`;
    }
  }
  // Tutor (gymhub/tutor.py): explain the last failure, review the code, one more hint.
  let tutorUsage = null;
  function showTab(name) {
    $("tabs").querySelector(`[data-tab="${name}"]`)?.click();
  }
  function lastFailure() {
    return (submissions || []).find((job) => ["test", "bench"].includes(job.action) && job.finished_at && !job.ok);
  }
  function renderTutor(content = "") {
    if (!gym.tutor) return;
    const failed = lastFailure();
    const budget = tutorUsage?.paused
      ? "The board has used this month's AI budget, so the tutor is paused until the 1st."
      : tutorUsage && tutorUsage.budget != null ? `$${tutorUsage.usage.usd.toFixed(2)} of $${tutorUsage.budget.toFixed(2)} used this month.` : "";
    $("tab-tutor").innerHTML = `<div class="tutor-actions">
        <button data-tutor="explain" ${failed ? "" : "disabled title='No failed submission yet'"}>Explain my last failure</button>
        <button data-tutor="review">Review my code</button>
        <button data-tutor="hint">Another hint</button></div>
      <p class="muted small">The tutor sees the statement, the hints and your code${failed ? ", and the output of your last failed submission" : ""}. It teaches rather than solves. ${esc(budget)}</p>
      <div id="tutor-answer">${content}</div>`;
    $("tab-tutor").querySelectorAll("[data-tutor]").forEach((button) => button.addEventListener("click", () => askTutor(button.dataset.tutor)));
  }
  async function askTutor(action) {
    showTab("tutor");
    let code = "";
    try {
      code = editor.getValue();
    } catch {}
    const failed = lastFailure();
    renderTutor(`<p class="muted">Thinking...</p>`);
    $("tab-tutor").querySelectorAll("[data-tutor]").forEach((b) => (b.disabled = true));
    try {
      const reply = await api(`/api/gyms/${encodeURIComponent(gymId)}/problems/${encodeURIComponent(slug)}/tutor`, {
        method: "POST", body: JSON.stringify({ action, lang, code, job_id: action === "explain" ? failed?.id : null }),
      });
      if (tutorUsage) tutorUsage.usage = reply.usage;
      renderTutor(`<div class="tutor-reply">${md(reply.text)}</div>`);
    } catch (error) {
      renderTutor(`<p class="error">${esc(error.message)}</p>`);
    }
  }
  if (gym.tutor) {
    renderTutor();
    api("/api/my/tutor")
      .then((info) => {
        tutorUsage = info;
        renderTutor($("tutor-answer")?.innerHTML || "");
      })
      .catch(() => {});
  }

  function renderHistory(jobs) {
    submissions = jobs;
    $("tab-history").innerHTML = jobs.length
      ? `<p class="muted small">Your submissions. Click a row for its output; <b>Code</b> shows what was submitted, compares it, and can restore it.</p>
        <table class="table"><tbody>${historyRows(jobs)}</tbody></table>`
      : `<p class="muted">No submissions yet. Runs are not listed: they are never recorded.</p>`;
  }
  $("tab-history").addEventListener("click", (event) => {
    const code = event.target.closest("[data-code]");
    if (code) {
      loadMonaco().then((monaco) => openSubmission(monaco, code.dataset.code, submissions, {
        language: (id) => langMeta[id]?.editor || "python",
        current: (id) => (id === lang ? editor.getValue() : current[id] ?? null),
        restore: (text, id) => {
          if (id !== lang) {
            lang = id;
            $("lang").value = id;
            loadLang();
          }
          editor.executeEdits("restore", [{ range: editor.getModel().getFullModelRange(), text }]);
          editor.pushUndoStop();
          editor.focus();
          showBanner("Restored a past submission into the editor. It is an unsaved draft until you Save or Submit.", [
            { label: "Undo", run: () => { editor.trigger("restore", "undo"); showBanner(null); } },
          ]);
        },
      })).catch((error) => alert(error.message));
      return;
    }
    const row = event.target.closest("tr[data-job]");
    if (row) showJob(row.dataset.job);
  });
  renderHistory(data.jobs);

  function updateLangStatus() {
    const s = langState({ exercises: { [slug]: data.progress } }, slug, lang);
    const bests = Object.entries(data.bests?.[lang] || {})
      .sort((a, b) => a[1].ms - b[1].ms)
      .map(([hardware, best]) => `${formatMs(best.ms)} on ${hardware}`);
    const title = bests.length && s.cls === "good" ? "solved" : s.title;
    const changed = s.cls === "good" ? data.tests_changed?.[lang] : null;
    $("lang-status").innerHTML = `<span class="mark ${s.cls}">${s.mark}</span> ${esc(title)}` +
      (changed ? ` <span class="tests-changed" title="Your passing submission ran against older tests">· tests changed since you solved${changed.version ? ` (on v${changed.version})` : ""}</span>` : "") +
      (bests.length ? ` <span class="muted" title="Best bench median per hardware">· ${esc(bests.join(" · "))}</span>` : "");
  }
  updateLangStatus();

  // Editor: Monaco, or on phones a plain textarea (Monaco is not loaded at all then).
  const monaco = plain ? null : await loadMonaco();
  if (monaco) registerCompletions(monaco);
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const editor = plain
    ? createPlainEditor($("editor"), {
        label: `Your solution: ${problem.title}`,
        commands: { run: () => submit("run"), test: () => submit("test"), save: () => submit("save") },
      })
    : monaco.editor.create($("editor"), {
    value: "",
    language: "python",
    theme: dark ? "vs-dark" : "vs",
    automaticLayout: true,
    minimap: { enabled: false },
    // Explicit metrics: Monaco sizes suggestion rows from what it measures, and a mismatch with the
    // font actually rendered (browser zoom, fallback fonts) clips the row text.
    fontFamily: 'Menlo, "SF Mono", "JetBrains Mono", "DejaVu Sans Mono", Consolas, monospace',
    fontSize: 14,
    lineHeight: 21,
    suggestFontSize: 13,
    suggestLineHeight: 24,
    scrollBeyondLastLine: false,
    tabSize: 4,
    // Suggestion and hover popups render above the page, so the panel edges cannot clip them.
    fixedOverflowWidgets: true,
    quickSuggestions: { other: true, comments: false, strings: false },
    wordBasedSuggestions: "currentDocument",
    acceptSuggestionOnEnter: "smart",
    suggest: { showWords: true, preview: true },
  });
  state.editor = editor;
  // Fonts can finish loading after the editor measured them; measure again so popups line up.
  if (monaco) document.fonts?.ready.then(() => monaco.editor.remeasureFonts());
  let suppress = false;

  const tryButton = browserRun.attach($("btn-try"), $("output"), () => ({ problem, lang, code: editor.getValue() }),
    (text) => ($("job-status").textContent = text));
  if (!state.me?.can_run && browserRun.available(problem, lang)) {
    $("output").textContent += "\nTry in browser runs the exercise's examples on your code right here.";
  }

  const cases = mountCases($("cases-host"), monaco, gymId, slug, problem.case_template, (count) => {
    $("cases-count").textContent = count ? `(${count})` : "";
  });
  state.disposers.push(() => cases.dispose());
  try {
    if (localStorage.getItem("gymhub:cases-open") === "1") $("cases-host").classList.remove("hidden");
  } catch {}
  $("btn-cases").addEventListener("click", () => {
    const hidden = $("cases-host").classList.toggle("hidden");
    try {
      localStorage.setItem("gymhub:cases-open", hidden ? "0" : "1");
    } catch {}
    cases.layout();
  });

  function showBanner(html, actions = []) {
    const banner = $("banner");
    if (!html) {
      banner.classList.add("hidden");
      return;
    }
    banner.innerHTML = `<span>${html}</span>` + actions.map((a, i) => `<button data-i="${i}" class="ghost">${esc(a.label)}</button>`).join("");
    banner.classList.remove("hidden");
    banner.querySelectorAll("button").forEach((b) => b.addEventListener("click", actions[Number(b.dataset.i)].run));
  }

  function setCode(text) {
    suppress = true;
    editor.setValue(text);
    suppress = false;
  }

  function loadLang() {
    const machine = current[lang] ?? "";
    const draft = readDraft(draftGym, slug, lang);
    // Nothing from a machine yet: begin from the starter (usually one solve(...) function).
    const start = machine || problem.starters?.[lang] || "";
    if (monaco) monaco.editor.setModelLanguage(editor.getModel(), langMeta[lang]?.editor || "python");
    if (draft && draft.code !== machine) {
      setCode(draft.code);
      if (draft.base !== machine) {
        showBanner("The file on your machine changed since you started this draft.", [
          { label: "Use the machine's version", run: () => { writeDraft(draftGym, slug, lang, null); loadLang(); } },
          { label: "Keep my draft", run: () => { writeDraft(draftGym, slug, lang, { code: editor.getValue(), base: machine }); showBanner(null); } },
        ]);
      } else {
        showBanner("Unsaved draft from this browser.", [
          { label: "Discard draft", run: () => { writeDraft(draftGym, slug, lang, null); loadLang(); } },
        ]);
      }
    } else {
      writeDraft(draftGym, slug, lang, null);
      setCode(start);
      showBanner(null);
    }
    updateLangStatus();
  }
  loadLang();

  editor.onDidChangeModelContent(() => {
    if (suppress) return;
    const machine = current[lang] ?? "";
    const code = editor.getValue();
    const existing = readDraft(draftGym, slug, lang);
    writeDraft(draftGym, slug, lang, code === machine ? null : { code, base: existing?.base ?? machine });
  });

  $("lang").addEventListener("change", () => {
    lang = $("lang").value;
    try { localStorage.setItem(`gymhub:lang:${gymId}`, lang); } catch {}
    loadLang();
    tryButton.refresh();
    if (state.me?.can_run) fillRunOn($("run-on"), gymId, slug, lang);
  });
  if (state.me?.can_run) fillRunOn($("run-on"), gymId, slug, lang);
  $("run-on").addEventListener("change", () => writeRunOn(gymId, runOnTarget($("run-on"))));

  // Jobs
  let activeJob = null;

  async function refreshProblem() {
    const fresh = await api(`/api/gyms/${encodeURIComponent(gymId)}/problems/${encodeURIComponent(slug)}`);
    if (!review) current = fresh.problem.current || {};
    data.progress = fresh.progress;
    data.bests = fresh.bests;
    renderHistory(fresh.jobs);
    updateLangStatus();
  }

  let queueWarning = "";
  function renderJob(job) {
    $("output").textContent = outputWithHost(job, job.status === "queued" ? queueWarning || "Waiting for the runner…" : "");
    $("output").scrollTop = $("output").scrollHeight;
    const elapsed = job.started_at ? ` · ${(((job.finished_at || Date.now() / 1000) - job.started_at)).toFixed(1)} s` : "";
    const mode = job.action === "run" ? " · not recorded" : "";
    $("job-status").innerHTML = `${statusBadge(job)} ${esc(actionLabel(job.action))} · ${esc(job.lang)}${elapsed}${mode}`;
    const live = job.status === "queued" || job.status === "running";
    $("btn-cancel").classList.toggle("hidden", !live);
    ["btn-run", "btn-test", "btn-bench", "btn-save"].forEach(
      (id) => ($(id).disabled = live || (id !== "btn-save" && !state.me?.can_run) || (id === "btn-bench" && !problem.benchable))
    );
  }

  async function poll(jobId) {
    try {
      const job = await followJob(jobId, renderJob);
      activeJob = null;
      if (review && job.action === "test" && job.ok) {
        showBanner("Review passed. It comes back later at a longer interval; your saved solution was not changed.", [
          { label: "Back to Home", run: () => (location.hash = "#/") },
        ]);
      }
      const savedCode = current[job.lang];
      await refreshProblem();
      if (job.lang === lang && current[lang] !== savedCode) {
        // The runner wrote our code (or restored the starter): it is no longer a draft.
        if (job.action === "reset") setCode(current[lang]);
      }
      if (current[lang] === editor.getValue()) {
        writeDraft(draftGym, slug, lang, null);
        showBanner(null);
      }
    } catch (error) {
      $("job-status").textContent = `Lost track of the job: ${error.message}`;
      state.pollTimer = setTimeout(() => poll(jobId), 3000);
    }
  }

  async function submit(action, extra = {}) {
    if (activeJob) return;
    const body = { gym: gymId, slug, lang, action, ...extra };
    if (["run", "test", "bench", "save"].includes(action)) body.code = editor.getValue();
    if (action === "bench" && $("bench-profile")?.checked) body.options = { ...(body.options || {}), profile: true };
    if (review && action === "test") body.options = { ...(body.options || {}), review: true };
    const target = ["run", "test", "bench"].includes(action) && runOnTarget($("run-on"));
    if (target) body.target = target;
    if (action === "run" && cases) {
      const custom = cases.read();
      if (custom.error) {
        $("cases-host").classList.remove("hidden");
        $("output").textContent = `Fix the custom cases first: ${custom.error}`;
        return;
      }
      if (custom.cases.length) body.options = { cases: custom.cases, only_custom: custom.only };
      else if (custom.only) {
        $("output").textContent = "Only my cases is ticked, but there are no custom cases. Add one or untick it.";
        return;
      }
    }
    try {
      const { id, warning } = await api("/api/jobs", { method: "POST", body: JSON.stringify(body) });
      activeJob = id;
      queueWarning = warning || "";
      renderJob({ status: "queued", action, lang, output: "" });
      poll(id);
    } catch (error) {
      $("output").textContent = error.message;
    }
  }

  async function showJob(jobId) {
    const job = await api(`/api/jobs/${jobId}`);
    renderJob(job);
  }

  // The ⋯ menu closes once one of its actions is chosen.
  $("more-menu").addEventListener("click", (event) => {
    if (event.target.closest("button")) $("more-menu").removeAttribute("open");
  });
  $("btn-editor-kind").addEventListener("click", () => {
    setEditorKind(plain ? "full" : "plain");
    route(); // the draft is kept in this browser, so nothing typed is lost
  });
  $("btn-ide")?.addEventListener("click", () => {
    setViewMode("ide");
    route();
  });
  $("btn-run").addEventListener("click", () => submit("run"));
  $("btn-test").addEventListener("click", () => submit("test"));
  $("btn-bench").addEventListener("click", () => submit("bench"));
  $("btn-save").addEventListener("click", () => submit("save"));
  $("btn-cancel").addEventListener("click", () => activeJob && api(`/api/jobs/${activeJob}/cancel`, { method: "POST" }));
  $("btn-machine").addEventListener("click", async () => {
    await refreshProblem();
    writeDraft(draftGym, slug, lang, null);
    loadLang();
  });
  $("btn-starter").addEventListener("click", () => {
    if (confirm("Replace your file with the blank starter? The old file is backed up on your machine.")) {
      writeDraft(draftGym, slug, lang, null);
      submit("reset");
    }
  });
  if (monaco) {
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => submit("run"));
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => submit("test"));
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => submit("save"));
  }

  // Keep the runner pill fresh and pick up edits made in a local editor.
  state.refreshTimer = setInterval(async () => {
    if (activeJob || document.hidden) return;
    try {
      const before = current[lang];
      await refreshProblem();
      if (current[lang] !== before) {
        const draft = readDraft(draftGym, slug, lang);
        if (!draft) setCode(current[lang]);
        else if (draft.base !== current[lang]) loadLang();
      }
    } catch {}
  }, 15000);

  const last = data.jobs.find((job) => job.lang === lang);
  if (last) showJob(last.id).catch(() => {});
}

// ---------- Run on: where a job runs, and where it ran ---------------------------------------------

// The "Run on" choice is kept per gym in this browser: null is Auto (any machine that can take it).
function readRunOn(gymId) {
  try {
    return JSON.parse(localStorage.getItem(`gymhub:run-on:${gymId}`) || "null");
  } catch {
    return null;
  }
}

function writeRunOn(gymId, target) {
  try {
    if (target) localStorage.setItem(`gymhub:run-on:${gymId}`, JSON.stringify(target));
    else localStorage.removeItem(`gymhub:run-on:${gymId}`);
  } catch {}
}

function targetName(target) {
  return target ? target.runner || target.pool_machine || "" : "";
}

// Fill a <select> with Auto and the machines this user may run the exercise on now
// (GET /api/run-targets). A saved choice that is not available now stays, marked offline, so the
// learner sees why a job waits. Hidden when there is nothing to choose from.
async function fillRunOn(select, gymId, slug, lang) {
  let choices = [];
  try {
    const params = new URLSearchParams({ gym: gymId, slug, lang });
    choices = (await api(`/api/run-targets?${params}`)).choices || [];
  } catch {}
  const saved = readRunOn(gymId);
  const key = (target) => JSON.stringify(target);
  const options = [`<option value="">Run on: Auto</option>`];
  for (const choice of choices) {
    const label = `${choice.name} · ${choice.hardware}`;
    options.push(`<option value="${esc(key(choice.target))}" title="${esc(choice.via ? `Pool machine, via ${choice.via}` : "Board runner")}">${esc(label)}</option>`);
  }
  const known = choices.some((choice) => key(choice.target) === key(saved));
  if (saved && !known) options.push(`<option value="${esc(key(saved))}">${esc(targetName(saved))} · offline</option>`);
  select.innerHTML = options.join("");
  select.value = saved ? key(saved) : "";
  select.classList.toggle("hidden", !choices.length && !saved);
  select.title = "Run on: Auto picks any machine that can run it";
}

function runOnTarget(select) {
  if (!select || !select.value) return null;
  try {
    return JSON.parse(select.value);
  } catch {
    return null;
  }
}

// "Ran on jetson (Orin) · via spark-runner", or "Running on jetson…" while it runs.
function whereLine(job) {
  if (!job.host) return "";
  const hardware = job.hardware ? ` (${job.hardware})` : "";
  const via = job.via ? ` · via ${job.via}` : "";
  if (job.status === "running") return `Running on ${job.host}${hardware}${via}…`;
  if (["done", "error", "cancelled"].includes(job.status)) return `Ran on ${job.host}${hardware}${via}`;
  return "";
}

// A job's output with where it runs or ran: a header while running, a footer once it ended.
function outputWithHost(job, empty) {
  const output = job.output || empty;
  const where = whereLine(job);
  if (!where) return output;
  if (job.status === "running") return `${where}\n\n${output}`;
  return `${output}${output && !output.endsWith("\n") ? "\n" : ""}\n— ${where}`;
}

// ---------- following a job ----------------------------------------------------------------------

// Follow a job until it ends: live Server-Sent Events, or polling if the stream is not available.
// Calls onUpdate(job) for every change and resolves with the final job.
function followJob(jobId, onUpdate) {
  return new Promise((resolve) => {
    let job = { id: jobId, output: "" };
    let done = false;
    let sawActive = false; // only a job seen queued or running notifies when it ends
    let source = null;
    const terminal = (item) => ["done", "error", "cancelled"].includes(item.status);
    const finish = (final) => {
      if (done) return;
      done = true;
      source?.close();
      if (sawActive && terminal(final)) notifyJobFinished(final);
      resolve(final);
    };
    const poll = async () => {
      if (done) return;
      try {
        job = await api(`/api/jobs/${jobId}`);
        if (!terminal(job)) sawActive = true;
        onUpdate(job);
        if (terminal(job)) return finish(job);
      } catch {}
      state.pollTimer = setTimeout(poll, 1000);
    };
    if (!window.EventSource) return poll();
    source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    state.eventSource = source;
    source.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === "gone") return finish(job);
      const output = "output" in data ? data.output : (job.output || "") + (data.append || "");
      job = { ...job, ...data, output };
      delete job.append;
      if (!terminal(job)) sawActive = true;
      onUpdate(job);
      if (terminal(job)) finish(job);
    };
    source.onerror = () => {
      if (done) return;
      source.close();
      poll();  // the stream broke or is not allowed here: fall back to polling
    };
  });
}

// ---------- custom cases (shared by the simple page and the IDE) --------------------------------

const CASES_HELP = `One object per case, with every argument by name. Plain values are JSON. Tensors:
{"tensor": [[1, 2], [3, 4]]}  {"randn": [64, 128], "seed": 1}  {"rand": [1024]}
{"randint": [16], "low": 0, "high": 10}  {"zeros": [4]}  {"ones": [8]}  {"empty": [32]}  {"arange": 10}
Add "dtype": "int64" (or float16, bool, ...) when needed. Optional "_name" labels a case.
Keep sizes consistent with the size arguments (N, M, ...). The reference computes the expected output.`;

function formatCases(cases) {
  if (!cases.length) return "[]";
  return "[\n" + cases.map((item) => "  " + JSON.stringify(item)).join(",\n") + "\n]";
}

// Mount the custom-cases editor in `host` for one exercise. Returns {read, count, dispose}.
// With no Monaco (the plain editor on phones), the cases are a plain textarea too.
function mountCases(host, monaco, gymId, slug, template, onChange = () => {}) {
  const key = `gymhub:cases:${gymId}:${slug}`;
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(key) || "{}");
  } catch {}
  host.innerHTML = `
    <div class="cases-bar">
      <b>Custom cases</b>
      <button data-act="add" ${template ? "" : "disabled"} title="Append the problem's example case as a starting point">Add example case</button>
      <button data-act="format">Format</button>
      <button data-act="clear">Clear</button>
      <label class="cases-only" title="Run only your cases, not the built-in ones"><input type="checkbox" data-act="only"> Only my cases</label>
      <span class="spacer"></span>
      <button data-act="help" class="ghost" title="Value format">Format help</button>
    </div>
    <pre class="cases-help hidden">${esc(CASES_HELP)}</pre>
    <div class="cases-editor"></div>
    <div class="cases-error"></div>`;
  const q = (selector) => host.querySelector(selector);
  q('[data-act="only"]').checked = Boolean(saved.only);
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const editor = !monaco
    ? createPlainEditor(q(".cases-editor"), { value: saved.text || "[]", label: "Custom cases (JSON)", minHeight: 110 })
    : monaco.editor.create(q(".cases-editor"), {
    value: saved.text || "[]",
    language: "json",
    theme: dark ? "vs-dark" : "vs",
    automaticLayout: true,
    fixedOverflowWidgets: true,
    minimap: { enabled: false },
    lineNumbers: "off",
    folding: false,
    fontSize: 13,
    scrollBeyondLastLine: false,
    wordWrap: "on",
    renderLineHighlight: "none",
    overviewRulerLanes: 0,
  });

  function parse() {
    const text = editor.getValue().trim();
    if (!text) return { cases: [] };
    try {
      let cases = JSON.parse(text);
      if (!Array.isArray(cases)) cases = [cases];
      if (cases.some((item) => typeof item !== "object" || item === null || Array.isArray(item)))
        return { error: "Each case must be an object of arguments by name." };
      if (cases.length > 20) return { error: "At most 20 custom cases." };
      return { cases };
    } catch (error) {
      return { error: `Not valid JSON: ${error.message}` };
    }
  }

  function persist() {
    try {
      localStorage.setItem(key, JSON.stringify({ text: editor.getValue(), only: q('[data-act="only"]').checked }));
    } catch {}
    const parsed = parse();
    q(".cases-error").textContent = parsed.error || "";
    onChange(parsed.cases ? parsed.cases.length : 0);
  }

  editor.onDidChangeModelContent(persist);
  q('[data-act="only"]').addEventListener("change", persist);
  q(".cases-bar").addEventListener("click", (event) => {
    const action = event.target.closest("button")?.dataset.act;
    const parsed = parse();
    if (action === "add") {
      const cases = parsed.cases || [];
      editor.setValue(formatCases([...cases, template]));
    } else if (action === "format" && parsed.cases) {
      editor.setValue(formatCases(parsed.cases));
    } else if (action === "clear") {
      editor.setValue("[]");
    } else if (action === "help") {
      q(".cases-help").classList.toggle("hidden");
    }
  });
  persist();

  return {
    // {cases, only} for a run, or {error} when the JSON is not usable.
    read() {
      const parsed = parse();
      if (parsed.error) return parsed;
      return { cases: parsed.cases, only: q('[data-act="only"]').checked };
    },
    count: () => parse().cases?.length || 0,
    layout: () => editor.layout(),
    dispose: () => editor.dispose(),
  };
}

// ---------- submission viewer (shared by the simple page and the IDE) ------------------------------

// Show a stored submission's code, diffed against the editor or the previous submission.
// opts: {language(lang) -> Monaco id, current(lang) -> code in the editor or null, restore(code, lang)}
async function openSubmission(monaco, jobId, submissions, opts) {
  const job = await api(`/api/jobs/${encodeURIComponent(jobId)}/code`);
  const older = submissions
    .filter((item) => item.has_code && item.lang === job.lang && item.created_at < job.created_at)
    .sort((a, b) => b.created_at - a.created_at)[0];
  const current = opts.current(job.lang);
  const overlay = document.createElement("div");
  overlay.className = "diff-modal";
  overlay.innerHTML = `
    <div class="diff-box" role="dialog" aria-label="Submission code">
      <div class="diff-head">
        <b>Submission</b>
        <span class="muted">${ago(job.created_at)} · ${esc(job.lang)} · ${esc(actionLabel(job.action))}</span>
        ${statusBadge(job)}
        <select class="diff-against">
          ${current != null ? `<option value="current">Compare with your editor</option>` : ""}
          ${older ? `<option value="previous">Compare with the previous submission (${ago(older.created_at)})</option>` : ""}
          <option value="none">Just this submission</option>
        </select>
        <span class="spacer"></span>
        <button class="primary diff-restore" title="Load this code into the editor as an unsaved draft">Restore into editor</button>
        <button class="diff-close" title="Close (Esc)">✕</button>
      </div>
      <div class="diff-legend muted small"></div>
      <div class="diff-body"></div>
    </div>`;
  document.body.appendChild(overlay);
  const q = (selector) => overlay.querySelector(selector);
  const language = opts.language(job.lang);
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  let widget = null;
  const models = [];
  const model = (text) => {
    const created = monaco.editor.createModel(text ?? "", language);
    models.push(created);
    return created;
  };

  async function show(mode) {
    widget?.dispose();
    models.splice(0).forEach((item) => item.dispose());
    const common = { readOnly: true, automaticLayout: true, theme: dark ? "vs-dark" : "vs", minimap: { enabled: false }, scrollBeyondLastLine: false, fontSize: 13 };
    if (mode === "none") {
      q(".diff-legend").textContent = "The code this submission ran.";
      widget = monaco.editor.create(q(".diff-body"), { ...common, model: model(job.code) });
      return;
    }
    let original = job.code;
    let modified = current;
    if (mode === "previous") {
      original = (await api(`/api/jobs/${encodeURIComponent(older.id)}/code`)).code;
      modified = job.code;
      q(".diff-legend").textContent = `Left: previous submission (${ago(older.created_at)}). Right: this submission.`;
    } else {
      q(".diff-legend").textContent = "Left: this submission. Right: the code in your editor now.";
    }
    const same = original === modified;
    if (same) q(".diff-legend").textContent += "  No differences.";
    widget = monaco.editor.createDiffEditor(q(".diff-body"), {
      ...common,
      renderSideBySide: window.innerWidth > 800,
      originalEditable: false,
      ignoreTrimWhitespace: false,
    });
    widget.setModel({ original: model(original), modified: model(modified) });
  }

  function close() {
    widget?.dispose();
    models.forEach((item) => item.dispose());
    overlay.remove();
    document.removeEventListener("keydown", onKey, true);
  }
  function onKey(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    }
  }
  document.addEventListener("keydown", onKey, true);
  q(".diff-close").addEventListener("click", close);
  overlay.addEventListener("click", (event) => event.target === overlay && close());
  q(".diff-against").addEventListener("change", (event) => show(event.target.value));
  q(".diff-restore").addEventListener("click", () => {
    opts.restore(job.code, job.lang);
    close();
  });
  await show(q(".diff-against").value);
}

function historyRows(jobs) {
  return jobs
    .map(
      (job) => `<tr class="clickable" tabindex="0" role="button" data-job="${esc(job.id)}"><td class="muted">${ago(job.created_at)}</td><td>${esc(job.lang)}</td>
      <td>${esc(actionLabel(job.action))}</td><td>${statusBadge(job)}</td><td class="mono muted small">${esc(job.headline)}</td>
      <td class="muted small">${esc([job.host, job.hardware].filter(Boolean).join(" · "))}</td>
      <td class="row-actions">${job.has_code ? `<button class="ghost small-button" data-code="${esc(job.id)}" title="See the code, compare it, restore it">Code</button>` : ""}</td></tr>`
    )
    .join("");
}

function viewMode() {
  try {
    return localStorage.getItem("gymhub:view") === "ide" ? "ide" : "simple";
  } catch {
    return "simple";
  }
}

function setViewMode(mode) {
  try {
    localStorage.setItem("gymhub:view", mode);
  } catch {}
}

// ---------- router ----------------------------------------------------------------------------

async function route() {
  if (!state.me) {
    renderSignIn();
    return;
  }
  clearTimers();
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  view.innerHTML = `<p class="muted loading">Loading…</p>`;
  try {
    if (parts[0] === "explore") {
      // Explore is part of the Gyms page now; old links land there.
      location.replace("#/gyms");
      return;
    }
    const section = ["gym", "gyms", "new-gym", "edit-gym"].includes(parts[0]) ? "gyms" : ["compute", "admin"].includes(parts[0]) ? parts[0] : "home";
    document.querySelectorAll("[data-nav]").forEach((link) => link.classList.toggle("active", link.dataset.nav === section));
    if (parts[0] === "admin") await renderAdmin();
    else if (parts[0] === "account") await renderAccount();
    else if (parts[0] === "review" && parts.length === 3) await renderProblem(parts[1], parts[2], true);
    else if (parts[0] === "new-gym") await renderEditor(null);
    else if (parts[0] === "edit-gym" && parts.length === 2) await renderEditor(parts[1]);
    else if (parts[0] === "join" && parts.length === 3) await renderJoin(parts[1], parts[2]);
    else if (parts[0] === "gyms") await renderGyms();
    else if (parts[0] === "compute") await renderCompute();
    else if (parts[0] === "gym" && parts.length === 2) await renderGym(parts[1]);
    else if (parts[0] === "gym" && parts.length >= 3) {
      await renderProblem(parts[1], parts[2]); // hands code exercises to the IDE view when that is chosen
    }
    else await renderHome();
  } catch (error) {
    fail(error);
  }
}

// Rows and tree nodes that act as buttons (role="button", tabindex="0") open with Enter or Space,
// as a real button would.
document.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const target = event.target;
  if (target.getAttribute?.("role") !== "button" || target.tagName === "BUTTON") return;
  event.preventDefault();
  target.click();
});

let booted = false;

async function boot() {
  // Invite links (#/invite/<code>) are remembered until sign-in uses them.
  const invite = location.hash.match(/^#\/invite\/([\w-]+)/);
  if (invite) {
    try { sessionStorage.setItem("gymhub:invite", invite[1]); } catch {}
    history.replaceState(null, "", "#/");
  }
  let me;
  try {
    me = await api("/api/me");
  } catch (error) {
    if (error.code === "not_invited") {
      // Behind IAP: Google already knows who this is. An invite link is all that is missing.
      const invite = pendingInvite();
      if (invite) {
        try {
          await api("/api/auth/accept-invite", { method: "POST", body: JSON.stringify({ invite }) });
          try { sessionStorage.removeItem("gymhub:invite"); } catch {}
          return boot();
        } catch (inviteError) {
          try { sessionStorage.removeItem("gymhub:invite"); } catch {}
          renderNoAccess(error.email, inviteError.message);
          return;
        }
      }
      renderNoAccess(error.email, error.message);
      return;
    }
    if (/sign in|session/i.test(error.message)) {
      renderSignIn();
      return;
    }
    fail(error);
    return;
  }
  state.me = me;
  renderUserMenu(me);
  watchMachines();
  if (!booted) {
    booted = true;
    await api("/api/overview").then((data) => setRunnerPill(data.runners)).catch(() => {});
    setInterval(() => !document.hidden && state.me && api("/api/overview").then((data) => setRunnerPill(data.runners)).catch(() => {}), 30000);
    window.addEventListener("hashchange", route);
  }
  route();
}

// ---------- sign-in, user menu, admin -------------------------------------------------------------

function pendingInvite() {
  try {
    return sessionStorage.getItem("gymhub:invite");
  } catch {
    return null;
  }
}

async function renderSignIn(message = "") {
  clearTimers();
  document.body.classList.remove("ide-mode");
  document.body.classList.add("signed-out");
  const config = await api("/api/auth/config").catch(() => ({}));
  const invite = pendingInvite();
  if (config.mode === "local") return renderLocalSignIn(config, invite, message);
  view.innerHTML = `
    <section class="signin landing-hero">
      <div class="signin-card">
        <span class="logo big"></span>
        <h1>Human Gym</h1>
        <p class="muted">Practice GPU kernels, LLM internals and algorithms, with tests and benchmarks on real hardware.</p>
        ${invite ? `<p class="invite-note">You have an invite. Sign in to accept it.</p>` : ""}
        <div id="google-button"></div>
        <p class="signin-error" id="signin-error">${esc(message)}</p>
        <p class="muted small">Invite-only. If you don't have access, ask the board's admin for an invite link.</p>
        <p class="muted small">By signing in you agree to the <a href="/terms">terms of use</a> and the <a href="/privacy">privacy policy</a>.</p>
      </div>
    </section>
    ${landingSections(config)}`;
  renderShowcase();
  if (!config.google_client_id) {
    document.getElementById("signin-error").textContent = "Sign-in is not configured on this board.";
    return;
  }
  await loadScript("https://accounts.google.com/gsi/client");
  google.accounts.id.initialize({
    client_id: config.google_client_id,
    callback: async ({ credential }) => {
      try {
        await api("/api/auth/google", { method: "POST", body: JSON.stringify({ credential, invite: pendingInvite() }) });
        try { sessionStorage.removeItem("gymhub:invite"); } catch {}
        document.body.classList.remove("signed-out");
        boot();
      } catch (error) {
        document.getElementById("signin-error").textContent = error.message;
      }
    },
    ux_mode: "popup",
  });
  google.accounts.id.renderButton(document.getElementById("google-button"), { theme: "outline", size: "large", text: "signin_with", shape: "pill" });
}

// GYMHUB_AUTH=local: a name and a password kept by the board. An invite link opens the account form
// (or, from a reset link, the new-password form); otherwise the sign-in form.
async function renderLocalSignIn(config, invite, message = "") {
  const info = invite ? await api(`/api/auth/invite/${encodeURIComponent(invite)}`).catch((error) => ({ valid: false, reason: error.message })) : null;
  if (info && !info.valid) {
    try { sessionStorage.removeItem("gymhub:invite"); } catch {}
    message = message || info.reason;
  }
  const joining = info && info.valid;
  const nameField = (fixed) => fixed
    ? `<label>Name to sign in with <input name="username" value="${esc(fixed)}" readonly></label>`
    : `<label>Name to sign in with <input name="username" autocomplete="username" autocapitalize="none" spellcheck="false" required
         pattern="[a-z0-9][a-z0-9._\\-]{1,31}" title="2 to 32 lowercase letters, digits, dots, dashes or underscores"></label>`;
  const form = !joining
    ? `<form id="local-form" class="signin-form">
         ${nameField(null)}
         <label>Password <input name="password" type="password" autocomplete="current-password" required></label>
         <button class="primary">Sign in</button>
       </form>`
    : info.reset
      ? `<p class="invite-note">Pick a new password for <b>${esc(info.for)}</b>.</p>
         <form id="local-form" class="signin-form">${nameField(info.for)}
           <label>New password <input name="password" type="password" autocomplete="new-password" minlength="8" required></label>
           <button class="primary">Set password and sign in</button></form>`
      : `<p class="invite-note">You have an invite${info.role === "admin" ? " to be an admin of this board" : ""}. Create your account.</p>
         <form id="local-form" class="signin-form">
           <label>Your name <input name="name" autocomplete="name" placeholder="shown to others on this board"></label>
           ${nameField(info.for)}
           <label>Password <input name="password" type="password" autocomplete="new-password" minlength="8" required></label>
           <button class="primary">Create account</button></form>`;
  view.innerHTML = `
    <section class="signin landing-hero">
      <div class="signin-card">
        <span class="logo big"></span>
        <h1>Human Gym</h1>
        <p class="muted">Practice GPU kernels, LLM internals and algorithms, with tests and benchmarks on real hardware.</p>
        ${form}
        <p class="signin-error" id="signin-error">${esc(message)}</p>
        <p class="muted small">${joining ? "Your name and password are kept by this board only." : "Invite-only. To join, or if you forgot your password, ask the board's admin for a link."}</p>
      </div>
    </section>
    ${landingSections(config)}`;
  renderShowcase();
  document.getElementById("local-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(event.target));
    const button = event.target.querySelector("button");
    button.disabled = true;
    try {
      if (joining) {
        await api("/api/auth/register", { method: "POST", body: JSON.stringify({ ...data, name: data.name || "", invite }) });
        try { sessionStorage.removeItem("gymhub:invite"); } catch {}
      } else {
        await api("/api/auth/password", { method: "POST", body: JSON.stringify(data) });
      }
      document.body.classList.remove("signed-out");
      boot();
    } catch (error) {
      document.getElementById("signin-error").textContent = error.message;
      button.disabled = false;
    }
  });
}

// What a signed-out visitor sees below the sign-in card: what is here and how it works.
function landingSections(config) {
  const source = config.source_url || "";
  return `
    <section class="landing">
      <h2>What you can practise</h2>
      <div class="landing-gyms" id="landing-gyms"><p class="muted">Loading the gyms on this board…</p></div>
    </section>
    <section class="landing">
      <h2>How it works</h2>
      <ol class="landing-steps">
        <li><b>Pick an exercise.</b> Each one has a statement, hints and a starter: a CUDA, Triton or PyTorch kernel, an LLM building block, an algorithm, or a question.</li>
        <li><b>Write it in the browser.</b> A code editor with completions, on a laptop or a phone. <i>Run</i> tries your own test cases; <i>Submit</i> runs the full tests.</li>
        <li><b>Get graded on real hardware.</b> Code runs in an isolated container on a machine with the right GPU or compiler. Kernels are benchmarked against the hardware's limits and PyTorch.</li>
        <li><b>Keep it.</b> Progress, best times, spaced review of what you solved, and optional leaderboards.</li>
      </ol>
    </section>
    <section class="landing landing-foot">
      <p class="muted small">Human Gym is open source (MIT). You can run your own board on your own machines, or write a gym in one Markdown file.
        ${source ? `<a href="${esc(source)}" target="_blank" rel="noopener">Source and docs</a>` : ""}</p>
    </section>`;
}

async function renderShowcase() {
  const slot = document.getElementById("landing-gyms");
  if (!slot) return;
  const data = await api("/api/showcase").catch(() => null);
  if (!document.body.contains(slot)) return;
  if (!data || !data.gyms.length) {
    slot.innerHTML = `<p class="muted">GPU kernels in CUDA, Triton and PyTorch; LLM internals in PyTorch; algorithms in Python and C++; ML and LLM fundamentals.</p>`;
    return;
  }
  const total = data.total_gyms > 1 ? `<p class="muted small landing-total">${data.total_exercises} exercises in ${data.total_gyms} gyms.</p>` : "";
  slot.innerHTML = data.gyms.map((gym) => `
    <div class="landing-gym">
      <div class="landing-gym-head"><b>${esc(gym.title)}</b>${gym.gpu ? ` <span class="badge">NVIDIA GPU</span>` : ""}</div>
      ${gym.description ? `<p class="muted small">${esc(gym.description)}</p>` : ""}
      <p class="small landing-gym-meta">${gym.problems} exercise${gym.problems === 1 ? "" : "s"}${gym.langs.length ? ` · ${gym.langs.map(esc).join(", ")}` : ""}</p>
    </div>`).join("") + total;
}

function renderNoAccess(email, message) {
  clearTimers();
  document.body.classList.add("signed-out");
  view.innerHTML = `
    <section class="signin">
      <div class="signin-card">
        <span class="logo big"></span>
        <h1>Human Gym</h1>
        <p>You are signed in as <b>${esc(email || "an unknown account")}</b>.</p>
        <p class="signin-error">${esc(message || "This board is invite-only.")}</p>
        <p class="muted small">Ask the board's admin for an invite link and open it while signed in with this account.</p>
        <p><a class="button" href="/_gcp_iap/clear_login_cookie">Use a different Google account</a></p>
      </div>
    </section>`;
}

function loadScript(src) {
  if (document.querySelector(`script[src="${src}"]`)) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.appendChild(script);
  });
}

function renderUserMenu(me) {
  const user = me.user;
  const slot = document.getElementById("user");
  const avatar = user.picture
    ? `<img class="avatar" src="${esc(user.picture)}" alt="" referrerpolicy="no-referrer">`
    : `<span class="avatar letter">${esc((user.name || user.email || "?")[0].toUpperCase())}</span>`;
  slot.innerHTML = `<details class="menu user-menu"><summary title="${esc(user.email)}">${avatar}</summary><div>
      <div class="user-who"><b>${esc(user.name || user.email)}</b><span class="muted small">${esc(user.email)} · ${esc(user.role)}</span>${usageLine()}</div>
      ${user.is_admin ? `<a href="#/admin">Admin</a>` : ""}
      ${me.auth !== "dev" ? `<a href="#/account">Your data</a>` : ""}
      ${me.auth === "app" || me.auth === "local" ? `<button id="sign-out">Sign out</button>` : ""}
      ${me.auth === "iap" ? `<a href="/_gcp_iap/clear_login_cookie">Switch Google account</a>` : ""}
    </div></details>`;
  const out = document.getElementById("sign-out");
  if (out)
    out.addEventListener("click", async () => {
      await api("/api/auth/logout", { method: "POST" }).catch(() => {});
      state.me = null;
      slot.innerHTML = "";
      try { google.accounts.id.disableAutoSelect(); } catch {}
      renderSignIn("You are signed out.");
    });
  slot.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => slot.querySelector("details").removeAttribute("open")));
}

// The report form opens under the gym's header; an admin sees it on the Admin page.
async function reportGym(gym, button) {
  const old = document.getElementById("report-form");
  if (old) return old.remove();
  const { reasons } = await api("/api/report-reasons");
  const form = document.createElement("form");
  form.id = "report-form";
  form.className = "panel admin-form";
  form.innerHTML = `<h3>Report "${esc(gym.title)}"</h3>
    <p class="muted small">Only this board's admins see reports, with your email. They can hide or remove the gym.</p>
    <label class="wide">What is wrong <select name="reason">${reasons.map((r) => `<option value="${esc(r.id)}">${esc(r.label)}</option>`).join("")}</select></label>
    <label class="wide">Details (optional unless you chose something else) <textarea name="details" rows="3" maxlength="2000" placeholder="Which exercise, and what you saw"></textarea></label>
    <button class="primary">Send report</button> <button type="button" class="ghost" data-cancel>Cancel</button> <span class="muted small" data-note></span>`;
  button.closest(".gym-intro").after(form);
  form.querySelector("[data-cancel]").addEventListener("click", () => form.remove());
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(form);
    try {
      await api(`/api/gyms/${encodeURIComponent(gym.id)}/report`, { method: "POST", body: JSON.stringify({ reason: data.get("reason"), details: data.get("details") || "" }) });
      form.innerHTML = `<p>Thanks. The board's admins will look at it.</p>`;
      button.disabled = true;
      button.textContent = "Reported";
    } catch (error) {
      form.querySelector("[data-note]").textContent = error.message;
    }
  });
}

function reportsPanel(gymReports) {
  if (!gymReports.reports.length) return "";
  const reason = (id) => gymReports.reasons[id] || id;
  return `<section class="panel"><h3>Reported gyms</h3>
    <p class="muted small">Hide makes a gym private and keeps it there until an admin makes it public again: its owner and editors still see it.
      Remove deletes it for everyone. Dismiss closes the reports and leaves the gym as it is.</p>
    <table class="table"><thead><tr><th>gym</th><th>reports</th><th>owner</th><th>now</th><th></th></tr></thead><tbody>
    ${gymReports.reports.map((g) => `<tr data-gym="${esc(g.id)}"><td><a href="#/gym/${encodeURIComponent(g.id)}">${esc(g.title)}</a>
        <ul class="small">${g.reports.map((r) => `<li><b>${esc(reason(r.reason))}</b>${r.details ? `: ${esc(r.details)}` : ""}
          <span class="muted">${esc(r.by || "")} ${r.at ? ago(r.at) : ""}</span></li>`).join("")}</ul></td>
        <td>${Number(g.count) || 0}</td><td>${esc(g.owner)}</td><td>${esc(g.hidden ? "hidden" : g.visibility)}</td>
        <td class="row-actions">${g.hidden ? "" : `<button class="small-button" data-moderate="hide">Hide</button> `}${g.uploaded ? `<button class="small-button ghost" data-moderate="remove">Remove</button> ` : ""}<button class="small-button ghost" data-moderate="dismiss">Dismiss</button></td></tr>`).join("")}
    </tbody></table></section>`;
}

// ---------- your data: export and account deletion ------------------------------------------------

async function renderAccount() {
  setCrumbs([["Your data"]]);
  const info = await api("/api/me/deletion");
  const gyms = info.gyms.map((gym) => {
    const people = gym.learners + gym.editors;
    return `<li><b>${esc(gym.title)}</b> <span class="muted small">${esc(gym.id)} · ${esc(gym.visibility)}${people ? ` · ${people} other ${people === 1 ? "person" : "people"}` : ""}</span></li>`;
  }).join("");
  const machines = info.machines.map((machine) => `<li>${esc(machine.name)}</li>`).join("");
  view.innerHTML = `
    <section class="panel account">
      <h2>Your data</h2>
      <p class="muted">What the board keeps about you, and why, is in the <a href="/privacy">privacy policy</a>.</p>
      <h3>Download a copy</h3>
      <p>One JSON file with your profile, progress, saved code, submissions and their output, the gyms you own or were invited to, your machines and tokens (without their secrets), and your usage.</p>
      <p><a class="button" id="export-data" href="/api/me/export" download>Download my data</a></p>
    </section>
    ${state.me?.auth === "local" ? `<section class="panel account">
      <h3>Change your password</h3>
      <form id="password-form" class="admin-form">
        <label>Current password <input name="old" type="password" autocomplete="current-password" required></label>
        <label>New password <input name="new" type="password" autocomplete="new-password" minlength="8" required></label>
        <button class="primary">Change password</button> <span id="password-saved" class="muted small"></span>
      </form>
    </section>` : ""}
    <section class="panel account danger-zone">
      <h3>Delete your account</h3>
      <p>This deletes your profile, progress, saved code, submissions, API tokens and usage records now. It cannot be undone. You can sign in again later only with a new invite, as a new member.</p>
      ${gyms ? `<p><b>Gyms you own are deleted for everyone</b>, including their versions. To keep one, transfer it to someone else first (on the gym's page):</p><ul>${gyms}</ul>` : ""}
      ${machines ? `<p><b>Machines you added are disconnected</b> and stop running jobs:</p><ul>${machines}</ul>` : ""}
      ${info.blocked
        ? `<p class="signin-error">${esc(info.blocked)}</p>`
        : `<label class="wide">Type your ${state.me?.auth === "local" ? "name" : "email"} (${esc(info.email)}) to confirm
             <input id="delete-confirm" autocomplete="off" spellcheck="false"></label>
           <p><button class="danger" id="delete-account" disabled>Delete my account</button> <span id="delete-error" class="signin-error"></span></p>`}
    </section>`;
  document.getElementById("password-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const saved = document.getElementById("password-saved");
    try {
      await api("/api/me/password", { method: "POST", body: JSON.stringify(Object.fromEntries(new FormData(event.target))) });
      event.target.reset();
      saved.textContent = "Changed.";
    } catch (error) {
      saved.textContent = error.message;
    }
  });
  const input = document.getElementById("delete-confirm");
  const button = document.getElementById("delete-account");
  if (!input) return;
  input.addEventListener("input", () => {
    button.disabled = input.value.trim().toLowerCase() !== info.email.toLowerCase();
  });
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api("/api/me", { method: "DELETE", body: JSON.stringify({ confirm: input.value }) });
    } catch (error) {
      document.getElementById("delete-error").textContent = error.message;
      button.disabled = false;
      return;
    }
    state.me = null;
    document.getElementById("user").innerHTML = "";
    try { google.accounts.id.disableAutoSelect(); } catch {}
    history.replaceState(null, "", "#/");
    renderSignIn("Your account and your data were deleted.");
  });
}

async function renderAdmin() {
  setCrumbs([]);
  if (!state.me?.user.is_admin) {
    view.innerHTML = `<section class="panel"><p>Only admins can see this page.</p></section>`;
    return;
  }
  const [users, invites, gymRequests, gymReports] = await Promise.all([
    api("/api/admin/users"), api("/api/admin/invites"), api("/api/admin/gym-requests"), api("/api/admin/gym-reports"),
  ]);
  const settings = users.settings;
  const base = `${location.origin}${location.pathname}#/invite/`;
  const local = state.me.auth === "local";
  const userRows = users.users
    .map(
      (user) => `<tr data-uid="${esc(user.uid)}">
        <td>${user.picture ? `<img class="avatar small" src="${esc(user.picture)}" alt="" referrerpolicy="no-referrer">` : ""} ${esc(user.name || "")}</td>
        <td>${esc(user.email)}</td>
        <td><select data-field="role"><option ${user.role === "member" ? "selected" : ""}>member</option><option ${user.role === "admin" ? "selected" : ""}>admin</option></select></td>
        <td><select data-field="status"><option ${user.status === "active" ? "selected" : ""}>active</option><option ${user.status === "disabled" ? "selected" : ""}>disabled</option></select></td>
        <td class="muted small">${esc(user.joined_by || "")}</td>
        <td class="muted small">${user.role === "admin" ? "no limits" : `${Math.round((user.usage?.seconds_today || 0) / 60)} min today · ${user.usage?.jobs_this_hour || 0} runs this hour`}</td>
        <td class="muted small">${user.last_login ? ago(user.last_login) : "never"}</td>
        ${local ? `<td class="row-actions">${user.uid.startsWith("l_") ? `<button class="small-button ghost" data-reset>Password reset link</button>` : ""}</td>` : ""}</tr>`
    )
    .join("");
  const inviteRows = invites.invites
    .map((invite) => {
      const expired = invite.expires_at < Date.now() / 1000;
      const used = invite.uses_left <= 0;
      const state_ = expired ? "expired" : used ? "used up" : `${invite.uses_left} of ${invite.uses_total} left`;
      return `<tr data-code="${esc(invite.code)}">
        <td><input class="invite-link" readonly value="${esc(base + invite.code)}"></td>
        <td>${esc(invite.email || "anyone")}</td><td class="muted small">${esc(invite.note || "")}</td>
        <td>${esc(state_)}</td><td class="muted small">${expired ? "expired" : `expires ${new Date(invite.expires_at * 1000).toLocaleDateString()}`}</td>
        <td class="row-actions"><button class="small-button" data-copy>Copy</button> <button class="small-button ghost" data-revoke>Revoke</button></td></tr>`;
    })
    .join("");
  view.innerHTML = `
    <h2 class="page-title">Admin</h2>
    <section class="panel" id="health-panel"><h3>Health</h3><p class="muted small">Loading…</p></section>
    <p class="muted small">Gym uploads and every member's versions are on the <a href="#/gyms">Gyms</a> page.</p>
    <section class="panel"><h3>Invite someone</h3>
      <form id="invite-form" class="admin-form">
        ${local
          ? `<label>Name <input name="email" autocapitalize="none" spellcheck="false" placeholder="optional: the account name it creates"></label>
             <label>Role <select name="role"><option>member</option><option>admin</option></select></label>`
          : `<label>Email <input name="email" type="email" placeholder="optional: only this account can use it"></label>`}
        <label>Uses <input name="uses" type="number" min="1" max="1000" value="1"></label>
        <label>Valid for (days) <input name="days" type="number" min="1" max="365" value="14"></label>
        <label>Note <input name="note" placeholder="optional"></label>
        <button class="primary">Create invite link</button>
      </form>
      <table class="table"><thead><tr><th>link</th><th>for</th><th>note</th><th>uses</th><th>expiry</th><th></th></tr></thead>
        <tbody>${inviteRows || `<tr><td colspan="6" class="muted">No invites yet.</td></tr>`}</tbody></table>
    </section>
    <section class="panel"><h3>Users</h3>
      <table class="table" id="users-table"><thead><tr><th>name</th><th>email</th><th>role</th><th>status</th><th>joined via</th><th>usage</th><th>last sign-in</th>${local ? "<th></th>" : ""}</tr></thead>
        <tbody>${userRows || `<tr><td colspan="7" class="muted">Nobody has signed in yet.</td></tr>`}</tbody></table>
      <p class="muted small">Always admins (set on the server): ${esc(users.admin_emails.join(", ") || "none")}</p>
    </section>
    <section class="panel"><h3>Limits for members</h3>
      <p class="muted small">Only running code counts (Run, Submit, Bench). Saving is never limited, and admins have no limits. 0 means no limit.</p>
      <form id="limits-form" class="admin-form">
        <label>Runs per hour <input name="jobs_per_hour" type="number" min="0" max="10000" value="${users.limits.jobs_per_hour}"></label>
        <label>Machine minutes per day (UTC) <input name="minutes_per_day" type="number" min="0" max="1440" value="${users.limits.minutes_per_day}"></label>
        <label>Jobs waiting at once <input name="max_waiting" type="number" min="0" max="100" value="${users.limits.max_waiting}"></label>
        <button class="primary">Save limits</button> <span id="limits-saved" class="muted small"></span>
      </form>
    </section>
    <section class="panel"><h3>AI spend</h3>
      <p class="muted small">${users.ai.enabled ? "" : "No Anthropic API key is set, so the tutor and essay grading are off. "}Everything the board spends on Claude this month (tutor and essay grading, admins included): <b>$${users.ai.spent.usd.toFixed(2)}</b> over ${users.ai.spent.requests} requests. At the cap, the tutor pauses and essays are self-marked until the 1st (UTC). 0 means no cap.</p>
      <form id="ai-form" class="admin-form">
        <label>Monthly cap (USD) <input name="monthly_usd" type="number" min="0" max="100000" step="0.01" value="${users.ai.monthly_usd}"></label>
        <button class="primary">Save cap</button> <span id="ai-saved" class="muted small"></span>
      </form>
    </section>
    <section class="panel"><h3>Access settings</h3>
      <form id="settings-form" class="admin-form">
        <label class="wide">Anyone with an email at these domains can sign in without an invite
          <input name="domains" placeholder="example.com, school.edu" value="${esc(settings.allowed_domains.join(", "))}"></label>
        <label class="check"><input type="checkbox" name="members_can_run" ${settings.members_can_run ? "checked" : ""}>
          Members can run code. Their jobs run on the compute pool, in containers: no network, no access to the
          machine owner's files, limited memory and processes. Their code and progress are kept separate.</label>
        <label class="check"><input type="checkbox" name="public_gyms_need_review" ${settings.public_gyms_need_review ? "checked" : ""}>
          An admin approves members' gyms before they become public (listed in Explore for everyone). On unless you turn it off.</label>
        <button class="primary">Save settings</button> <span id="settings-saved" class="muted small"></span>
      </form>
    </section>
    ${gymRequests.requests.length ? `<section class="panel"><h3>Gyms waiting to go public</h3>
      <table class="table"><thead><tr><th>gym</th><th>exercises</th><th>asked by</th><th>now</th><th></th></tr></thead><tbody>
      ${gymRequests.requests.map((r) => `<tr data-gym="${esc(r.id)}"><td><a href="#/gym/${encodeURIComponent(r.id)}">${esc(r.title)}</a>
          <div class="muted small">${esc(r.description || "")}</div></td><td>${Number(r.problems) || 0}</td>
          <td>${esc(r.requested_by || "")} <span class="muted small">${r.requested_at ? ago(r.requested_at) : ""}</span></td><td>${esc(r.visibility)}</td>
          <td class="row-actions"><button class="small-button" data-approve="1">Approve</button> <button class="small-button ghost" data-approve="0">Decline</button></td></tr>`).join("")}
      </tbody></table></section>` : ""}
    ${reportsPanel(gymReports)}`;
  view.querySelectorAll("[data-moderate]").forEach((button) =>
    button.addEventListener("click", async () => {
      const gymId = button.closest("tr").dataset.gym;
      const action = button.dataset.moderate;
      let note = "";
      if (action === "hide" && (note = prompt(`Hide ${gymId}? Its owner sees this note:`, "Hidden after reports")) === null) return;
      if (action === "remove" && prompt(`Delete ${gymId} for everyone? Learners keep their progress. Type its id to confirm.`) !== gymId) return;
      try {
        await api(`/api/admin/gym-reports/${encodeURIComponent(gymId)}`, { method: "POST", body: JSON.stringify({ action, note }) });
        renderAdmin();
      } catch (error) {
        alert(error.message);
      }
    }),
  );
  view.querySelectorAll("[data-approve]").forEach((button) =>
    button.addEventListener("click", async () => {
      const gymId = button.closest("tr").dataset.gym;
      try {
        await api(`/api/admin/gym-requests/${encodeURIComponent(gymId)}`, { method: "POST", body: JSON.stringify({ approve: button.dataset.approve === "1" }) });
        renderAdmin();
      } catch (error) {
        alert(error.message);
      }
    }),
  );

  renderHealth().catch(() => {});
  document.getElementById("invite-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    try {
      const invite = await api("/api/admin/invites", {
        method: "POST",
        body: JSON.stringify({ email: form.get("email") || null, role: form.get("role") || "member", uses: Number(form.get("uses")), days: Number(form.get("days")), note: form.get("note") || "" }),
      });
      await navigator.clipboard?.writeText(base + invite.code).catch(() => {});
      renderAdmin();
    } catch (error) {
      alert(error.message);
    }
  });
  view.querySelectorAll("tr[data-code]").forEach((row) => {
    row.querySelector("[data-copy]").addEventListener("click", async () => {
      const input = row.querySelector(".invite-link");
      input.select();
      await navigator.clipboard?.writeText(input.value).catch(() => document.execCommand("copy"));
      row.querySelector("[data-copy]").textContent = "Copied";
    });
    row.querySelector("[data-revoke]").addEventListener("click", async () => {
      if (!confirm("Revoke this invite link?")) return;
      await api(`/api/admin/invites/${encodeURIComponent(row.dataset.code)}`, { method: "DELETE" });
      renderAdmin();
    });
  });
  view.querySelectorAll("#users-table [data-reset]").forEach((button) =>
    button.addEventListener("click", async () => {
      const uid = button.closest("tr").dataset.uid;
      try {
        const reset = await api(`/api/admin/users/${encodeURIComponent(uid)}/password-reset`, { method: "POST" });
        await navigator.clipboard?.writeText(base + reset.code).catch(() => {});
        renderAdmin();
      } catch (error) {
        alert(error.message);
      }
    })
  );
  view.querySelectorAll("#users-table select").forEach((select) =>
    select.addEventListener("change", async () => {
      const uid = select.closest("tr").dataset.uid;
      try {
        await api(`/api/admin/users/${encodeURIComponent(uid)}`, { method: "POST", body: JSON.stringify({ [select.dataset.field]: select.value }) });
      } catch (error) {
        alert(error.message);
        renderAdmin();
      }
    })
  );
  document.getElementById("limits-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    try {
      await api("/api/admin/limits", {
        method: "PUT",
        body: JSON.stringify({
          jobs_per_hour: Number(form.get("jobs_per_hour")),
          minutes_per_day: Number(form.get("minutes_per_day")),
          max_waiting: Number(form.get("max_waiting")),
        }),
      });
      document.getElementById("limits-saved").textContent = "Saved.";
    } catch (error) {
      alert(error.message);
    }
  });
  document.getElementById("ai-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    try {
      await api("/api/admin/ai", { method: "PUT", body: JSON.stringify({ monthly_usd: Number(form.get("monthly_usd")) }) });
      document.getElementById("ai-saved").textContent = "Saved.";
    } catch (error) {
      alert(error.message);
    }
  });
  document.getElementById("settings-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    try {
      await api("/api/admin/settings", {
        method: "PUT",
        body: JSON.stringify({
          allowed_domains: String(form.get("domains") || "").split(/[\s,]+/).filter(Boolean),
          members_can_run: form.get("members_can_run") === "on",
          public_gyms_need_review: form.get("public_gyms_need_review") === "on",
        }),
      });
      document.getElementById("settings-saved").textContent = "Saved.";
      const me = await api("/api/me");
      state.me = me;
    } catch (error) {
      alert(error.message);
    }
  });
}

// ---------- health (admin) ---------------------------------------------------------------------------

async function renderHealth() {
  const host = document.getElementById("health-panel");
  if (!host) return;
  const h = await api("/api/admin/health");
  const sec = (value) => (value == null ? "-" : value < 1 ? `${Math.round(value * 1000)} ms` : value < 120 ? `${value.toFixed(1)} s` : `${(value / 60).toFixed(1)} min`);
  const pct = (value) => (value == null ? "-" : `${(value * 100).toFixed(1)}%`);
  const label = (w) => (w.hours === 1 ? "last hour" : w.hours % 24 === 0 ? `last ${w.hours === 24 ? "day" : `${w.hours / 24} days`}` : `last ${w.hours} h`);
  const windows = h.windows;
  const row = (name, cell) => `<tr><th>${name}</th>${windows.map((w) => `<td>${cell(w)}</td>`).join("")}</tr>`;
  const spread = (s) => (s.n ? `${sec(s.p50)} <span class="muted small">p95 ${sec(s.p95)}</span>` : "-");
  const metrics = [
    row("jobs", (w) => `${w.jobs} <span class="muted small">${esc(Object.entries(w.by_action).map(([a, n]) => `${n} ${a}`).join(", "))}</span>`),
    row("queue wait, p50", (w) => spread(w.queue_wait_s)),
    row("run time, p50", (w) => spread(w.run_s)),
    row("error rate", (w) => (w.finished ? `${pct(w.error_rate)} <span class="muted small">${w.errors} of ${w.finished}</span>` : "-")),
    row("cancelled", (w) => `${w.cancelled}`),
  ].join("");
  const ids = [...new Set(windows.flatMap((w) => Object.keys(w.machines)))];
  const machines = ids
    .map((id) => {
      const first = windows.map((w) => w.machines[id]).find(Boolean);
      const cells = windows
        .map((w) => {
          const m = w.machines[id];
          if (!m) return `<td class="muted">-</td>`;
          const online = m.online_s == null ? "online time unknown" : `${sec(m.busy_s)} busy of ${sec(m.online_s)} online`;
          return `<td><b>${pct(m.utilization)}</b> <span class="muted small">${online}; ${m.jobs} jobs, ${m.errors} errors</span></td>`;
        })
        .join("");
      return `<tr><td>${esc(first.name)}</td><td class="muted small">${esc(first.hardware || "")}</td>${cells}</tr>`;
    })
    .join("");
  const heads = windows.map((w) => `<th>${label(w)}</th>`).join("");
  host.innerHTML = `<h3>Health</h3>
    <div class="health-grid">
      <div><b>${h.queued_now}</b><span>waiting now</span></div>
      <div><b>${h.running_now}</b><span>running now</span></div>
    </div>
    <table class="table health-table"><thead><tr><th></th>${heads}</tr></thead><tbody>${metrics}</tbody></table>
    ${machines ? `<h4>Machine utilization <span class="muted small">(busy time / online time)</span></h4>
      <table class="table health-table"><thead><tr><th>machine</th><th>hardware</th>${heads}</tr></thead><tbody>${machines}</tbody></table>` : ""}
    <p class="muted small">From the newest ${h.sampled} jobs${h.complete ? "" : " (fewer than all jobs of the period: the numbers cover only those)"}. Error rate counts machine jobs that ended in error. Every step of a job is also logged: in Cloud Logging, filter on jsonPayload.event="job_finished" (or job_created, job_leased, job_requeued).</p>`;
}

// ---------- uploaded gyms (admin) -------------------------------------------------------------

const VERSION_BADGE = { validating: "wait", published: "good", failed: "bad" };

function versionDetails(version) {
  const checks = version.checks || [];
  const failed = checks.filter((check) => !check.ok);
  const parts = [];
  if (version.status === "validating") parts.push(`<p class="muted small">A machine is checking this version…</p>`);
  if (checks.length) parts.push(`<p class="small">${checks.length - failed.length} of ${checks.length} checks passed.</p>`);
  if ((version.errors || []).length)
    parts.push(`<ul class="version-errors">${version.errors.map((error) => `<li>${esc(error)}</li>`).join("")}</ul>`);
  if ((version.warnings || []).length)
    parts.push(`<ul class="version-warnings muted small">${version.warnings.map((warning) => `<li>${esc(warning)}</li>`).join("")}</ul>`);
  return parts.join("");
}

async function loadVersions() {
  const box = document.getElementById("versions");
  if (!box) return false;
  const { versions } = await api("/api/gym-versions");
  if (!versions.length) {
    box.innerHTML = `<p class="muted small">No gyms uploaded yet.</p>`;
    return false;
  }
  box.innerHTML = `<table class="table versions-table"><thead><tr><th>gym</th><th>version</th><th>status</th><th>exercises</th><th>uploaded</th><th></th></tr></thead><tbody>
    ${versions
      .map(
        (version) => `<tr>
          <td>${version.status === "published" && version.live ? `<a href="#/gym/${encodeURIComponent(version.gym)}">${esc(version.title || version.gym)}</a>` : esc(version.title || version.gym)}<div class="muted small">${esc(version.gym)}</div></td>
          <td>v${version.version}${version.live ? ` <span class="badge good">live</span>` : ""}</td>
          <td><span class="badge ${VERSION_BADGE[version.status] || ""}">${esc(version.status)}</span></td>
          <td>${version.exercises ?? ""}</td>
          <td class="muted small">${esc(version.created_by || "")}<br>${ago(version.created_at)}</td>
          <td class="version-detail">${versionDetails(version)}</td></tr>`
      )
      .join("")}</tbody></table>`;
  return versions.some((version) => version.status === "validating");
}

function setupGymUploads() {
  let timer = null;
  const refresh = async () => {
    try {
      const pending = await loadVersions();
      if (pending) timer = setTimeout(refresh, 3000);
    } catch (error) {
      const box = document.getElementById("versions");
      if (box) box.innerHTML = `<p class="error small">${esc(error.message)}</p>`;
    }
  };
  state.disposers.push(() => clearTimeout(timer));
  refresh();
  document.getElementById("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = event.target.bundle.files[0];
    const result = document.getElementById("upload-result");
    const button = event.target.querySelector("button");
    if (!file) return;
    button.disabled = true;
    result.innerHTML = `<p class="muted small">Uploading ${esc(file.name)}…</p>`;
    try {
      const reply = await api("/api/gym-versions", { method: "POST", body: file,
        headers: { "Content-Type": /\.md$/i.test(file.name) ? "text/markdown" : "application/gzip" } });
      const version = reply.version;
      result.innerHTML = version.duplicate
        ? `<p class="small">This exact bundle is already v${version.version} (${esc(version.status)}).</p>`
        : `<p class="small">Uploaded ${esc(version.gym)} v${version.version}. Validation is queued.</p>` +
          (reply.warning ? `<p class="warning small">${esc(reply.warning)}</p>` : "");
      event.target.reset();
      clearTimeout(timer);
      refresh();
    } catch (error) {
      result.innerHTML = `<p class="error small">${esc(error.message)}</p>` +
        (error.errors?.length ? `<ul class="version-errors">${error.errors.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : "");
    } finally {
      button.disabled = false;
    }
  });
}

boot();
