"use strict";
// VS Code-style workspace for one gym: explorer, preview + code editor groups, output panel, status bar.
// Uses helpers from app.js (api, esc, md, ago, statusBadge, langState, actionLabel, loadMonaco,
// readDraft, writeDraft, state). Loaded before app.js; only called after it has run.

const IDE_GYM_ICONS = { gpu: "GPU", llm: "LLM", algo: "ALG" };
const FILE_ICONS = { cuda: "CU", cpp: "C++", pytorch: "PY", python: "PY", triton: "TR", md: "M↓", hints: "?", editorial: "✎" };

async function renderIde(gymId, slug) {
  document.body.classList.add("ide-mode");
  const ide = {
    gymId,
    slug,
    gym: null,
    problems: [],
    progress: { exercises: {} },
    cache: new Map(), // slug -> {problem, progress, jobs}
    expanded: new Set([slug]),
    codeTabs: [], // {id, slug, lang, name}
    active: null, // id of active code tab
    preview: "problem", // problem | hints
    hintsShown: new Map(),
    models: new Map(), // tab id -> monaco model
    viewStates: new Map(),
    activeJob: null,
    panelTab: "output",
    sidebar: true,
    panel: true,
    disposed: false,
  };
  state.ide = ide;
  state.cleanup = () => {
    ide.disposed = true;
    document.body.classList.remove("ide-mode");
    window.removeEventListener("keydown", onKey, true);
    for (const model of ide.models.values()) model.dispose();
    ide.cases?.dispose();
  };

  const overview = await api(`/api/gyms/${encodeURIComponent(gymId)}`);
  ide.gym = overview.gym;
  ide.problems = overview.problems;
  ide.progress = overview.progress || { exercises: {} };
  let gyms = [];
  try {
    gyms = (await api("/api/overview")).gyms;
  } catch {}
  const langMeta = Object.fromEntries((ide.gym.langs || []).map((lang) => [lang.id, lang]));

  view.innerHTML = `
    <div class="ide">
      <div class="ide-title">
        <a href="#/" class="ide-home" title="Control board">◆ Human Gym</a>
        <div class="ide-title-center" id="ide-title"></div>
        <button id="ide-simple" class="ide-switch" title="Back to the simple editor (remembered in this browser)">Simple view</button>
        <span id="ide-runner" class="ide-title-right"></span>
      </div>
      <div class="ide-body">
        <nav class="ide-activity">
          ${gyms
            .map(
              (gym) => `<a href="#/gym/${esc(gym.id)}" class="ide-act ${gym.id === gymId ? "active" : ""}" data-gym="${esc(gym.id)}" title="${esc(gym.title)}">${esc(IDE_GYM_ICONS[gym.id] || gym.id.slice(0, 3).toUpperCase())}</a>`
            )
            .join("")}
          <div class="spacer"></div>
          <button class="ide-act" id="ide-toggle-sidebar" title="Toggle sidebar (Ctrl+B)">☰</button>
          <a class="ide-act" href="#/" title="Control board">⌂</a>
        </nav>
        <aside class="ide-sidebar" id="ide-sidebar">
          <div class="ide-side-head">EXPLORER</div>
          <div class="ide-side-section">${esc((ide.gym.title || gymId).toUpperCase())}</div>
          <div class="ide-tree" id="ide-tree"></div>
        </aside>
        <div class="ide-resizer v" data-resize="sidebar"></div>
        <section class="ide-main">
          <div class="ide-groups" id="ide-groups">
            <div class="ide-group preview" id="ide-preview-group">
              <div class="ide-tabs" id="ide-preview-tabs"></div>
              <div class="ide-preview markdown" id="ide-preview"></div>
            </div>
            <div class="ide-resizer v" data-resize="preview"></div>
            <div class="ide-group code">
              <div class="ide-tabs" id="ide-code-tabs">
                <div class="ide-tablist" id="ide-tablist"></div>
                <div class="ide-actions">
                  <select id="ide-run-on" class="run-on hidden" aria-label="Run on"></select>
                  <button id="ide-run" title="Run (Ctrl+Enter): every case with its output, nothing saved or recorded">▷ Run</button>
                  <button id="ide-try" class="hidden">Try in browser</button>
                  <button id="ide-submit" class="primary" title="Submit (Ctrl+Shift+Enter): save, full tests, record progress">Submit</button>
                  <button id="ide-bench" title="Benchmark">Bench</button>
                  <details class="ide-more"><summary title="More actions">⋯</summary><div>
                    <button id="ide-save">Save to machine (Ctrl+S)</button>
                    <button id="ide-reload">Load the file from my machine</button>
                    <button id="ide-starter">Reset to starter</button>
                    <button id="ide-toggle-preview">Toggle problem pane</button>
                  </div></details>
                </div>
              </div>
              <div class="ide-crumbs" id="ide-crumbs"></div>
              <div class="ide-editor" id="ide-editor"></div>
            </div>
          </div>
          <div class="ide-resizer h" data-resize="panel"></div>
          <div class="ide-panel" id="ide-panel">
            <div class="ide-panel-tabs">
              <button data-panel="output" class="active">OUTPUT</button>
              <button data-panel="runs">RUNS</button>
              <button data-panel="cases">CASES <span id="ide-cases-count"></span></button>
              <div class="spacer"></div>
              <span id="ide-job" class="ide-job"></span>
              <button id="ide-cancel" class="hidden" title="Cancel the running job">■ Cancel</button>
              <button id="ide-panel-close" title="Toggle panel (Ctrl+J)">✕</button>
            </div>
            <pre class="ide-output" id="ide-output">Run (Ctrl+Enter) tries the code in the editor on every case and shows each output. Nothing is saved or recorded.
Submit (Ctrl+Shift+Enter) saves the file, runs the full tests, and records your progress.</pre>
            <div class="ide-runs hidden" id="ide-runs"></div>
            <div class="ide-cases cases hidden" id="ide-cases"></div>
          </div>
        </section>
      </div>
      <footer class="ide-status">
        <span id="st-gym"></span>
        <span id="st-runner"></span>
        <span id="st-job"></span>
        <div class="spacer"></div>
        <span id="st-pos"></span>
        <span id="st-lang"></span>
        <span id="st-solved"></span>
      </footer>
      <div class="ide-quick hidden" id="ide-quick">
        <input id="ide-quick-input" placeholder="Go to exercise or file…" autocomplete="off" spellcheck="false">
        <div id="ide-quick-list"></div>
      </div>
      <div class="ide-toasts" id="ide-toasts"></div>
    </div>`;

  const $ = (id) => document.getElementById(id);
  let tryButton = null; // "Try in browser" (browser-run.js), set up with the other buttons below

  // ---------- data ----------------------------------------------------------------------------

  async function loadProblem(target, force = false) {
    if (!force && ide.cache.has(target)) return ide.cache.get(target);
    const data = await api(`/api/gyms/${encodeURIComponent(gymId)}/problems/${encodeURIComponent(target)}`);
    ide.cache.set(target, data);
    if (data.progress) ide.progress.exercises[target] = data.progress;
    if (!ide.hintsShown.has(target)) ide.hintsShown.set(target, data.progress?.hints_seen || 0);
    return data;
  }

  const current = (target) => ide.cache.get(target)?.problem.current || {};
  const tabOf = (id) => ide.codeTabs.find((tab) => tab.id === id);
  const activeTab = () => tabOf(ide.active);

  // ---------- explorer ------------------------------------------------------------------------

  function renderTree() {
    const tracks = new Map();
    for (const problem of ide.problems) {
      if (!tracks.has(problem.track)) tracks.set(problem.track, []);
      tracks.get(problem.track).push(problem);
    }
    let html = "";
    for (const [track, problems] of tracks) {
      html += `<div class="ide-tree-track">${esc(track)}</div>`;
      for (const problem of problems) {
        const solved = (problem.langs || []).some((lang) => langState(ide.progress, problem.slug, lang).cls === "good");
        const tried = (problem.langs || []).some((lang) => langState(ide.progress, problem.slug, lang).cls === "bad");
        const mark = solved ? `<span class="good">✓</span>` : tried ? `<span class="bad">✗</span>` : `<span class="none">·</span>`;
        const open = ide.expanded.has(problem.slug);
        html += `<div class="ide-node exercise ${problem.slug === ide.slug ? "current" : ""}" tabindex="0" role="button" data-slug="${esc(problem.slug)}">
          <span class="twisty">${open ? "▾" : "▸"}</span>${mark}<span class="name">${esc(problem.slug)}</span>
          <span class="level ${esc(problem.difficulty)}">${esc(problem.difficulty[0].toUpperCase())}</span></div>`;
        if (open) {
          const files = [
            { kind: "problem", name: "PROBLEM.md", icon: FILE_ICONS.md },
            { kind: "hints", name: "HINTS.md", icon: FILE_ICONS.hints },
            ...(hasEditorial(problem.slug) ? [{ kind: "editorial", name: "EDITORIAL.md", icon: FILE_ICONS.editorial }] : []),
            ...(problem.langs || []).map((lang) => ({
              kind: "code",
              lang,
              name: langMeta[lang]?.file || lang,
              icon: FILE_ICONS[lang] || "··",
            })),
          ];
          for (const file of files) {
            const id = file.kind === "code" ? `${problem.slug}/${file.lang}` : null;
            const isActive =
              (file.kind === "code" && id === ide.active) ||
              (file.kind !== "code" && problem.slug === ide.slug && ide.preview === file.kind && ide.previewVisible !== false);
            const s = file.kind === "code" ? langState(ide.progress, problem.slug, file.lang) : null;
            html += `<div class="ide-node file ${isActive ? "active" : ""}" tabindex="0" role="button" data-slug="${esc(problem.slug)}" data-kind="${file.kind}" data-lang="${esc(file.lang || "")}">
              <span class="ficon ${esc(file.lang || file.kind)}">${esc(file.icon)}</span><span class="name">${esc(file.name)}</span>
              ${s ? `<span class="fmark ${s.cls}" title="${esc(s.title)}">${s.mark === "·" ? "" : s.mark}</span>` : ""}</div>`;
          }
        }
      }
    }
    $("ide-tree").innerHTML = html;
  }

  $("ide-tree").addEventListener("click", async (event) => {
    const node = event.target.closest(".ide-node");
    if (!node) return;
    const target = node.dataset.slug;
    if (node.classList.contains("exercise")) {
      if (ide.expanded.has(target) && target === ide.slug) ide.expanded.delete(target);
      else {
        ide.expanded.add(target);
        await openExercise(target);
      }
      renderTree();
      return;
    }
    if (node.dataset.kind === "code") await openCode(target, node.dataset.lang);
    else {
      await selectExercise(target);
      showPreview(node.dataset.kind);
    }
  });

  // ---------- preview group (problem / hints) -------------------------------------------------

  ide.previewVisible = true;

  function hasEditorial(target) {
    const problem = ide.cache.get(target)?.problem;
    return Boolean(problem && (problem.editorial || problem.editorial_locked));
  }

  function renderPreviewTabs() {
    const tabs = [
      ["problem", "PROBLEM.md", FILE_ICONS.md],
      ["hints", "HINTS.md", FILE_ICONS.hints],
      ...(hasEditorial(ide.slug) ? [["editorial", "EDITORIAL.md", FILE_ICONS.editorial]] : []),
    ];
    $("ide-preview-tabs").innerHTML =
      `<div class="ide-tablist">` +
      tabs
        .map(
          ([kind, name, icon]) =>
            `<div class="ide-tab ${ide.preview === kind ? "active" : ""}" data-preview="${kind}"><span class="ficon md">${icon}</span>${esc(name)}<span class="muted small"> · ${esc(ide.slug)}</span></div>`
        )
        .join("") +
      `</div><div class="ide-actions"><button id="ide-preview-close" title="Close the problem pane">✕</button></div>`;
  }

  function renderPreview() {
    const data = ide.cache.get(ide.slug);
    if (!data) return;
    if (ide.preview === "editorial" && !hasEditorial(ide.slug)) ide.preview = "problem";
    renderPreviewTabs();
    const problem = data.problem;
    if (ide.preview === "problem") {
      $("ide-preview").innerHTML = md(problem.statement);
    } else if (ide.preview === "editorial") {
      refreshEditorial($("ide-preview"), gymId, ide.slug, problem);
    } else {
      const hints = problem.hints || [];
      const shown = ide.hintsShown.get(ide.slug) || 0;
      $("ide-preview").innerHTML = hints.length
        ? hints
            .slice(0, shown)
            .map((hint, index) => `<div class="hint"><h4>Hint ${index + 1} of ${hints.length}</h4>${md(hint)}</div>`)
            .join("") +
          (shown < hints.length ? `<button id="ide-hint-next">Reveal hint ${shown + 1}</button>` : `<p class="muted">That was the last hint.</p>`)
        : `<p class="muted">This exercise has no hints.</p>`;
      const button = $("ide-hint-next");
      if (button)
        button.addEventListener("click", () => {
          const count = (ide.hintsShown.get(ide.slug) || 0) + 1;
          ide.hintsShown.set(ide.slug, count);
          renderPreview();
          api("/api/jobs", { method: "POST", body: JSON.stringify({ gym: gymId, slug: ide.slug, action: "hint", options: { count } }) }).catch(() => {});
        });
    }
    $("ide-preview").scrollTop = 0;
  }

  function showPreview(kind) {
    ide.preview = kind;
    setPreviewVisible(true);
    renderPreview();
    renderTree();
  }

  function setPreviewVisible(visible) {
    ide.previewVisible = visible;
    $("ide-preview-group").classList.toggle("hidden", !visible);
    document.querySelector('[data-resize="preview"]').classList.toggle("hidden", !visible);
  }

  $("ide-preview-tabs").addEventListener("click", (event) => {
    if (event.target.closest("#ide-preview-close")) {
      setPreviewVisible(false);
      renderTree();
      return;
    }
    const tab = event.target.closest("[data-preview]");
    if (tab) showPreview(tab.dataset.preview);
  });

  // ---------- code editor -----------------------------------------------------------------------

  const monaco = await loadMonaco();
  if (ide.disposed) return;
  registerCompletions(monaco);
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const editor = monaco.editor.create($("ide-editor"), {
    model: null,
    theme: dark ? "vs-dark" : "vs",
    automaticLayout: true,
    fixedOverflowWidgets: true,
    fontFamily: 'Menlo, "SF Mono", "JetBrains Mono", "DejaVu Sans Mono", Consolas, monospace',
    fontSize: 14,
    lineHeight: 21,
    suggestFontSize: 13,
    suggestLineHeight: 24,
    minimap: { enabled: true, scale: 1, renderCharacters: false },
    scrollBeyondLastLine: false,
    tabSize: 4,
    renderWhitespace: "selection",
    bracketPairColorization: { enabled: true },
    guides: { bracketPairs: true, indentation: true },
    stickyScroll: { enabled: true },
    smoothScrolling: true,
    cursorBlinking: "smooth",
    quickSuggestions: { other: true, comments: false, strings: false },
    wordBasedSuggestions: "allDocuments",
    suggest: { showWords: true, preview: true, snippetsPreventQuickSuggestions: false },
    snippetSuggestions: "top",
    acceptSuggestionOnEnter: "smart",
  });
  state.editor = editor;
  document.fonts?.ready.then(() => monaco.editor.remeasureFonts());

  function dirty(tab) {
    const model = ide.models.get(tab.id);
    return model && model.getValue() !== (current(tab.slug)[tab.lang] ?? "");
  }

  function renderCodeTabs() {
    $("ide-tablist").innerHTML = ide.codeTabs
      .map((tab) => {
        const isDirty = dirty(tab);
        return `<div class="ide-tab ${tab.id === ide.active ? "active" : ""}" data-tab="${esc(tab.id)}" title="${esc(tab.slug)}/${esc(tab.name)}">
          <span class="ficon ${esc(tab.lang)}">${esc(FILE_ICONS[tab.lang] || "··")}</span>${esc(tab.name)}
          <span class="muted small">${esc(tab.slug)}</span>
          <span class="ide-tab-close ${isDirty ? "dirty" : ""}" data-close="${esc(tab.id)}" title="${isDirty ? "Unsaved changes (kept as a draft)" : "Close"}">${isDirty ? "●" : "✕"}</span></div>`;
      })
      .join("");
    const tab = activeTab();
    $("ide-crumbs").innerHTML = tab
      ? `${esc(ide.gym.title)} <span>›</span> ${esc(tab.slug)} <span>›</span> ${esc(tab.name)}`
      : `<span class="muted">Open a solution file from the explorer.</span>`;
  }

  $("ide-tablist").addEventListener("click", (event) => {
    const close = event.target.closest("[data-close]");
    if (close) {
      closeTab(close.dataset.close);
      event.stopPropagation();
      return;
    }
    const tab = event.target.closest("[data-tab]");
    if (tab) activate(tab.dataset.tab);
  });
  $("ide-tablist").addEventListener("auxclick", (event) => {
    const tab = event.target.closest("[data-tab]");
    if (tab && event.button === 1) closeTab(tab.dataset.tab);
  });

  function closeTab(id) {
    const index = ide.codeTabs.findIndex((tab) => tab.id === id);
    if (index < 0) return;
    ide.codeTabs.splice(index, 1);
    // The draft stays in this browser; closing a tab never loses work.
    ide.models.get(id)?.dispose();
    ide.models.delete(id);
    ide.viewStates.delete(id);
    if (ide.active === id) {
      const next = ide.codeTabs[Math.min(index, ide.codeTabs.length - 1)];
      ide.active = null;
      if (next) activate(next.id);
      else editor.setModel(null);
    }
    renderCodeTabs();
    renderTree();
    updateStatus();
  }

  function modelFor(tab) {
    if (ide.models.has(tab.id)) return ide.models.get(tab.id);
    const machine = current(tab.slug)[tab.lang] ?? "";
    const draft = readDraft(gymId, tab.slug, tab.lang);
    const value = draft && draft.code !== machine ? draft.code : machine;
    if (!(draft && draft.code !== machine)) writeDraft(gymId, tab.slug, tab.lang, null);
    const uri = monaco.Uri.parse(`file:///${gymId}/${tab.slug}/${tab.name}`);
    const model = monaco.editor.createModel(value, langMeta[tab.lang]?.editor || "python", uri);
    model.onDidChangeContent(() => {
      if (model.__programmatic) return;
      const machineNow = current(tab.slug)[tab.lang] ?? "";
      const code = model.getValue();
      const existing = readDraft(gymId, tab.slug, tab.lang);
      writeDraft(gymId, tab.slug, tab.lang, code === machineNow ? null : { code, base: existing?.base ?? machineNow });
      renderCodeTabs();
    });
    ide.models.set(tab.id, model);
    if (draft && draft.code !== machine && draft.base !== machine) {
      toast(`${tab.slug}/${tab.name}: the file on your machine changed since you started this draft.`, [
        { label: "Use machine version", run: () => replaceWithMachine(tab) },
        { label: "Keep my draft", run: () => writeDraft(gymId, tab.slug, tab.lang, { code: model.getValue(), base: machine }) },
      ]);
    }
    return model;
  }

  function setModelValue(model, value) {
    model.__programmatic = true;
    model.pushEditOperations([], [{ range: model.getFullModelRange(), text: value }], () => null);
    model.__programmatic = false;
  }

  function replaceWithMachine(tab) {
    writeDraft(gymId, tab.slug, tab.lang, null);
    const model = ide.models.get(tab.id);
    if (model) setModelValue(model, current(tab.slug)[tab.lang] ?? "");
    renderCodeTabs();
  }

  function activate(id) {
    const tab = tabOf(id);
    if (!tab) return;
    if (ide.active && editor.getModel()) ide.viewStates.set(ide.active, editor.saveViewState());
    ide.active = id;
    editor.setModel(modelFor(tab));
    const view = ide.viewStates.get(id);
    if (view) editor.restoreViewState(view);
    editor.focus();
    if (tab.slug !== ide.slug) selectExercise(tab.slug);
    try { localStorage.setItem(`gymhub:lang:${gymId}`, tab.lang); } catch {}
    renderCodeTabs();
    renderTree();
    updateStatus();
    renderRuns();
    tryButton?.refresh();
    // Run on: the machines that can run this exercise in this language
    if (state.me?.can_run && ide.runOnFor !== `${tab.slug}/${tab.lang}`) {
      ide.runOnFor = `${tab.slug}/${tab.lang}`;
      fillRunOn($("ide-run-on"), gymId, tab.slug, tab.lang);
    }
  }

  async function openCode(target, lang) {
    await loadProblem(target);
    const id = `${target}/${lang}`;
    if (!tabOf(id)) ide.codeTabs.push({ id, slug: target, lang, name: langMeta[lang]?.file || lang });
    activate(id);
  }

  function mountExerciseCases() {
    ide.cases?.dispose();
    const problem = ide.cache.get(ide.slug)?.problem;
    if (!problem) return;
    ide.cases = mountCases($("ide-cases"), monaco, gymId, ide.slug, problem.case_template, (count) => {
      $("ide-cases-count").textContent = count ? `(${count})` : "";
    });
  }

  async function selectExercise(target) {
    await loadProblem(target);
    if (ide.slug !== target || !ide.cases) {
      ide.slug = target;
      mountExerciseCases();
      ide.expanded.add(target);
      history.replaceState(null, "", `#/gym/${encodeURIComponent(gymId)}/${encodeURIComponent(target)}`);
      renderPreview();
      renderRuns();
    }
    $("ide-title").textContent = `${target} — ${ide.gym.title}`;
    renderTree();
  }

  async function openExercise(target) {
    await selectExercise(target);
    const data = ide.cache.get(target);
    const langs = data.problem.langs || [];
    let lang = langs[0];
    try {
      const saved = localStorage.getItem(`gymhub:lang:${gymId}`);
      if (saved && langs.includes(saved)) lang = saved;
    } catch {}
    if (lang) await openCode(target, lang);
    renderPreview();
  }

  // ---------- jobs -----------------------------------------------------------------------------

  function renderJob(job) {
    const output = $("ide-output");
    const atBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 40;
    output.textContent = outputWithHost(job, job.status === "queued" ? ide.queueWarning || "Waiting for the runner…" : "");
    if (atBottom || job.status === "running" || job.status === "queued") output.scrollTop = output.scrollHeight;
    const elapsed = job.started_at ? ` · ${((job.finished_at || Date.now() / 1000) - job.started_at).toFixed(1)} s` : "";
    const mode = job.action === "run" ? " · not recorded" : "";
    $("ide-job").innerHTML = `${statusBadge(job)} ${esc(actionLabel(job.action))} · ${esc(job.slug || "")} · ${esc(job.lang)}${elapsed}${mode}`;
    const live = job.status === "queued" || job.status === "running";
    $("ide-cancel").classList.toggle("hidden", !live);
    for (const id of ["ide-run", "ide-submit", "ide-bench"]) $(id).disabled = live || !state.me?.can_run;
    $("st-job").innerHTML = live ? `<span class="spin">◌</span> ${esc(actionLabel(job.action))}…` : job.status ? `${job.ok ? "✓" : "✗"} ${esc(actionLabel(job.action))}` : "";
  }

  async function poll(jobId, tab) {
    if (ide.disposed) return;
    try {
      const job = await followJob(jobId, (update) => !ide.disposed && renderJob(update));
      if (ide.disposed) return;
      ide.activeJob = null;
      await refreshExercise(tab.slug, job.action === "reset" ? tab : null);
      if (job.action !== "run") {
        const fresh = await api(`/api/gyms/${encodeURIComponent(gymId)}`).catch(() => null);
        if (fresh) {
          ide.progress = fresh.progress;
          ide.problems = fresh.problems;
          renderTree();
        }
      }
    } catch (error) {
      $("ide-job").textContent = `Lost track of the job: ${error.message}`;
      state.pollTimer = setTimeout(() => poll(jobId, tab), 3000);
    }
  }

  async function submit(action) {
    const tab = activeTab();
    if (!tab || ide.activeJob) {
      if (!tab) toast("Open a solution file first.");
      return;
    }
    const body = { gym: gymId, slug: tab.slug, lang: tab.lang, action };
    if (["run", "test", "bench", "save"].includes(action)) body.code = ide.models.get(tab.id).getValue();
    const target = ["run", "test", "bench"].includes(action) && runOnTarget($("ide-run-on"));
    if (target) body.target = target;
    if (action === "run" && ide.cases && tab.slug === ide.slug) {
      const custom = ide.cases.read();
      if (custom.error) {
        showPanel("cases");
        toast(`Fix the custom cases first: ${custom.error}`);
        return;
      }
      if (custom.cases.length) body.options = { cases: custom.cases, only_custom: custom.only };
      else if (custom.only) {
        toast("Only my cases is ticked, but there are no custom cases.");
        return;
      }
    }
    showPanel("output");
    try {
      const { id, warning } = await api("/api/jobs", { method: "POST", body: JSON.stringify(body) });
      ide.activeJob = id;
      ide.queueWarning = warning || "";
      if (warning) toast(warning);
      renderJob({ status: "queued", action, lang: tab.lang, slug: tab.slug, output: "" });
      poll(id, tab);
    } catch (error) {
      $("ide-output").textContent = error.message;
    }
  }

  async function refreshExercise(target, resetTab = null) {
    const data = await loadProblem(target, true);
    for (const tab of ide.codeTabs.filter((t) => t.slug === target)) {
      const model = ide.models.get(tab.id);
      const machine = data.problem.current?.[tab.lang] ?? "";
      if (!model) continue;
      if (resetTab && resetTab.id === tab.id) setModelValue(model, machine);
      if (model.getValue() === machine) writeDraft(gymId, tab.slug, tab.lang, null);
      else if (!readDraft(gymId, tab.slug, tab.lang)) setModelValue(model, machine); // edited locally, no draft here
    }
    renderCodeTabs();
    renderRuns();
    updateStatus();
  }

  function renderRuns() {
    const data = ide.cache.get(ide.slug);
    const jobs = data?.jobs || [];
    $("ide-runs").innerHTML = jobs.length
      ? `<table class="table"><tbody>${historyRows(jobs)}</tbody></table>`
      : `<p class="muted">No submissions for ${esc(ide.slug)} yet. Runs are not listed: they are never recorded.</p>`;
  }

  $("ide-runs").addEventListener("click", async (event) => {
    const code = event.target.closest("[data-code]");
    if (code) {
      const slugNow = ide.slug;
      openSubmission(monaco, code.dataset.code, ide.cache.get(slugNow)?.jobs || [], {
        language: (id) => langMeta[id]?.editor || "python",
        current: (id) => ide.models.get(`${slugNow}/${id}`)?.getValue() ?? current(slugNow)[id] ?? null,
        restore: async (text, id) => {
          await openCode(slugNow, id);
          const model = ide.models.get(`${slugNow}/${id}`);
          editor.executeEdits("restore", [{ range: model.getFullModelRange(), text }]);
          editor.pushUndoStop();
          toast("Restored a past submission. It is an unsaved draft until you Save or Submit. Ctrl+Z undoes it.");
        },
      }).catch((error) => toast(error.message));
      return;
    }
    const row = event.target.closest("[data-job]");
    if (!row) return;
    const job = await api(`/api/jobs/${row.dataset.job}`);
    showPanel("output");
    renderJob(job);
  });

  // ---------- panel, sidebar, resizers ---------------------------------------------------------

  function showPanel(name) {
    ide.panelTab = name;
    ide.panel = true;
    $("ide-panel").classList.remove("hidden");
    document.querySelector('[data-resize="panel"]').classList.remove("hidden");
    document.querySelectorAll(".ide-panel-tabs [data-panel]").forEach((b) => b.classList.toggle("active", b.dataset.panel === name));
    $("ide-output").classList.toggle("hidden", name !== "output");
    $("ide-runs").classList.toggle("hidden", name !== "runs");
    $("ide-cases").classList.toggle("hidden", name !== "cases");
    if (name === "cases") ide.cases?.layout();
  }

  function togglePanel() {
    ide.panel = !ide.panel;
    $("ide-panel").classList.toggle("hidden", !ide.panel);
    document.querySelector('[data-resize="panel"]').classList.toggle("hidden", !ide.panel);
  }

  function toggleSidebar() {
    ide.sidebar = !ide.sidebar;
    $("ide-sidebar").classList.toggle("hidden", !ide.sidebar);
    document.querySelector('[data-resize="sidebar"]').classList.toggle("hidden", !ide.sidebar);
  }

  document.querySelector(".ide-panel-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("[data-panel]");
    if (button) showPanel(button.dataset.panel);
  });
  $("ide-panel-close").addEventListener("click", togglePanel);
  $("ide-toggle-sidebar").addEventListener("click", toggleSidebar);

  document.querySelectorAll(".ide-resizer").forEach((handle) => {
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      const kind = handle.dataset.resize;
      const target = { sidebar: $("ide-sidebar"), preview: $("ide-preview-group"), panel: $("ide-panel") }[kind];
      const start = kind === "panel" ? event.clientY : event.clientX;
      const size = kind === "panel" ? target.offsetHeight : target.offsetWidth;
      const move = (e) => {
        const delta = (kind === "panel" ? start - e.clientY : e.clientX - start);
        const value = Math.max(kind === "panel" ? 60 : 160, size + delta);
        target.style.flexBasis = `${value}px`;
      };
      const up = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
  });

  // ---------- status bar, toasts, quick open ----------------------------------------------------

  function updateStatus() {
    const tab = activeTab();
    $("st-gym").textContent = `⎇ ${gymId}`;
    const runners = state.runners || [];
    const runner = runners.find((r) => r.online);
    $("st-runner").innerHTML = runner ? `● ${esc(runner.name)}${gpuLabel(runner) ? ` · ${esc(gpuLabel(runner))}` : ""}${runner.busy ? " · busy" : ""}` : `<span class="warn">● runner offline</span>`;
    $("ide-runner").innerHTML = runner ? `<span class="good">●</span> ${esc(runner.name)}` : `<span class="bad">●</span> runner offline`;
    if (tab) {
      const position = editor.getPosition();
      $("st-pos").textContent = position ? `Ln ${position.lineNumber}, Col ${position.column}` : "";
      $("st-lang").textContent = langMeta[tab.lang]?.label || tab.lang;
      const s = langState(ide.progress, tab.slug, tab.lang);
      $("st-solved").textContent = `${s.mark} ${s.title}`;
    } else {
      $("st-pos").textContent = $("st-lang").textContent = $("st-solved").textContent = "";
    }
  }
  editor.onDidChangeCursorPosition(() => {
    const position = editor.getPosition();
    if (position) $("st-pos").textContent = `Ln ${position.lineNumber}, Col ${position.column}`;
  });

  function toast(message, actions = []) {
    const element = document.createElement("div");
    element.className = "ide-toast";
    element.innerHTML = `<span>${esc(message)}</span><div>${actions.map((a, i) => `<button data-i="${i}">${esc(a.label)}</button>`).join("")}<button data-i="x">✕</button></div>`;
    element.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      if (button.dataset.i !== "x") actions[Number(button.dataset.i)].run();
      element.remove();
    });
    $("ide-toasts").appendChild(element);
    if (!actions.length) setTimeout(() => element.remove(), 5000);
  }

  function quickOpen() {
    const items = [];
    for (const problem of ide.problems) {
      for (const lang of problem.langs || []) items.push({ label: `${problem.slug}/${langMeta[lang]?.file || lang}`, detail: problem.title, run: () => openCode(problem.slug, lang) });
      items.push({ label: `${problem.slug}/PROBLEM.md`, detail: problem.title, run: async () => { await selectExercise(problem.slug); showPreview("problem"); } });
    }
    const box = $("ide-quick");
    const input = $("ide-quick-input");
    let selected = 0;
    let shown = items;
    function draw() {
      const query = input.value.toLowerCase().replace(/\s+/g, "");
      shown = items.filter((item) => fuzzy(item.label.toLowerCase(), query)).slice(0, 40);
      selected = Math.min(selected, Math.max(0, shown.length - 1));
      $("ide-quick-list").innerHTML = shown
        .map((item, i) => `<div class="ide-quick-item ${i === selected ? "sel" : ""}" data-i="${i}">${esc(item.label)} <span class="muted small">${esc(item.detail)}</span></div>`)
        .join("");
    }
    function close() {
      box.classList.add("hidden");
      input.onkeydown = input.oninput = null;
      editor.focus();
    }
    input.value = "";
    box.classList.remove("hidden");
    input.focus();
    draw();
    input.oninput = () => { selected = 0; draw(); };
    input.onkeydown = (event) => {
      if (event.key === "ArrowDown") { selected = Math.min(selected + 1, shown.length - 1); draw(); event.preventDefault(); }
      else if (event.key === "ArrowUp") { selected = Math.max(selected - 1, 0); draw(); event.preventDefault(); }
      else if (event.key === "Enter") { const item = shown[selected]; close(); item?.run(); }
      else if (event.key === "Escape") close();
    };
    $("ide-quick-list").onclick = (event) => {
      const row = event.target.closest("[data-i]");
      if (row) { const item = shown[Number(row.dataset.i)]; close(); item?.run(); }
    };
    input.onblur = () => setTimeout(() => !box.contains(document.activeElement) && box.classList.add("hidden"), 150);
  }

  function fuzzy(text, query) {
    let position = 0;
    for (const char of query) {
      position = text.indexOf(char, position);
      if (position < 0) return false;
      position++;
    }
    return true;
  }

  // ---------- commands and keys -------------------------------------------------------------------

  $("ide-simple").addEventListener("click", () => {
    setViewMode("simple");
    route();
  });
  $("ide-run-on").addEventListener("change", () => writeRunOn(gymId, runOnTarget($("ide-run-on"))));
  $("ide-run").addEventListener("click", () => submit("run"));
  tryButton = browserRun.attach($("ide-try"), $("ide-output"), () => {
    const tab = activeTab();
    return { problem: tab && ide.cache.get(tab.slug)?.problem, lang: tab?.lang, code: tab ? ide.models.get(tab.id).getValue() : "" };
  }, (text) => {
    showPanel("output");
    $("st-job").textContent = text;
  });
  $("ide-submit").addEventListener("click", () => submit("test"));
  $("ide-bench").addEventListener("click", () => submit("bench"));
  $("ide-save").addEventListener("click", () => submit("save"));
  $("ide-cancel").addEventListener("click", () => ide.activeJob && api(`/api/jobs/${ide.activeJob}/cancel`, { method: "POST" }));
  $("ide-reload").addEventListener("click", async () => {
    const tab = activeTab();
    if (!tab) return;
    await loadProblem(tab.slug, true);
    replaceWithMachine(tab);
  });
  $("ide-starter").addEventListener("click", () => {
    const tab = activeTab();
    if (tab && confirm(`Replace ${tab.slug}/${tab.name} with the blank starter? The old file is backed up on your machine.`)) {
      writeDraft(gymId, tab.slug, tab.lang, null);
      submit("reset");
    }
  });
  $("ide-toggle-preview").addEventListener("click", () => { setPreviewVisible(!ide.previewVisible); renderTree(); });
  document.querySelectorAll(".ide-more button").forEach((b) => b.addEventListener("click", () => b.closest("details").removeAttribute("open")));

  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => submit("run"));
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => submit("test"));
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => submit("save"));
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyP, quickOpen);
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyB, toggleSidebar);
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyJ, togglePanel);

  function onKey(event) {
    if (editor.hasTextFocus() || !(event.ctrlKey || event.metaKey)) return;
    const key = event.key.toLowerCase();
    const actions = {
      enter: () => submit(event.shiftKey ? "test" : "run"),
      s: () => submit("save"),
      p: quickOpen,
      b: toggleSidebar,
      j: togglePanel,
    };
    if (actions[key]) {
      event.preventDefault();
      actions[key]();
    }
  }
  window.addEventListener("keydown", onKey, true);

  // Pick up edits made on the machine, and keep the runner status fresh.
  state.refreshTimer = setInterval(async () => {
    if (ide.activeJob || document.hidden) return;
    try {
      const slugs = new Set(ide.codeTabs.map((tab) => tab.slug));
      for (const target of slugs) await refreshExercise(target);
      const overview = await api("/api/overview");
      setRunnerPill(overview.runners);
      updateStatus();
    } catch {}
  }, 15000);

  // ---------- start ---------------------------------------------------------------------------

  if (!state.me?.owner) $("ide-reload").textContent = "Reload my saved code";

  if (!state.me?.can_run) {
    for (const id of ["ide-run", "ide-submit", "ide-bench"]) {
      $(id).disabled = true;
      $(id).title = "Running code is not enabled for members yet.";
    }
    $("ide-output").textContent =
      "You can read, edit and save your code. Running it is not enabled for members on this board yet; ask an admin.";
  }

  if (!ide.problems.some((problem) => problem.slug === slug)) {
    ide.slug = slug = ide.problems[0]?.slug;
    ide.expanded = new Set([slug]);
  }
  if (slug) {
    await openExercise(slug);
    const last = ide.cache.get(slug)?.jobs?.[0];
    if (last) api(`/api/jobs/${last.id}`).then((job) => !ide.activeJob && renderJob(job)).catch(() => {});
  }
  renderTree();
  renderCodeTabs();
  updateStatus();
}

// ---------- completions: CUDA, C++, Python, PyTorch, Triton --------------------------------------

const COMPLETIONS = {
  cpp: {
    keywords: "auto bool break case char class const constexpr continue default delete do double else enum explicit extern false float for if inline int long namespace new nullptr operator private protected public return short signed sizeof static static_cast struct switch template this true typedef typename unsigned using virtual void volatile while".split(" "),
    functions: {
      // CUDA
      __syncthreads: "barrier for all threads in the block",
      __syncwarp: "barrier for the threads of a warp",
      __shfl_down_sync: "(mask, value, delta) read value from lane + delta",
      __shfl_xor_sync: "(mask, value, laneMask) butterfly exchange",
      __shfl_sync: "(mask, value, srcLane) broadcast from a lane",
      atomicAdd: "(address, value) atomic add, returns old value",
      atomicMax: "(address, value) atomic max (int)",
      atomicMin: "(address, value) atomic min (int)",
      atomicCAS: "(address, compare, value) compare and swap",
      atomicExch: "(address, value) atomic exchange",
      cudaDeviceSynchronize: "wait for all launched work",
      cudaMalloc: "(void** ptr, size_t bytes)",
      cudaFree: "(void* ptr)",
      cudaMemcpy: "(dst, src, bytes, kind)",
      cudaMemset: "(ptr, value, bytes)",
      cudaGetLastError: "error from the last launch",
      __ldg: "read-only cached load",
      __expf: "fast exp (lower precision)",
      __fdividef: "fast division",
      expf: "float exp", logf: "float log", sqrtf: "float sqrt", rsqrtf: "1 / sqrt(x)", fmaxf: "float max", fminf: "float min",
      fabsf: "float abs", tanhf: "float tanh", powf: "float pow", fmaf: "fused multiply-add",
      make_float2: "", make_float4: "", make_int2: "",
      // C++ standard library
      sort: "(first, last[, comp])", stable_sort: "(first, last[, comp])", reverse: "(first, last)",
      lower_bound: "(first, last, value) first >= value", upper_bound: "(first, last, value) first > value",
      binary_search: "(first, last, value)", max_element: "(first, last)", min_element: "(first, last)",
      accumulate: "(first, last, init)", iota: "(first, last, start)", fill: "(first, last, value)",
      count: "(first, last, value)", find: "(first, last, value)", unique: "(first, last)", next_permutation: "(first, last)",
      swap: "(a, b)", max: "(a, b)", min: "(a, b)", abs: "(x)", make_pair: "(a, b)", move: "(x)", to_string: "(number)",
      memset: "(ptr, value, bytes)", printf: "(format, ...)",
    },
    types: "vector string unordered_map unordered_set map set multiset priority_queue deque queue stack pair tuple array bitset greater less numeric_limits size_t int64_t uint32_t int32_t dim3 float2 float4 int2 int4 __half".split(" "),
    constants: "threadIdx blockIdx blockDim gridDim warpSize INT_MAX INT_MIN LLONG_MAX LLONG_MIN FLT_MAX".split(" "),
    qualifiers: "__global__ __device__ __host__ __shared__ __constant__ __restrict__ __forceinline__".split(" "),
    members: {
      _default: "push_back emplace_back pop_back size empty clear begin end rbegin rend front back insert erase find count reserve resize assign data at substr length top push pop first second emplace contains".split(" "),
      threadIdx: ["x", "y", "z"], blockIdx: ["x", "y", "z"], blockDim: ["x", "y", "z"], gridDim: ["x", "y", "z"],
    },
    snippets: {
      kernel: ["CUDA kernel", "__global__ void ${1:kernel}(const float* ${2:input}, float* ${3:output}, int ${4:N}) {\n\tint i = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (i < ${4:N}) {\n\t\t$0\n\t}\n}"],
      launch: ["kernel launch", "int threads = ${1:256};\nint blocks = (${2:N} + threads - 1) / threads;\n${3:kernel}<<<blocks, threads>>>(${4});\ncudaDeviceSynchronize();"],
      gridstride: ["grid-stride loop", "for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ${1:N}; i += blockDim.x * gridDim.x) {\n\t$0\n}"],
      shared: ["shared memory tile", "__shared__ float ${1:tile}[${2:256}];"],
      warpreduce: ["warp sum reduction", "for (int offset = 16; offset > 0; offset /= 2)\n\t${1:value} += __shfl_down_sync(0xffffffff, ${1:value}, offset);"],
      for: ["for loop", "for (int ${1:i} = 0; ${1:i} < ${2:n}; ++${1:i}) {\n\t$0\n}"],
      forr: ["range for", "for (auto& ${1:x} : ${2:items}) {\n\t$0\n}"],
      inc: ["#include", "#include <${1:vector}>"],
      main: ["using namespace std", "#include <bits/stdc++.h>\nusing namespace std;\n"],
    },
  },
  python: {
    keywords: "and as assert break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield".split(" "),
    functions: {
      len: "", range: "(stop) / (start, stop[, step])", enumerate: "(iterable, start=0)", zip: "(*iterables)", sorted: "(iterable, key=None, reverse=False)",
      reversed: "", min: "", max: "", sum: "", abs: "", list: "", dict: "", set: "", tuple: "", print: "", isinstance: "", int: "", float: "", str: "",
      map: "", filter: "", any: "", all: "", divmod: "", pow: "", round: "",
    },
    types: [],
    constants: ["torch", "tl", "triton", "F", "heapq", "collections", "bisect", "math", "itertools", "functools"],
    qualifiers: [],
    members: {
      torch: "empty empty_like zeros zeros_like ones ones_like full arange linspace tensor rand randn matmul mm bmm einsum softmax log_softmax exp log sqrt rsqrt tanh sigmoid relu sum mean max min amax amin argmax cumsum cumprod topk sort argsort where gather scatter cat stack split chunk tril triu outer allclose isclose float32 float16 bfloat16 int64 int32 bool cuda no_grad compile".split(" "),
      F: "softmax log_softmax relu gelu silu cross_entropy linear layer_norm rms_norm scaled_dot_product_attention pad embedding dropout one_hot".split(" "),
      tl: "program_id num_programs arange load store zeros full sum max min argmax exp exp2 log sqrt rsqrt sigmoid where dot maximum minimum cdiv constexpr float32 float16 int32 atomic_add reshape trans broadcast_to cumsum".split(" "),
      triton: "jit cdiv autotune Config next_power_of_2 heuristics".split(" "),
      heapq: "heappush heappop heapify heappushpop nlargest nsmallest".split(" "),
      collections: "deque defaultdict Counter OrderedDict namedtuple".split(" "),
      bisect: "bisect_left bisect_right insort".split(" "),
      math: "inf sqrt log exp ceil floor gcd isqrt comb".split(" "),
      _default: "append extend pop popleft appendleft insert remove sort reverse index count keys values items get setdefault update add discard shape dtype device view reshape transpose permute contiguous unsqueeze squeeze expand clone copy_ fill_ zero_ masked_fill float half sum mean max min item tolist numel size stride data_ptr".split(" "),
    },
    snippets: {
      def: ["function", "def ${1:name}(${2:args}):\n\t$0"],
      for: ["for loop", "for ${1:i} in range(${2:n}):\n\t$0"],
      tritonkernel: ["Triton kernel", "@triton.jit\ndef ${1:kernel}(x_ptr, y_ptr, n, BLOCK: tl.constexpr):\n\tpid = tl.program_id(0)\n\toffsets = pid * BLOCK + tl.arange(0, BLOCK)\n\tmask = offsets < n\n\tx = tl.load(x_ptr + offsets, mask=mask)\n\ttl.store(y_ptr + offsets, ${2:x}, mask=mask)"],
      tritonlaunch: ["Triton launch", "BLOCK = ${1:1024}\ngrid = (triton.cdiv(${2:n}, BLOCK),)\n${3:kernel}[grid](${4}, BLOCK=BLOCK)"],
      imports: ["common imports", "import torch\nimport triton\nimport triton.language as tl\n"],
    },
  },
};

let completionsRegistered = false;

function registerCompletions(monaco) {
  if (completionsRegistered) return;
  completionsRegistered = true;
  const Kind = monaco.languages.CompletionItemKind;
  const Rule = monaco.languages.CompletionItemInsertTextRule;

  for (const language of ["cpp", "python"]) {
    const data = COMPLETIONS[language];
    monaco.languages.registerCompletionItemProvider(language, {
      triggerCharacters: ["."],
      provideCompletionItems(model, position) {
        const word = model.getWordUntilPosition(position);
        const range = { startLineNumber: position.lineNumber, endLineNumber: position.lineNumber, startColumn: word.startColumn, endColumn: word.endColumn };
        const before = model.getValueInRange({ startLineNumber: position.lineNumber, startColumn: 1, endLineNumber: position.lineNumber, endColumn: word.startColumn });
        const member = before.match(/([A-Za-z_][A-Za-z0-9_]*)\s*(?:\.|->)\s*$/) || before.match(/[\])]\s*(?:\.|->)\s*$/);
        const item = (label, kind, detail = "", insertText = label, snippet = false) => ({
          label, kind, detail, insertText, range, insertTextRules: snippet ? Rule.InsertAsSnippet : undefined,
        });
        if (member) {
          const object = member[1];
          const names = data.members[object] || data.members._default;
          const kind = data.members[object] ? Kind.Function : Kind.Method;
          return { suggestions: names.map((name) => item(name, kind, object ? `${object}.${name}` : "")) };
        }
        const suggestions = [];
        for (const name of data.keywords) suggestions.push(item(name, Kind.Keyword));
        for (const [name, detail] of Object.entries(data.functions)) suggestions.push(item(name, Kind.Function, detail));
        for (const name of data.types) suggestions.push(item(name, Kind.Class));
        for (const name of data.constants) suggestions.push(item(name, Kind.Variable));
        for (const name of data.qualifiers) suggestions.push(item(name, Kind.Keyword, "CUDA"));
        for (const [name, [detail, body]] of Object.entries(data.snippets)) suggestions.push(item(name, Kind.Snippet, detail, body, true));
        return { suggestions };
      },
    });
  }
}
