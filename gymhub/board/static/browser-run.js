// "Try in browser": the exercise's examples run on the code in the editor right here, with no
// runner and nothing recorded (browser-worker.js, gymkit/examples.py). Submit still grades.
"use strict";

const browserRun = (() => {
  const RUN_SECONDS = 20; // after Python has loaded; there are no time limits, but no endless loops either
  let worker = null;
  let seq = 0;
  const pending = new Map();

  function spawn() {
    worker = new Worker("/static/browser-worker.js");
    worker.onmessage = ({ data }) => {
      const job = pending.get(data.id);
      if (!job) return;
      if (data.stage === "running") {
        job.onStage?.("running");
        job.timer = setTimeout(() => stop(data.id, `Stopped after ${RUN_SECONDS} s: the code may never finish. Check for an endless loop.`), RUN_SECONDS * 1000);
        return;
      }
      finish(data.id, data.error ? { error: data.error } : data.result);
    };
    worker.onerror = (event) => {
      event.preventDefault();
      for (const id of [...pending.keys()]) stop(id, `Python could not start in this browser: ${event.message || "unknown error"}`);
    };
  }

  function finish(id, value) {
    const job = pending.get(id);
    if (!job) return;
    clearTimeout(job.timer);
    pending.delete(id);
    job.resolve(value);
  }

  function stop(id, message) {
    worker?.terminate(); // the only way to stop Python mid-run; the next run starts a fresh one
    worker = null;
    for (const other of [...pending.keys()]) finish(other, { error: other === id ? message : "Stopped." });
  }

  function count(problem) {
    try {
      return JSON.parse(problem.examples).cases.length;
    } catch {
      return 0;
    }
  }

  const available = (problem, lang) =>
    lang === "python" && typeof problem?.examples === "string" && typeof Worker === "function" && count(problem) > 0;

  // Wires a "Try in browser" button: shown when the exercise has examples in this language, writes
  // the run's text to `output`. get() returns {problem, lang, code}; status(text) is optional.
  function attach(button, output, get, status = () => {}) {
    let busy = false;
    const refresh = () => {
      const { problem, lang } = get();
      const n = count(problem);
      button.classList.toggle("hidden", !available(problem, lang));
      button.title = `Run the ${n} example${n === 1 ? "" : "s"} on the code in the editor, here in your browser: instant, no runner, not recorded. Submit runs the full tests.`;
    };
    button.addEventListener("click", async () => {
      const { problem, lang, code } = get();
      if (busy || !available(problem, lang)) return;
      busy = true;
      button.disabled = true;
      const n = count(problem);
      output.textContent = worker ? `Running ${n} examples in this browser…` : "Loading Python in this browser (the first time takes a while: about 10 MB, then cached)…";
      status("◌ Trying in browser…");
      try {
        const result = await browserRun.run(code, problem.examples, () => (output.textContent = `Running ${n} examples in this browser…`));
        output.textContent = result.error ? `Could not run the examples in this browser.\n${result.error}` : result.text;
        status(result.error ? "✗ Try in browser" : `${result.passed === result.total ? "✓" : "✗"} ${result.passed}/${result.total} examples`);
      } finally {
        busy = false;
        button.disabled = false;
      }
    });
    refresh();
    return { refresh };
  }

  return {
    // Only Python, and only exercises published with examples.
    available,
    attach,
    count,
    // Resolves to {text, passed, total} or {error}. onStage("running") once Python has loaded.
    run(source, examples, onStage) {
      if (!worker) spawn();
      const id = ++seq;
      return new Promise((resolve) => {
        pending.set(id, { resolve, onStage, timer: null });
        worker.postMessage({ id, source, examples });
      });
    },
  };
})();
