// Runs the code in the editor on an exercise's examples, in this browser, with Pyodide.
// The examples are inputs and expected outputs only (gymkit/examples.py builds them where the grader
// is); the board never sends a grader here. Started by browser-run.js; one worker per page, reused.
"use strict";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
let booting = null;

async function boot() {
  importScripts(`${PYODIDE}pyodide.js`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE });
  const dir = "/home/pyodide/gymkit_browser";
  pyodide.FS.mkdirTree(dir);
  for (const name of ["compare.py", "examples.py"]) {
    const response = await fetch(`/browser/${name}`, { credentials: "same-origin" });
    if (!response.ok) throw new Error(`could not load ${name} (HTTP ${response.status})`);
    pyodide.FS.writeFile(`${dir}/${name}`, await response.text());
  }
  pyodide.runPython(`import sys\nsys.path.insert(0, ${JSON.stringify(dir)})`);
  return pyodide;
}

self.onmessage = async ({ data }) => {
  const { id, source, examples } = data;
  try {
    if (!booting) booting = boot();
    let pyodide;
    try {
      pyodide = await booting;
    } catch (error) {
      booting = null; // try again next time (a flaky network, say)
      throw error;
    }
    // NumPy and whatever else the code imports that Pyodide ships; unknown modules fail on import.
    try {
      const needsNumpy = /"\$":"(ndarray|npscalar)"/.test(examples) ? "\nimport numpy" : "";
      await pyodide.loadPackagesFromImports(source + needsNumpy);
    } catch {}
    self.postMessage({ id, stage: "running" });
    const runText = pyodide.pyimport("examples").run_text;
    const out = runText(source, examples);
    const result = out.toJs({ dict_converter: Object.fromEntries });
    out.destroy();
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: String(error?.message || error) });
  }
};
