"use strict";
// The gym editor: write a small Python gym in the browser, save drafts, publish versions.
// Server side: gymhub/board/authoring.py and gymhub/gymdraft.py. Uses helpers from app.js.

const DIFFICULTIES = ["easy", "medium", "hard"];
const PUBLISH_WAIT_MS = 15 * 60 * 1000; // stop polling a validating version after this long

// A test cell whose text is not valid JSON yet: kept as typed (never turned into null) until it is fixed.
// Only ever in the editor's memory; a draft with one is not saved.
class RawCell {
  constructor(text) {
    this.text = text;
  }
}

function jsonCell(value) {
  if (value instanceof RawCell) return value.text;
  return value === undefined ? "" : JSON.stringify(value);
}

function cellClass(value) {
  return value instanceof RawCell ? " invalid" : "";
}

// Integers past 2^53 change value when the browser parses them, so they cannot round-trip through the editor.
function hasUnsafeInteger(value) {
  if (typeof value === "number") return Number.isInteger(value) && !Number.isSafeInteger(value);
  if (Array.isArray(value)) return value.some(hasUnsafeInteger);
  if (value && typeof value === "object") return Object.values(value).some(hasUnsafeInteger);
  return false;
}

function exerciseCard(exercise, index) {
  const params = exercise.params || [];
  const rows = (exercise.cases || [])
    .map(
      (testCase) => `<tr>
        <td><input class="case-name" value="${esc(testCase.name || "")}" placeholder="name"></td>
        ${params.map((param) => `<td><input class="case-arg mono${cellClass(testCase.args?.[param])}" data-param="${esc(param)}" value="${esc(jsonCell(testCase.args?.[param]))}" placeholder="JSON"></td>`).join("")}
        <td><input class="case-expected mono${cellClass(testCase.expected)}" value="${esc(jsonCell(testCase.expected))}" placeholder="blank: use the reference"></td>
        <td><button type="button" class="small-button ghost" data-remove-case>×</button></td></tr>`
    )
    .join("");
  return `<details class="panel exercise-card" data-index="${index}" ${exercise._open ? "open" : ""}>
    <summary><b>${esc(exercise.title || "New exercise")}</b> <span class="muted small">${esc(exercise.slug || "")} · ${esc(exercise.difficulty || "")}</span>
      <span class="spacer"></span><button type="button" class="small-button ghost" data-remove-exercise>Remove</button></summary>
    <div class="admin-form">
      <label>Id <input name="slug" value="${esc(exercise.slug || "")}" placeholder="two_sum"></label>
      <label class="grow">Title <input name="title" value="${esc(exercise.title || "")}"></label>
      <label>Difficulty <select name="difficulty">${DIFFICULTIES.map((d) => `<option ${exercise.difficulty === d ? "selected" : ""}>${d}</option>`).join("")}</select></label>
      <label class="wide">One-line summary <input name="summary" value="${esc(exercise.summary || "")}"></label>
      <label class="wide">Statement (Markdown) <textarea name="statement" rows="8">${esc(exercise.statement || "")}</textarea></label>
      <label class="wide">Hints, revealed one at a time (separate hints with a line containing only ---)
        <textarea name="hints" rows="3">${esc((exercise.hints || []).join("\n---\n"))}</textarea></label>
      <label class="wide">Parameters of solve(...), comma separated <input name="params" class="mono" value="${esc(params.join(", "))}" placeholder="nums, target"></label>
      <label class="half">Starter: what learners begin with <textarea name="starter" class="mono" rows="7" spellcheck="false">${esc(exercise.starter || "")}</textarea></label>
      <label class="half">Reference solution: never shown; proves the tests can be passed <textarea name="solution" class="mono" rows="7" spellcheck="false">${esc(exercise.solution || "")}</textarea></label>
    </div>
    <h4>Tests</h4>
    <p class="muted small">Each cell is JSON: <code>[1, 2]</code>, <code>"text"</code>, <code>3</code>. Leave "expected" blank to compute it with the reference solution, which suits large or random inputs.</p>
    <table class="table cases-table"><thead><tr><th>name</th>${params.map((p) => `<th>${esc(p)}</th>`).join("")}<th>expected</th><th></th></tr></thead>
      <tbody>${rows}</tbody></table>
    <button type="button" class="small-button" data-add-case>Add a test</button>
  </details>`;
}

function readEditor(spec) {
  const errors = [];
  const form = document.getElementById("gym-editor");
  spec.id = form.querySelector("[name=gym-id]").value.trim();
  spec.title = form.querySelector("[name=gym-title]").value.trim();
  spec.description = form.querySelector("[name=gym-description]").value.trim();
  spec.exercises = [...form.querySelectorAll(".exercise-card")].map((card, index) => {
    const field = (name) => card.querySelector(`[name=${name}]`).value;
    const params = field("params").split(",").map((p) => p.trim()).filter(Boolean);
    const cases = [...card.querySelectorAll(".cases-table tbody tr")].map((row, rowIndex) => {
      const testCase = { args: {} };
      const name = row.querySelector(".case-name").value.trim();
      if (name) testCase.name = name;
      const where = `${field("slug") || `exercise ${index + 1}`}, test ${rowIndex + 1}`;
      const parse = (input, label) => {
        const bad = (message) => {
          input.classList.add("invalid");
          errors.push(`${where}: ${label} ${message}`);
          return new RawCell(input.value);
        };
        if (!input.value.trim()) return bad("is empty");
        let value;
        try {
          value = JSON.parse(input.value);
        } catch {
          return bad("is not valid JSON");
        }
        if (hasUnsafeInteger(value))
          return bad("has an integer too large for the browser to keep exactly (over 2^53); upload a packed gym for values that big");
        return value;
      };
      row.querySelectorAll(".case-arg").forEach((input) => {
        input.classList.remove("invalid");
        if (params.includes(input.dataset.param)) testCase.args[input.dataset.param] = parse(input, input.dataset.param);
      });
      const expected = row.querySelector(".case-expected");
      expected.classList.remove("invalid");
      if (expected.value.trim()) testCase.expected = parse(expected, "expected");
      return testCase;
    });
    return {
      slug: field("slug").trim(), title: field("title").trim(), difficulty: field("difficulty"), summary: field("summary").trim(),
      statement: field("statement"), hints: field("hints").split(/^\s*---\s*$/m).map((h) => h.trim()).filter(Boolean),
      params, starter: field("starter"), solution: field("solution"), cases, _open: card.open,
    };
  });
  return errors;
}

function cleanSpec(spec) {
  return { ...spec, exercises: spec.exercises.map(({ _open, ...exercise }) => exercise) };
}

async function renderEditor(gymId) {
  // New until the first save or publish succeeds; from then on the id is fixed and saves update that draft.
  let isNew = !gymId;
  const loaded = await api(`/api/drafts/${isNew ? "new" : encodeURIComponent(gymId)}`);
  const spec = loaded.spec;
  if (isNew) spec.exercises.forEach((exercise) => (exercise._open = true));
  let pollTimer = null;
  let statusHtml = ""; // survives re-renders (adding a test while a version is being checked keeps its status)
  state.disposers.push(() => clearTimeout(pollTimer));

  const setStatus = (html) => {
    statusHtml = html;
    const box = document.getElementById("editor-status");
    if (box) box.innerHTML = html;
  };
  const showErrors = (title, errors) =>
    setStatus(`<section class="panel"><p class="error">${esc(title)}</p>${errors.length ? `<ul class="version-errors">${errors.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : ""}</section>`);
  const saved = () => {
    // The first successful save: this is now the draft of spec.id, at its own address.
    if (!isNew) return;
    isNew = false;
    history.replaceState(null, "", `#/edit-gym/${encodeURIComponent(spec.id)}`);
    render();
  };

  const watch = (version, note) => {
    clearTimeout(pollTimer);
    const started = Date.now();
    const show = (current, message) => {
      const status = current.status;
      const live = status === "published";
      setStatus(`<section class="panel"><p>v${current.version}: <span class="badge ${VERSION_BADGE[status] || "wait"}">${esc(status)}</span>
          <span class="muted small">${esc(message)}</span></p>${status === "validating" ? "" : versionDetails(current)}
          ${live ? `<p><a href="#/gym/${encodeURIComponent(spec.id)}">Open the gym →</a></p>` : ""}</section>`);
    };
    show(version, note);
    const poll = async () => {
      let current = null;
      let otherLive = false;
      try {
        const { versions } = await api(`/api/gym-versions?gym=${encodeURIComponent(spec.id)}`);
        current = versions.find((item) => item.version === version.version) || null;
        otherLive = versions.some((item) => item.live && item.version !== version.version);
      } catch (error) {
        if (Date.now() - started < PUBLISH_WAIT_MS) {
          show(version, `Could not check its status (${error.message}); trying again.`);
          pollTimer = setTimeout(poll, 10000);
        } else show(version, `Could not check its status (${error.message}). Its result appears on the Gyms page.`);
        return;
      }
      if (!current || current.status === "validating") {
        if (Date.now() - started < PUBLISH_WAIT_MS) pollTimer = setTimeout(poll, 2500);
        else show(current || version, "Still waiting for a machine to check it. Its result appears on the Gyms page.");
        return;
      }
      const failed = otherLive
        ? "Not published: learners still see the previous version."
        : "Not published: fix what is listed below, then publish again.";
      show(current, current.status === "published" ? "Live: learners see this version." : failed);
    };
    if (version.status === "validating") pollTimer = setTimeout(poll, 2500);
  };

  const render = () => {
    setCrumbs([["Gyms", "#/gyms"], [isNew ? "New gym" : `Edit ${spec.id}`, null]]);
    view.innerHTML = `
      <h2 class="page-title">${isNew ? "Create a gym" : `Edit ${esc(spec.title || spec.id)}`}</h2>
      <p class="muted">Python exercises: learners write <code>solve(...)</code>, and your tests compare its answer with the expected output.
        A machine checks every version before anyone sees it: the starter must fail the tests and your reference solution must pass them.
        New gyms are private until you share them.</p>
      <form id="gym-editor" onsubmit="return false">
        <section class="panel"><div class="admin-form">
          <label>Gym id <input name="gym-id" value="${esc(spec.id)}" ${isNew ? "" : "readonly"} placeholder="my-gym"
            title="${isNew ? "Lowercase letters, digits and -. It cannot change after the first save." : "The id is fixed once saved."}"></label>
          <label class="grow">Title <input name="gym-title" value="${esc(spec.title)}"></label>
          <label class="wide">Description <input name="gym-description" value="${esc(spec.description || "")}"></label>
        </div></section>
        ${spec.exercises.map(exerciseCard).join("")}
        <div class="editor-actions">
          <button type="button" id="add-exercise">Add an exercise</button><span class="spacer"></span>
          <button type="button" id="save-draft">Save draft</button>
          <button type="button" class="primary" id="publish">Publish a version</button>
        </div>
        <div id="editor-status">${statusHtml}</div>
      </form>`;
    const sync = () => {
      const errors = readEditor(spec);
      if (errors.length) showErrors("Fix these first:", errors);
      return !errors.length;
    };
    const rerender = () => {
      readEditor(spec);
      render();
    };
    view.querySelectorAll(".exercise-card").forEach((card) => {
      const exercise = () => spec.exercises[Number(card.dataset.index)];
      card.querySelector("[data-add-case]").addEventListener("click", () => {
        readEditor(spec);
        exercise().cases.push({ args: {} });
        exercise()._open = true;
        render();
      });
      card.querySelector("[data-remove-exercise]").addEventListener("click", (event) => {
        event.preventDefault();
        if (!confirm("Remove this exercise from the draft?")) return;
        readEditor(spec);
        spec.exercises.splice(Number(card.dataset.index), 1);
        render();
      });
      card.querySelectorAll("[data-remove-case]").forEach((button, rowIndex) =>
        button.addEventListener("click", () => {
          readEditor(spec);
          exercise().cases.splice(rowIndex, 1);
          render();
        })
      );
      // New parameters: new columns, and a starter that matches when it was still the default.
      card.querySelector("[name=params]").addEventListener("change", () => {
        const before = exercise().params.join(", ");
        readEditor(spec);
        const item = exercise();
        if (!item.starter.trim() || item.starter.trim() === `def solve(${before}):\n    pass`)
          item.starter = `def solve(${item.params.join(", ")}):\n    pass\n`;
        item._open = true;
        render();
      });
    });
    document.getElementById("add-exercise").addEventListener("click", () => {
      readEditor(spec);
      const number = spec.exercises.length + 1;
      spec.exercises.push({
        slug: `exercise_${number}`, title: "", difficulty: "easy", summary: "", statement: "", hints: [], params: ["x"],
        starter: "def solve(x):\n    pass\n", solution: "", cases: [{ args: {} }], _open: true,
      });
      rerender();
    });
    const send = (path, method) =>
      api(`/api/drafts/${encodeURIComponent(spec.id)}${path}`, { method, body: JSON.stringify({ spec: cleanSpec(spec), create: isNew }) });
    document.getElementById("save-draft").addEventListener("click", async () => {
      if (!sync()) return;
      try {
        const reply = await send("", "PUT");
        saved();
        if (reply.errors.length) showErrors("Saved. Before you can publish:", reply.errors);
        else setStatus(`<p class="muted small">Saved ${new Date().toLocaleTimeString()}.</p>`);
      } catch (error) {
        showErrors(error.message, error.errors || []);
      }
    });
    document.getElementById("publish").addEventListener("click", async (event) => {
      if (!sync()) return;
      const button = event.currentTarget;
      button.disabled = true;
      try {
        let reply;
        try {
          reply = await send("/publish", "POST");
        } catch (error) {
          if (error.saved) saved(); // refused after the draft was saved (bad tests, quota...): it is ours now
          throw error;
        }
        saved();
        const version = reply.version;
        const note = version.duplicate
          ? `This exact gym is already v${version.version}.`
          : reply.warning || "A machine is checking it.";
        watch(version, note);
      } catch (error) {
        showErrors(error.message, error.errors || []);
      } finally {
        const current = document.getElementById("publish");
        if (current) current.disabled = false;
      }
    });
  };
  render();
}

// ---------- API tokens and GitHub (shown on the Gyms page and in a gym's Manage panel) ----------

async function renderTokens(host) {
  const { tokens } = await api("/api/my/tokens");
  host.innerHTML = `<h4>Publish from a terminal or CI</h4>
    <p class="muted small">Create an API token, then <code>GYMHUB_BOARD=${esc(location.origin)} GYMHUB_TOKEN=gp_… python -m gymhub.cli gym push my-gym.gym.md --wait</code> (or a gym folder).
      A token can only upload gym versions and read their status. It works when the board uses in-app sign-in, not behind IAP.</p>
    <div id="new-token"></div>
    <table class="table"><tbody>${tokens
      .map((token) => `<tr><td>${esc(token.name)}</td><td class="muted small">created ${ago(token.created_at)}</td>
        <td class="row-actions"><button class="small-button ghost" data-revoke-token="${esc(token.id)}">Revoke</button></td></tr>`)
      .join("") || `<tr><td class="muted small">No tokens.</td></tr>`}</tbody></table>
    <button class="small-button" id="create-token">Create a token</button>`;
  document.getElementById("create-token").addEventListener("click", async () => {
    const name = prompt("Name this token (where you will use it):", "laptop");
    if (!name) return;
    const reply = await api("/api/my/tokens", { method: "POST", body: JSON.stringify({ name }) });
    await renderTokens(host);
    document.getElementById("new-token").innerHTML = `<p class="small">Copy it now; it is not shown again:</p><input readonly class="mono token-value" value="${esc(reply.token)}">`;
  });
  host.querySelectorAll("[data-revoke-token]").forEach((button) =>
    button.addEventListener("click", async () => {
      if (!confirm("Revoke this token? Anything using it stops working.")) return;
      await api(`/api/my/tokens/${encodeURIComponent(button.dataset.revokeToken)}`, { method: "DELETE" });
      renderTokens(host);
    })
  );
}

async function renderGitHub(gymId, host) {
  const { source } = await api(`/api/gyms/${encodeURIComponent(gymId)}/github`);
  const last = source?.last;
  host.innerHTML = `<h4>Publish from GitHub</h4>
    <p class="muted small">Each push to a matching branch or tag of a <b>public</b> repository publishes a new version (for private repositories, run the CLI from GitHub Actions instead).</p>
    <form id="github-form" class="admin-form">
      <label>Repository <input name="repo" value="${esc(source?.repo || "")}" placeholder="owner/name"></label>
      <label>Folder with gym.json <input name="path" value="${esc(source?.path || "")}" placeholder="(repository root)"></label>
      <label>Ref <input name="ref" value="${esc(source?.ref || "refs/heads/main")}" placeholder="refs/heads/main or refs/tags/v*"></label>
      <button class="primary">${source ? "Update" : "Connect"}</button>
      ${source ? `<button type="button" class="ghost" id="github-disconnect">Disconnect</button>` : ""}
    </form>
    ${source ? `<p class="small">In the repository's Settings → Webhooks, add: payload URL <code>${esc(source.webhook_url)}</code>,
      content type <code>application/json</code>, secret <code>${esc(source.secret)}</code>, "Just the push event".</p>` : ""}
    ${last ? `<p class="small">Last push: <code>${esc((last.sha || "").slice(0, 8))}</code> ${esc(last.ref || "")} ${ago(last.at)}: ${
      last.error ? `<span class="error-text">${esc(last.error)}</span>` : `v${last.version}`}</p>` : ""}`;
  document.getElementById("github-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    try {
      await api(`/api/gyms/${encodeURIComponent(gymId)}/github`, {
        method: "PUT", body: JSON.stringify({ repo: form.get("repo"), path: form.get("path"), ref: form.get("ref") }),
      });
      renderGitHub(gymId, host);
    } catch (error) {
      alert(error.message);
    }
  });
  document.getElementById("github-disconnect")?.addEventListener("click", async () => {
    await api(`/api/gyms/${encodeURIComponent(gymId)}/github`, { method: "DELETE" });
    renderGitHub(gymId, host);
  });
}
