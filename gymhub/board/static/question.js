"use strict";
// Exercises that are not code: answer, choice, essay, checklist (gymhub/kinds.py). Graded by the board.
// Uses helpers from app.js (api, esc, md, ago, view, state, setCrumbs, statusBadge).

const KIND_LABEL = {
  answer: "Answer", choice: "Choice", essay: "Written answer", checklist: "Checklist",
  lesson: "Lesson", order: "Put in order", match: "Match", fill: "Fill in the blanks",
};

// The same shuffle for everyone, seeded by the exercise: items never show in the author's order.
function seededShuffle(items, seed) {
  let state = [...String(seed)].reduce((hash, char) => (hash * 31 + char.charCodeAt(0)) >>> 0, 7) || 1;
  const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const shuffled = [...items];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// Markdown without its wrapping <p>, for text that sits inline (list items, blanks in a sentence).
function mdInline(text) {
  return md(text).replace(/^\s*<p>([\s\S]*)<\/p>\s*$/, "$1");
}

function questionWidget(problem, last, progress) {
  const kind = problem.kind;
  if (kind === "lesson") return "";
  if (kind === "order") {
    const byId = Object.fromEntries((problem.items || []).map((item) => [item.id, item]));
    const ids = Array.isArray(last) && last.every((id) => byId[id]) && last.length === problem.items.length
      ? last : seededShuffle(Object.keys(byId), problem.slug);
    return `<p class="muted small">Drag the items, or use the arrows, so the first step is at the top.</p>
      <ol class="q-order" id="q-order">${ids
        .map((id) => `<li draggable="true" data-id="${esc(id)}"><span class="grip" aria-hidden="true">⋮⋮</span><span class="q-order-text">${mdInline(byId[id].text)}</span>
          <span class="q-order-moves"><button type="button" class="small-button ghost" data-move="-1" aria-label="Move up">↑</button><button type="button" class="small-button ghost" data-move="1" aria-label="Move down">↓</button></span></li>`)
        .join("")}</ol>`;
  }
  if (kind === "match") {
    const options = problem.options || [];
    const picked = last && typeof last === "object" ? last : {};
    const letter = (index) => LETTERS[index] || String(index + 1);
    return `<div class="q-match">
      <ol class="q-match-options" type="A">${options.map((option) => `<li>${mdInline(option.text)}</li>`).join("")}</ol>
      <div class="q-match-rows">${(problem.prompts || [])
        .map((prompt) => `<label class="q-match-row"><span>${mdInline(prompt.text)}</span>
          <select data-prompt="${esc(prompt.id)}"><option value="">Choose…</option>${options
            .map((option, index) => `<option value="${esc(option.id)}" ${picked[prompt.id] === option.id ? "selected" : ""}>${letter(index)}</option>`)
            .join("")}</select></label>`)
        .join("")}</div></div>`;
  }
  if (kind === "fill") {
    const given = last && typeof last === "object" ? last : {};
    const input = (id) => `<input class="q-blank" data-blank="${esc(id)}" aria-label="Blank ${esc(id)}" autocomplete="off" spellcheck="false" value="${esc(given[id] ?? "")}">`;
    const text = problem.text || "";
    if (problem.code) {
      // Code: escaped as is, monospace, with the inputs in place of the blanks.
      return `<pre class="q-fill-code">${esc(text).replace(/\[\[([a-z0-9_]+)\]\]/g, (_, id) => input(id))}</pre>`;
    }
    return `<div class="q-fill markdown">${md(text.replace(/\[\[([a-z0-9_]+)\]\]/g, "@@BLANK$1@@")).replace(/@@BLANK([a-z0-9_]+)@@/g, (_, id) => input(id))}</div>`;
  }
  if (kind === "answer")
    return `<input id="q-answer" class="q-input" autocomplete="off" placeholder="${esc(problem.placeholder || "Your answer")}" value="${esc(typeof last === "string" ? last : "")}">`;
  if (kind === "choice") {
    const picked = new Set(Array.isArray(last) ? last : last ? [last] : []);
    const type = problem.multiple ? "checkbox" : "radio";
    return `<p class="muted small">${problem.multiple ? "Select every correct answer." : "Select one answer."}</p>
      <div class="q-choices">${(problem.choices || [])
        .map((choice) => `<label class="q-choice"><input type="${type}" name="q-choice" value="${esc(choice.id)}" ${picked.has(choice.id) ? "checked" : ""}>
          <span>${md(choice.text)}</span></label>`)
        .join("")}</div>`;
  }
  if (kind === "essay") {
    const limits = [problem.min_words ? `at least ${problem.min_words}` : "", problem.max_words ? `at most ${problem.max_words}` : ""].filter(Boolean).join(", ");
    return `<textarea id="q-essay" class="q-essay" rows="12" placeholder="Write your answer">${esc(typeof last === "string" ? last : "")}</textarea>
      <p class="muted small"><span id="q-words">0</span> words${limits ? ` (${esc(limits)})` : ""}</p>`;
  }
  if (kind === "checklist") {
    const checked = new Set(progress?.langs?.checklist?.checked || []);
    return `<div class="q-choices">${(problem.items || [])
      .map((item, index) => `<label class="q-choice"><input type="checkbox" name="q-item" value="${index}" ${checked.has(index) ? "checked" : ""}><span>${md(item)}</span></label>`)
      .join("")}</div>`;
  }
  return `<p>This kind of exercise is not supported here.</p>`;
}

function readResponse(kind) {
  if (kind === "lesson") return true;
  if (kind === "order") return [...document.querySelectorAll("#q-order li")].map((item) => item.dataset.id);
  if (kind === "match") return Object.fromEntries([...document.querySelectorAll(".q-match select")].map((select) => [select.dataset.prompt, select.value || null]));
  if (kind === "fill") return Object.fromEntries([...document.querySelectorAll(".q-blank")].map((input) => [input.dataset.blank, input.value]));
  if (kind === "answer") return document.getElementById("q-answer").value;
  if (kind === "choice") {
    const picked = [...document.querySelectorAll("input[name=q-choice]:checked")].map((input) => input.value);
    return document.querySelector("input[name=q-choice][type=radio]") ? picked[0] ?? null : picked;
  }
  if (kind === "essay") return document.getElementById("q-essay").value;
  if (kind === "checklist") return [...document.querySelectorAll("input[name=q-item]:checked")].map((input) => Number(input.value));
  return null;
}

function resultHtml(result) {
  if (!result) return "";
  const badge = result.ok === true ? `<span class="badge good">correct</span>` : result.ok === false ? `<span class="badge bad">not yet</span>` : `<span class="badge wait">submitted</span>`;
  const criteria = (result.criteria || [])
    .map((item) => `<li class="${item.met ? "met" : "unmet"}">${item.met ? "✓" : "✗"} <b>${esc(item.criterion)}</b>${item.comment ? `: ${esc(item.comment)}` : ""}</li>`)
    .join("");
  const rubric = result.rubric
    ? `<div class="q-rubric"><h4>Rubric</h4>${md(result.rubric)}<p>Does your answer cover it?</p>
        <button class="primary" data-self="1">Yes, mark it done</button> <button data-self="0">Not yet</button></div>`
    : "";
  return `<div class="q-result">${badge} ${esc(result.feedback || "")}${result.score != null ? ` <span class="muted small">score ${Math.round(result.score * 100)}%</span>` : ""}
    ${criteria ? `<ul class="q-criteria">${criteria}</ul>` : ""}${rubric}</div>`;
}

async function renderQuestion(gymId, slug, data) {
  const { problem, gym } = data;
  setCrumbs([["Gyms", "#/gyms"], [gym.title || gymId, `#/gym/${gymId}`], [problem.title, null]]);
  const progress = data.progress || {};
  const solved = Object.values(progress.langs || {}).some((entry) => entry.first_solved);
  let hintsShown = Math.max(0, progress.hints_seen || 0);
  const hints = problem.hints || [];
  const changed = Object.keys(data.tests_changed || {}).length;
  // One column: the question reads top to bottom, then the answer; hints and past runs fold below.
  view.innerHTML = `
    <div class="question-page">
      <section class="panel q-panel">
        <div class="q-head"><span class="badge">${esc(KIND_LABEL[problem.kind] || problem.kind)}</span>
          ${problem.difficulty ? `<span class="level ${esc(problem.difficulty)}">${esc(problem.difficulty)}</span>` : ""}
          ${solved ? `<span class="mark good">✓</span> <span class="muted small">solved</span>` : ""}
          ${changed && solved ? `<span class="tests-changed small">· the answer key changed since you solved it</span>` : ""}
          ${exerciseMeta(problem)}</div>
        <div class="markdown q-statement">${md(problem.statement)}</div>
        <form id="q-form" onsubmit="return false">${questionWidget({ ...problem, slug }, problem.last_answer, progress)}
          <div class="q-actions"><button class="primary" id="q-submit" ${state.me ? "" : "disabled"}>${
            { checklist: "Save progress", lesson: solved ? "Read ✓" : "Mark as read" }[problem.kind] || "Submit"
          }</button></div>
        </form>
        <div id="q-result"></div>
      </section>
      ${!hints.length ? "" : `<details class="panel fold"><summary><h3>Hints</h3> <span class="muted">${hints.length}</span></summary><div id="q-tab-hints"></div></details>`}
      ${problem.kind === "lesson" ? "" : `<details class="panel fold"><summary><h3>Runs</h3></summary><div id="q-tab-history"></div></details>`}
    </div>`;
  const $ = (id) => document.getElementById(id);
  setupOrdering($("q-order"));
  view.querySelectorAll(".q-blank").forEach((input) => {
    input.addEventListener("input", () => input.classList.remove("invalid"));
    input.addEventListener("keydown", (event) => event.key === "Enter" && send(readResponse(problem.kind)));
  });
  const renderHints = () => {
    if (!$("q-tab-hints")) return;
    $("q-tab-hints").innerHTML = hints.length
      ? hints.slice(0, hintsShown).map((hint, i) => `<h4>Hint ${i + 1}</h4>${md(hint)}`).join("") +
        (hintsShown < hints.length ? `<button id="q-more-hint">${hintsShown ? "Next hint" : "Show the first hint"}</button>` : "")
      : `<p class="muted">No hints for this exercise.</p>`;
    $("q-more-hint")?.addEventListener("click", async () => {
      hintsShown += 1;
      renderHints();
      api("/api/jobs", { method: "POST", body: JSON.stringify({ gym: gymId, slug, lang: "", action: "hint", options: { count: hintsShown } }) }).catch(() => {});
    });
  };
  renderHints();
  const renderHistory = (jobs) => {
    if (!$("q-tab-history")) return;
    $("q-tab-history").innerHTML = jobs.length
      ? `<table class="table"><tbody>${jobs
          .map((job) => `<tr><td class="muted">${ago(job.created_at)}</td><td>${statusBadge(job)}</td><td class="small">${esc(job.headline)}</td></tr>`)
          .join("")}</tbody></table>`
      : `<p class="muted">Nothing submitted yet.</p>`;
  };
  renderHistory(data.jobs || []);
  const words = $("q-essay");
  if (words) {
    const count = () => ($("q-words").textContent = words.value.trim() ? words.value.trim().split(/\s+/).length : 0);
    words.addEventListener("input", count);
    count();
  }
  const send = async (response) => {
    const button = $("q-submit");
    button.disabled = true;
    $("q-result").innerHTML = `<p class="muted small">${problem.kind === "essay" && typeof response === "string" ? "Grading…" : "Checking…"}</p>`;
    try {
      const result = await api(`/api/gyms/${encodeURIComponent(gymId)}/problems/${encodeURIComponent(slug)}/answer`, {
        method: "POST", body: JSON.stringify({ response }),
      });
      $("q-result").innerHTML = problem.kind === "lesson" ? "" : resultHtml(result);
      if (problem.kind === "lesson" && result.ok) $("q-submit").textContent = "Read ✓";
      const wrong = new Set(result.wrong || []);
      view.querySelectorAll(".q-blank").forEach((input) => input.classList.toggle("invalid", wrong.has(input.dataset.blank)));
      $("q-result").querySelectorAll("[data-self]").forEach((b) =>
        b.addEventListener("click", () => send({ self_assessed: b.dataset.self === "1" }))
      );
      const fresh = await api(`/api/gyms/${encodeURIComponent(gymId)}/problems/${encodeURIComponent(slug)}`);
      renderHistory(fresh.jobs || []);
      if (result.ok && !solved) {
        const head = document.querySelector(".q-head");
        if (head && !head.querySelector(".mark")) head.insertAdjacentHTML("beforeend", ` <span class="mark good">✓</span> <span class="muted small">solved</span>`);
      }
    } catch (error) {
      $("q-result").innerHTML = `<p class="error">${esc(error.message)}</p>`;
    } finally {
      button.disabled = false;
    }
  };
  $("q-submit").addEventListener("click", () => send(readResponse(problem.kind)));
  $("q-answer")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") send(readResponse(problem.kind));
  });
}

// An "order" list: drag items, or move them with the ↑ / ↓ buttons (keyboard and touch).
function setupOrdering(list) {
  if (!list) return;
  list.addEventListener("click", (event) => {
    const button = event.target.closest("[data-move]");
    if (!button) return;
    const item = button.closest("li");
    const target = Number(button.dataset.move) < 0 ? item.previousElementSibling : item.nextElementSibling?.nextElementSibling;
    if (Number(button.dataset.move) < 0 && !target) return;
    list.insertBefore(item, target || null);
    item.querySelector(`[data-move="${button.dataset.move}"]`).focus(); // keep the keyboard on the item it moved
  });
  let dragged = null;
  list.addEventListener("dragstart", (event) => {
    dragged = event.target.closest("li");
    dragged?.classList.add("dragging");
    event.dataTransfer.effectAllowed = "move";
  });
  list.addEventListener("dragend", () => {
    dragged?.classList.remove("dragging");
    dragged = null;
  });
  list.addEventListener("dragover", (event) => {
    if (!dragged) return;
    event.preventDefault();
    const over = event.target.closest("li");
    if (!over || over === dragged) return;
    const box = over.getBoundingClientRect();
    list.insertBefore(dragged, event.clientY > box.top + box.height / 2 ? over.nextElementSibling : over);
  });
}
