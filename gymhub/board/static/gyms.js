"use strict";
// Gyms page (with search and tags), creating gyms, a gym's Manage panel (sharing, stats, versions), forks and join links.
// Uses helpers from app.js and home.js (api, esc, ago, view, state, setCrumbs, setRunnerPill, gymRow,
// runnerBanner, setupGymUploads). Loaded before app.js; only called after it has run.

const VISIBILITY_LABEL = {
  private: "Private: only you and your editors",
  shared: "Shared: people you name, or who use the join link",
  unlisted: "Unlisted: anyone on this board who has the join link",
  public: "Public: everyone on this board, listed in Explore",
};

function visibilityBadge(gym) {
  if (!gym.visibility || gym.visibility === "public") return "";
  return `<span class="badge vis-${esc(gym.visibility)}">${esc(gym.visibility)}</span>`;
}

function joinLink(gymId, code) {
  return `${location.origin}${location.pathname}#/join/${encodeURIComponent(gymId)}/${encodeURIComponent(code)}`;
}

// ---------- Gyms page ---------------------------------------------------------------------------

function uploadPanel() {
  return `<details class="panel fold" id="create-gym" ${state.createOpen ? "open" : ""}><summary><h3>Create or upload a gym</h3></summary>
      <p><a class="button-link primary-link" href="#/new-gym">Write a gym in the browser</a>
        <span class="muted small">Python exercises with your own tests: no files, no repo.</span></p>
      <h4>Or upload a file</h4>
      <p class="muted small">A whole gym fits in one <a href="/docs/gym-markdown" target="_blank" rel="noopener">Gym Markdown</a> file
        (<code>.gym.md</code>): lessons, every kind of question, and Python exercises with their tests, written in plain Markdown,
        by you or by an agent. Start one with <code>python -m gymhub.cli gym new my-gym.gym.md</code>.
        For C++, custom graders or data files, a gym is a folder with <code>gym.json</code> and <code>problems/</code>; pack it with
        <code>python -m gymhub.packages pack path/to/gym</code> and upload the .tar.gz. A machine checks every version first: every
        grader loads, every starter fails its tests and every reference solution passes them. New gyms start private; share them
        from the gym's Manage panel.</p>
      <form id="upload-form" class="admin-form">
        <label class="wide">Gym Markdown (.gym.md) or bundle (.tar.gz) <input name="bundle" type="file" accept=".md,.tar.gz,.tgz,application/gzip,text/markdown" required></label>
        <button class="primary">Upload and validate</button>
      </form>
      <div id="upload-result"></div>
      <div id="versions"><p class="muted small">Loading uploads…</p></div>
      <div id="tokens"></div>
    </details>`;
}

// Gyms written in the browser that are not live yet (never published, or every version failed its checks).
// Published ones are reached from their gym's Manage panel.
async function renderDrafts(host) {
  const { drafts } = await api("/api/my/drafts");
  const pending = drafts.filter((draft) => !draft.published);
  if (!pending.length || !host.isConnected) return;
  host.innerHTML = `<h3 class="section-title">Your drafts</h3><section class="panel gym-list">${pending
    .map((draft) => {
      const latest = draft.latest;
      const badge = latest
        ? `<span class="badge ${VERSION_BADGE[latest.status] || ""}">v${latest.version} ${esc(latest.status)}</span>`
        : `<span class="badge">not published</span>`;
      return `<p><a href="#/edit-gym/${encodeURIComponent(draft.id)}">${esc(draft.title || draft.id)}</a> ${badge}
        <span class="muted small">${esc(draft.id)} · saved ${ago(draft.updated_at)}</span></p>`;
    })
    .join("")}</section>`;
}

// Every gym this person can open, with Explore's search and tags; creating and uploading below.
async function renderGyms() {
  setCrumbs([]);
  const data = await api("/api/home");
  setRunnerPill(data.runners);
  const gyms = data.gyms;
  const tags = [...new Set(gyms.flatMap((gym) => gym.tags || []))].sort();
  let query = state.gymQuery || "";
  let tag = state.gymTag || "";
  view.innerHTML = `${runnerBanner(data.runners)}
    <div class="page-head"><h2 class="page-title">Gyms</h2>
      <input id="gym-search" class="gym-search" type="search" placeholder="Search title, description, tag" value="${esc(query)}" aria-label="Search gyms"></div>
    ${tags.length ? `<div class="explore-tags" id="gym-tags">${tags.map((t) => `<button class="tag" data-tag="${esc(t)}">${esc(t)}</button>`).join("")}</div>` : ""}
    <div id="gym-lists"></div>
    <div id="my-drafts"></div>
    ${state.me?.can_run ? uploadPanel() : ""}`;
  renderDrafts(document.getElementById("my-drafts")).catch(() => {});

  const mine = (gym) => gym.source === "bundle" && (gym.role === "owner" || gym.role === "editor") && gym.owner_uid;
  function showLists() {
    const words = query.trim().toLowerCase();
    const shown = gyms.filter((gym) => {
      const text = [gym.id, gym.title, gym.description, ...(gym.tags || [])].join(" ").toLowerCase();
      return (!words || text.includes(words)) && (!tag || (gym.tags || []).includes(tag));
    });
    const list = (items) => `<section class="panel gym-list">${items.map((gym) => gymRow(gym, true)).join("")}</section>`;
    const own = shown.filter(mine);
    const rest = shown.filter((gym) => !mine(gym));
    document.getElementById("gym-lists").innerHTML = !shown.length
      ? `<p class="muted">No gym matches${words ? ` “${esc(query)}”` : ""}${tag ? ` with the tag ${esc(tag)}` : ""}.</p>`
      : own.length
      ? `<h3 class="section-title">Yours</h3>${list(own)}${rest.length ? `<h3 class="section-title">Others</h3>${list(rest)}` : ""}`
      : list(rest);
    view.querySelectorAll("#gym-tags [data-tag]").forEach((button) => button.classList.toggle("on", button.dataset.tag === tag));
  }
  showLists();
  document.getElementById("gym-search").addEventListener("input", (event) => {
    query = state.gymQuery = event.target.value;
    showLists();
  });
  view.querySelectorAll("#gym-tags [data-tag]").forEach((button) =>
    button.addEventListener("click", () => {
      tag = state.gymTag = button.dataset.tag === tag ? "" : button.dataset.tag;
      showLists();
    })
  );
  if (state.me?.can_run) {
    const create = document.getElementById("create-gym");
    create.addEventListener("toggle", () => (state.createOpen = create.open));
    setupGymUploads();
    renderTokens(document.getElementById("tokens")).catch(() => {});
  }
}

// ---------- fork and join ---------------------------------------------------------------------

async function forkGym(gymId, title) {
  const id = prompt(`Fork "${title}" into a new private gym. Its id (lowercase letters, digits and -):`, `my-${gymId}`.slice(0, 40));
  if (!id) return;
  try {
    const reply = await api(`/api/gyms/${encodeURIComponent(gymId)}/fork`, { method: "POST", body: JSON.stringify({ id }) });
    alert(`Forked into ${reply.version.gym}. A machine is checking it; it appears under Your gyms once it passes.` + (reply.warning ? `\n\n${reply.warning}` : ""));
    location.hash = "#/gyms";
  } catch (error) {
    alert(error.message);
  }
}

async function renderJoin(gymId, code) {
  setCrumbs([]);
  try {
    await api(`/api/gyms/${encodeURIComponent(gymId)}/join`, { method: "POST", body: JSON.stringify({ code }) });
    location.hash = `#/gym/${encodeURIComponent(gymId)}`;
  } catch (error) {
    view.innerHTML = `<section class="panel"><h3>Cannot join</h3><p>${esc(error.message)}</p></section>`;
  }
}

// ---------- a gym's Manage panel (owner and editors) ------------------------------------------

function statsTable(stats) {
  if (!stats.learners) return `<p class="muted small">Nobody has practised here yet.</p>`;
  const cell = (value, suffix = "") => (value == null ? `<span class="muted">-</span>` : `${value}${suffix}`);
  const rows = stats.problems
    .map(
      (row) => `<tr><td>${esc(row.title)}</td><td>${row.started}</td><td>${row.solved}</td>
        <td>${row.solve_rate == null ? cell(null) : `${Math.round(row.solve_rate * 100)}%`}</td>
        <td class="${row.stuck ? "stuck" : ""}">${row.stuck}</td><td>${cell(row.attempts_to_solve)}</td>
        <td>${cell(row.attempts_while_stuck)}</td><td>${cell(row.hints_per_learner)}</td></tr>`
    )
    .join("");
  return `<p class="muted small">${stats.learners} ${stats.learners === 1 ? "person has" : "people have"} practised here. Counts only, never names.</p>
    <table class="table"><thead><tr><th>exercise</th><th>started</th><th>solved</th><th>solve rate</th><th>stuck</th>
      <th title="Average submissions by those who solved it">tries to solve</th><th title="Average submissions by those still stuck">tries while stuck</th><th>hints</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
}

function pendingNote(emails) {
  if (!emails?.length) return "";
  return `<p class="muted small">Not signed in yet: ${esc(emails.join(", "))}. They get access when they first sign in
    (they still need a board invite, or an email domain this board allows).</p>`;
}

async function renderManage(gym, host) {
  const [settings, stats] = await Promise.all([
    api(`/api/gyms/${encodeURIComponent(gym.id)}/settings`),
    api(`/api/gyms/${encodeURIComponent(gym.id)}/stats`),
  ]);
  const owner = settings.role === "owner";
  const disabled = owner ? "" : "disabled";
  host.innerHTML = `<details class="panel manage" ${state.manageOpen === gym.id ? "open" : ""}><summary><h3>Manage this gym</h3>
      <span class="muted small">${esc(VISIBILITY_LABEL[settings.visibility].split(":")[0])} · you are its ${esc(settings.role)}</span></summary>
    <form id="gym-settings" class="admin-form">
      <fieldset class="visibility" ${disabled}><legend>Who can see it</legend>
        ${Object.entries(VISIBILITY_LABEL)
          .map(([value, label]) => `<label class="check"><input type="radio" name="visibility" value="${value}" ${settings.visibility === value ? "checked" : ""}> ${esc(label)}</label>`)
          .join("")}
      </fieldset>
      ${settings.hidden ? `<div class="notice">An admin of this board hid this gym after reports${settings.hidden.note ? `: ${esc(settings.hidden.note)}` : ""}.
        It stays private until an admin makes it public again; choose Public and save to ask for that.</div>` : ""}
      ${settings.visibility_requested === "public" ? `<p class="muted small">You asked to make it public: waiting for an admin of this board to approve.</p>`
        : settings.public_needs_review && settings.visibility !== "public" && !state.me?.user.is_admin ? `<p class="muted small">On this board an admin approves gyms before they become public.</p>` : ""}
      <label class="wide">Shared with (emails)
        <input name="learners" ${disabled} value="${esc([...settings.learners, ...(settings.pending_learners || [])].join(", "))}" placeholder="friend@example.com"></label>
      ${pendingNote(settings.pending_learners)}
      ${settings.uploaded ? `<label class="wide">Editors: may upload new versions and see stats
        <input name="editors" ${disabled} value="${esc([...settings.editors, ...(settings.pending_editors || [])].join(", "))}" placeholder="co-author@example.com"></label>
      ${pendingNote(settings.pending_editors)}
      <label class="check"><input type="checkbox" name="allow_fork" ${settings.allow_fork ? "checked" : ""} ${disabled}>
        Allow forks. Whoever forks gets a full copy, graders and hidden tests included.</label>` : ""}
      <label class="check"><input type="checkbox" name="tutor" ${settings.tutor ? "checked" : ""} ${disabled}>
        AI tutor for learners: explains failures, reviews code, gives extra hints${settings.tutor_available ? "" : " (not set up on this board yet)"}</label>
      <label class="check"><input type="checkbox" name="leaderboard" ${settings.leaderboard ? "checked" : ""} ${disabled}>
        Leaderboards: fastest bench per exercise and GPU. Only benchmarks run on trusted machines count (yours and the board's).</label>
      <label class="wide">Tags (comma separated, shown in Explore)
        <input name="tags" ${disabled} value="${esc(settings.tags.join(", "))}" placeholder="graphs, interview prep"></label>
      ${owner ? `<button class="primary">Save</button> <span id="settings-note" class="muted small"></span>` : `<p class="muted small">Only the owner changes these.</p>`}
    </form>
    ${owner && settings.share_code && settings.visibility !== "private" ? `<div class="share-link"><label class="wide">Join link
        <input readonly id="join-link" value="${esc(joinLink(gym.id, settings.share_code))}"></label>
      <button class="small-button" id="copy-join">Copy</button> <button class="small-button ghost" id="new-join">New link</button></div>` : ""}
    <h4>Progress</h4>${statsTable(stats)}
    ${settings.uploaded ? `<h4>Versions</h4>${settings.has_draft ? `<p><a class="button-link" href="#/edit-gym/${encodeURIComponent(gym.id)}">Edit in the browser</a></p>` : ""}<div id="gym-versions"></div>` : ""}
    ${gym.can_fork && state.me?.can_run ? `<p><button class="small-button" id="manage-fork">Fork into a new gym</button></p>` : ""}
    ${owner && settings.uploaded ? `<div id="github-host"></div>
      <h4>Ownership</h4>
      <p><button class="small-button" id="transfer-gym">Transfer to someone else</button>
        <button class="small-button danger" id="delete-gym">Delete this gym</button></p>` : ""}
  </details>`;
  const details = host.querySelector("details");
  details.addEventListener("toggle", () => (state.manageOpen = details.open ? gym.id : null));
  document.getElementById("manage-fork")?.addEventListener("click", () => forkGym(gym.id, gym.title));
  if (settings.uploaded) {
    const { versions } = await api(`/api/gym-versions?gym=${encodeURIComponent(gym.id)}`);
    document.getElementById("gym-versions").innerHTML = versions
      .map((v) => `<p class="small">v${v.version} <span class="badge ${VERSION_BADGE[v.status] || ""}">${esc(v.status)}</span>${v.live ? ` <span class="badge good">live</span>` : ""}
        <span class="muted">${esc(v.created_by || "")} · ${ago(v.created_at)}</span>${(v.errors || []).length ? `<br><span class="error-text">${esc(v.errors[0])}</span>` : ""}</p>`)
      .join("");
  }
  if (owner && settings.uploaded) renderGitHub(gym.id, document.getElementById("github-host")).catch(() => {});
  if (!owner) return;
  const split = (value) => String(value || "").split(/[\s,]+/).filter(Boolean);
  document.getElementById("gym-settings").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    try {
      const saved = await api(`/api/gyms/${encodeURIComponent(gym.id)}/settings`, {
        method: "PUT",
        body: JSON.stringify({
          visibility: form.get("visibility"),
          learner_emails: split(form.get("learners")),
          editor_emails: split(form.get("editors")),
          allow_fork: form.get("allow_fork") === "on",
          tutor: form.get("tutor") === "on",
          leaderboard: form.get("leaderboard") === "on",
          tags: String(form.get("tags") || "").split(",").map((tag) => tag.trim()).filter(Boolean),
        }),
      });
      state.manageOpen = gym.id;
      await renderGym(gym.id); // the header shows the new visibility too
      if (saved.kept_learners) alert(`${saved.kept_learners} ${saved.kept_learners === 1 ? "person who was" : "people who were"} practising here keep access.`);
    } catch (error) {
      document.getElementById("settings-note").textContent = error.message;
    }
  });
  document.getElementById("transfer-gym")?.addEventListener("click", async () => {
    const email = prompt("Email of the new owner (a member of this board). You stay on as an editor.");
    if (!email) return;
    try {
      await api(`/api/gyms/${encodeURIComponent(gym.id)}/transfer`, { method: "POST", body: JSON.stringify({ email }) });
      await renderGym(gym.id);
    } catch (error) {
      alert(error.message);
    }
  });
  document.getElementById("delete-gym")?.addEventListener("click", async () => {
    const typed = prompt(`Delete "${gym.title}" for everyone? Learners keep their progress, but nobody can open it again.\nType its id (${gym.id}) to confirm.`);
    if (typed === null) return;
    try {
      await api(`/api/gyms/${encodeURIComponent(gym.id)}`, { method: "DELETE", body: JSON.stringify({ confirm: typed }) });
      location.hash = "#/gyms";
    } catch (error) {
      alert(error.message);
    }
  });
  document.getElementById("copy-join")?.addEventListener("click", async () => {
    const input = document.getElementById("join-link");
    input.select();
    await navigator.clipboard?.writeText(input.value).catch(() => document.execCommand("copy"));
    document.getElementById("copy-join").textContent = "Copied";
  });
  document.getElementById("new-join")?.addEventListener("click", async () => {
    if (!confirm("Make a new join link? The current one stops working; people who already joined keep access.")) return;
    await api(`/api/gyms/${encodeURIComponent(gym.id)}/share-code`, { method: "POST" });
    state.manageOpen = gym.id;
    renderManage(gym, host);
  });
}
