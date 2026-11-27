"use strict";
// Home, Gyms and Compute pages. Uses helpers from app.js (api, esc, ago, statusBadge, actionLabel,
// setRunnerPill, setCrumbs, view). Loaded before app.js; only called after it has run.

const DAY_MS = 86400000;

// progress.json timestamps are naive local times ("2026-11-20T10:31:02"); `new Date` reads them as local.
function localDate(text) {
  if (!text) return null;
  const date = new Date(text);
  return Number.isNaN(date.getTime()) ? null : date;
}

function dayKey(date) {
  return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
}

// The exercises that count toward "n of m solved": not warm-ups, not lessons (which are reading).
function practiceProblems(gym) {
  return pathProblems(gym).filter((problem) => problem.kind !== "lesson");
}

// Everything on a gym's path, lessons included: where "next" comes from.
function pathProblems(gym) {
  return (gym.problem_list || []).filter((problem) => !problem.warmup);
}

function isSolved(gym, slug) {
  const langs = gym.progress?.exercises?.[slug]?.langs || {};
  return Object.values(langs).some((entry) => entry.first_solved);
}

// One pass over every gym's progress: what was touched, solved and improved, and when.
function summarize(gyms) {
  const days = new Map();
  const bump = (date) => date && days.set(dayKey(date), (days.get(dayKey(date)) || 0) + 1);
  let last = null;
  let solved = 0;
  let total = 0;
  let solvedThisWeek = 0;
  const bests = [];
  const weekAgo = Date.now() - 7 * DAY_MS;

  for (const gym of gyms) {
    const titles = Object.fromEntries((gym.problem_list || []).map((problem) => [problem.slug, problem]));
    for (const problem of practiceProblems(gym)) {
      total++;
      if (isSolved(gym, problem.slug)) solved++;
    }
    for (const [slug, exercise] of Object.entries(gym.progress?.exercises || {})) {
      if (!titles[slug]) continue;
      let firstSolved = null;
      for (const [lang, entry] of Object.entries(exercise.langs || {})) {
        const lastRun = localDate(entry.last_run);
        const solvedAt = localDate(entry.first_solved);
        bump(lastRun);
        if (solvedAt && solvedAt.getTime() !== lastRun?.getTime()) bump(solvedAt);
        for (const sample of entry.history || []) {
          const at = localDate(sample.at);
          if (at && at.getTime() !== lastRun?.getTime()) bump(at);
        }
        if (solvedAt && (!firstSolved || solvedAt < firstSolved)) firstSolved = solvedAt;
        if (lastRun && (!last || lastRun > last.at)) {
          last = { at: lastRun, gym, problem: titles[slug], lang, status: entry.last_status, solved: Boolean(entry.first_solved) };
        }
        const bestAt = localDate(entry.best_at);
        if (bestAt && entry.best_ms != null) {
          const before = (entry.history || []).filter((sample) => {
            const at = localDate(sample.at);
            return at && at < bestAt;
          });
          const previous = before.length ? Math.min(...before.map((sample) => sample.ms)) : null;
          bests.push({ at: bestAt, gym, problem: titles[slug], lang, ms: entry.best_ms, previous });
        }
      }
      if (firstSolved && firstSolved.getTime() >= weekAgo && !titles[slug].warmup && titles[slug].kind !== "lesson") solvedThisWeek++;
    }
  }

  // Streak: consecutive days with activity, ending today (or yesterday, if nothing yet today).
  let streak = 0;
  const cursor = new Date();
  if (!days.has(dayKey(cursor))) cursor.setDate(cursor.getDate() - 1);
  while (days.has(dayKey(cursor))) {
    streak++;
    cursor.setDate(cursor.getDate() - 1);
  }
  bests.sort((a, b) => b.at - a.at);
  return { days, last, solved, total, solvedThisWeek, streak, bests: bests.slice(0, 4) };
}

// Members see how much of their running allowance they have used (admins have no limits).
function usageLine() {
  const me = state.me;
  if (!me?.usage || !me.can_run) return "";
  const minutes = Math.round(me.usage.seconds_today / 60);
  const parts = [];
  if (me.limits.minutes_per_day) parts.push(`${minutes} of ${me.limits.minutes_per_day} machine minutes today`);
  if (me.limits.jobs_per_hour) parts.push(`${me.usage.jobs_this_hour} of ${me.limits.jobs_per_hour} runs this hour`);
  return parts.length ? `<div class="usage-line">${esc(parts.join(" · "))}</div>` : "";
}

function upNext(gym) {
  return pathProblems(gym).find((problem) => !isSolved(gym, problem.slug)) || null;
}

function heatmap(days, weeks = 16) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const start = new Date(today);
  start.setDate(start.getDate() - (weeks * 7 - 1));
  // Align columns to weeks starting on Monday.
  start.setDate(start.getDate() - ((start.getDay() + 6) % 7));
  let cells = "";
  for (let date = new Date(start); date <= today; date.setDate(date.getDate() + 1)) {
    const count = days.get(dayKey(date)) || 0;
    const level = count === 0 ? 0 : count < 2 ? 1 : count < 4 ? 2 : count < 7 ? 3 : 4;
    cells += `<span class="cell l${level}" title="${date.toDateString()}: ${count ? `${count} action${count > 1 ? "s" : ""}` : "no activity"}"></span>`;
  }
  return `<div class="heatmap">${cells}</div>`;
}

function levelChip(level) {
  return `<span class="level ${esc(level)}">${esc(level)}</span>`;
}

function runnerBanner(runners) {
  if ((runners || []).some((runner) => runner.online)) return "";
  return `<div class="notice">Your GPU machine is offline. You can still read and edit; runs wait until it reconnects.
    <a href="#/compute">Compute</a></div>`;
}

// One line per gym: name, progress, what is next. With `detail` (the Gyms page) also the
// description and what it needs.
function gymRow(gym, detail = false) {
  const practice = practiceProblems(gym);
  const solved = practice.filter((problem) => isSolved(gym, problem.slug)).length;
  const next = upNext(gym);
  const percent = practice.length ? Math.round((100 * solved) / practice.length) : 0;
  const meta = [
    `${practice.length} exercises`,
    ...(gym.langs || []).map((lang) => lang.label),
    gym.requires?.gpu ? "needs an NVIDIA GPU" : "",
    gym.forked_from ? `fork of ${gym.forked_from.gym}` : "",
  ].filter(Boolean);
  return `<a class="gym-row" href="#/gym/${esc(gym.id)}">
      <div class="gym-row-main"><b>${esc(gym.title)}</b> ${visibilityBadge(gym)}
        ${detail ? `<p>${esc(gym.description)}</p><div class="muted small">${esc(meta.join(" · "))}</div>` : ""}</div>
      <div class="gym-row-progress" title="${solved} of ${practice.length} solved">
        <div class="bar"><span style="width:${percent}%"></span></div><span class="muted small">${solved} / ${practice.length}</span></div>
      <div class="gym-row-next small">${next ? `<span class="muted">Next:</span> ${esc(next.title)}` : `<span class="muted">All solved</span>`}</div>
    </a>`;
}

// ---------- Home ------------------------------------------------------------------------------

// Shown only when something is due; what comes up later is a line under the cards.
function reviewSection(reviews) {
  if (!reviews?.due?.length) return "";
  const days = (ts) => Math.max(0, Math.round((Date.now() / 1000 - ts) / 86400));
  const cards = reviews.due
    .map(
      (item) => `<a class="next-card review-card" href="#/review/${encodeURIComponent(item.gym)}/${encodeURIComponent(item.slug)}">
        <span class="muted small">${esc(item.gym_title)} · review ${item.stage + 1} of ${item.of}</span>
        <b>${esc(item.title)}</b>
        <span class="muted small">solved ${days(item.solved_at)} days ago · re-solve from scratch</span></a>`
    )
    .join("");
  const more = reviews.due_count > reviews.due.length ? ` and ${reviews.due_count - reviews.due.length} more` : "";
  const later = reviews.upcoming
    ? `<p class="muted small">${reviews.upcoming} more coming up${reviews.next_due_at ? `, the next ${new Date(reviews.next_due_at * 1000).toLocaleDateString()}` : ""}.</p>`
    : "";
  return `<h3 class="section-title">Review <span class="badge wait">${reviews.due_count} due</span></h3>
    <section class="next-grid">${cards}</section>${more ? `<p class="muted small">${more.trim()}.</p>` : ""}${later}`;
}

async function renderHome() {
  setCrumbs([]);
  const [data, me] = await Promise.all([api("/api/home"), api("/api/me").catch(() => null)]);
  if (me) {
    state.me = me;
    renderUserMenu(me); // fresh usage numbers in the menu
  }
  setRunnerPill(data.runners);
  const gyms = data.gyms;
  if (!gyms.length) {
    view.innerHTML = `<section class="panel empty"><h2>Welcome</h2>
      <p>No gyms are published yet. Start the runner on your GPU machine; it publishes every local gym here.</p>
      <pre>source .venv/bin/activate\npython -m gymhub.runner</pre></section>`;
    return;
  }
  const summary = summarize(gyms);

  let continueCard;
  // Gyms a machine that is up can grade come first (the board says so per gym, "gradable"): a
  // learner without a GPU starts with an exercise that runs, not a CUDA kernel. When none can, the
  // usual order.
  const byGradable = (list) => [...list.filter((gym) => gym.gradable !== false), ...list.filter((gym) => gym.gradable === false)];
  // The last exercise touched is solved: continue with the next unsolved one, in that gym first.
  const after = summary.last?.solved
    ? [summary.last.gym, ...byGradable(gyms.filter((gym) => gym !== summary.last.gym))].map((gym) => ({ gym, problem: upNext(gym) })).find((item) => item.problem)
    : null;
  if (after) {
    const { gym, problem } = after;
    continueCard = `<a class="continue" href="#/gym/${esc(gym.id)}/${esc(problem.slug)}">
        <div class="eyebrow">Up next</div>
        <h2>${esc(problem.title)}</h2>
        <div class="continue-meta"><span>${esc(gym.title)}</span>${levelChip(problem.difficulty)}</div>
        <p class="muted">${esc(problem.summary)}</p>
        <p class="muted small">You solved ${esc(summary.last.problem.title)} ${ago(summary.last.at.getTime() / 1000)}.</p>
        <span class="button primary">Start →</span></a>`;
  } else if (summary.last) {
    const { gym, problem, lang, solved, status, at } = summary.last;
    const result = solved ? `<span class="badge good">solved</span>` : status === "fail" ? `<span class="badge bad">not passing yet</span>` : "";
    continueCard = `<a class="continue" href="#/gym/${esc(gym.id)}/${esc(problem.slug)}">
        <div class="eyebrow">Continue where you left off</div>
        <h2>${esc(problem.title)}</h2>
        <div class="continue-meta"><span>${esc(gym.title)}</span>${levelChip(problem.difficulty)}<span class="muted">${esc(lang)}</span>${result}<span class="muted">${ago(at.getTime() / 1000)}</span></div>
        <p class="muted">${esc(problem.summary)}</p>
        <span class="button primary">Resume →</span></a>`;
  } else {
    const first = byGradable(gyms).map((gym) => ({ gym, problem: upNext(gym) })).find((item) => item.problem);
    continueCard = first
      ? `<a class="continue" href="#/gym/${esc(first.gym.id)}/${esc(first.problem.slug)}">
          <div class="eyebrow">Start here</div><h2>${esc(first.problem.title)}</h2>
          <div class="continue-meta"><span>${esc(first.gym.title)}</span>${levelChip(first.problem.difficulty)}</div>
          <p class="muted">${esc(first.problem.summary)}</p><span class="button primary">Start →</span></a>`
      : `<div class="continue"><h2>Everything is solved</h2><p class="muted">Beat your best times with Bench.</p></div>`;
  }

  const bests = summary.bests.length
    ? `<h4>Recent personal bests</h4><ul class="bests">${summary.bests
        .map(
          (best) => `<li><a href="#/gym/${esc(best.gym.id)}/${esc(best.problem.slug)}">${esc(best.problem.title)}</a>
          <span class="muted small">${esc(best.lang)}</span>
          <span class="mono">${best.previous != null ? `${formatMs(best.previous)} → ` : ""}<b>${formatMs(best.ms)}</b></span>
          <span class="muted small">${ago(best.at.getTime() / 1000)}</span></li>`
        )
        .join("")}</ul>`
    : "";

  // The numbers stay in view; the heatmap and personal bests fold under them.
  const stats = `<section class="panel stats-card">
      <div class="stat-row">
        <div><b>${summary.solved}</b><span>of ${summary.total} solved</span></div>
        <div><b>${summary.solvedThisWeek}</b><span>solved this week</span></div>
        <div><b>${summary.streak}</b><span>day streak</span></div>
      </div>
      <details class="fold" ${state.homeActivityOpen ? "open" : ""}><summary>Activity${summary.bests.length ? " and personal bests" : ""}</summary>
        ${heatmap(summary.days)}
        <div class="muted small">Activity over the last 16 weeks</div>
        ${bests}
      </details>
    </section>`;

  view.innerHTML = `
    ${runnerBanner(data.runners)}
    <section class="home-top">${continueCard}${stats}</section>
    ${reviewSection(data.reviews)}
    <h3 class="section-title">Your gyms</h3>
    <section class="panel gym-list">${gyms.map((gym) => gymRow(gym)).join("")}</section>`;
  view.querySelector(".stats-card details").addEventListener("toggle", (event) => (state.homeActivityOpen = event.target.open));
}

// Bench times: three significant figures (four on leaderboards, where close times are compared),
// in the unit that keeps them readable.
function formatMs(ms, digits = 3) {
  if (ms == null) return "";
  if (ms >= 1000) return `${(ms / 1000).toPrecision(digits)} s`;
  if (ms < 0.1) return `${(ms * 1000).toPrecision(digits)} µs`;
  return `${ms.toPrecision(digits)} ms`;
}

// ---------- Gyms ------------------------------------------------------------------------------

// ---------- Compute ---------------------------------------------------------------------------

// Machines that grade are pool agents (#135). A gymhub runner is either the bridge that connects the
// board to the pool (a dispatcher: it sends every job to the pool and runs nothing itself), shown as
// one status line, or an old runner that executes jobs itself, folded away under "Old runners" (#162).
const OLD_RUNNER_HINT = "Old runner: it runs jobs on its own machine. Add the machine to the pool instead (Machines that grade, above), then stop the old runner (systemctl --user disable --now gymhub-runner).";

function isDispatcher(runner) {
  return Boolean(runner.capabilities?.pool?.dispatch_only || runner.capabilities?.dispatch_only);
}

function isBridge(runner) {
  return Boolean(isDispatcher(runner) || runner.capabilities?.pool);
}

function runnerKind(runner, poolConfigured) {
  if (isDispatcher(runner)) return ` <span class="badge" title="Connects the board to the pool: it sends every job to the pool and runs nothing itself">pool bridge</span>`;
  if (!poolConfigured) return "";
  return ` <span class="badge" title="${esc(OLD_RUNNER_HINT)}">old runner</span>`;
}

async function renderCompute() {
  setCrumbs([]);
  const [data, pool, mine] = await Promise.all([
    api("/api/overview"),
    api("/api/pool/machines").catch(() => ({ configured: false, machines: [] })),
    api("/api/my/workers").catch(() => ({ workers: [] })),
  ]);
  setRunnerPill(data.runners);
  state.computeRunners = data.runners;
  state.poolConfigured = Boolean(pool.configured);
  // A machine you added is already listed, with its actions, under "Your machines": list it once.
  // With a pool, the bridge is the status line above the pool machines, not a row here.
  const mineIds = new Set(mine.workers.map((worker) => worker.id));
  const others = data.runners.filter((runner) => !mineIds.has(runner.worker_id) && !(pool.configured && isBridge(runner)));
  const othersTitle = pool.configured ? "Board runners" : mineIds.size ? "Other machines" : state.me?.user.is_admin ? "All machines" : "Machines";
  const rows = others
    .map(
      (runner) => `<tr><td><span class="dot ${runner.online ? "on" : "off"}"></span>${esc(runner.name)}${runnerKind(runner, pool.configured)}</td>
          <td>${esc(gpuLabel(runner))}${capabilityNote(runner.capabilities)}</td><td>${esc(runner.platform)}</td><td>${esc((runner.gyms || []).join(", "))}</td>
          <td>${runner.online ? (runner.busy ? "running a job" : "idle") : "offline"}</td>
          <td class="muted">${ago(runner.heartbeat_at)}</td></tr>`
    )
    .join("");
  const runnersTable = (empty) => `<table class="table"><thead><tr><th>name</th><th>GPU</th><th>platform</th><th>gyms</th><th>state</th><th>heartbeat</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="6" class="muted">${empty}</td></tr>`}</tbody></table>`;
  const jobs = data.jobs.length ? data.jobs.map(jobRow).join("") : `<tr><td colspan="6" class="muted">No submissions yet.</td></tr>`;
  // With a pool, old runners and worker tokens are history: folded, after the machines that grade.
  const oldCount = others.length + mine.workers.length;
  const machines = pool.configured
    ? `<section id="pool-machines"></section>
       ${oldCount ? `<details class="panel fold old-runners"><summary><h3>Old runners</h3> <span class="muted small">${oldCount}</span></summary>
         <p class="muted small">Before the pool, a gymhub runner on a computer took jobs from the board and ran them there. These are those
           runners and their tokens. They are not needed to grade jobs any more; a runner here that is offline is not a pool machine being down.</p>
         <section id="my-machines"></section>
         ${others.length ? `<h4 class="subhead">${othersTitle}</h4>${runnersTable("")}` : ""}
       </details>` : ""}`
    : `<section id="pool-machines"></section>
       <section id="my-machines"></section>
       <section class="panel"><h3>${othersTitle}</h3>
         ${runnersTable(mineIds.size ? "No other machine has connected." : "No machine has connected yet. Start one with <code>python -m gymhub.runner</code>.")}</section>`;
  view.innerHTML = `
    <h2 class="page-title">Compute</h2>
    ${machines}
    <section class="panel"><h3>Recent jobs</h3>
      <table class="table"><thead><tr><th>when</th><th>exercise</th><th>action</th><th>result</th><th>ran on</th><th>summary</th></tr></thead>
      <tbody>${jobs}</tbody></table></section>
    ${notificationsPanel()}
    <section id="worker-tokens"></section>`;
  setupNotificationsPanel();
  if (state.me?.user.is_admin) renderWorkerTokens();
  renderPoolMachines(pool);
  if (document.getElementById("my-machines")) renderMyMachines(null, pool.configured);
}

function poolGpus(machine) {
  return machine.gpus.length
    ? machine.gpus.map((gpu) => `${gpu.model}${gpu.memory_gb ? ` (${Math.round(gpu.memory_gb)} GB)` : ""}`).join(", ")
    : "CPU only";
}

function poolState(machine, stale, partial) {
  if (partial) return "up";
  if (stale) return `unknown (was ${machine.online ? (machine.running ? "running" : "idle") : "offline"})`;
  if (!machine.online) return machine.last_seen ? `offline since ${ago(machine.last_seen)}` : "never connected";
  if (machine.drain) return "draining (takes no new job)";
  if (machine.running) return `running ${machine.running} job${machine.running > 1 ? "s" : ""}`;
  return "idle";
}

// One line on how the board reaches the pool: the board itself, or the bridge runner (dispatcher).
function poolBridgeLine(pool) {
  const bridge = pool.bridge;
  const via = bridge
    ? ` through <b>${esc(bridge.name || "a runner")}</b>${bridge.dispatcher ? ", which sends every job to the pool and runs none itself" : ""}`
    : "";
  if (!bridge) return pool.source === "board" ? "" : "Board connected to pool: <b>no</b>. No runner connects this board to the pool.";
  if (!bridge.online) return `Board connected to pool: <b>no</b>. The runner that connects them, <b>${esc(bridge.name || "the pool bridge")}</b>, was last heard from ${ago(bridge.heartbeat_at)}. Jobs wait until it is back.`;
  if (!bridge.reaches_pool) return `Board connected to pool: <b>no</b>. ${esc(bridge.name || "The bridge runner")} is up but cannot reach the pool server.`;
  return `Board connected to pool: <b>yes</b>${via}. Last heartbeat ${ago(bridge.heartbeat_at)}.`;
}

// "Where would my next exercise run?", answered from the machines that are up.
function poolWhereLine(machines, stale) {
  if (stale) return "Where a job would run is not known until the pool bridge is back: the states below are from its last report.";
  const up = machines.filter((machine) => machine.online && !machine.drain);
  if (!up.length) return "No pool machine is up: new jobs wait until one connects.";
  const gpu = up.filter((machine) => machine.gpus.length).map((machine) => `${machine.name} (${machine.gpus[0].model})`);
  const cpu = up.map((machine) => machine.name);
  return `A GPU exercise can run on ${gpu.length ? `<b>${esc(gpu.join(", "))}</b>` : "<b>no machine</b> (none that is up has a GPU)"}.
    A CPU exercise can run on <b>${esc(cpu.join(", "))}</b>.`;
}

function poolIslandsLine(pool) {
  const groups = (pool.islands || []).map((island) => island.machines).filter((names) => names.length > 1);
  if (!groups.length) return "";
  return `Fast links (one LAN): ${groups.map((names) => esc(names.join(" ↔ "))).join("; ")}. Multi-machine jobs stay within one of these.`;
}

// Whose jobs the pool grades: the bridge's owner's only, until they share it (gymhub/board/workers.py
// may_run_for). Its owner gets a "Share the pool" form; everyone else sees whether their jobs go there.
function poolSharingLine(pool) {
  const grades = pool.bridge?.grades_for;
  if (!grades) return "";
  if (!grades.mine)
    return grades.yours
      ? `<p class="small">The pool grades your jobs: its owner shared it with ${grades.who === "everyone" ? "everyone on this board" : "you"}.</p>`
      : `<p class="small">The pool grades only its owner's jobs. Ask them to share it with you to run code here.</p>`;
  const who = grades.who === "everyone" ? "everyone on this board" : grades.emails?.length ? `you and ${grades.emails.join(", ")}` : "only you";
  return `<p class="small">Grades jobs for: <b>${esc(who)}</b>. <button class="small-button" id="share-pool-toggle">Share the pool…</button></p>
    <form class="admin-form share-form hidden" id="share-pool" data-worker="${esc(grades.worker)}">
      <label class="wide">Also grade the jobs of these people (emails of board members, comma separated)
        <input name="emails" value="${esc((grades.emails || []).join(", "))}" placeholder="friend@example.com"></label>
      <label class="check"><input type="checkbox" name="everyone" ${grades.who === "everyone" ? "checked" : ""}> Everyone on this board</label>
      <button class="primary">Save sharing</button> <span class="share-error signin-error"></span>
      <p class="muted small">Private by default: your pool machines grade only your own jobs. Shared jobs run in the pool's containers, with no network.</p>
    </form>`;
}

function bindPoolSharing(host) {
  const toggle = host.querySelector("#share-pool-toggle");
  const form = host.querySelector("#share-pool");
  if (!toggle || !form) return;
  toggle.addEventListener("click", () => form.classList.toggle("hidden"));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = new FormData(form);
    try {
      await api(`/api/my/workers/${encodeURIComponent(form.dataset.worker)}/sharing`, {
        method: "POST",
        body: JSON.stringify({ emails: String(values.get("emails") || "").split(/[\s,]+/).filter(Boolean), everyone: values.get("everyone") === "on" }),
      });
      renderCompute();
    } catch (error) {
      form.querySelector(".share-error").textContent = error.message;
    }
  });
}

// No pool known to the board. With no pool bridge either (a board with its own runner, such as
// `gymhub-board`), lead with what grades the runs: the runners themselves. How to connect a pool
// folds away under them. A bridge that is down or sees no pool machine is said first (#162).
function noPoolPanel(pool, runners = []) {
  const bridge = pool.bridge;
  if (!bridge) {
    const graders = runners.filter((runner) => runner.online && !isBridge(runner));
    const local = state.me?.auth === "dev" && graders.length === 1;
    const names = graders.map((runner) => `<b>${esc(runner.name)}</b> (${esc(gpuLabel(runner) || "no GPU")})`).join(", ");
    const lead = !graders.length
      ? `<b>No machine is grading right now.</b> Runs wait in the queue until a runner connects: <code>gymhub-board</code> starts one with the board,
          or start it on its own with <code>python -m gymhub.runner</code>.`
      : local
      ? `<b>This machine grades your runs</b>: ${names}.`
      : `${names} ${graders.length === 1 ? "grades" : "grade"} the runs on this board.`;
    const noGpu = graders.length && !graders.some((runner) => runner.capabilities?.gpus?.length)
      ? `<p class="muted small">${graders.length === 1 ? "It has" : "They have"} no NVIDIA GPU, so exercises that need one (the GPU kernels) wait in the queue
          until a machine with a GPU connects. Everything else runs here.</p>`
      : "";
    return `<section class="panel"><h3>Machines that grade</h3>
        <p class="small">${lead}</p>${noGpu}
        <details class="fold"><summary>Grade on a compute pool instead</summary>
          <p class="muted small">A compute pool grades runs on machines you register, in containers with no network, and lets other people's
            code run without trusting it to this machine. To use one, start a pool bridge: a runner with <code>--dispatch-only</code> on a machine
            that reaches the pool server (for example <code>python -m gymhub.runner --api ${esc(location.origin)} --dispatch-only</code>). See docs/pool.md.</p>
        </details></section>`;
  }
  const why = !bridge.online
    ? `The pool bridge <b>${esc(bridge.name || "")}</b> was last heard from ${ago(bridge.heartbeat_at)}, so the board cannot see the pool.
        Check it on its machine: <code>systemctl --user status gymhub-runner</code>.`
    : `The pool bridge <b>${esc(bridge.name || "")}</b> is up but reports no pool machine: it cannot reach the pool server, or no pool machine is online.
        Its log says which: <code>journalctl --user -u gymhub-runner -n 30</code>.`;
  return `<section class="panel"><h3>Machines that grade</h3>
      <p class="small">Board connected to pool: <b>no</b>.</p>
      <p class="muted small">${why}</p></section>`;
}

function renderPoolMachines(pool, code = null) {
  const host = document.getElementById("pool-machines");
  if (!host) return;
  if (!pool.configured) {
    host.innerHTML = noPoolPanel(pool, state.computeRunners);
    return;
  }
  const rows = pool.machines
    .map(
      (machine) => `<tr title="${esc(Object.entries(machine.labels || {}).map(([key, value]) => `${key}=${value}`).join(" "))}">
        <td><span class="dot ${machine.online && !pool.stale ? "on" : "off"}"></span>${esc(machine.name)}${machine.owner ? ` <span class="muted small">${esc(machine.owner)}</span>` : ""}</td>
        <td>${esc(poolGpus(machine))}</td><td class="muted small">${esc(machine.platform || machine.arch || "")}</td>
        <td>${esc(poolState(machine, pool.stale, pool.partial))}</td><td class="small">${esc((machine.gyms || []).join(", "))}</td>
        <td class="muted">${machine.last_seen ? ago(machine.last_seen) : pool.partial ? "" : "never"}</td></tr>`
    )
    .join("");
  const reveal = code
    ? `<div class="token-reveal"><p><b>Run this on the machine</b> (Linux, in a terminal):</p>
        <pre>${esc(code.install)}</pre>
        ${code.sha256 ? `<p class="muted small">SHA-256 of the agent package the installer fetches: <span class="mono">${esc(code.sha256)}</span></p>` : ""}
        <p class="muted small">The code works once${code.expires_at ? `, until ${esc(new Date(code.expires_at * 1000).toLocaleString())}` : ""}.
          The machine appears here when its agent connects.</p>
        <button class="small-button" id="copy-pool-install">Copy command</button></div>`
    : "";
  const admin = state.me?.user.is_admin;
  const add = pool.can_add
    ? `<form id="add-pool-machine" class="admin-form"><label>Machine name <input name="name" maxlength="40"
        pattern="[A-Za-z0-9._-]*" placeholder="e.g. home-4090 (optional)"></label><button class="primary">Add a machine</button></form>`
    : admin
    ? `<p class="muted small">To add a machine: on a computer that reaches the pool server, run <code>pool machine add --name &lt;name&gt;</code>,
        then run the install command it prints on the new machine. (This board cannot reach the pool server itself, so it cannot hand out the code.)</p>`
    : `<p class="muted small">Ask an admin of this board to add a machine.</p>`;
  const reported = pool.source === "bridge" && pool.reported_at ? ` As reported by the pool bridge ${ago(pool.reported_at)}.` : "";
  const partial = pool.partial
    ? `<div class="notice"><span>${esc(pool.bridge?.name || "The pool bridge")} runs an older version, so only the machines that are up are listed,
        without their state. Update it and restart it (<code>git pull</code>, then <code>systemctl --user restart gymhub-runner</code>) to list every machine.</span></div>`
    : "";
  const lines = [poolBridgeLine(pool), poolWhereLine(pool.machines, pool.stale), poolIslandsLine(pool)].filter(Boolean);
  host.innerHTML = `<section class="panel"><h3>Machines that grade${pool.pool ? ` <span class="muted small">pool ${esc(pool.pool)}</span>` : ""}</h3>
      ${lines.map((line) => `<p class="small">${line}</p>`).join("")}
      ${poolSharingLine(pool)}
      ${pool.error ? `<div class="notice">${esc(pool.error)}</div>` : ""}
      ${partial}
      <table class="table"><thead><tr><th>name</th><th>GPU</th><th>platform</th><th>state</th><th>gyms</th><th>last seen</th></tr></thead>
        <tbody>${rows || `<tr><td colspan="6" class="muted">No pool machine yet.</td></tr>`}</tbody></table>
      <p class="muted small">Pool machines grade jobs in containers with no network: their owner's, and those of people the owner shares the pool with.${reported} A new machine needs Linux and Docker
        (plus the NVIDIA Container Toolkit for a GPU; the installer adds both), and no checkout of this repository.</p>
      ${reveal || add}
    </section>`;
  bindPoolSharing(host);
  const form = host.querySelector("#add-pool-machine");
  if (form)
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        const reply = await api("/api/pool/machine-codes", { method: "POST", body: JSON.stringify({ name: new FormData(form).get("name") || "" }) });
        renderPoolMachines(pool, reply);
      } catch (error) {
        alert(error.message);
      }
    });
  const copy = host.querySelector("#copy-pool-install");
  if (copy) copy.addEventListener("click", async () => { await navigator.clipboard?.writeText(code.install).catch(() => {}); copy.textContent = "Copied"; });
}

async function renderMyMachines(registration = null, poolConfigured = Boolean(state.poolConfigured)) {
  const host = document.getElementById("my-machines");
  if (!host) return;
  const data = await api("/api/my/workers");
  const sharedLabel = (worker) =>
    worker.shared_everyone ? "everyone" : worker.shared_emails.length ? worker.shared_emails.join(", ") : "only you";
  // Other people's code runs only on the compute pool (gymhub/runner/pooled.py); a runner has no sandbox
  // of its own. "sandbox" is what runners from before that change report.
  const othersLabel = (worker) =>
    worker.pool
      ? "on the pool"
      : worker.sandbox
      ? `<span title="${esc(OLD_RUNNER_HINT)}">in its sandbox (old runner)</span>`
      : `<span class="muted" title="Other people's code runs only on the compute pool; this machine has no pool member, so it runs only your jobs">not here</span>`;
  const rows = data.workers
    .map(
      (worker) => `<tr data-id="${esc(worker.id)}"><td><span class="dot ${worker.online ? "on" : "off"}"></span>${esc(worker.name)}${
        poolConfigured && !worker.dispatcher ? ` <span class="badge" title="${esc(OLD_RUNNER_HINT)}">old runner</span>` : ""}</td>
        <td>${esc(gpuLabel(worker))}${capabilityNote(worker.capabilities)}</td><td class="muted small">${esc(worker.platform || "")}</td>
        <td>${worker.online ? (worker.busy ? "running a job" : "idle") : worker.last_seen ? "offline" : `<span title="The installer registered this machine, but it has not connected since. Run the installer again on it, or remove it.">never connected</span>`}</td>
        <td class="small">${othersLabel(worker)}</td>
        <td class="small">${esc(sharedLabel(worker))}</td>
        <td class="muted">${worker.last_seen ? ago(worker.last_seen) : "never"}</td>
        <td class="row-actions"><button class="small-button" data-share>Share…</button> <button class="small-button ghost" data-remove>Remove</button></td></tr>
        <tr class="share-row hidden" data-share-for="${esc(worker.id)}"><td colspan="8">
          <form class="admin-form share-form">
            <label class="wide">Share with these people (emails of board members, comma separated)
              <input name="emails" value="${esc(worker.shared_emails.join(", "))}" placeholder="friend@example.com"></label>
            <label class="check"><input type="checkbox" name="everyone" ${worker.shared_everyone ? "checked" : ""}> Everyone on this board</label>
            <button class="primary">Save sharing</button> <span class="share-error signin-error"></span>
            <p class="muted small">${worker.pool ? "Their jobs are graded on the compute pool, in its containers, never on this machine itself." : "Their jobs run only on the compute pool, and this machine has no pool member to send them with, so they will not run here."}</p>
          </form></td></tr>`
    )
    .join("");
  const forOthers = data.workers.flatMap((worker) => worker.jobs_for_others.map((job) => ({ ...job, machine: worker.name })));
  const othersTable = forOthers.length
    ? `<h4 class="subhead">Recent jobs your machines ran for others</h4><table class="table"><tbody>${forOthers
        .map((job) => `<tr><td class="muted">${ago(job.when)}</td><td>${esc(job.host && job.host !== job.machine ? `${job.host} via ${job.machine}` : job.machine)}</td><td>${esc(job.who)}</td><td>${esc(job.exercise)}</td>
          <td>${esc(actionLabel(job.action))}</td><td>${statusBadge(job)}</td></tr>`)
        .join("")}</tbody></table>`
    : "";
  const command = registration ? `curl -fsSL ${registration.install_url || location.origin}/install.sh | sh -s -- --code ${registration.code}` : "";
  const intro = poolConfigured
    ? `Runner tokens you created. A runner with your token runs <b>${esc(data.runs)}</b>, unless you share it, on its own machine.
        To grade on a new computer, add it to the pool instead. Remove a token that never connected.`
    : `Run exercises on your own computer. A machine you add runs <b>${esc(data.runs)}</b>, unless you share it.
        The installer sets up the runner; it needs Python 3.10+, and an NVIDIA GPU only for GPU exercises.`;
  host.innerHTML = `<section class="panel"><h3>${poolConfigured ? "Your runner tokens" : "Your machines"}</h3>
      <p class="muted small">${intro}</p>
      ${
        data.can_add === false
          ? `<div class="notice"><span>Adding a machine is not available yet: this board is behind Google IAP, which only lets
              signed-in browsers through, and its worker-only service is not set up.
              ${state.me?.user.is_admin ? "Run <code>deploy/gcp.sh deploy</code> to create it, or switch the board to in-app sign-in (<code>AUTH_MODE=app</code>, see “Sign-in and inviting people” in docs/self-hosting.md)." : "Ask the board's admin to enable it."}</span></div>`
          : registration
          ? `<div class="token-reveal"><p><b>Run this on the machine</b> (in a terminal, as your normal user, no sudo):</p>
              <pre>${esc(command)}</pre>
              <p class="muted small">The code works once and expires in <span id="reg-left">10:00</span>.
              The machine appears below when the installer finishes.</p>
              <button class="small-button" id="copy-install">Copy command</button></div>`
          : `<form id="add-machine" class="admin-form"><label>Machine name <input name="name" required maxlength="40"
              pattern="[A-Za-z0-9._-]+" placeholder="e.g. home-4090"></label><button class="primary">${poolConfigured ? "Add a runner" : "Add a machine"}</button></form>`
      }
      <table class="table"><thead><tr><th>name</th><th>GPU</th><th>platform</th><th>state</th><th>others' jobs</th><th>shared with</th><th>last seen</th><th></th></tr></thead>
        <tbody>${rows || `<tr><td colspan="8" class="muted">You have not added a machine.</td></tr>`}</tbody></table>
      ${othersTable}
    </section>`;
  const form = host.querySelector("#add-machine");
  if (form)
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        const reply = await api("/api/my/workers/registration", { method: "POST", body: JSON.stringify({ name: new FormData(form).get("name") }) });
        renderMyMachines({ ...reply, until: Date.now() + reply.expires_in * 1000 }, poolConfigured);
      } catch (error) {
        alert(error.message);
      }
    });
  const copy = host.querySelector("#copy-install");
  if (copy) copy.addEventListener("click", async () => { await navigator.clipboard?.writeText(command).catch(() => {}); copy.textContent = "Copied"; });
  if (registration) {
    clearInterval(state.registrationTimer);
    state.registrationTimer = setInterval(async () => {
      const left = Math.max(0, Math.round((registration.until - Date.now()) / 1000));
      const label = document.getElementById("reg-left");
      if (!label) return clearInterval(state.registrationTimer);
      label.textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
      const fresh = await api("/api/my/workers").catch(() => null);
      if (left === 0 || (fresh && fresh.workers.length > data.workers.length)) {
        clearInterval(state.registrationTimer);
        renderMyMachines();
      }
    }, 3000);
  }
  host.querySelectorAll("[data-share]").forEach((button) =>
    button.addEventListener("click", () => host.querySelector(`[data-share-for="${button.closest("tr").dataset.id}"]`).classList.toggle("hidden"))
  );
  host.querySelectorAll(".share-form").forEach((form) =>
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const id = form.closest("tr").dataset.shareFor;
      const values = new FormData(form);
      try {
        await api(`/api/my/workers/${encodeURIComponent(id)}/sharing`, {
          method: "POST",
          body: JSON.stringify({ emails: String(values.get("emails") || "").split(/[\s,]+/).filter(Boolean), everyone: values.get("everyone") === "on" }),
        });
        renderMyMachines();
      } catch (error) {
        form.querySelector(".share-error").textContent = error.message;
      }
    })
  );
  host.querySelectorAll("[data-remove]").forEach((button) =>
    button.addEventListener("click", async () => {
      if (!confirm("Remove this machine? It stops running jobs at its next request.")) return;
      await api(`/api/my/workers/${encodeURIComponent(button.closest("tr").dataset.id)}`, { method: "DELETE" });
      renderMyMachines();
    })
  );
}

async function renderWorkerTokens(created = null) {
  const host = document.getElementById("worker-tokens");
  if (!host) return;
  const { workers } = await api("/api/workers");
  const board = location.origin;
  const rows = workers
    .map(
      (worker) => `<tr data-id="${esc(worker.id)}"><td>${esc(worker.name)}</td><td class="mono muted small">gw_${esc(worker.id)}_…</td>
        <td class="muted small">${esc(worker.owner_email)}</td><td class="muted">${worker.last_seen ? ago(worker.last_seen) : "never"}</td>
        <td class="row-actions"><button class="small-button ghost" data-revoke>Revoke</button></td></tr>`
    )
    .join("");
  const reveal = created
    ? `<div class="token-reveal">
        <p><b>Token for ${esc(created.name)}.</b> Copy it now: it is not shown again.</p>
        <pre>${esc(created.token)}</pre>
        <p class="muted small">On that machine, in the repo:</p>
        <pre>mkdir -p ~/.config/gymhub && umask 077 && echo '${esc(created.token)}' > ~/.config/gymhub/worker-token
.venv/bin/python -m gymhub.runner --api ${esc(board)}</pre>
      </div>`
    : "";
  // Folded unless a token was just created: it is shown once and must not be hidden.
  host.innerHTML = `<details class="panel fold" ${created ? "open" : ""}><summary><h3>Worker tokens</h3> <span class="muted small">${workers.length}</span></summary>
      <p class="muted small">A token lets one machine run jobs through the board's API, with no cloud credentials on it.
        Revoking a token stops that machine at its next request.</p>
      ${reveal}
      <form id="token-form" class="admin-form"><label>Machine name <input name="name" required maxlength="60" placeholder="e.g. spark-10dc"></label>
        <button class="primary">Create token</button></form>
      <table class="table"><thead><tr><th>name</th><th>token</th><th>owner</th><th>last seen</th><th></th></tr></thead>
        <tbody>${rows || `<tr><td colspan="5" class="muted">No worker tokens yet.</td></tr>`}</tbody></table>
    </details>`;
  host.querySelector("#token-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = new FormData(event.target).get("name");
    try {
      renderWorkerTokens(await api("/api/workers", { method: "POST", body: JSON.stringify({ name }) }));
    } catch (error) {
      alert(error.message);
    }
  });
  host.querySelectorAll("[data-revoke]").forEach((button) =>
    button.addEventListener("click", async () => {
      if (!confirm("Revoke this token? The machine using it stops running jobs.")) return;
      await api(`/api/workers/${encodeURIComponent(button.closest("tr").dataset.id)}`, { method: "DELETE" });
      renderWorkerTokens();
    })
  );
}

function capabilityNote(caps) {
  if (!caps) return "";
  const parts = [];
  const gpu = (caps.gpus || [])[0];
  if (gpu?.compute_capability) parts.push(`sm ${gpu.compute_capability}`);
  if (caps.cuda) parts.push(`CUDA ${caps.cuda}`);
  if (caps.cpus) parts.push(`${caps.cpus} CPUs`);
  if (caps.memory_gb) parts.push(`${caps.memory_gb} GB`);
  if (caps.tools?.length) parts.push(caps.tools.join(", "));
  return parts.length ? `<div class="muted small">${esc(parts.join(" · "))}</div>` : "";
}

function jobRow(job) {
  // Validating an uploaded version is a job about the whole gym, with no exercise.
  const what = job.slug
    ? `<a href="#/gym/${esc(job.gym)}/${esc(job.slug)}">${esc(job.gym)}/${esc(job.slug)}</a> <span class="muted">${esc(job.lang)}</span>`
    : `${esc(job.gym)} <span class="muted">uploaded version</span>`;
  return `<tr><td class="muted">${ago(job.created_at)}</td>
    <td>${what}</td>
    <td>${esc(actionLabel(job.action))}</td><td>${statusBadge(job)}</td><td class="muted small">${esc(job.host ? `${job.host}${job.via ? ` via ${job.via}` : ""}` : "")}</td>
    <td class="mono muted">${esc(job.headline)}</td></tr>`;
}
