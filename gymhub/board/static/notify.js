"use strict";
// Browser notifications (issue #23): a job you started finished while this tab was in the background,
// or one of your machines went offline. Settings live on the board (users/{uid}.notifications);
// the browser's own permission is asked for once, from the Compute page.
// Uses helpers from app.js (api, esc, state, actionLabel).

const NOTIFY_DEFAULTS = { jobs: true, machines: true };
let machineWatch = null;
let lastOnline = null;

function notifySettings() {
  return { ...NOTIFY_DEFAULTS, ...(state.me?.notifications || {}) };
}

function notificationsAllowed() {
  return "Notification" in window && Notification.permission === "granted";
}

function notify(title, body, href) {
  if (!notificationsAllowed()) return;
  try {
    const note = new Notification(title, { body, tag: href || title });
    note.onclick = () => {
      window.focus();
      if (href) location.hash = href;
      note.close();
    };
  } catch {
    /* some browsers only allow notifications from a service worker: nothing to do */
  }
}

// Called by followJob when a job it watched reach a final state.
function notifyJobFinished(job) {
  if (!document.hidden || !notifySettings().jobs) return;
  const verdict = job.status === "error" ? "error" : job.status === "cancelled" ? "cancelled" : job.ok ? "passed" : "failed";
  const headline = (job.output || "").split("\n").find((line) => line.trim()) || "";
  notify(`${actionLabel(job.action)} ${verdict}: ${job.slug} [${job.lang}]`, headline.slice(0, 160), `#/gym/${job.gym}/${job.slug}`);
}

// Called with every runner list the page fetches (setRunnerPill), and by the background watch.
function noteRunners(runners) {
  const me = state.me?.user;
  if (!me) return;
  const mine = (runners || []).filter((runner) => me.is_admin || runner.owner_email === me.email);
  const online = new Set(mine.filter((runner) => runner.online).map((runner) => runner.name));
  if (lastOnline && notifySettings().machines) {
    for (const name of lastOnline) {
      if (!online.has(name)) notify(`${name} went offline`, "Jobs that need it wait in the queue until it is back.", "#/compute");
    }
  }
  lastOnline = online;
}

// The page refreshes machines every 30 s while visible; in the background, check every 2 minutes.
function watchMachines() {
  clearInterval(machineWatch);
  if (!notifySettings().machines || !notificationsAllowed()) return;
  machineWatch = setInterval(() => {
    if (document.hidden && state.me) api("/api/overview").then((data) => noteRunners(data.runners)).catch(() => {});
  }, 120000);
}

function notificationsPanel() {
  const settings = notifySettings();
  const supported = "Notification" in window;
  const permission = supported ? Notification.permission : "unsupported";
  const status =
    permission === "granted" ? "Allowed in this browser."
      : permission === "denied" ? "Blocked in this browser: allow notifications for this site in the browser's settings."
        : permission === "unsupported" ? "This browser does not support notifications." : "";
  const summary = permission === "granted" ? "on in this browser" : permission === "denied" ? "blocked in this browser" : permission === "unsupported" ? "not supported here" : "off";
  return `<details class="panel fold" id="notify-panel"><summary><h3>Notifications</h3> <span class="muted small">${esc(summary)}</span></summary>
    <form class="admin-form" id="notify-form">
      <label class="check"><input type="checkbox" name="jobs" ${settings.jobs ? "checked" : ""}> When a job I started finishes while the board is in a background tab</label>
      <label class="check"><input type="checkbox" name="machines" ${settings.machines ? "checked" : ""}> When one of my machines goes offline</label>
      ${permission === "default" ? `<button type="button" id="notify-allow">Allow notifications in this browser</button>` : ""}
      <span class="muted small" id="notify-status">${esc(status)}</span>
    </form></details>`;
}

function setupNotificationsPanel() {
  const form = document.getElementById("notify-form");
  if (!form) return;
  form.addEventListener("change", async () => {
    const body = { jobs: form.jobs.checked, machines: form.machines.checked };
    try {
      await api("/api/me/notifications", { method: "PUT", body: JSON.stringify(body) });
      state.me.notifications = body;
      document.getElementById("notify-status").textContent = "Saved.";
      watchMachines();
    } catch (error) {
      document.getElementById("notify-status").textContent = error.message;
    }
  });
  document.getElementById("notify-allow")?.addEventListener("click", async () => {
    const result = await Notification.requestPermission();
    document.getElementById("notify-status").textContent = result === "granted" ? "Allowed in this browser." : "Not allowed.";
    document.getElementById("notify-allow")?.remove();
    if (result === "granted") {
      notify("Notifications are on", "You will hear from the board when a background job finishes.", "#/compute");
      watchMachines();
    }
  });
}
