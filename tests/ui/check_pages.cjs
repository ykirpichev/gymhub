// Open board pages in headless Chromium and report what went wrong on each (tests/test_ui.py runs this).
//
//   node tests/ui/check_pages.cjs [--devices=desktop,iPhone 13,Pixel 7] <base url> <hash> [<hash> ...]
//
// Prints one JSON object: {"<device>": {"<hash>": [problems...]}}, "desktop" being a plain desktop
// window and any other name a Playwright device (viewport, touch, user agent). A problem is an uncaught
// page error, a console error, a failed request to the board itself, or the board's own "Something went
// wrong" panel; on phones also a page wider than the screen (it scrolls sideways) and, on exercise pages,
// Run and Submit off screen. Exit code 0 even with problems: the test decides; 2 when the browser could
// not run at all.

const { chromium, devices } = require("playwright");

async function checkPage(context, base, hash, phone) {
  const page = await context.newPage();
  const problems = [];
  page.on("pageerror", (error) => problems.push(`page error: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(`console error: ${message.text()} (${message.location().url || "?"})`);
  });
  page.on("response", (response) => {
    if (response.url().startsWith(base) && response.status() >= 400) problems.push(`HTTP ${response.status()} ${response.url()}`);
  });
  await page.goto(`${base}/${hash}`, { waitUntil: "load" });
  // Pages render after their API calls: wait for the "Loading…" placeholder to go, then a moment
  // more for late requests (Monaco, polling).
  await page.waitForFunction(() => !document.querySelector("#view .loading"), null, { timeout: 15000 }).catch(() => {
    problems.push("still loading after 15 s");
  });
  await page.waitForTimeout(1500);
  const failed = await page.$eval("#view", (view) => view.querySelector(".panel.error")?.innerText || "").catch(() => "");
  if (failed) problems.push(`board error panel: ${failed.replace(/\s+/g, " ")}`);
  if (phone) {
    const layout = await page.evaluate(() => {
      const root = document.documentElement;
      const onScreen = (selector) => {
        const element = document.querySelector(selector);
        if (!element) return null;
        const box = element.getBoundingClientRect();
        return box.width > 0 && box.top >= 0 && box.bottom <= window.innerHeight && box.right <= root.clientWidth;
      };
      return { scrollWidth: root.scrollWidth, clientWidth: root.clientWidth, run: onScreen("#btn-run"), submit: onScreen("#btn-test") };
    });
    if (layout.scrollWidth > layout.clientWidth + 1)
      problems.push(`scrolls sideways: ${layout.scrollWidth} px wide on a ${layout.clientWidth} px screen`);
    if (layout.run === false || layout.submit === false) problems.push("Run or Submit is not on screen");
  }
  await page.close();
  return problems;
}

async function main() {
  let args = process.argv.slice(2);
  let names = ["desktop"];
  const option = args.find((arg) => arg.startsWith("--devices="));
  if (option) {
    names = option.slice("--devices=".length).split(",").map((name) => name.trim()).filter(Boolean);
    args = args.filter((arg) => arg !== option);
  }
  const [base, ...hashes] = args;
  for (const name of names) if (name !== "desktop" && !devices[name]) throw new Error(`unknown Playwright device: ${name}`);
  const proxy = process.env.HTTPS_PROXY || process.env.https_proxy;
  // Behind an HTTP proxy (some CI and sandboxes), the board's CDN scripts go through it; the board itself does not.
  const browser = await chromium.launch(proxy ? { proxy: { server: proxy, bypass: "127.0.0.1,localhost" } } : {});
  const report = {};
  const contexts = [];
  try {
    // One context per device; the pages are checked a few at a time (most of each check is waiting).
    const checks = [];
    for (const name of names) {
      const phone = name !== "desktop";
      const context = await browser.newContext(phone ? devices[name] : {});
      contexts.push(context);
      report[name] = {};
      for (const hash of hashes) checks.push({ name, context, hash, phone });
    }
    const concurrency = Math.max(1, Number(process.env.UI_CHECK_CONCURRENCY) || 12);
    let next = 0;
    const worker = async () => {
      while (next < checks.length) {
        const { name, context, hash, phone } = checks[next++];
        report[name][hash] = await checkPage(context, base, hash, phone);
      }
    };
    await Promise.all(Array.from({ length: Math.min(concurrency, checks.length) }, worker));
    for (const context of contexts) await context.close();
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify(report));
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exit(2);
});
