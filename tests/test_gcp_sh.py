"""deploy/gcp.sh budget and ttl (issue #24), run against a fake gcloud that records its arguments."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

FAKE_GCLOUD = """#!/usr/bin/env bash
echo "$*" >> "$GCLOUD_LOG"
case "$*" in
  "billing projects describe"*) echo "billingAccounts/AAA-BBB" ;;
  "billing budgets list"*) cat "$GCLOUD_BUDGETS" 2> /dev/null || true ;;
esac
"""


@pytest.fixture()
def gcp(tmp_path):
    if not shutil.which("bash") or (REPO / "deploy" / "gcp.env").exists():
        pytest.skip("needs bash, and no deploy/gcp.env (it would override the test's project)")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gcloud").write_text(FAKE_GCLOUD)
    (bin_dir / "gcloud").chmod(0o755)
    log = tmp_path / "gcloud.log"

    def run(*args, **env):
        log.write_text("")
        result = subprocess.run(
            ["bash", str(REPO / "deploy" / "gcp.sh"), *args], capture_output=True, text=True, timeout=60,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "GCLOUD_LOG": str(log),
                 "GCLOUD_BUDGETS": str(tmp_path / "budgets"), "PROJECT_ID": "test-project", **env},
        )
        return result, log.read_text().splitlines()

    run.budgets = tmp_path / "budgets"
    return run


def test_budget_from_env_with_alert_thresholds(gcp):
    result, calls = gcp("budget", BUDGET_USD="12.50", BUDGET_ALERT_CHANNEL="projects/p/notificationChannels/7")
    assert result.returncode == 0, result.stderr
    create = next(call for call in calls if call.startswith("billing budgets create"))
    assert "--billing-account=AAA-BBB" in create and "--budget-amount=12.50USD" in create
    assert "--filter-projects=projects/test-project" in create
    for rule in ("percent=0.5", "percent=0.9", "percent=1.0", "percent=1.0,basis=forecasted-spend"):
        assert f"--threshold-rule={rule} " in create + " "
    assert "--notifications-rule-monitoring-notification-channels=projects/p/notificationChannels/7" in create
    assert "$12.50/month" in result.stdout


def test_budget_argument_wins_and_rerun_updates(gcp):
    result, calls = gcp("budget", "7", BUDGET_USD="12")
    assert "--budget-amount=7USD" in next(call for call in calls if call.startswith("billing budgets create"))
    gcp.budgets.write_text("billingAccounts/AAA-BBB/budgets/b1\n")
    result, calls = gcp("budget", "9")
    assert result.returncode == 0, result.stderr
    assert not any(call.startswith("billing budgets create") for call in calls)
    assert any(call.startswith("billing budgets update billingAccounts/AAA-BBB/budgets/b1") and "--budget-amount=9USD" in call
               for call in calls)
    result, calls = gcp("budget", "lots")
    assert result.returncode != 0 and not any("budgets" in call for call in calls)


def test_ttl_policies_for_production_and_staging(gcp):
    from gymhub.store import EXPIRING

    result, calls = gcp("ttl")
    assert result.returncode == 0, result.stderr
    groups = sorted(call.split("--collection-group=")[1].split()[0] for call in calls)
    assert groups == sorted([*EXPIRING, *(f"staging-{name}" for name in EXPIRING)])
    assert all(call.startswith("firestore fields ttls update expire_at") and "--enable-ttl" in call for call in calls)


def test_deploy_workflow_staging_on_green_main_production_on_tags():
    yaml = pytest.importorskip("yaml")
    text = (REPO / ".github" / "workflows" / "deploy.yml").read_text()
    workflow = yaml.safe_load(text)
    triggers = workflow[True]  # YAML 1.1 reads the key `on` as True
    assert triggers["push"]["tags"] == ["v*"] and "branches" not in triggers["push"]
    assert triggers["workflow_run"]["workflows"] == ["tests"] and triggers["workflow_run"]["branches"] == ["main"]
    job = workflow["jobs"]["deploy"]
    assert "startsWith(github.ref, 'refs/tags/v')" in job["if"]
    assert "github.event_name == 'push' && 'production'" in job["env"]["TARGET"]
    gate = job["steps"][0]
    assert gate["if"] == "github.event_name == 'push'" and "tests.yml/runs?head_sha=$SHA&event=push&branch=main" in gate["run"]
    # The deployer's identity accepts exactly what the workflow deploys from: main and v* tags.
    script = (REPO / "deploy" / "gcp.sh").read_text()
    assert ("assertion.repository == '$repo' && (assertion.ref == 'refs/heads/main' || "
            "assertion.ref.startsWith('refs/tags/v'))") in script
