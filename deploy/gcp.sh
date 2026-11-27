#!/usr/bin/env bash
# Human Gym on Google Cloud: a new project, Firestore, the board on Cloud Run behind IAP.
#
#   deploy/gcp.sh create <BILLING_ACCOUNT_ID>   new project, APIs, Firestore, service account
#   deploy/gcp.sh deploy [staging]              build and deploy the board (AUTH_MODE=app: in-app Google
#                                               sign-in with invites; AUTH_MODE=iap: only your account via IAP).
#                                               "staging" deploys gymhub-board-staging: same project and database,
#                                               every collection prefixed "staging-", bundles under staging/
#   deploy/gcp.sh github-ci [owner/repo]        let GitHub Actions deploy (Workload Identity Federation, no key):
#                                               staging after every green push to main, production on a tag v*
#                                               or on request
#   deploy/gcp.sh iap-client <client.json>      give IAP its OAuth client (projects without an organization)
#   deploy/gcp.sh runner                        how to connect this machine's runner with a worker token
#   deploy/gcp.sh budget [USD]                  monthly budget with alerts at 50/90/100% and on forecast
#                                               (amount: argument, else BUDGET_USD, else 5)
#   deploy/gcp.sh ttl                           let Firestore delete scratch documents (their expire_at passed)
#   deploy/gcp.sh anthropic-key                 store an Anthropic API key (read from stdin) so essays are
#                                               graded by Claude; takes effect on the next deploy
#   deploy/gcp.sh status                        print the URL and what is configured
#
# Settings are kept in deploy/gcp.env (git-ignored). Every step is safe to re-run.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
ENV_FILE="$HERE/gcp.env"
SERVICE=gymhub-board
BOARD_SA_NAME=gymhub-board
DEPLOYER_SA_NAME=gymhub-deployer
WIF_POOL=github
WIF_PROVIDER=human-gym
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
# DEPLOY_CI=1 (GitHub Actions): only build and deploy. Buckets, secrets and IAM were set up by a deploy
# from your machine, and the deployer account is not allowed to change them.
DEPLOY_CI="${DEPLOY_CI:-}"

[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
REGION="${REGION:-us-central1}"

save_env() {
  cat > "$ENV_FILE" << EOF
PROJECT_ID=$PROJECT_ID
REGION=$REGION
OWNER_EMAIL=$OWNER_EMAIL
AUTH_MODE=${AUTH_MODE:-app}
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}
IAP_ACCESS=${IAP_ACCESS:-owner}
BUDGET_USD=${BUDGET_USD:-}
BUDGET_ALERT_CHANNEL=${BUDGET_ALERT_CHANNEL:-}
EOF
}

need_project() {
  if [[ -z "${PROJECT_ID:-}" ]]; then
    echo "No project yet. Run: deploy/gcp.sh create <BILLING_ACCOUNT_ID>" >&2
    exit 1
  fi
}

cmd_create() {
  local billing="${1:-}"
  if [[ -z "$billing" ]]; then
    echo "usage: deploy/gcp.sh create <BILLING_ACCOUNT_ID>   (see: gcloud billing accounts list)" >&2
    exit 1
  fi
  OWNER_EMAIL="${OWNER_EMAIL:-$(gcloud config get-value account 2> /dev/null)}"
  if [[ -z "$OWNER_EMAIL" ]]; then
    echo "Sign in first: gcloud auth login" >&2
    exit 1
  fi
  PROJECT_ID="${PROJECT_ID:-human-gym-$(head -c 3 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
  save_env

  if gcloud projects describe "$PROJECT_ID" > /dev/null 2>&1; then
    echo "Project $PROJECT_ID exists."
  else
    echo "Creating project $PROJECT_ID"
    gcloud projects create "$PROJECT_ID" --name="Human gym"
  fi
  gcloud billing projects link "$PROJECT_ID" --billing-account="$billing" > /dev/null
  gcloud config set project "$PROJECT_ID" > /dev/null

  echo "Enabling APIs (takes a minute)"
  gcloud services enable --project "$PROJECT_ID" \
    run.googleapis.com firestore.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com iap.googleapis.com iam.googleapis.com \
    cloudresourcemanager.googleapis.com

  if ! gcloud firestore databases describe --project "$PROJECT_ID" --database='(default)' > /dev/null 2>&1; then
    echo "Creating Firestore (native mode) in $REGION"
    gcloud firestore databases create --project "$PROJECT_ID" --location="$REGION" --type=firestore-native
  fi

  local sa="$BOARD_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "$sa" --project "$PROJECT_ID" > /dev/null 2>&1; then
    gcloud iam service-accounts create "$BOARD_SA_NAME" --project "$PROJECT_ID" \
      --display-name="Human Gym board (Cloud Run)"
  fi
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$sa" \
    --role=roles/datastore.user --condition=None > /dev/null

  # Image repository for `run deploy --source`. On a brand-new project the owner's permissions
  # take a minute or two to reach Artifact Registry, so retry.
  local tries=0
  until gcloud artifacts repositories describe cloud-run-source-deploy --location "$REGION" --project "$PROJECT_ID" > /dev/null 2>&1 \
    || gcloud artifacts repositories create cloud-run-source-deploy --repository-format=docker \
      --location "$REGION" --project "$PROJECT_ID" --description="Human Gym board images" > /dev/null 2>&1; do
    tries=$((tries + 1))
    if ((tries > 15)); then
      echo "Could not create the Artifact Registry repository. Re-run this step in a few minutes." >&2
      exit 1
    fi
    echo "Waiting for Artifact Registry permissions..."
    sleep 20
  done
  # Keep only the 5 newest board images; every deploy adds one.
  local policy
  policy="$(mktemp --suffix=.json)"
  echo '[{"name":"keep-newest-5","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":5}},
         {"name":"delete-older","action":{"type":"Delete"},"condition":{"tagState":"any","olderThan":"1d"}}]' > "$policy"
  gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy --location "$REGION" --project "$PROJECT_ID" \
    --policy="$policy" --no-dry-run > /dev/null
  rm -f "$policy"
  echo "Project ready: $PROJECT_ID. Next: deploy/gcp.sh deploy"
}

cmd_deploy() {
  need_project
  local number sa stage target="${1:-production}" service="$SERVICE" prefix="" max=2
  case "$target" in
    production) ;;
    staging) service="$SERVICE-staging"; prefix="staging-"; max=1 ;;
    *) echo "deploy takes no argument (production) or 'staging', not '$target'" >&2; exit 1 ;;
  esac
  number="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
  sa="$BOARD_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

  # Upload only what the board image needs.
  stage="$(mktemp -d)"
  trap 'rm -rf "$stage"' RETURN
  mkdir -p "$stage/gymhub"
  cp "$HERE/Dockerfile.board" "$stage/Dockerfile"
  cp "$HERE/requirements-board.txt" "$stage/"
  cp "$REPO/gymhub/__init__.py" "$REPO/gymhub/store.py" "$REPO/gymhub/jobqueue.py" "$REPO/gymhub/quotas.py" "$REPO/gymhub/bundle.py" "$REPO/gymhub/routing.py" "$REPO/gymhub/packages.py" "$REPO/gymhub/gymversions.py" "$REPO/gymhub/blobs.py" "$REPO/gymhub/gymaccess.py" "$REPO/gymhub/gymdraft.py" "$REPO/gymhub/gymmd.py" "$REPO/gymhub/GYM_MARKDOWN.md" "$REPO/gymhub/kinds.py" "$REPO/gymhub/tutor.py" "$REPO/gymhub/reviews.py" "$REPO/gymhub/leaderboard.py" "$REPO/gymhub/health.py" "$REPO/gymhub/poolview.py" "$REPO/gymhub/accounts.py" "$stage/gymhub/"
  (cd "$REPO" && "$PYTHON" -m gymhub.bundle "$stage/worker-bundle.tar.gz")
  rsync -a --exclude __pycache__ "$REPO/gymhub/board" "$stage/gymhub/"
  mkdir -p "$stage/pool" && cp "$REPO/pool/__init__.py" "$REPO/pool/client.py" "$stage/pool/"  # the Compute page reads the pool server
  mkdir -p "$stage/gymkit" && cp "$REPO/gymkit/compare.py" "$REPO/gymkit/examples.py" "$stage/gymkit/"  # served to browsers (/browser/)

  [[ -n "$DEPLOY_CI" ]] || ensure_bundle_bucket "$sa"
  # Optional: an Anthropic API key for essay grading (deploy/gcp.sh anthropic-key).
  local claude_secret=""
  if gcloud secrets describe gymhub-anthropic-key --project "$PROJECT_ID" --quiet < /dev/null > /dev/null 2>&1; then
    [[ -n "$DEPLOY_CI" ]] || gcloud secrets add-iam-policy-binding gymhub-anthropic-key --project "$PROJECT_ID" \
      --member="serviceAccount:$sa" --role=roles/secretmanager.secretAccessor --condition=None > /dev/null
    claude_secret="ANTHROPIC_API_KEY=gymhub-anthropic-key:latest"
  fi
  local env="GYMHUB_STORE=firestore:$PROJECT_ID|GOOGLE_CLOUD_PROJECT=$PROJECT_ID|GYMHUB_BLOBS=gs://$PROJECT_ID-gym-bundles${prefix:+/staging}"
  [[ -z "$prefix" ]] || env="$env|GYMHUB_STORE_PREFIX=$prefix"
  if [[ "${AUTH_MODE:-app}" == "app" ]]; then
    if [[ -z "${GOOGLE_CLIENT_ID:-}" ]]; then
      echo "Set GOOGLE_CLIENT_ID in deploy/gcp.env (the OAuth client's ID; see docs/self-hosting.md 'Sign-in')." >&2
      exit 1
    fi
    [[ -n "$DEPLOY_CI" ]] || ensure_session_secret "$sa"
    echo "Building and deploying $service with in-app Google sign-in (Cloud Build, a few minutes)"
    gcloud run deploy "$service" \
      --project "$PROJECT_ID" --region "$REGION" \
      --source "$stage" \
      --service-account "$sa" \
      --allow-unauthenticated \
      --no-iap \
      --min-instances 0 --max-instances "$max" \
      --cpu 1 --memory 512Mi --concurrency 40 --timeout 3600 \
      --set-env-vars "^|^GYMHUB_AUTH=app|$env|GYMHUB_GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID|GYMHUB_ADMIN_EMAILS=$OWNER_EMAIL" \
      --set-secrets "GYMHUB_SESSION_KEY=gymhub-session-key:latest${claude_secret:+,$claude_secret}" \
      --quiet
  else
    echo "Building and deploying $service behind IAP (Cloud Build, a few minutes)"
    gcloud run deploy "$service" \
      --project "$PROJECT_ID" --region "$REGION" \
      --source "$stage" \
      --service-account "$sa" \
      --no-allow-unauthenticated \
      --iap \
      --min-instances 0 --max-instances "$max" \
      --cpu 1 --memory 512Mi --concurrency 40 --timeout 3600 \
      --set-env-vars "^|^GYMHUB_AUTH=iap|$env|GYMHUB_ADMIN_EMAILS=$OWNER_EMAIL|GYMHUB_IAP_AUDIENCE=/projects/$number/locations/$REGION/services/$service" \
      ${claude_secret:+--set-secrets "$claude_secret"} \
      --quiet
    # Machines cannot pass IAP without a Google login, so they get the same image a second time,
    # serving only the worker API (GYMHUB_SURFACE=workers) and authenticated by worker tokens.
    deploy_workers "$service" "$sa" "$max" "$env" "$number"
    if [[ -n "$DEPLOY_CI" ]]; then
      cmd_status "$service"
      return
    fi
    echo "Locking the board to $OWNER_EMAIL"
    # IAP's service agent may invoke the service; nobody else can call Cloud Run directly.
    gcloud run services add-iam-policy-binding "$service" --project "$PROJECT_ID" --region "$REGION" \
      --member="serviceAccount:service-$number@gcp-sa-iap.iam.gserviceaccount.com" \
      --role=roles/run.invoker > /dev/null
    gcloud run services remove-iam-policy-binding "$service" --project "$PROJECT_ID" --region "$REGION" \
      --member=allUsers --role=roles/run.invoker > /dev/null 2>&1 || true
    # Who gets past IAP. IAP_ACCESS=everyone: any Google account; the board then admits only admins,
    # invited users and allowed domains. Otherwise only your account.
    gcloud iap web add-iam-policy-binding --project "$PROJECT_ID" --region "$REGION" \
      --resource-type=cloud-run --service="$service" \
      --member="user:$OWNER_EMAIL" --role=roles/iap.httpsResourceAccessor --condition=None > /dev/null
    if [[ "${IAP_ACCESS:-owner}" == "everyone" ]]; then
      gcloud iap web add-iam-policy-binding --project "$PROJECT_ID" --region "$REGION" \
        --resource-type=cloud-run --service="$service" \
        --member=allAuthenticatedUsers --role=roles/iap.httpsResourceAccessor --condition=None > /dev/null
    else
      gcloud iap web remove-iam-policy-binding --project "$PROJECT_ID" --region "$REGION" \
        --resource-type=cloud-run --service="$service" \
        --member=allAuthenticatedUsers --role=roles/iap.httpsResourceAccessor --condition=None > /dev/null 2>&1 || true
    fi
  fi
  cmd_status "$service"
}

deploy_workers() {
  local service="$1" sa="$2" max="$3" env="$4" number="$5" workers="$1-workers" image url
  image="$(gcloud run services describe "$service" --project "$PROJECT_ID" --region "$REGION" \
    --format='value(spec.template.spec.containers[0].image)')"
  echo "Deploying $workers: only the worker API, without IAP, so a machine needs just its worker token"
  gcloud run deploy "$workers" \
    --project "$PROJECT_ID" --region "$REGION" \
    --image "$image" \
    --service-account "$sa" \
    --allow-unauthenticated \
    --no-iap \
    --min-instances 0 --max-instances "$max" \
    --cpu 1 --memory 512Mi --concurrency 40 --timeout 3600 \
    --set-env-vars "^|^GYMHUB_AUTH=iap|GYMHUB_SURFACE=workers|$env|GYMHUB_ADMIN_EMAILS=$OWNER_EMAIL|GYMHUB_IAP_AUDIENCE=/projects/$number/locations/$REGION/services/$service" \
    --quiet
  url="$(gcloud run services describe "$workers" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')"
  # The board shows machines this URL in the install command (Compute page).
  gcloud run services update "$service" --project "$PROJECT_ID" --region "$REGION" \
    --update-env-vars "GYMHUB_WORKER_URL=$url" --quiet > /dev/null
  echo "Worker URL: $url  (RUNNER_API in deploy/gcp.env for deploy/install-runner.sh)"
}

# Uploaded gym bundles (gymhub/blobs.py). us-central1 Standard storage is in the free tier up to 5 GB.
ensure_bundle_bucket() {
  local sa="$1" bucket="gs://$PROJECT_ID-gym-bundles"
  if ! gcloud storage buckets describe "$bucket" --project "$PROJECT_ID" > /dev/null 2>&1; then
    echo "Creating $bucket for uploaded gyms"
    gcloud services enable storage.googleapis.com --project "$PROJECT_ID"
    gcloud storage buckets create "$bucket" --project "$PROJECT_ID" --location="$REGION" \
      --uniform-bucket-level-access --public-access-prevention > /dev/null
  fi
  gcloud storage buckets add-iam-policy-binding "$bucket" --member="serviceAccount:$sa" \
    --role=roles/storage.objectAdmin > /dev/null
}

ensure_session_secret() {
  # A random key that signs session cookies. Created once; rotate it to sign everyone out.
  local sa="$1"
  gcloud services enable secretmanager.googleapis.com --project "$PROJECT_ID" > /dev/null
  if ! gcloud secrets describe gymhub-session-key --project "$PROJECT_ID" > /dev/null 2>&1; then
    head -c 48 /dev/urandom | base64 | tr -d '\n' |
      gcloud secrets create gymhub-session-key --project "$PROJECT_ID" --replication-policy=automatic --data-file=- > /dev/null
  fi
  gcloud secrets add-iam-policy-binding gymhub-session-key --project "$PROJECT_ID" \
    --member="serviceAccount:$sa" --role=roles/secretmanager.secretAccessor --condition=None > /dev/null
}

cmd_runner() {
  # The runner talks to the board with a worker token and holds no Google login (issue #52).
  need_project
  local url
  url="$(gcloud run services describe "$SERVICE-workers" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)' 2> /dev/null || true)"
  [[ -n "$url" ]] || url="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)' 2> /dev/null || true)"
  echo "The runner uses a worker token, not this machine's Google login."
  echo "  1. On the board's Compute page, create a worker token for this machine."
  echo "  2. Save it:  install -m 600 /dev/stdin ~/.config/gymhub/worker-token   (paste, then Ctrl+D)"
  echo "  3. Add to deploy/gcp.env:  RUNNER_API=${url:-<the worker URL from deploy/gcp.sh status>}"
  echo "  4. Install the service:  deploy/install-runner.sh"
  if gcloud auth application-default print-access-token > /dev/null 2>&1; then
    echo "  5. Remove the Google login the runner used before:  gcloud auth application-default revoke"
  fi
}

cmd_iap_client() {
  # Projects outside a Google Workspace organization need their own OAuth client for IAP.
  # Create it in the console (see docs/self-hosting.md), download its JSON, then run this with the file.
  need_project
  local file="${1:-}" settings
  if [[ ! -f "$file" ]]; then
    echo "usage: deploy/gcp.sh iap-client <downloaded-oauth-client.json>" >&2
    exit 1
  fi
  settings="$(mktemp --suffix=.yaml)"
  trap 'rm -f "$settings"' RETURN
  chmod 600 "$settings"
  python3 - "$file" > "$settings" << 'PY'
import json, sys
web = json.load(open(sys.argv[1]))["web"]
print("access_settings:\n  oauth_settings:")
print(f"    client_id: {web['client_id']}\n    client_secret: {web['client_secret']}")
PY
  local service
  for service in "$SERVICE" "$SERVICE-staging"; do  # every board that exists; run again after a first staging deploy
    gcloud run services describe "$service" --project "$PROJECT_ID" --region "$REGION" > /dev/null 2>&1 || continue
    gcloud iap settings set "$settings" --project "$PROJECT_ID" --region "$REGION" \
      --resource-type=cloud-run --service="$service" > /dev/null
    echo "IAP sign-in configured for $service"
  done
  GOOGLE_CLIENT_ID="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["web"]["client_id"])' "$file")"
  save_env
  echo "IAP now uses OAuth client $GOOGLE_CLIENT_ID (also saved for in-app sign-in)"
}

cmd_anthropic_key() {
  need_project
  gcloud services enable secretmanager.googleapis.com --project "$PROJECT_ID" > /dev/null
  echo "Paste the Anthropic API key, then press Enter (it is not echoed):" >&2
  local key
  read -rs key
  [[ "$key" == sk-ant-* ]] || { echo "That does not look like an Anthropic API key (sk-ant-...)." >&2; exit 1; }
  if gcloud secrets describe gymhub-anthropic-key --project "$PROJECT_ID" --quiet < /dev/null > /dev/null 2>&1; then
    printf '%s' "$key" | gcloud secrets versions add gymhub-anthropic-key --project "$PROJECT_ID" --data-file=- > /dev/null
  else
    printf '%s' "$key" | gcloud secrets create gymhub-anthropic-key --project "$PROJECT_ID" --replication-policy=automatic --data-file=- > /dev/null
  fi
  echo "Stored. Run deploy/gcp.sh deploy to start grading essays with Claude."
}

cmd_budget() {
  # A monthly budget for this project. The billing account's admins get emails at 50%, 90% and 100%
  # of actual spend, and when the month's forecast passes 100%. BUDGET_ALERT_CHANNEL (a Cloud
  # Monitoring notification channel, projects/<p>/notificationChannels/<id>) also notifies there.
  # The amount: the argument, else BUDGET_USD (deploy/gcp.env or the environment), else 5.
  # Re-running changes the amount of the existing budget instead of adding a second one.
  need_project
  local amount="${1:-${BUDGET_USD:-5}}" billing name existing
  if ! [[ "$amount" =~ ^[0-9]+([.][0-9]{1,2})?$ ]]; then
    echo "The budget is an amount in USD, like 5 or 12.50, not '$amount'" >&2
    exit 1
  fi
  billing="$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingAccountName)' | sed 's|billingAccounts/||')"
  if [[ -z "$billing" ]]; then
    echo "Project $PROJECT_ID has no billing account" >&2
    exit 1
  fi
  name="Human gym (gymhub)"  # the name earlier versions created, so re-running finds theirs too
  gcloud services enable billingbudgets.googleapis.com --project "$PROJECT_ID" > /dev/null
  existing="$(gcloud billing budgets list --billing-account="$billing" --billing-project="$PROJECT_ID" \
    --filter="displayName=\"$name\"" --format='value(name)' | head -n 1)"
  if [[ -n "$existing" ]]; then
    gcloud billing budgets update "$existing" --billing-account="$billing" --billing-project="$PROJECT_ID" \
      --budget-amount="${amount}USD" > /dev/null
    echo "Budget updated to \$$amount/month."
    return
  fi
  gcloud billing budgets create --billing-account="$billing" --billing-project="$PROJECT_ID" \
    --display-name="$name" --budget-amount="${amount}USD" --calendar-period=month \
    --filter-projects="projects/$PROJECT_ID" \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 \
    --threshold-rule=percent=1.0,basis=forecasted-spend \
    ${BUDGET_ALERT_CHANNEL:+--notifications-rule-monitoring-notification-channels="$BUDGET_ALERT_CHANNEL"} > /dev/null
  echo "Budget of \$$amount/month set; alerts go to the billing account's admins${BUDGET_ALERT_CHANNEL:+ and $BUDGET_ALERT_CHANNEL}."
}

# Collections whose documents may carry expire_at: keep equal to EXPIRING in gymhub/store.py
# (tests/test_retention.py checks). Submissions and everything else never get the field, so stay.
TTL_COLLECTIONS="jobs usage attempts throttle registrations"

cmd_ttl() {
  # A TTL policy per collection group: Firestore deletes a document some time (usually within a day)
  # after its expire_at timestamp. Staging's collections are prefixed, so they get their own policies.
  # Creating a policy takes a while on the server side; --async returns at once. Safe to re-run.
  need_project
  local prefix collection
  for prefix in "" "staging-"; do
    for collection in $TTL_COLLECTIONS; do
      echo "TTL on $prefix$collection.expire_at"
      gcloud firestore fields ttls update expire_at --project "$PROJECT_ID" \
        --collection-group="$prefix$collection" --enable-ttl --async --quiet > /dev/null
    done
  done
  echo "Check progress with: gcloud firestore fields ttls list --project $PROJECT_ID"
}

cmd_status() {
  need_project
  local url service="${1:-$SERVICE}"
  url="$(gcloud run services describe "$service" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)' 2> /dev/null || true)"
  echo
  echo "project  $PROJECT_ID ($REGION)"
  echo "owner    $OWNER_EMAIL"
  echo "board    ${url:-not deployed} ($service)"
  local workers_url
  workers_url="$(gcloud run services describe "$service-workers" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)' 2> /dev/null || true)"
  [[ -z "$workers_url" ]] || echo "workers  $workers_url ($service-workers: worker API only, RUNNER_API for machines)"
  echo "IAP members:"
  gcloud iap web get-iam-policy --project "$PROJECT_ID" --region "$REGION" --resource-type=cloud-run \
    --service="$service" --flatten='bindings[].members' --format='value(bindings.members)' 2> /dev/null || echo "  (none yet)"
}

cmd_github_ci() {
  # GitHub Actions deploys as gymhub-deployer through Workload Identity Federation: GitHub's OIDC token
  # for this repository is exchanged for short-lived credentials, so no key is stored anywhere.
  # The deployer may build and deploy the two board services, and nothing else: buckets, secrets
  # and who may sign in stay with deploys from your machine.
  need_project
  local repo="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}" number deployer board build_sa
  number="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
  deployer="$DEPLOYER_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
  board="$BOARD_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
  build_sa="$number-compute@developer.gserviceaccount.com"  # Cloud Build's default for --source deploys
  gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project "$PROJECT_ID"
  if ! gcloud iam service-accounts describe "$deployer" --project "$PROJECT_ID" > /dev/null 2>&1; then
    gcloud iam service-accounts create "$DEPLOYER_SA_NAME" --project "$PROJECT_ID" \
      --display-name="GitHub Actions: deploys the board" > /dev/null
  fi
  # storage.bucketViewer: bucket metadata only (no objects), so `run deploy --source` can find its source bucket.
  local role
  for role in roles/run.admin roles/cloudbuild.builds.editor roles/artifactregistry.writer \
    roles/serviceusage.serviceUsageConsumer roles/secretmanager.viewer roles/logging.viewer \
    roles/storage.bucketViewer; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$deployer" --role="$role" \
      --condition=None > /dev/null
  done
  # roles/storage.bucketViewer (above): `gcloud run deploy --source` lists the project's buckets to find
  # its upload bucket. It sees bucket names and settings, never objects.
  # Source uploads only: not the bundle bucket, where learners' gyms live.
  local sources="gs://run-sources-$PROJECT_ID-$REGION"
  gcloud storage buckets describe "$sources" > /dev/null 2>&1 ||
    gcloud storage buckets create "$sources" --project "$PROJECT_ID" --location="$REGION" --uniform-bucket-level-access > /dev/null
  gcloud storage buckets add-iam-policy-binding "$sources" --member="serviceAccount:$deployer" \
    --role=roles/storage.admin > /dev/null
  for sa in "$board" "$build_sa"; do  # deploy as the board's account; Cloud Build runs as the build account
    gcloud iam service-accounts add-iam-policy-binding "$sa" --project "$PROJECT_ID" \
      --member="serviceAccount:$deployer" --role=roles/iam.serviceAccountUser > /dev/null
  done
  if ! gcloud iam workload-identity-pools describe "$WIF_POOL" --project "$PROJECT_ID" --location=global > /dev/null 2>&1; then
    gcloud iam workload-identity-pools create "$WIF_POOL" --project "$PROJECT_ID" --location=global \
      --display-name="GitHub Actions" > /dev/null
  fi
  # Only workflows of this repository, running on main or on a tag v* (production), may become the deployer.
  local condition="assertion.repository == '$repo' && (assertion.ref == 'refs/heads/main' || assertion.ref.startsWith('refs/tags/v'))"
  if gcloud iam workload-identity-pools providers describe "$WIF_PROVIDER" --project "$PROJECT_ID" --location=global \
    --workload-identity-pool="$WIF_POOL" > /dev/null 2>&1; then
    gcloud iam workload-identity-pools providers update-oidc "$WIF_PROVIDER" --project "$PROJECT_ID" --location=global \
      --workload-identity-pool="$WIF_POOL" --attribute-condition="$condition" > /dev/null
  else
    gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" --project "$PROJECT_ID" --location=global \
      --workload-identity-pool="$WIF_POOL" --issuer-uri="https://token.actions.githubusercontent.com" \
      --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
      --attribute-condition="$condition" > /dev/null
  fi
  gcloud iam service-accounts add-iam-policy-binding "$deployer" --project "$PROJECT_ID" \
    --role=roles/iam.workloadIdentityUser \
    --member="principalSet://iam.googleapis.com/projects/$number/locations/global/workloadIdentityPools/$WIF_POOL/attribute.repository/$repo" > /dev/null
  # The workflow reads these. deploy/gcp.env holds no secrets, but it names your account, so it is a secret.
  gh variable set GCP_WIF_PROVIDER --repo "$repo" \
    --body "projects/$number/locations/global/workloadIdentityPools/$WIF_POOL/providers/$WIF_PROVIDER"
  gh variable set GCP_DEPLOYER --repo "$repo" --body "$deployer"
  gh secret set GCP_ENV --repo "$repo" < "$ENV_FILE"
  echo "GitHub Actions in $repo can now deploy: staging after every green push to main,"
  echo "production on a pushed tag v* (git tag v1.2.3 && git push origin v1.2.3) or from the Actions tab."
  echo "Tip: protect v* tags (Settings > Rules > Tag rulesets) so only you can create them."
}

case "${1:-}" in
  create) shift; cmd_create "$@" ;;
  deploy) shift; cmd_deploy "$@" ;;
  github-ci) shift; cmd_github_ci "$@" ;;
  runner) cmd_runner ;;
  iap-client) shift; cmd_iap_client "$@" ;;
  budget) shift; cmd_budget "$@" ;;
  ttl) cmd_ttl ;;
  anthropic-key) cmd_anthropic_key ;;
  status) cmd_status ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
