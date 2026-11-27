"""The board's privacy policy and terms of use: public pages, readable before signing in.

The text is in gymhub/board/legal/*.html (a DRAFT for the operator to review; not legal advice). Who
runs the board and how to reach them come from the environment, so a self-hosted board names its own
operator:

    GYMHUB_OPERATOR        who runs the board, as people should read it  (default: "the board's admins")
    GYMHUB_CONTACT_EMAIL   where privacy and terms questions go          (default: the board's admins)
    GYMHUB_LEGAL_UPDATED   the "Last updated" date                        (default: LAST_UPDATED)
"""

import html
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

TEXTS = Path(__file__).parent / "legal"
LAST_UPDATED = "26 November 2026"
PAGES = {"privacy": "Privacy policy", "terms": "Terms of use"}

router = APIRouter()


def _contact() -> str:
    email = os.environ.get("GYMHUB_CONTACT_EMAIL", "").strip()
    if not email:
        return "the board's admins"
    quoted = html.escape(email)
    return f'<a href="mailto:{quoted}">{quoted}</a>'


def render(name: str) -> str:
    body = (TEXTS / f"{name}.html").read_text()
    fields = {
        "operator": html.escape(os.environ.get("GYMHUB_OPERATOR", "").strip() or "the board's admins"),
        "contact": _contact(),
        "updated": html.escape(os.environ.get("GYMHUB_LEGAL_UPDATED", "").strip() or LAST_UPDATED),
    }
    for key, value in fields.items():
        body = body.replace("{{" + key + "}}", value)
    links = " · ".join(f'<a href="/{page}">{title}</a>' for page, title in PAGES.items() if page != name)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{PAGES[name]} · Human Gym</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body class="legal-page">
  <header class="app-header"><a class="brand" href="/"><span class="logo"></span>Human Gym</a></header>
  <main class="legal">
{body}
  <p class="muted small legal-links"><a href="/">Back to the board</a> · {links}</p>
  </main>
</body>
</html>"""


def _page(name: str):
    def page():
        return HTMLResponse(render(name), headers={"Cache-Control": "no-cache"})

    page.__name__ = f"{name}_page"
    return page


for _name in PAGES:
    router.add_api_route(f"/{_name}", _page(_name), methods=["GET"], include_in_schema=False)
