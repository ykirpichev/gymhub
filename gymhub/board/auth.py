"""Who is calling, and what they may do.

GYMHUB_AUTH=app   (default) Sign in with Google in the board. The browser gets a Google ID token,
                  the API verifies it, creates or finds the user, and sets a signed session cookie.
                  Needs GYMHUB_GOOGLE_CLIENT_ID and GYMHUB_SESSION_KEY; GYMHUB_ADMIN_EMAILS lists
                  accounts that become admins on first sign-in.
GYMHUB_AUTH=iap   The board sits behind Identity-Aware Proxy. IAP proves who the visitor is (its JWT is
                  verified against GYMHUB_IAP_AUDIENCE); the user model below decides what they may do,
                  exactly as with in-app sign-in. New visitors accept an invite with /api/auth/accept-invite.
GYMHUB_AUTH=local A name and a password kept by the board, no Google: for a board on a LAN, a Tailscale
                  network or behind `ssh -L`. Every account starts from an invite link (the admin's too:
                  `gymhub-board --lan` prints one for the owner). GYMHUB_ADMIN_EMAILS lists the names of
                  admins, which only an invite made for that name can claim. Needs GYMHUB_SESSION_KEY.
GYMHUB_AUTH=dev   No checks; every request is a local admin. Only for running the board on localhost.

Who may sign in (app mode): an existing active user; an email in GYMHUB_ADMIN_EMAILS; someone
holding a valid invite; or an email whose domain is allowed in settings/auth.

Firestore:
  users/{uid}       email, name, picture, role (admin|member), status (active|disabled), created_at, last_login
  invites/{code}    created_by, email (optional: only this address; in local mode only this name), uses_left,
                    expires_at, created_at; local mode: role (admin invites), reset_uid (a password reset link)
  passwords/{uid}   local mode: uid, hash (scrypt). Apart from users/ so no user listing can carry it.
  settings/auth     allowed_domains [], members_can_run (bool)
"""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass

import requests
from fastapi import Depends, HTTPException, Request, Response
from google.auth import jwt

from gymhub import gymaccess

IAP_CERTS_URL = "https://www.gstatic.com/iap/verify/public_key"
IAP_ISSUER = "https://cloud.google.com/iap"
SESSION_COOKIE = "gymhub_session"
SESSION_SECONDS = 30 * 86400
USER_CACHE_SECONDS = 30

_certs: dict = {}
_certs_at = 0.0
_lock = threading.Lock()
_user_cache: dict[str, tuple[float, dict | None]] = {}


@dataclass
class User:
    uid: str
    email: str
    name: str = ""
    picture: str = ""
    role: str = "member"  # admin | member

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def public(self) -> dict:
        return asdict(self) | {"is_admin": self.is_admin}


def mode() -> str:
    return os.environ.get("GYMHUB_AUTH", "app")


def _store():
    from gymhub.board.app import store

    return store


def admin_emails() -> set[str]:
    # GYMHUB_ALLOWED_EMAILS is the older IAP setting; those accounts are admins too.
    names = os.environ.get("GYMHUB_ADMIN_EMAILS", "") + "," + os.environ.get("GYMHUB_ALLOWED_EMAILS", "")
    return {email.strip().lower() for email in names.split(",") if email.strip()}


def auth_settings() -> dict:
    settings = _store().get("settings", "auth") or {}
    settings.setdefault("allowed_domains", [])
    settings.setdefault("members_can_run", False)
    settings.setdefault("public_gyms_need_review", True)  # #121: an admin approves gyms before they go public (on unless turned off)
    return settings


# ---- session cookie --------------------------------------------------------------------------


def _session_key() -> bytes:
    key = os.environ.get("GYMHUB_SESSION_KEY", "")
    if len(key) < 32:
        raise HTTPException(500, "Board is missing GYMHUB_SESSION_KEY")
    return key.encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_session(uid: str) -> str:
    payload = _b64(json.dumps({"uid": uid, "exp": int(time.time()) + SESSION_SECONDS}).encode())
    signature = _b64(hmac.new(_session_key(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}"


def read_session(token: str) -> str | None:
    try:
        payload, signature = token.split(".", 1)
        expected = _b64(hmac.new(_session_key(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        data = json.loads(_unb64(payload))
        if data.get("exp", 0) < time.time():
            return None
        return str(data["uid"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def set_session_cookie(response: Response, request: Request, uid: str) -> None:
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        SESSION_COOKIE, make_session(uid), max_age=SESSION_SECONDS, httponly=True, secure=secure, samesite="lax", path="/"
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


# ---- users -----------------------------------------------------------------------------------


def load_user(uid: str, fresh: bool = False) -> dict | None:
    now = time.time()
    cached = _user_cache.get(uid)
    if cached and not fresh and now - cached[0] < USER_CACHE_SECONDS:
        return cached[1]
    doc = _store().get("users", uid)
    _user_cache[uid] = (now, doc)
    return doc


def forget_user(uid: str) -> None:
    _user_cache.pop(uid, None)


def verify_google_token(credential: str) -> dict:
    """Claims of a Google ID token issued for our client. Replaced in tests."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    client_id = os.environ.get("GYMHUB_GOOGLE_CLIENT_ID")
    if not client_id:
        raise HTTPException(500, "Board is missing GYMHUB_GOOGLE_CLIENT_ID")
    try:
        return id_token.verify_oauth2_token(credential, google_requests.Request(), client_id)
    except ValueError as exc:
        raise HTTPException(401, f"Google sign-in could not be verified: {exc}") from exc


def _invite_usable(invite: dict | None, email: str) -> str | None:
    """None when the invite can be used by `email`, else the reason it cannot."""
    if invite is None:
        return "This invite link is not valid."
    if invite.get("expires_at") and invite["expires_at"] < time.time():
        return "This invite link has expired."
    if invite.get("uses_left", 1) <= 0:
        return "This invite link has already been used."
    if invite.get("email") and invite["email"].lower() != email:
        return f"This invite is for {invite['email']}. Sign in with that account."
    return None


def sign_in_google(credential: str, invite_code: str | None) -> tuple[str, dict]:
    """Verify the Google token and return (uid, user document), creating the user when allowed."""
    return admit(verify_google_token(credential), invite_code)


def admit(claims: dict, invite_code: str | None) -> tuple[str, dict]:
    """A verified Google identity (from in-app sign-in or IAP): the user document, created when allowed."""
    if not claims.get("email_verified"):
        raise HTTPException(403, "Your Google account's email is not verified.")
    email = str(claims["email"]).lower()
    uid = f"g_{claims['sub']}"
    store = _store()
    now = time.time()
    profile = {"email": email, "name": claims.get("name", ""), "picture": claims.get("picture", ""), "last_login": now}
    existing = store.get("users", uid)
    if existing:
        if existing.get("status") == "disabled":
            raise HTTPException(403, "Your account has been disabled. Ask the board's admin.")
        store.set("users", uid, profile, merge=True)
        forget_user(uid)
        return uid, {**existing, **profile}

    role = None
    joined_by = None
    if email in admin_emails():
        role, joined_by = "admin", "admin list"
    elif invite_code:
        invite = store.get("invites", invite_code)
        reason = _invite_usable(invite, email) or (
            "This is a password reset link, not an invite." if invite.get("reset_uid") else None)
        if reason:
            raise HTTPException(403, reason)
        # Spend one use atomically, so a single-use link cannot admit two people.
        _spend_invite(invite_code, invite)
        role, joined_by = invite.get("role", "member"), f"invite {invite_code[:6]}…"
    else:
        domain = email.rsplit("@", 1)[-1]
        if domain in {d.lower() for d in auth_settings()["allowed_domains"]}:
            role, joined_by = "member", f"domain {domain}"
    if role is None:
        raise HTTPException(403, "This board is invite-only. Ask its admin for an invite link.")
    doc = {**profile, "role": role, "status": "active", "created_at": now, "joined_by": joined_by}
    store.set("users", uid, doc)
    forget_user(uid)
    gymaccess.claim_invites(store, uid, email)  # gyms shared with this email before they signed in
    return uid, doc


# ---- local accounts (GYMHUB_AUTH=local) -------------------------------------------------------------

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
MIN_PASSWORD = 8
SIGNIN_FAILURES = 10  # wrong passwords per name per window before sign-in pauses for that name
SIGNIN_WINDOW = 15 * 60


def local_uid(username: str) -> str:
    return f"l_{username}"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${_b64(salt)}${_b64(digest)}"


def check_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        got = hashlib.scrypt(password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p), dklen=len(_unb64(digest)))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, _unb64(digest))


def clean_username(username: str) -> str:
    name = (username or "").strip().lower()
    if not USERNAME_RE.match(name):
        raise HTTPException(400, "Pick a name of 2 to 32 characters: lowercase letters, digits, dots, dashes and "
                                 "underscores, starting with a letter or digit.")
    return name


def _check_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise HTTPException(400, f"Use a password of at least {MIN_PASSWORD} characters.")


def invite_info(code: str) -> dict:
    """What the account form needs to know about an invite link, before anyone signs in."""
    invite = _store().get("invites", code)
    reason = _invite_usable(invite, (invite or {}).get("email") or "")
    if reason:
        return {"valid": False, "reason": reason}
    return {"valid": True, "for": invite.get("email"), "reset": bool(invite.get("reset_uid")), "role": invite.get("role", "member")}


def _spend_invite(code: str, invite: dict) -> None:
    if not _store().claim("invites", code, {"uses_left": invite["uses_left"]}, {"uses_left": invite["uses_left"] - 1}):
        raise HTTPException(409, "This invite was just used. Try again.")


def register_local(username: str, password: str, name: str, invite_code: str) -> tuple[str, dict]:
    """Create an account from an invite link, or set a new password from a reset link."""
    store = _store()
    username = clean_username(username)
    _check_new_password(password)
    invite = store.get("invites", invite_code) if invite_code else None
    reason = _invite_usable(invite, username)
    if reason:
        raise HTTPException(403, reason)
    uid = local_uid(username)
    now = time.time()
    existing = store.get("users", uid)
    if invite.get("reset_uid"):
        if invite["reset_uid"] != uid or existing is None:
            raise HTTPException(403, "This password reset link is for another account.")
        if existing.get("status") == "disabled":
            raise HTTPException(403, "Your account has been disabled. Ask the board's admin.")
        _spend_invite(invite_code, invite)
        store.set("passwords", uid, {"uid": uid, "hash": hash_password(password), "updated_at": now})
        store.set("users", uid, {"last_login": now}, merge=True)
        forget_user(uid)
        return uid, {**existing, "last_login": now}
    if existing is not None:
        raise HTTPException(409, f"The name {username} is taken. Pick another.")
    if username in admin_emails() and (invite.get("email") or "").lower() != username:
        raise HTTPException(403, f"The name {username} is reserved for this board's admin.")
    role = "admin" if username in admin_emails() or invite.get("role") == "admin" else "member"
    _spend_invite(invite_code, invite)
    store.set("passwords", uid, {"uid": uid, "hash": hash_password(password), "updated_at": now})
    doc = {"email": username, "name": (name or "").strip()[:100] or username, "picture": "", "role": role,
           "status": "active", "created_at": now, "last_login": now, "joined_by": f"invite {invite_code[:6]}…"}
    store.set("users", uid, doc)
    forget_user(uid)
    gymaccess.claim_invites(store, uid, username)  # gyms shared with this name before the account existed
    return uid, doc


def _in_window(doc: dict | None, now: float) -> bool:
    """The window opens at a name's first wrong password, so it never resets mid-burst at a clock boundary."""
    return bool(doc) and now - doc.get("since", 0) < SIGNIN_WINDOW


def _count_failure(username: str) -> int:
    now = time.time()
    name = hashlib.sha256(f"signin:{username}".encode()).hexdigest()[:32]

    def apply(doc):
        from gymhub.store import expires_in

        if _in_window(doc, now):
            return {"since": doc["since"], "count": doc["count"] + 1, "expire_at": expires_in(86400)}
        return {"since": now, "count": 1, "expire_at": expires_in(86400)}

    return _store().transact("throttle", name, apply)["count"]


def _failures(username: str) -> int:
    doc = _store().get("throttle", hashlib.sha256(f"signin:{username}".encode()).hexdigest()[:32])
    return doc["count"] if _in_window(doc, time.time()) else 0


def sign_in_local(username: str, password: str) -> tuple[str, dict]:
    store = _store()
    username = (username or "").strip().lower()
    wrong = HTTPException(401, "Wrong name or password.")
    if not USERNAME_RE.match(username):
        raise wrong
    if _failures(username) >= SIGNIN_FAILURES:
        raise HTTPException(429, "Too many wrong passwords for this name. Wait 15 minutes, or ask an admin for a reset link.")
    uid = local_uid(username)
    stored = store.get("passwords", uid)
    if not stored or not check_password(password, stored.get("hash", "")):
        _count_failure(username)
        raise wrong
    doc = store.get("users", uid)
    if not doc or doc.get("status") != "active":
        raise HTTPException(403, "Your account has been disabled. Ask the board's admin.")
    store.set("users", uid, {"last_login": time.time()}, merge=True)
    forget_user(uid)
    return uid, doc


def change_password(uid: str, old: str, new: str) -> None:
    stored = _store().get("passwords", uid)
    if not stored or not check_password(old, stored.get("hash", "")):
        raise HTTPException(403, "Your current password is not right.")
    _check_new_password(new)
    _store().set("passwords", uid, {"uid": uid, "hash": hash_password(new), "updated_at": time.time()})


# ---- request dependencies ----------------------------------------------------------------------


def _iap_certs() -> dict:
    global _certs, _certs_at
    with _lock:
        if not _certs or time.time() - _certs_at > 3600:
            response = requests.get(IAP_CERTS_URL, timeout=10)
            response.raise_for_status()
            _certs = response.json()
            _certs_at = time.time()
        return _certs


def iap_claims(request: Request) -> dict:
    """The Google identity IAP vouches for, as claims shaped like a Google ID token's."""
    token = request.headers.get("x-goog-iap-jwt-assertion")
    if not token:
        raise HTTPException(status_code=401, detail="Missing IAP assertion")
    audience = os.environ.get("GYMHUB_IAP_AUDIENCE")
    if not audience:
        raise HTTPException(status_code=500, detail="Board is missing GYMHUB_IAP_AUDIENCE")
    try:
        claims = jwt.decode(token, certs=_iap_certs(), audience=audience)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid IAP assertion: {exc}") from exc
    if claims.get("iss") != IAP_ISSUER:
        raise HTTPException(status_code=401, detail="Wrong token issuer")
    # IAP's subject is "accounts.google.com:<id>": the same <id> Google sign-in uses, so uids match.
    return {"sub": str(claims["sub"]).split(":")[-1], "email": claims.get("email", ""), "email_verified": True}


def _user_from(uid: str, doc: dict) -> User:
    return User(uid=uid, email=doc["email"], name=doc.get("name", ""), picture=doc.get("picture", ""), role=doc.get("role", "member"))


def _iap_user(request: Request) -> User:
    """IAP checked who this is; the board's user model decides what they may do (admins, invites, domains)."""
    claims = iap_claims(request)
    uid = f"g_{claims['sub']}"
    doc = load_user(uid)
    if doc and doc.get("status") == "active":
        return _user_from(uid, doc)
    try:
        uid, doc = admit(claims, None)  # admins and allowed domains get in on their first visit
    except HTTPException as exc:
        raise HTTPException(
            403, {"code": "not_invited", "email": claims["email"].lower(), "message": exc.detail}
        ) from exc
    return _user_from(uid, doc)


def current_user(request: Request) -> User:
    if mode() == "dev":
        return User(uid="local-dev", email="local-dev", name="Local dev", role="admin")
    if mode() == "iap":
        return _iap_user(request)
    token = request.cookies.get(SESSION_COOKIE)
    uid = read_session(token) if token else None
    if not uid:
        raise HTTPException(status_code=401, detail="Sign in to continue")
    doc = load_user(uid)
    if not doc or doc.get("status") != "active":
        raise HTTPException(status_code=401, detail="Your session is no longer valid. Sign in again.")
    return _user_from(uid, doc)


# ---- personal API tokens (gymhub.cli gym push, GitHub Actions) ---------------------------------
# api_tokens/{id}   uid, name, hash (sha256 of the secret), created_at. Token: gp_<id>_<secret>.
# Accepted only where `publisher` is the dependency: publishing gym versions, and what gymhub.cli needs
# (a gym, an exercise, creating, reading and cancelling jobs). Never for settings, admin or sharing.


def token_hash(secret: str) -> str:
    import hashlib

    return hashlib.sha256(secret.encode()).hexdigest()


def token_user(token: str) -> User:
    import hmac

    parts = token.split("_", 2)  # gp_<hex id>_<secret>; the secret may contain "_"
    if len(parts) != 3 or parts[0] != "gp":
        raise HTTPException(401, "Malformed API token")
    doc = _store().get("api_tokens", parts[1])
    if not doc or doc.get("revoked") or not hmac.compare_digest(doc.get("hash", ""), token_hash(parts[2])):
        raise HTTPException(401, "This API token is not valid (revoked or wrong)")
    user = load_user(doc["uid"])
    if not user or user.get("status") != "active":
        raise HTTPException(401, "The owner of this API token is disabled")
    return _user_from(doc["uid"], user)


def publisher(request: Request) -> User:
    """The signed-in user, or the owner of a personal API token (Authorization: Bearer gp_...)."""
    header = request.headers.get("authorization", "")
    if header.startswith("Bearer gp_"):
        return token_user(header.removeprefix("Bearer ").strip())
    return current_user(request)


def require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Only admins can do this")
    return user


def can_run(user: User) -> bool:
    """Members may run code only when an admin turned it on (until per-user sandboxing exists)."""
    return user.is_admin or bool(auth_settings()["members_can_run"])


def new_invite_code() -> str:
    return secrets.token_urlsafe(18)
