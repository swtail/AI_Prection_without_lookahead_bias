"""Small HTTP client with an on-disk cache and per-host rate limiting.

SEC EDGAR asks automated clients to stay under 10 requests/second and to send
a User-Agent identifying the client (https://www.sec.gov/os/accessing-edgar-data).
Set ``EARNINGS_USER_AGENT`` (e.g. "MyLab research contact@mylab.org").
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

DEFAULT_UA = os.environ.get("EARNINGS_USER_AGENT", "nolookahead-earnings research client")
CACHE_DIR = Path(os.environ.get("EARNINGS_CACHE_DIR", Path.home() / ".cache" / "earnings"))
_MIN_INTERVAL = {"data.sec.gov": 0.12, "www.sec.gov": 0.12}
_last_call: dict[str, float] = {}
_session = requests.Session()


def _throttle(host: str) -> None:
    gap = _MIN_INTERVAL.get(host, 0.0)
    wait = _last_call.get(host, 0.0) + gap - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.monotonic()


def get(url: str, *, max_age: float | None = 86400, headers: dict | None = None,
        retries: int = 4) -> bytes:
    """GET ``url``; reuse a cached copy younger than ``max_age`` seconds (None = forever)."""
    key = hashlib.sha256(url.encode()).hexdigest()
    path = CACHE_DIR / key[:2] / key
    if path.exists() and (max_age is None or time.time() - path.stat().st_mtime < max_age):
        return path.read_bytes()

    host = urlparse(url).netloc
    hdrs = {"User-Agent": DEFAULT_UA, "Accept-Encoding": "gzip, deflate"}
    if host == "api.nasdaq.com":
        hdrs.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)", "Accept": "application/json"})
    hdrs.update(headers or {})
    err = None
    for attempt in range(retries):
        _throttle(host)
        try:
            r = _session.get(url, headers=hdrs, timeout=60)
            if r.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
                return r.content
            if r.status_code == 404:
                raise FileNotFoundError(url)
            err = RuntimeError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:  # network hiccup: back off and retry
            err = e
        time.sleep(2 ** attempt)
    raise err  # type: ignore[misc]


def get_json(url: str, **kw):
    return json.loads(get(url, **kw))
