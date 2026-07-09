"""External verification: probe the public hostname from outside the tunnel.

"Verify from the edge, not from 'config saved'." The classifier turns an edge
response into the code the taxonomy keys on, parsing Cloudflare 1xxx error codes
(e.g. 1033) out of the HTML error body when present.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request

from ..types import EdgeResult

_CF_CODE_RE = re.compile(r"(?:error[\s:]+|error code[\s:]+)(\d{3,4})", re.IGNORECASE)


def classify_edge(http_status: int | None, body: str) -> EdgeResult:
    """Turn a raw edge response into a structured :class:`EdgeResult`.

    A Cloudflare 1xxx code found in the body (e.g. ``Error 1033``) takes
    precedence over the wrapping HTTP status, because that is the code that
    identifies the failure (1033 = connector down, wrapped as HTTP 530).
    """
    cf_code = None
    m = _CF_CODE_RE.search(body or "")
    if m:
        val = int(m.group(1))
        if 1000 <= val <= 1999:  # Cloudflare 1xxx family
            cf_code = val
    return EdgeResult(
        reachable=http_status is not None,
        http_status=http_status,
        cf_error_code=cf_code,
        body_snippet=(body or "")[:200],
    )


def probe(url: str, *, timeout: float = 10.0) -> EdgeResult:
    """Fetch ``url`` from outside and classify the result."""
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": "terracegate-probe/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode(errors="replace")
            return classify_edge(resp.status, body)
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode(errors="replace")
        return classify_edge(exc.code, body)
    except urllib.error.URLError:
        return EdgeResult(reachable=False)


def is_live(result: EdgeResult) -> bool:
    """A hostname is live only when the edge returns a real 2xx/3xx and there is
    no Cloudflare 1xxx error."""
    if result.cf_error_code is not None:
        return False
    return result.http_status is not None and 200 <= result.http_status < 400
