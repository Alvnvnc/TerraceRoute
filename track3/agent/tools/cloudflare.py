"""A minimal, verified Cloudflare API client (stdlib urllib only).

Only the endpoints the agent actually needs (plan.md section 3):
create/configure/delete a remotely-managed tunnel, and create/find/delete the
proxied CNAME that routes a hostname into it.

Set ``dry_run=True`` to exercise the whole expose/teardown flow with no network
and no credentials — it returns deterministic fake ids so the CLI and rollback
logic are testable offline.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

API_BASE = "https://api.cloudflare.com/client/v4"


class CloudflareError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class CloudflareClient:
    def __init__(
        self,
        account_id: str,
        zone_id: str,
        token: str,
        *,
        dry_run: bool = False,
        base: str = API_BASE,
        timeout: float = 15.0,
    ):
        self.account_id = account_id
        self.zone_id = zone_id
        self._token = token
        self.dry_run = dry_run
        self.base = base
        self.timeout = timeout

    # ---- tunnel lifecycle -------------------------------------------------- #
    def create_tunnel(self, name: str) -> dict[str, str]:
        """Create a remotely-managed tunnel. Returns ``{id, token}``.

        ``config_src=cloudflare`` keeps the ingress config in Cloudflare (no
        local YAML); the run token comes back inline on creation.
        """
        if self.dry_run:
            return {"id": f"dryrun-{name}", "token": f"dryrun-token-{name}"}
        res = self._request(
            "POST",
            f"/accounts/{self.account_id}/cfd_tunnel",
            {"name": name, "config_src": "cloudflare"},
        )
        return {"id": res["id"], "token": res.get("token", "")}

    def get_tunnel_token(self, tunnel_id: str) -> str:
        if self.dry_run:
            return f"dryrun-token-{tunnel_id}"
        res = self._request(
            "GET", f"/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/token"
        )
        # This endpoint returns ``result`` as a bare string.
        return res if isinstance(res, str) else str(res)

    def put_configuration(self, tunnel_id: str, ingress: list[dict]) -> dict:
        if self.dry_run:
            return {"ingress": ingress}
        return self._request(
            "PUT",
            f"/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/configurations",
            {"config": {"ingress": ingress}},
        )

    def get_tunnel_status(self, tunnel_id: str) -> str:
        """Returns one of: healthy | degraded | down | inactive."""
        if self.dry_run:
            return "healthy"
        res = self._request(
            "GET", f"/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}"
        )
        return res.get("status", "inactive")

    def delete_tunnel_connections(self, tunnel_id: str) -> None:
        if self.dry_run:
            return
        self._request(
            "DELETE",
            f"/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/connections",
        )

    def delete_tunnel(self, tunnel_id: str) -> None:
        if self.dry_run:
            return
        self._request("DELETE", f"/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}")

    # ---- DNS --------------------------------------------------------------- #
    def create_dns_cname(self, name: str, tunnel_id: str, proxied: bool = True) -> str:
        """Create the proxied CNAME ``name`` -> ``{tunnel_id}.cfargotunnel.com``.
        Returns the record id (needed for cleanup)."""
        content = f"{tunnel_id}.cfargotunnel.com"
        if self.dry_run:
            return f"dryrun-dns-{name}"
        res = self._request(
            "POST",
            f"/zones/{self.zone_id}/dns_records",
            {"type": "CNAME", "name": name, "content": content, "proxied": proxied},
        )
        return res["id"]

    def find_dns_record(self, name: str, rtype: str = "CNAME") -> Optional[str]:
        if self.dry_run:
            return None
        res = self._request(
            "GET",
            f"/zones/{self.zone_id}/dns_records?type={rtype}&name={name}",
        )
        records = res if isinstance(res, list) else []
        return records[0]["id"] if records else None

    def delete_dns_record(self, record_id: str) -> None:
        if self.dry_run:
            return
        self._request("DELETE", f"/zones/{self.zone_id}/dns_records/{record_id}")

    # ---- helpers ----------------------------------------------------------- #
    @staticmethod
    def build_ingress(hostname: str, service: str) -> list[dict]:
        """Ingress rules for one hostname, with the mandatory catch-all 404."""
        return [
            {"hostname": hostname, "service": service},
            {"service": "http_status:404"},
        ]

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode()
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            raise CloudflareError(
                _first_error(raw) or f"HTTP {exc.code} on {method} {path}",
                status=exc.code,
                body=raw,
            ) from exc
        except urllib.error.URLError as exc:
            raise CloudflareError(f"network error on {method} {path}: {exc.reason}")

        payload = json.loads(raw) if raw else {}
        if isinstance(payload, dict) and payload.get("success") is False:
            raise CloudflareError(_first_error(raw) or "Cloudflare API error",
                                  body=raw)
        return payload.get("result") if isinstance(payload, dict) else payload


def _first_error(raw: str) -> str:
    try:
        errs = json.loads(raw).get("errors") or []
        if errs:
            return f"{errs[0].get('code')}: {errs[0].get('message')}"
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass
    return ""
