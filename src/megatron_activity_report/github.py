"""Minimal GitHub REST client with bounded retries and secret-safe errors."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class GitHubError(RuntimeError):
    """A sanitized GitHub API failure."""


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        timeout: int = 60,
        max_retries: int = 5,
        api_url: str = "https://api.github.com",
    ):
        self.token = token or _token_from_environment()
        if not self.token:
            raise GitHubError(
                "MEGATRON_GH_TOKEN, GH_TOKEN, or GITHUB_TOKEN is required"
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_url = api_url.rstrip("/")

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        return self.request("POST", path, payload=payload)

    def patch(self, path: str, payload: dict[str, Any]) -> Any:
        return self.request("PATCH", path, payload=payload)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        accept: str = "application/vnd.github+json",
    ) -> Any:
        url = path if path.startswith("https://") else f"{self.api_url}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": accept,
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "megatron-lm-activity-report",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = response.read()
                    return json.loads(data) if data else None
            except urllib.error.HTTPError as exc:
                response_body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {429, 500, 502, 503, 504}
                if exc.code == 403 and exc.headers.get("X-RateLimit-Remaining") == "0":
                    retryable = True
                if not retryable or attempt >= self.max_retries:
                    raise GitHubError(
                        f"GitHub {method} {urllib.parse.urlparse(url).path} failed "
                        f"with HTTP {exc.code}: {_message(response_body)}"
                    ) from exc
                time.sleep(_retry_delay(exc.headers, attempt))
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise GitHubError(f"GitHub request failed: {exc.reason}") from exc
                time.sleep(min(30.0, 2**attempt + random.random()))
        raise AssertionError("retry loop must return or raise")

    def paginate(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str = "application/vnd.github+json",
        max_pages: int | None = None,
    ) -> list[Any]:
        page = 1
        result: list[Any] = []
        while max_pages is None or page <= max_pages:
            query = dict(params or {})
            query.update({"per_page": 100, "page": page})
            batch = self.request("GET", path, params=query, accept=accept)
            if not isinstance(batch, list):
                raise GitHubError(f"expected list response from {path}")
            result.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return result


def _token_from_environment() -> str | None:
    return (
        os.environ.get("MEGATRON_GH_TOKEN")
        or os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )


def _message(body: str) -> str:
    try:
        payload = json.loads(body)
        return str(payload.get("message") or "unknown error")[:500]
    except json.JSONDecodeError:
        return body[:500]


def _retry_delay(headers: Any, attempt: int) -> float:
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return min(120.0, float(retry_after))
        except ValueError:
            pass
    reset = headers.get("X-RateLimit-Reset")
    if reset and headers.get("X-RateLimit-Remaining") == "0":
        try:
            return min(120.0, max(1.0, float(reset) - time.time() + 1.0))
        except ValueError:
            pass
    return min(30.0, 2**attempt + random.random())
