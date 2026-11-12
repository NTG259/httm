"""HTTP client lịch sự: tôn trọng robots.txt, giới hạn tốc độ theo domain, retry."""
from __future__ import annotations

import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

from diabetes_mas.config import settings


class RobotsDisallowed(Exception):
    pass


class PoliteClient:
    def __init__(self, timeout_s: float | None = None, retries: int | None = None) -> None:
        cfg = settings()["http"]
        self.user_agent: str = cfg["user_agent"]
        self.min_interval: float = float(cfg["min_interval_s"])
        self.timeout: float = float(timeout_s if timeout_s is not None else cfg["timeout_s"])
        self.retries: int = int(retries if retries is not None else cfg["retries"])
        self.session = requests.Session()
        self.session.headers["User-Agent"] = self.user_agent
        self._last_call: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = self._raw_get(f"{base}/robots.txt")
                rp.parse(resp.text.splitlines() if resp.ok else [])
            except requests.RequestException:
                rp.parse([])
            self._robots[base] = rp
        return self._robots[base]

    def allowed(self, url: str) -> bool:
        return self._robots_for(url).can_fetch(self.user_agent, url)

    def _wait(self, netloc: str) -> None:
        last = self._last_call.get(netloc)
        if last is not None:
            delta = time.monotonic() - last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
        self._last_call[netloc] = time.monotonic()

    def _raw_get(self, url: str) -> requests.Response:
        self._wait(urlparse(url).netloc)
        return self.session.get(url, timeout=self.timeout, allow_redirects=True)

    def get(self, url: str) -> requests.Response:
        if not self.allowed(url):
            raise RobotsDisallowed(url)
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._raw_get(url)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} {url}")
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(self.min_interval * (attempt + 1))
        raise last_exc  # type: ignore[misc]
