from __future__ import annotations

import os
from typing import Any, Self

import httpx
from dotenv import load_dotenv

BASE_URL = "https://webapi.bps.go.id/v1/api"
DEFAULT_DOMAIN = "0000"
DEFAULT_TIMEOUT = 30.0

load_dotenv()


class BPSAPIError(RuntimeError):
    pass


class BPSClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        domain: str = DEFAULT_DOMAIN,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key or os.getenv("BPS_API_KEY")

        if not self.api_key:
            raise ValueError("BPS_API_KEY environment variable is required")

        self.domain = domain
        self.client = httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        request_params = {**params, "domain": self.domain, "key": self.api_key}

        response = self.client.get(f"{BASE_URL}/list", params=request_params)

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") != "OK":
            raise BPSAPIError(payload.get("message", "Unknown BPS API error"))

        return payload

    def list_table_statistics(self, *, page: int = 1, perpage: int = 100) -> dict[str, Any]:
        return self._get({"model": "tablestatistic", "page": page, "perpage": perpage})

    def view_table_statistics(self, table_id: str) -> dict[str, Any]:
        params = {
            "model": "tablestatistic",
            "lang": "ind",
            "domain": self.domain,
            "id": table_id,
            "key": self.api_key,
        }

        response = self.client.get(f"{BASE_URL}/view", params=params)

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") != "OK":
            raise BPSAPIError(payload.get("message", "Unknown BPS API error"))

        return payload
