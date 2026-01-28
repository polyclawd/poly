import os
import httpx
from typing import Literal

Scenario = Literal["neutral", "bull", "bear"]

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")

SCENARIO_KEY = "scenario:current"
DEFAULT_SCENARIO: Scenario = "neutral"


def _headers():
    return {
        "Authorization": f"Bearer {KV_REST_API_TOKEN}",
        "Accept": "application/json",
    }


async def get_scenario() -> Scenario:
    """
    Читает текущий сценарий из Upstash.
    Если ключа нет или ошибка — возвращает neutral.
    """
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        return DEFAULT_SCENARIO

    url = f"{KV_REST_API_URL}/get/{SCENARIO_KEY}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_headers())

    if r.status_code != 200:
        return DEFAULT_SCENARIO

    data = r.json()
    value = data.get("result")

    if value in ("neutral", "bull", "bear"):
        return value  # type: ignore

    return DEFAULT_SCENARIO


async def set_scenario(value: Scenario) -> bool:
    """
    Устанавливает сценарий в Upstash.
    """
    if value not in ("neutral", "bull", "bear"):
        return False

    url = f"{KV_REST_API_URL}/set/{SCENARIO_KEY}/{value}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=_headers())

    return r.status_code == 200
