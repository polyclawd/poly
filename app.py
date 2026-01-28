import os
import time
import uuid
from typing import Optional, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel


# ----------------------------
# Config
# ----------------------------
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")

RUN_TIMEOUT_SECONDS = float(os.getenv("EXECUTOR_RUN_TIMEOUT_SECONDS", "25"))


# ----------------------------
# Auth (Swagger "Authorize")
# ----------------------------
bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured: EXECUTOR_SECRET is empty")

    if creds is None:
        raise HTTPException(status_code=403, detail="Not authenticated")

    token = creds.credentials
    if token != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return True


# ----------------------------
# Upstash REST helpers
# ----------------------------
def _upstash_headers() -> dict:
    if not KV_REST_API_TOKEN:
        return {"Accept": "application/json"}
    return {
        "Authorization": f"Bearer {KV_REST_API_TOKEN}",
        "Accept": "application/json",
    }


async def upstash_get(key: str) -> Optional[str]:
    if not KV_REST_API_URL:
        return None

    url = f"{KV_REST_API_URL}/get/{key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=_upstash_headers())

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()
    data = r.json()
    # Upstash возвращает {"result": "..."} либо {"result": null}
    return data.get("result")


async def upstash_set(key: str, value: str) -> None:
    if not KV_REST_API_URL:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL is empty")
    if not KV_REST_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_TOKEN is empty")

    url = f"{KV_REST_API_URL}/set/{key}/{value}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers())

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()


async def upstash_setnx(key: str, value: str) -> int:
    if not KV_REST_API_URL:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL is empty")
    if not KV_REST_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_TOKEN is empty")

    url = f"{KV_REST_API_URL}/setnx/{key}/{value}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers())

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()
    data = r.json()
    return int(data.get("result", 0))


async def upstash_del(key: str) -> None:
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        return

    url = f"{KV_REST_API_URL}/del/{key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers())
    # unlock не должен валить всё
    if r.status_code in (200, 201, 204):
        return


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id)
    return lock_id if ok == 1 else None


async def unlock(_: str) -> None:
    await upstash_del(LOCK_KEY)


# ----------------------------
# Scenario API (dropdown в Swagger)
# ----------------------------
ScenarioName = Literal["neutral", "bull", "bear"]


class ScenarioSetRequest(BaseModel):
    scenario: ScenarioName


DEFAULT_SCENARIO: ScenarioName = "neutral"


async def get_scenario_value() -> ScenarioName:
    raw = await upstash_get(SCENARIO_KEY)
    if raw in ("neutral", "bull", "bear"):
        return raw  # type: ignore
    return DEFAULT_SCENARIO


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Polyclawd Executor", version="0.2.0")


@app.get("/")
async def root():
    return {"ok": True, "service": "executor", "docs": "/docs", "health": "/healthz"}


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "has_executor_secret": bool(EXECUTOR_SECRET),
        "has_kv_url": bool(KV_REST_API_URL),
        "has_kv_token": bool(KV_REST_API_TOKEN),
        "lock_key": LOCK_KEY,
        "scenario_key": SCENARIO_KEY,
    }


@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    scenario = await get_scenario_value()
    return {"ok": True, "scenario": scenario, "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_set(body: ScenarioSetRequest, _: bool = Depends(require_auth)):
    await upstash_set(SCENARIO_KEY, body.scenario)
    return {"ok": True, "scenario": body.scenario, "saved_to": SCENARIO_KEY}


@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    # lock
    lock_id = await try_lock()
    if not lock_id:
        return {
            "ok": True,
            "skipped": True,
            "reason": "locked",
            "message": "Another executor run is in progress",
        }

    try:
        scenario = await get_scenario_value()

        # пока заглушка, но сценарий уже участвует
        duration = round(time.time() - started, 3)
        return {
            "ok": True,
            "skipped": False,
            "scenario": scenario,
            "message": "tick executed",
            "duration_sec": duration,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass
