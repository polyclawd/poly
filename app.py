import os
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

app = FastAPI(title="PolyClawd Executor", version="0.1.0")

# --- Auth (this makes Swagger show the 🔒 Authorize button) ---
bearer_scheme = HTTPBearer(auto_error=True)


def _env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing env var: {name}")
    return val


KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", "AoCOAAIgcDLO-2qllWynMs01ogYqMD8JqSY3FvxGj_ywnYHKeHFZxg")

EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")

# Keys used in KV
LOCK_KEY = "executor:lock"
LAST_RUN_KEY = "executor:last_run"


def _kv_headers(read_only: bool = False) -> dict:
    token = KV_REST_API_READ_ONLY_TOKEN if read_only and KV_REST_API_READ_ONLY_TOKEN else KV_REST_API_TOKEN
    if not token:
        raise RuntimeError("Missing KV token (KV_REST_API_TOKEN or KV_REST_API_READ_ONLY_TOKEN)")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


async def kv_get(key: str) -> Optional[str]:
    if not KV_REST_API_URL:
        raise RuntimeError("Missing KV_REST_API_URL")
    url = f"{KV_REST_API_URL}/get/{key}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_kv_headers(read_only=True))
        r.raise_for_status()
        data = r.json()
        # Upstash typically returns {"result": "..."} or {"result": None}
        return data.get("result")


async def kv_set(key: str, value: str) -> None:
    if not KV_REST_API_URL:
        raise RuntimeError("Missing KV_REST_API_URL")
    url = f"{KV_REST_API_URL}/set/{key}/{value}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=_kv_headers(read_only=False))
        r.raise_for_status()


async def kv_setnx(key: str, value: str) -> bool:
    """
    Atomic lock acquire using Upstash SETNX.
    """
    if not KV_REST_API_URL:
        raise RuntimeError("Missing KV_REST_API_URL")
    url = f"{KV_REST_API_URL}/setnx/{key}/{value}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=_kv_headers(read_only=False))
        r.raise_for_status()
        data = r.json()
        # Upstash returns {"result": 1} if set, {"result": 0} if not set
        return bool(data.get("result"))


async def kv_del(key: str) -> None:
    if not KV_REST_API_URL:
        raise RuntimeError("Missing KV_REST_API_URL")
    url = f"{KV_REST_API_URL}/del/{key}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=_kv_headers(read_only=False))
        r.raise_for_status()


async def try_lock(ttl_seconds: int = 60) -> bool:
    """
    We implement a simple lock:
    - lock value is "<uuid>:<expires_at_epoch>"
    - acquire with SETNX
    - if lock exists and expired -> delete and retry once
    """
    now = int(time.time())
    lock_id = str(uuid.uuid4())
    expires_at = now + ttl_seconds
    value = f"{lock_id}:{expires_at}"

    acquired = await kv_setnx(LOCK_KEY, value)
    if acquired:
        return True

    existing = await kv_get(LOCK_KEY)
    if not existing:
        return False

    try:
        _lock_id, exp_str = existing.split(":", 1)
        exp = int(exp_str)
    except Exception:
        # malformed lock -> clear it
        await kv_del(LOCK_KEY)
        return await kv_setnx(LOCK_KEY, value)

    if exp <= now:
        # expired -> clear and retry once
        await kv_del(LOCK_KEY)
        return await kv_setnx(LOCK_KEY, value)

    return False


async def unlock() -> None:
    # best-effort unlock
    try:
        await kv_del(LOCK_KEY)
    except Exception:
        pass


@app.get("/")
async def root():
    return {"ok": True, "service": "executor", "docs": "/docs"}


@app.get("/healthz")
async def healthz():
    """
    Health check endpoint. Also useful to verify KV connectivity.
    """
    status = {"ok": True}
    # If KV configured, try a cheap read to confirm auth works.
    if KV_REST_API_URL and (KV_REST_API_TOKEN or KV_REST_API_READ_ONLY_TOKEN):
        try:
            _ = await kv_get(LAST_RUN_KEY)
            status["kv"] = "ok"
        except Exception as e:
            status["kv"] = f"error: {type(e).__name__}"
    else:
        status["kv"] = "not_configured"
    return status


@app.post("/run")
async def run(creds: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    """
    Protected endpoint:
    Header must be: Authorization: Bearer <EXECUTOR_SECRET>
    """
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="Server missing EXECUTOR_SECRET")

    token = creds.credentials  # <-- already stripped from "Bearer "
    if token != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Acquire lock to avoid overlapping runs
    try:
        locked = await try_lock(ttl_seconds=90)
    except httpx.HTTPStatusError as e:
        # Most common: 401 from Upstash because token/url mismatch
        raise HTTPException(status_code=500, detail=f"KV error: {e.response.status_code} {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KV error: {type(e).__name__}: {e}")

    if not locked:
        return {"ok": True, "message": "Already running (lock held). Skipping tick."}

    try:
        # ---- PLACE YOUR REAL EXECUTION LOGIC HERE ----
        # For now we just write last_run timestamp to KV.
        ts = str(int(time.time()))
        await kv_set(LAST_RUN_KEY, ts)

        return {"ok": True, "message": "Tick executed", "last_run": ts}
    finally:
        await unlock()
