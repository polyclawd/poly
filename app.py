import os
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET")
SITE_BASE_URL = os.getenv("SITE_BASE_URL")  # e.g. https://your-vercel-site.vercel.app

LOCK_KEY = "executor:lock"
LOCK_TTL_SECONDS = 60

app = FastAPI()


def _auth_or_401(authorization: Optional[str]):
    if not EXECUTOR_SECRET:
        raise HTTPException(500, "EXECUTOR_SECRET missing on executor")
    if authorization != f"Bearer {EXECUTOR_SECRET}":
        raise HTTPException(401, "Unauthorized")


async def kv_get(key: str) -> Any:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        raise HTTPException(500, "Upstash env missing on executor")
    url = f"{UPSTASH_REDIS_REST_URL}/get/{key}"
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data.get("result")


async def kv_set(key: str, value: Any) -> None:
    url = f"{UPSTASH_REDIS_REST_URL}/set/{key}"
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=value)
        r.raise_for_status()


async def kv_set_ex(key: str, ttl_seconds: int, value: Any) -> None:
    url = f"{UPSTASH_REDIS_REST_URL}/setex/{key}/{ttl_seconds}"
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=value)
        r.raise_for_status()


async def try_lock() -> bool:
    """
    Poor-man distributed lock using SETNX + EX.
    Upstash REST supports SETNX via /setnx/<key>.
    If not available in your plan, we can emulate later.
    """
    url = f"{UPSTASH_REDIS_REST_URL}/setnx/{LOCK_KEY}"
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=str(int(time.time())))
        r.raise_for_status()
        ok = bool(r.json().get("result"))
        if ok:
            # set ttl
            await kv_set_ex(LOCK_KEY, LOCK_TTL_SECONDS, "1")
        return ok


async def release_lock() -> None:
    url = f"{UPSTASH_REDIS_REST_URL}/del/{LOCK_KEY}"
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(url, headers=headers)


async def post_executor_status(last_result: str, message: str):
    """
    Update the site so UI shows executor status.
    """
    if not SITE_BASE_URL:
        return
    url = f"{SITE_BASE_URL}/api/executor/status"
    headers = {"Authorization": f"Bearer {EXECUTOR_SECRET}"}
    payload = {
        "lastRunAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lastResult": last_result,
        "message": message[:200],
    }
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            await client.post(url, headers=headers, json=payload)
        except Exception:
            # don't crash run just because UI status update failed
            pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


async def append_agent_event(agent_id: int, event: Dict[str, Any], limit: int = 50):
    key = f"replay:agent:{agent_id}:events"
    existing = await kv_get(key) or []
    if not isinstance(existing, list):
        existing = []
    existing.append(event)
    existing = existing[-limit:]
    await kv_set(key, existing)


async def load_agents() -> List[Dict[str, Any]]:
    agents = await kv_get("replay:agents") or []
    return agents if isinstance(agents, list) else []


async def save_agents(agents: List[Dict[str, Any]]):
    await kv_set("replay:agents", agents)


async def load_state() -> Dict[str, Any]:
    state = await kv_get("replay:state") or {}
    return state if isinstance(state, dict) else {}


async def load_agent_market(agent_id: int) -> Optional[Dict[str, Any]]:
    m = await kv_get(f"pm:agent:{agent_id}:market")
    return m if isinstance(m, dict) else None


class RunResponse(BaseModel):
    ok: bool
    result: str
    message: str


@app.post("/run", response_model=RunResponse)
async def run(authorization: Optional[str] = Header(default=None)):
    _auth_or_401(authorization)

    locked = await try_lock()
    if not locked:
        await post_executor_status("skipped", "Lock active: previous run still in progress")
        return RunResponse(ok=True, result="skipped", message="Lock active, skipping")

    try:
        state = await load_state()
        trade_mode = state.get("tradeMode", "PAPER")
        kill = bool(state.get("killSwitch", True))

        if kill:
            await post_executor_status("skipped", "Kill switch enabled")
            return RunResponse(ok=True, result="skipped", message="Kill switch enabled")

        # DRY RUN only in this step — do not place real orders yet
        agents = await load_agents()
        if not agents:
            await post_executor_status("error", "No agents in KV")
            return RunResponse(ok=False, result="error", message="No agents in KV")

        for a in agents:
            agent_id = int(a["id"])
            market = await load_agent_market(agent_id)
            question = (market or {}).get("question") or a.get("marketTitle") or f"Agent {agent_id}"

            # example: generate a fake action now, replace with real strategy later
            action = "HOLD"
            size_usd = 0

            # you can make it deterministic per agent to look stable
            if agent_id % 3 == 0:
                action = "BUY"
                size_usd = 10
            elif agent_id % 5 == 0:
                action = "SELL"
                size_usd = 10

            msg = f"[DRY RUN] {action} {a.get('outcome','YES')} ${size_usd} on '{question}' (mode={trade_mode})"
            await append_agent_event(agent_id, {
                "id": f"live-{agent_id}-{int(time.time())}",
                "agentId": agent_id,
                "ts": now_iso(),
                "type": "info",
                "message": msg
            })

            # update agent status so UI shows activity
            a["status"] = "active" if action != "HOLD" else "idle"
            a["lastRunAt"] = now_iso()

        await save_agents(agents)
        await post_executor_status("ok", f"Ran DRY cycle for {len(agents)} agents")
        return RunResponse(ok=True, result="ok", message="Dry run cycle executed")
    except Exception as e:
        await post_executor_status("error", str(e))
        raise
    finally:
        await release_lock()


@app.get("/healthz")
async def healthz():
    return {"ok": True}
