import os
import time
import uuid
from typing import Optional, Literal, List, Dict, Any

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from prebaked_data import get_prebaked_snapshot
from scenario import get_scenario, set_scenario

# ============================
# Config
# ============================
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")

# ============================
# Auth
# ============================
bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="EXECUTOR_SECRET missing")

    if creds is None:
        raise HTTPException(status_code=403, detail="Not authenticated")

    if creds.credentials != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return True


# ============================
# Upstash lock
# ============================
def _headers():
    return {"Authorization": f"Bearer {KV_REST_API_TOKEN}", "Accept": "application/json"}


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    url = f"{KV_REST_API_URL}/setnx/{LOCK_KEY}/{lock_id}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=_headers())
    if r.status_code == 200 and r.json().get("result") == 1:
        return lock_id
    return None


async def unlock():
    url = f"{KV_REST_API_URL}/del/{LOCK_KEY}"
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(url, headers=_headers())


# ============================
# Agents
# ============================
Signal = Literal["BUY", "SELL", "HOLD"]

AGENTS: List[Dict[str, str]] = [
    {"id": f"A{i:02d}", "role": f"Agent {i}"} for i in range(1, 16)
]


def agent_decide(agent: Dict[str, str], snap: Dict[str, Any]) -> Dict[str, Any]:
    scenario = snap["scenario"]
    if scenario == "bull":
        signal, conf = "BUY", 0.7
    elif scenario == "bear":
        signal, conf = "SELL", 0.7
    else:
        signal, conf = "HOLD", 0.55

    return {
        "agent_id": agent["id"],
        "role": agent["role"],
        "signal": signal,
        "confidence": conf,
        "notes": [snap["headline"]],
        "evidence": {
            "source": "prebaked",
            "items": snap["news"][:1] + snap["social"][:1],
        },
    }


def aggregate(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    score = sum(
        (1 if a["signal"] == "BUY" else -1 if a["signal"] == "SELL" else 0) * a["confidence"]
        for a in agents
    )
    avg_conf = sum(a["confidence"] for a in agents) / len(agents)

    if score > 2.5:
        decision = "BUY"
    elif score < -2.5:
        decision = "SELL"
    else:
        decision = "HOLD"

    return {"decision": decision, "score": round(score, 3), "avg_conf": round(avg_conf, 3)}


# ============================
# App
# ============================
app = FastAPI(title="Polyclawd Executor", version="0.4.0")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/scenario/{value}")
async def change_scenario(value: Literal["neutral", "bull", "bear"], _: bool = Depends(require_auth)):
    ok = await set_scenario(value)
    return {"ok": ok, "scenario": value}


@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    lock = await try_lock()
    if not lock:
        return {"ok": True, "skipped": True}

    try:
        scenario = await get_scenario()
        snap = get_prebaked_snapshot(scenario)

        agents_out = [agent_decide(a, snap) for a in AGENTS]
        agg = aggregate(agents_out)

        return {
            "ok": True,
            "scenario": scenario,
            "tick_index": snap["tick_index"],
            "decision": agg["decision"],
            "score": agg["score"],
            "avg_conf": agg["avg_conf"],
            "agents": agents_out,
        }
    finally:
        await unlock()
