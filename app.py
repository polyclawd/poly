import os
import time
import uuid
import json
from typing import Optional, Any, Dict, List
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# ----------------------------
# Config
# ----------------------------
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")
KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "").strip()
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", "").strip()

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))
RUN_TIMEOUT_SECONDS = float(os.getenv("EXECUTOR_RUN_TIMEOUT_SECONDS", "25"))

DEFAULT_MARKET_ID = os.getenv("DEFAULT_MARKET_ID", "POLY:DEMO_001")

SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")
LATEST_PREFIX = os.getenv("EXECUTOR_LATEST_PREFIX", "executor:latest:")
HISTORY_PREFIX = os.getenv("EXECUTOR_HISTORY_PREFIX", "executor:history:")

HISTORY_MAX = int(os.getenv("EXECUTOR_HISTORY_MAX", "200"))


# ----------------------------
# Auth (Swagger "Authorize")
# ----------------------------
bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured: EXECUTOR_SECRET is empty")
    if creds is None:
        raise HTTPException(status_code=403, detail="Not authenticated")
    if creds.credentials != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# ----------------------------
# Upstash REST helpers
# ----------------------------
def _upstash_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _enc(v: str) -> str:
    # Upstash Redis REST uses path params; must URL-encode
    return quote(v, safe="")


async def _upstash_post(path: str, token: str) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / token is empty")
    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))
    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV tokens")
    r.raise_for_status()
    return r.json()


async def _upstash_get(path: str, token: str) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / token is empty")
    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=_upstash_headers(token))
    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV tokens")
    r.raise_for_status()
    return r.json()


async def upstash_set(key: str, value: str, token: str) -> None:
    await _upstash_post(f"set/{_enc(key)}/{_enc(value)}", token)


async def upstash_get(key: str, token: str) -> Optional[str]:
    data = await _upstash_get(f"get/{_enc(key)}", token)
    return data.get("result")


async def upstash_setnx(key: str, value: str, token: str) -> int:
    data = await _upstash_post(f"setnx/{_enc(key)}/{_enc(value)}", token)
    return int(data.get("result", 0))


async def upstash_del(key: str, token: str) -> int:
    data = await _upstash_post(f"del/{_enc(key)}", token)
    return int(data.get("result", 0))


async def upstash_lpush(key: str, value: str, token: str) -> int:
    data = await _upstash_post(f"lpush/{_enc(key)}/{_enc(value)}", token)
    return int(data.get("result", 0))


async def upstash_ltrim(key: str, start: int, stop: int, token: str) -> str:
    data = await _upstash_post(f"ltrim/{_enc(key)}/{start}/{stop}", token)
    return str(data.get("result", ""))


async def upstash_lrange(key: str, start: int, stop: int, token: str) -> List[str]:
    data = await _upstash_get(f"lrange/{_enc(key)}/{start}/{stop}", token)
    return data.get("result") or []


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id, KV_REST_API_TOKEN)
    return lock_id if ok == 1 else None


async def unlock(_: str) -> None:
    await upstash_del(LOCK_KEY, KV_REST_API_TOKEN)


# ----------------------------
# Demo agents (fake)
# ----------------------------
AGENTS_PRESET = [
    {"agent_id": "agent_01", "name": "Market Microstructure"},
    {"agent_id": "agent_02", "name": "Orderbook Reader"},
    {"agent_id": "agent_03", "name": "News Synthesizer"},
    {"agent_id": "agent_04", "name": "Social Sentiment"},
    {"agent_id": "agent_05", "name": "Volatility Watch"},
    {"agent_id": "agent_06", "name": "Flow Tracker"},
    {"agent_id": "agent_07", "name": "Whale Radar"},
    {"agent_id": "agent_08", "name": "Risk Manager"},
    {"agent_id": "agent_09", "name": "Mean Reversion"},
    {"agent_id": "agent_10", "name": "Momentum"},
    {"agent_id": "agent_11", "name": "Macro Pulse"},
    {"agent_id": "agent_12", "name": "Liquidity Guard"},
    {"agent_id": "agent_13", "name": "Event Scanner"},
    {"agent_id": "agent_14", "name": "Pricing Edge"},
    {"agent_id": "agent_15", "name": "Consensus Builder"},
]


def _vote_for(i: int, scenario: str) -> Dict[str, Any]:
    base = (i * 37 + int(time.time()) // 60) % 100

    if scenario == "bull":
        vote = "BUY" if base < 75 else "HOLD"
    elif scenario == "bear":
        vote = "HOLD" if base < 65 else "BUY"
    else:
        vote = "BUY" if base < 55 else "HOLD"

    conf = round(0.45 + (base % 40) / 100, 2)

    reasons_buy = [
        "Liquidity acceptable; expected value positive this tick.",
        "Edge detected; scenario tailwind supports entry.",
        "Pricing looks favorable vs. implied probability.",
        "Momentum aligns; small probe position is justified.",
    ]
    reasons_hold = [
        "Signals conflict; avoid overtrading.",
        "No clear edge; wait for better price.",
        "Volatility elevated; reduce risk this tick.",
        "Consensus weak; hold for confirmation.",
    ]
    reason = reasons_buy[base % len(reasons_buy)] if vote == "BUY" else reasons_hold[base % len(reasons_hold)]

    signals = {
        "price": round(((base % 100) / 100), 3),
        "implied_prob": round(((base * 13) % 100) / 100, 3),
        "sentiment": round((((base * 7) % 200) - 100) / 100, 3),
        "volatility": round(((base * 9) % 100) / 100, 3),
    }
    return {"vote": vote, "confidence": conf, "reason": reason, "signals": signals}


def compute_tick(market_id: str, scenario: str) -> Dict[str, Any]:
    agents = []
    buy_count = 0
    hold_count = 0

    for idx, a in enumerate(AGENTS_PRESET, start=1):
        out = _vote_for(idx, scenario)
        if out["vote"] == "BUY":
            buy_count += 1
        else:
            hold_count += 1
        agents.append({**a, **out})

    rule = "BUY if buy_count >= 9 else HOLD"
    decision = "BUY" if buy_count >= 9 else "HOLD"
    avg_conf = round(sum(x["confidence"] for x in agents) / len(agents), 3)

    top_buy_reasons = [x["reason"] for x in agents if x["vote"] == "BUY"][:3]

    tick = int(time.time() // 60)

    return {
        "market_id": market_id,
        "ts": int(time.time()),
        "tick": tick,
        "scenario": scenario,
        "summary": {
            "decision": decision,
            "buy_count": buy_count,
            "hold_count": hold_count,
            "avg_confidence": avg_conf,
            "rule": rule,
        },
        "digest": {"buy_count": buy_count, "top_buy_reasons": top_buy_reasons},
        "agents": agents,
    }


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Polyclawd Executor", version="0.1.0")


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
        "has_kv_ro_token": bool(KV_REST_API_READ_ONLY_TOKEN),
        "lock_key": LOCK_KEY,
        "scenario_key": SCENARIO_KEY,
        "history_max": HISTORY_MAX,
    }


# ----------------------------
# Scenario endpoints
# ----------------------------
@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    s = await upstash_get(SCENARIO_KEY, KV_REST_API_TOKEN)
    return {"ok": True, "scenario": (s or "neutral"), "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_set(body: Dict[str, Any], _: bool = Depends(require_auth)):
    scenario = str(body.get("scenario", "neutral")).strip().lower()
    if scenario not in ("neutral", "bull", "bear"):
        raise HTTPException(status_code=422, detail="scenario must be one of: neutral, bull, bear")
    await upstash_set(SCENARIO_KEY, scenario, KV_REST_API_TOKEN)
    return {"ok": True, "scenario": scenario}


# ----------------------------
# Public endpoints for UI
# ----------------------------
@app.get("/public/latest/{market_id}")
async def public_latest(market_id: str):
    key = f"{LATEST_PREFIX}{market_id}"
    ro = KV_REST_API_READ_ONLY_TOKEN or KV_REST_API_TOKEN
    raw = await upstash_get(key, ro)
    if not raw:
        return {"ok": False, "market_id": market_id, "error": "no data yet"}
    try:
        data = json.loads(raw)
    except Exception:
        return {"ok": False, "market_id": market_id, "error": "corrupted json"}
    return {"ok": True, "market_id": market_id, "data": data}


@app.get("/public/history/{market_id}")
async def public_history(
    market_id: str,
    minutes: int = Query(60, ge=1, le=24 * 60),
    limit: int = Query(200, ge=1, le=500),
):
    key = f"{HISTORY_PREFIX}{market_id}"
    ro = KV_REST_API_READ_ONLY_TOKEN or KV_REST_API_TOKEN

    items = await upstash_lrange(key, 0, limit - 1, ro)

    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - minutes * 60 * 1000

    parsed: List[Dict[str, Any]] = []
    for raw in items:
        try:
            obj = json.loads(raw)
        except Exception:
            continue

        ts = obj.get("ts")
        if ts is None:
            parsed.append(obj)
            continue

        ts = int(ts)
        ts_ms = ts * 1000 if ts < 10_000_000_000 else ts
        if ts_ms >= cutoff_ms:
            parsed.append(obj)

    parsed.sort(key=lambda x: x.get("ts", 0), reverse=True)

    return {
        "ok": True,
        "market_id": market_id,
        "minutes": minutes,
        "limit": limit,
        "raw_count": len(items),
        "count": len(parsed),
        "items": parsed,
    }


# ----------------------------
# Executor tick
# ----------------------------
@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked", "message": "Another executor run is in progress"}

    try:
        scenario = (await upstash_get(SCENARIO_KEY, KV_REST_API_TOKEN)) or "neutral"
        market_id = DEFAULT_MARKET_ID
        payload = compute_tick(market_id, scenario)

        latest_key = f"{LATEST_PREFIX}{market_id}"
        await upstash_set(latest_key, json.dumps(payload), KV_REST_API_TOKEN)

        history_key = f"{HISTORY_PREFIX}{market_id}"
        await upstash_lpush(history_key, json.dumps(payload), KV_REST_API_TOKEN)
        await upstash_ltrim(history_key, 0, HISTORY_MAX - 1, KV_REST_API_TOKEN)

        duration = round(time.time() - started, 3)
        return {
            "ok": True,
            "skipped": False,
            "scenario": scenario,
            "market_id": market_id,
            "tick": payload["tick"],
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
