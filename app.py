import os
import time
import uuid
import random
import json
from typing import Optional, Dict, Any, List

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field


# ============================================================
# Config
# ============================================================
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", "") or KV_REST_API_TOKEN

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")

# где хранить последний расчёт для сайта
PUBLIC_LATEST_KEY = os.getenv("PUBLIC_LATEST_KEY", "public:latest")

LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))
RUN_TIMEOUT_SECONDS = float(os.getenv("EXECUTOR_RUN_TIMEOUT_SECONDS", "25"))

POSITION_ID = os.getenv("POSITION_ID", "POLYMARKET:DEMO_POSITION_001")

ALLOWED_SCENARIOS = {"bull", "neutral", "bear"}


# ============================================================
# Auth (Swagger "Authorize")
# ============================================================
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


# ============================================================
# Upstash REST helpers
# ============================================================
def _upstash_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


async def _upstash_get(path: str, token: str) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="KV misconfigured: KV_REST_API_URL / token missing")

    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")
    r.raise_for_status()
    return r.json()


async def _upstash_post(path: str, token: str) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="KV misconfigured: KV_REST_API_URL / token missing")

    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")
    r.raise_for_status()
    return r.json()


async def kv_set(key: str, value: str, token: str) -> bool:
    data = await _upstash_post(f"set/{key}/{value}", token)
    return str(data.get("result", "")).upper() == "OK"


async def kv_get(key: str, token: str) -> Optional[str]:
    data = await _upstash_get(f"get/{key}", token)
    v = data.get("result", None)
    if v is None:
        return None
    return str(v)


async def kv_setnx(key: str, value: str, token: str) -> int:
    data = await _upstash_post(f"setnx/{key}/{value}", token)
    return int(data.get("result", 0))


async def kv_del(key: str, token: str) -> int:
    try:
        data = await _upstash_post(f"del/{key}", token)
        return int(data.get("result", 0))
    except Exception:
        return 0


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await kv_setnx(LOCK_KEY, lock_id, KV_REST_API_TOKEN)
    return lock_id if ok == 1 else None


async def unlock(_: str) -> None:
    await kv_del(LOCK_KEY, KV_REST_API_TOKEN)


# ============================================================
# Scenario storage
# ============================================================
async def get_scenario() -> str:
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        return "neutral"

    value = await kv_get(SCENARIO_KEY, KV_REST_API_READ_ONLY_TOKEN)
    if value in ALLOWED_SCENARIOS:
        return value
    return "neutral"


async def set_scenario(value: str) -> str:
    value = (value or "").strip().lower()
    if value not in ALLOWED_SCENARIOS:
        raise HTTPException(status_code=422, detail=f"Invalid scenario. Allowed: {sorted(ALLOWED_SCENARIOS)}")

    ok = await kv_set(SCENARIO_KEY, value, KV_REST_API_TOKEN)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save scenario to KV")

    return value


class ScenarioBody(BaseModel):
    scenario: str = Field(..., description="bull | neutral | bear")


# ============================================================
# Fake Agents
# ============================================================
AGENT_NAMES = [
    "Market Microstructure",
    "Orderbook Reader",
    "News Synthesizer",
    "Social Sentiment",
    "Risk Manager",
    "Volatility Analyst",
    "Momentum Trader",
    "Contrarian",
    "Macro Lens",
    "Technical Analyst",
    "Liquidity Watch",
    "Event Forecaster",
    "Correlation Scout",
    "Position Sizing",
    "Final Gatekeeper",
]


def _agent_decision(agent_idx: int, scenario: str) -> Dict[str, Any]:
    seed = (agent_idx + 1) * 1000 + {"bull": 1, "neutral": 2, "bear": 3}[scenario]
    rnd = random.Random(seed)

    if scenario == "bull":
        buy_prob = 0.72
    elif scenario == "bear":
        buy_prob = 0.22
    else:
        buy_prob = 0.45

    vote_buy = rnd.random() < buy_prob

    confidence = round(0.55 + rnd.random() * 0.4, 2)
    if not vote_buy:
        confidence = round(max(0.5, confidence - 0.15), 2)

    reason_templates_buy = [
        "Implied probability looks undervalued vs. our prior; small edge detected.",
        "Short-term momentum supports entry; risk bounded by size limits.",
        "Market drift aligns with scenario; expected value positive on this tick.",
        "Spread/liquidity acceptable; timing ok for a small probe position.",
    ]
    reason_templates_hold = [
        "Edge not clear this tick; better to wait for confirmation.",
        "Conflicting signals; avoid overtrading in mixed conditions.",
        "Risk/reward not compelling given scenario; hold capital.",
        "Liquidity/spread not favorable; skip this tick.",
    ]

    reason = rnd.choice(reason_templates_buy if vote_buy else reason_templates_hold)

    signals = {
        "price": round(0.35 + rnd.random() * 0.55, 3),
        "implied_prob": round(0.35 + rnd.random() * 0.55, 3),
        "sentiment": round(-1 + rnd.random() * 2, 3),
        "volatility": round(0.05 + rnd.random() * 0.4, 3),
    }

    return {
        "agent_id": f"agent_{agent_idx+1:02d}",
        "name": AGENT_NAMES[agent_idx],
        "vote": "BUY" if vote_buy else "HOLD",
        "confidence": confidence,
        "reason": reason,
        "signals": signals,
    }


def run_fake_agents(scenario: str) -> List[Dict[str, Any]]:
    return [_agent_decision(i, scenario) for i in range(15)]


def aggregate_decision(agent_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    buys = [a for a in agent_reports if a["vote"] == "BUY"]
    holds = [a for a in agent_reports if a["vote"] != "BUY"]

    buy_count = len(buys)
    hold_count = len(holds)

    decision = "BUY" if buy_count >= 9 else "HOLD"

    avg_conf = round(sum(a["confidence"] for a in agent_reports) / max(1, len(agent_reports)), 3)
    buy_conf = round(sum(a["confidence"] for a in buys) / max(1, len(buys)), 3) if buys else 0.0

    return {
        "decision": decision,
        "buy_count": buy_count,
        "hold_count": hold_count,
        "avg_confidence": avg_conf,
        "avg_buy_confidence": buy_conf,
        "rule": "BUY if buy_count >= 9 else HOLD",
    }


# ============================================================
# Public latest storage
# ============================================================
def _now_ts() -> int:
    return int(time.time())


def _tick_index(ts: int) -> int:
    # 5-min buckets
    return int(ts / 300)


async def save_public_latest(payload: Dict[str, Any]) -> None:
    """
    Сохраняем JSON в KV.
    Важно: Upstash /set/<key>/<value> использует value в path,
    поэтому JSON нужно сериализовать и закодировать.
    Здесь мы используем /set/<key>/<value> напрямую -> это ломается на спецсимволах.
    Поэтому мы сохраняем через base64 безопаснее.
    """
    # безопасно сохраняем JSON как base64, чтобы не ломать URL
    import base64

    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).decode("utf-8")

    ok = await kv_set(PUBLIC_LATEST_KEY, b64, KV_REST_API_TOKEN)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to persist public:latest")


async def load_public_latest() -> Optional[Dict[str, Any]]:
    import base64

    b64 = await kv_get(PUBLIC_LATEST_KEY, KV_REST_API_READ_ONLY_TOKEN)
    if not b64:
        return None
    try:
        raw = base64.urlsafe_b64decode(b64.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


# ============================================================
# App
# ============================================================
app = FastAPI(title="Polyclawd Executor", version="0.3.0")


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
        "public_latest_key": PUBLIC_LATEST_KEY,
        "position_id": POSITION_ID,
    }


# ---------- Scenario control (admin) ----------
@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    s = await get_scenario()
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_post(body: ScenarioBody, _: bool = Depends(require_auth)):
    s = await set_scenario(body.scenario)
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


# ---------- Public endpoints (no auth) ----------
@app.get("/public/health")
async def public_health():
    latest = await load_public_latest()
    return {"ok": True, "has_latest": bool(latest), "latest_ts": latest.get("ts") if latest else None}


@app.get("/public/latest")
async def public_latest():
    latest = await load_public_latest()
    if not latest:
        return {"ok": True, "empty": True, "message": "No data yet. Wait for /run to execute at least once."}
    return {"ok": True, "data": latest}


# ---------- Executor tick ----------
@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked", "message": "Another run is in progress"}

    try:
        scenario = await get_scenario()

        agents = run_fake_agents(scenario)
        summary = aggregate_decision(agents)

        ts = _now_ts()
        payload = {
            "ts": ts,
            "tick": _tick_index(ts),
            "position_id": POSITION_ID,
            "scenario": scenario,
            "summary": summary,
            "agents": agents,
        }

        await save_public_latest(payload)

        duration = round(time.time() - started, 3)
        return {
            "ok": True,
            "skipped": False,
            "message": "tick executed",
            "duration_sec": duration,
            "saved_to": PUBLIC_LATEST_KEY,
            "data": payload,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass
