import os
import time
import uuid
import random
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
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", "")  # optional

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")

LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))
RUN_TIMEOUT_SECONDS = float(os.getenv("EXECUTOR_RUN_TIMEOUT_SECONDS", "25"))

# "конкретная позиция" (пока фейк) — позже заменим на реальную
POSITION_ID = os.getenv("POSITION_ID", "POLYMARKET:DEMO_POSITION_001")


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


async def _upstash_post(path: str, token: str) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN missing")

    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()
    return r.json()


async def upstash_set(key: str, value: str, token: str) -> bool:
    # Upstash REST: SET key value
    data = await _upstash_post(f"set/{key}/{value}", token)
    # usually: {"result":"OK"}
    return str(data.get("result", "")).upper() == "OK"


async def upstash_get(key: str, token: str) -> Optional[str]:
    # Upstash REST: GET key
    data = await _upstash_post(f"get/{key}", token)
    # usually: {"result": "value"} or {"result": null}
    v = data.get("result", None)
    if v is None:
        return None
    return str(v)


async def upstash_setnx(key: str, value: str, token: str) -> int:
    # Upstash REST: SETNX key value
    data = await _upstash_post(f"setnx/{key}/{value}", token)
    return int(data.get("result", 0))


async def upstash_del(key: str, token: str) -> int:
    # Upstash REST: DEL key
    try:
        data = await _upstash_post(f"del/{key}", token)
        return int(data.get("result", 0))
    except Exception:
        return 0


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id, KV_REST_API_TOKEN)
    if ok == 1:
        return lock_id
    return None


async def unlock(_: str) -> None:
    await upstash_del(LOCK_KEY, KV_REST_API_TOKEN)


# ============================================================
# Scenario storage
# ============================================================
ALLOWED_SCENARIOS = {"bull", "neutral", "bear"}


async def get_scenario() -> str:
    # если KV не настроен — живём дефолтом
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        return "neutral"

    # читаем только обычным токеном (RO токен — опционально, можно расширить потом)
    value = await upstash_get(SCENARIO_KEY, KV_REST_API_TOKEN)
    if value in ALLOWED_SCENARIOS:
        return value
    return "neutral"


async def set_scenario(value: str) -> str:
    if value not in ALLOWED_SCENARIOS:
        raise HTTPException(status_code=422, detail=f"Invalid scenario. Allowed: {sorted(ALLOWED_SCENARIOS)}")

    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        raise HTTPException(status_code=500, detail="KV not configured: cannot persist scenario")

    ok = await upstash_set(SCENARIO_KEY, value, KV_REST_API_TOKEN)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save scenario to KV")

    return value


class ScenarioBody(BaseModel):
    scenario: str = Field(..., description="bull | neutral | bear")


# ============================================================
# Fake Agents (A1)
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
    """
    Фейковая логика:
    - В bull большинство склоняется к BUY
    - В bear большинство склоняется к HOLD (или SELL, но пока делаем BUY/HOLD)
    - В neutral смешанно
    Плюс небольшой псевдо-рандом, но детерминированный на (agent_idx, scenario)
    """
    seed = (agent_idx + 1) * 1000 + {"bull": 1, "neutral": 2, "bear": 3}[scenario]
    rnd = random.Random(seed)

    if scenario == "bull":
        buy_prob = 0.72
    elif scenario == "bear":
        buy_prob = 0.22
    else:
        buy_prob = 0.45

    vote_buy = rnd.random() < buy_prob

    confidence = round(0.55 + rnd.random() * 0.4, 2)  # 0.55..0.95
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

    # Фейковые "сигналы" (заготовленные числа, просто чтобы выглядело как анализ)
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

    # Простое правило (пока):
    # BUY если >= 9 агентов за BUY, иначе HOLD.
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
# App
# ============================================================
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
        "position_id": POSITION_ID,
    }


@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    s = await get_scenario()
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_post(body: ScenarioBody, _: bool = Depends(require_auth)):
    s = await set_scenario(body.scenario)
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    """
    Запускается GitHub Actions cron-ом.
    Требует: Authorization: Bearer <EXECUTOR_SECRET>
    """
    started = time.time()

    # lock, чтобы не было параллельных тиков
    lock_id = await try_lock()
    if not lock_id:
        return {
            "ok": True,
            "skipped": True,
            "reason": "locked",
            "message": "Another executor run is in progress",
        }

    try:
        scenario = await get_scenario()

        # A1: фейковые агенты
        agents = run_fake_agents(scenario)
        summary = aggregate_decision(agents)

        duration = round(time.time() - started, 3)
        return {
            "ok": True,
            "skipped": False,
            "message": "tick executed",
            "duration_sec": duration,
            "position_id": POSITION_ID,
            "scenario": scenario,
            "summary": summary,
            "agents": agents,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass
