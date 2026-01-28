import os
import time
import uuid
import random
from typing import Optional, Dict, Any, List

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# ----------------------------
# Config
# ----------------------------
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", "") or KV_REST_API_TOKEN

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")

LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))
RUN_TIMEOUT_SECONDS = float(os.getenv("EXECUTOR_RUN_TIMEOUT_SECONDS", "25"))

DEFAULT_SCENARIO = os.getenv("EXECUTOR_DEFAULT_SCENARIO", "neutral")
ALLOWED_SCENARIOS = ["neutral", "bull", "bear"]


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
def _upstash_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


async def upstash_get(key: str, token: str) -> Optional[str]:
    if not KV_REST_API_URL or not token:
        return None

    url = f"{KV_REST_API_URL}/get/{key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        return None
    r.raise_for_status()
    data = r.json()
    # Upstash обычно: {"result":"value"} или {"result":null}
    val = data.get("result", None)
    return val


async def upstash_set(key: str, value: str, token: str) -> bool:
    if not KV_REST_API_URL or not token:
        return False

    url = f"{KV_REST_API_URL}/set/{key}/{value}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        return False
    r.raise_for_status()
    return True


async def upstash_setnx(key: str, value: str, token: str) -> int:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN is empty")

    url = f"{KV_REST_API_URL}/setnx/{key}/{value}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()
    data = r.json()
    return int(data.get("result", 0))


async def upstash_del(key: str, token: str) -> int:
    if not KV_REST_API_URL or not token:
        return 0

    url = f"{KV_REST_API_URL}/del/{key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        return 0

    r.raise_for_status()
    data = r.json()
    return int(data.get("result", 0))


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id, KV_REST_API_TOKEN)
    if ok == 1:
        return lock_id
    return None


async def unlock(lock_id: str) -> None:
    await upstash_del(LOCK_KEY, KV_REST_API_TOKEN)


async def get_scenario() -> str:
    s = await upstash_get(SCENARIO_KEY, KV_REST_API_READ_ONLY_TOKEN)
    if not s:
        return DEFAULT_SCENARIO
    s = str(s).strip().lower()
    return s if s in ALLOWED_SCENARIOS else DEFAULT_SCENARIO


async def set_scenario(s: str) -> str:
    s = (s or "").strip().lower()
    if s not in ALLOWED_SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Invalid scenario. Allowed: {ALLOWED_SCENARIOS}")
    ok = await upstash_set(SCENARIO_KEY, s, KV_REST_API_TOKEN)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to set scenario (Upstash misconfigured?)")
    return s


# ----------------------------
# Fake data for 15 agents
# ----------------------------
# Это "заготовленная информация" — позже вынесем в KV/файл.
FAKE_AGENT_PROFILES: List[Dict[str, Any]] = [
    {"id": 1, "name": "NewsPulse", "weight": 1.0},
    {"id": 2, "name": "OrderBook", "weight": 1.1},
    {"id": 3, "name": "Volatility", "weight": 0.9},
    {"id": 4, "name": "MacroMood", "weight": 1.0},
    {"id": 5, "name": "SocialBuzz", "weight": 0.8},
    {"id": 6, "name": "WhaleWatch", "weight": 1.2},
    {"id": 7, "name": "Liquidity", "weight": 1.0},
    {"id": 8, "name": "Momentum", "weight": 1.1},
    {"id": 9, "name": "MeanRevert", "weight": 0.9},
    {"id": 10, "name": "Sentiment", "weight": 1.0},
    {"id": 11, "name": "ArbFinder", "weight": 0.7},
    {"id": 12, "name": "RiskGuard", "weight": 1.3},
    {"id": 13, "name": "EdgeCalc", "weight": 1.0},
    {"id": 14, "name": "Timing", "weight": 0.8},
    {"id": 15, "name": "Consensus", "weight": 1.4},
]


def fake_agent_signal(scenario: str, agent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Фейковая "аналитика" агента:
    - генерим score в зависимости от сценария
    - решение BUY если score >= threshold
    """
    base = {"bull": 0.65, "neutral": 0.50, "bear": 0.35}.get(scenario, 0.50)
    noise = random.uniform(-0.20, 0.20)
    weight = float(agent.get("weight", 1.0))

    raw_score = base + noise
    score = max(0.0, min(1.0, raw_score * (0.90 + 0.10 * weight)))

    threshold = 0.60 if scenario == "bull" else (0.55 if scenario == "neutral" else 0.65)
    decision = "BUY" if score >= threshold else "SKIP"

    reason = (
        "scenario tailwind" if scenario == "bull" and decision == "BUY" else
        "mixed signals" if scenario == "neutral" else
        "risk-off regime"
    )

    return {
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "scenario": scenario,
        "score": round(score, 3),
        "threshold": threshold,
        "decision": decision,
        "reason": reason,
    }


def aggregate_decision(agent_reports: List[Dict[str, Any]], scenario: str) -> Dict[str, Any]:
    buys = [a for a in agent_reports if a["decision"] == "BUY"]
    buy_count = len(buys)
    total = len(agent_reports)

    # простая логика консенсуса:
    # bull -> нужно >= 6 BUY
    # neutral -> >= 8 BUY
    # bear -> >= 10 BUY (почти никогда)
    required = 6 if scenario == "bull" else (8 if scenario == "neutral" else 10)
    final = "BUY" if buy_count >= required else "SKIP"

    avg_score = round(sum(a["score"] for a in agent_reports) / max(1, total), 3)

    return {
        "final_decision": final,
        "buy_count": buy_count,
        "total_agents": total,
        "required_buys": required,
        "avg_score": avg_score,
    }


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
        "default_scenario": DEFAULT_SCENARIO,
    }


@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    s = await get_scenario()
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_post(payload: Dict[str, Any], _: bool = Depends(require_auth)):
    s = await set_scenario(str(payload.get("scenario", "")))
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    scenario = await get_scenario()

    lock_id = await try_lock()
    if not lock_id:
        return {
            "ok": True,
            "skipped": True,
            "scenario": scenario,
            "reason": "locked",
            "message": "Another executor run is in progress",
        }

    try:
        # --- 15 agents ---
        agent_reports = [fake_agent_signal(scenario, a) for a in FAKE_AGENT_PROFILES]
        summary = aggregate_decision(agent_reports, scenario)

        duration = round(time.time() - started, 3)

        return {
            "ok": True,
            "skipped": False,
            "scenario": scenario,
            "summary": summary,
            "agents": agent_reports,
            "duration_sec": duration,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass
