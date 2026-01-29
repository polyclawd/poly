import os
import time
import uuid
import json
import random
from typing import Optional, List, Dict, Any

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# ----------------------------
# Config
# ----------------------------
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", "")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))

SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")
TICK_KEY = os.getenv("EXECUTOR_TICK_KEY", "executor:tick")

DEFAULT_MARKET_ID = os.getenv("EXECUTOR_DEFAULT_MARKET_ID", "POLY:DEMO_001")
BUY_THRESHOLD = int(os.getenv("EXECUTOR_BUY_THRESHOLD", "9"))

# где будет лежать пресет агентов в репе (рядом с app.py)
AGENTS_PRESET_FILE = os.getenv("EXECUTOR_AGENTS_PRESET_FILE", "agents_preset.json")


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
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


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
    return data.get("result")


async def upstash_set(key: str, value: str, token: str) -> bool:
    if not KV_REST_API_URL or not token:
        return False
    # /set/<key>/<value> — value надо url-safe, но JSON лучше передавать через body.
    # Upstash поддерживает /set/<key> + body (raw). Чтобы не усложнять — base64 не делаем,
    # а просто используем /set/<key>/<value> ТОЛЬКО для простых строк.
    url = f"{KV_REST_API_URL}/set/{key}/{value}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))
    if r.status_code == 401:
        return False
    r.raise_for_status()
    return True


async def upstash_set_json(key: str, obj: Any, token: str) -> bool:
    if not KV_REST_API_URL or not token:
        return False
    url = f"{KV_REST_API_URL}/set/{key}"
    payload = json.dumps(obj, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token), content=payload.encode("utf-8"))
    if r.status_code == 401:
        return False
    r.raise_for_status()
    return True


async def upstash_lpush_json(key: str, obj: Any, token: str) -> bool:
    if not KV_REST_API_URL or not token:
        return False
    url = f"{KV_REST_API_URL}/lpush/{key}"
    payload = json.dumps(obj, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token), content=payload.encode("utf-8"))
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
    return lock_id if ok == 1 else None


async def unlock(_: str) -> None:
    await upstash_del(LOCK_KEY, KV_REST_API_TOKEN)


# ----------------------------
# Agent preset loading + fake generation
# ----------------------------
def load_agents_preset() -> List[Dict[str, Any]]:
    try:
        with open(AGENTS_PRESET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "agents" in data and isinstance(data["agents"], list):
            return data["agents"]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def make_fake_agent_result(agent_def: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """
    agent_def ожидаем вида:
    {
      "agent_id": "agent_01",
      "name": "...",
      "bias": "BUY" | "HOLD" | "MIXED"
    }
    """
    agent_id = agent_def.get("agent_id", "agent_x")
    name = agent_def.get("name", "Agent")
    bias = agent_def.get("bias", "MIXED").upper()

    # небольшой сценарный сдвиг голосов
    scenario = (scenario or "neutral").lower()
    bull_boost = 0.15 if scenario == "bull" else 0.0
    bear_boost = 0.15 if scenario == "bear" else 0.0

    if bias == "BUY":
        p_buy = 0.75 + bull_boost - bear_boost
    elif bias == "HOLD":
        p_buy = 0.25 + bull_boost - bear_boost
    else:
        p_buy = 0.55 + bull_boost - bear_boost

    p_buy = max(0.05, min(0.95, p_buy))

    vote = "BUY" if random.random() < p_buy else "HOLD"
    confidence = round(random.uniform(0.52, 0.9) if vote == "BUY" else random.uniform(0.45, 0.78), 3)

    price = round(random.uniform(0.2, 0.95), 3)
    implied_prob = round(random.uniform(0.2, 0.95), 3)
    sentiment = round(random.uniform(-1.0, 1.0), 3)
    volatility = round(random.uniform(0.1, 0.9), 3)

    if vote == "BUY":
        reason = random.choice(
            [
                "Edge detected; scenario tailwind supports entry.",
                "Liquidity acceptable; expected value positive this tick.",
                "Pricing looks favorable vs. implied probability.",
                "Signal alignment across factors suggests positive EV.",
            ]
        )
    else:
        reason = random.choice(
            [
                "Signals conflict; avoid overtrading.",
                "No clear edge at current price.",
                "Volatility not attractive for entry this tick.",
                "Hold: wait for better setup.",
            ]
        )

    return {
        "agent_id": agent_id,
        "name": name,
        "vote": vote,
        "confidence": confidence,
        "reason": reason,
        "signals": {
            "price": price,
            "implied_prob": implied_prob,
            "sentiment": sentiment,
            "volatility": volatility,
        },
    }


def summarize(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy = [a for a in agents if a.get("vote") == "BUY"]
    hold = [a for a in agents if a.get("vote") == "HOLD"]
    avg_conf = round(sum(float(a.get("confidence", 0)) for a in agents) / max(1, len(agents)), 3)

    decision = "BUY" if len(buy) >= BUY_THRESHOLD else "HOLD"
    rule = f"BUY if buy_count >= {BUY_THRESHOLD} else HOLD"

    # топ причины BUY
    top_buy_reasons = []
    for a in sorted(buy, key=lambda x: float(x.get("confidence", 0)), reverse=True)[:3]:
        if a.get("reason"):
            top_buy_reasons.append(a["reason"])

    return {
        "decision": decision,
        "buy_count": len(buy),
        "hold_count": len(hold),
        "avg_confidence": avg_conf,
        "rule": rule,
        "digest": {
            "buy_count": len(buy),
            "top_buy_reasons": top_buy_reasons,
        },
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
        "lock_key": LOCK_KEY,
        "scenario_key": SCENARIO_KEY,
        "tick_key": TICK_KEY,
        "agents_preset_file": AGENTS_PRESET_FILE,
        "buy_threshold": BUY_THRESHOLD,
    }


# -------- Scenario endpoints (GET + POST) --------
@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    scenario = await upstash_get(SCENARIO_KEY, KV_REST_API_TOKEN)
    scenario = scenario or "neutral"
    return {"ok": True, "scenario": scenario, "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_post(body: Dict[str, Any], _: bool = Depends(require_auth)):
    # ожидаем {"scenario": "neutral"|"bull"|"bear"}
    scenario = (body.get("scenario") or "").strip().lower()
    if scenario not in {"neutral", "bull", "bear"}:
        raise HTTPException(status_code=400, detail="scenario must be one of: neutral, bull, bear")

    ok = await upstash_set(SCENARIO_KEY, scenario, KV_REST_API_TOKEN)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to write scenario to KV")
    return {"ok": True, "scenario": scenario}


# -------- Public read endpoints --------
@app.get("/public/latest/{market_id}")
async def public_latest(market_id: str):
    key = f"public:latest:{market_id}"
    data = await upstash_get(key, KV_REST_API_TOKEN)
    if not data:
        return {"ok": False, "market_id": market_id, "error": "no data yet"}
    try:
        return {"ok": True, "market_id": market_id, "data": json.loads(data)}
    except Exception:
        return {"ok": True, "market_id": market_id, "data_raw": data}


@app.get("/public/history/{market_id}")
async def public_history(market_id: str):
    # простая заглушка: возвращаем последние N сохраненных снапшотов,
    # но так как Upstash REST list read неудобен без /lrange — пока не трогаем.
    return {"ok": True, "market_id": market_id, "hint": "history is stored as list public:history:<market_id>"}


# -------- Main cron endpoint --------
@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked", "message": "Another executor run is in progress"}

    try:
        # scenario
        scenario = await upstash_get(SCENARIO_KEY, KV_REST_API_TOKEN)
        scenario = scenario or "neutral"

        # tick (инкрементим чтобы было видно обновление)
        tick_raw = await upstash_get(TICK_KEY, KV_REST_API_TOKEN)
        try:
            tick = int(tick_raw) if tick_raw is not None else 0
        except Exception:
            tick = 0
        tick += 1
        await upstash_set(TICK_KEY, str(tick), KV_REST_API_TOKEN)

        # agents preset → 15 результатов
        preset = load_agents_preset()
        if not preset:
            # fallback: если файла нет
            preset = [{"agent_id": f"agent_{i:02d}", "name": f"Agent {i:02d}", "bias": "MIXED"} for i in range(1, 16)]

        agents = [make_fake_agent_result(a, scenario) for a in preset[:15]]

        # summary
        summary = summarize(agents)

        market_id = DEFAULT_MARKET_ID
        ts = int(time.time())

        snapshot = {
            "ts": ts,
            "tick": tick,
            "market_id": market_id,
            "scenario": scenario,
            "summary": summary,
            "agents": agents,
        }

        # save public latest + history (history в список)
        latest_key = f"public:latest:{market_id}"
        hist_key = f"public:history:{market_id}"

        await upstash_set_json(latest_key, snapshot, KV_REST_API_TOKEN)
        await upstash_lpush_json(hist_key, snapshot, KV_REST_API_TOKEN)

        duration = round(time.time() - started, 3)

        # ответ /run (короткий)
        return {
            "ok": True,
            "skipped": False,
            "message": "tick executed",
            "scenario": scenario,
            "market_id": market_id,
            "tick": tick,
            "buy_count": summary["buy_count"],
            "decision": summary["decision"],
            "duration_sec": duration,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass
