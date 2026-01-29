import os
import time
import uuid
import json
from typing import Optional, Dict, Any, List
from urllib.parse import quote

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

DEFAULT_MARKET_ID = os.getenv("DEFAULT_MARKET_ID", "POLY:DEMO_001")

SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")
DEFAULT_SCENARIO = os.getenv("DEFAULT_SCENARIO", "neutral")
ALLOWED_SCENARIOS = ["neutral", "bull", "bear"]

HISTORY_TTL_SECONDS = int(os.getenv("EXECUTOR_HISTORY_TTL_SECONDS", "3600"))  # 1 час
HISTORY_MAX_ITEMS = int(os.getenv("EXECUTOR_HISTORY_MAX_ITEMS", "120"))       # ограничим размер

HTTP_TIMEOUT = float(os.getenv("EXECUTOR_HTTP_TIMEOUT", "12"))


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="Polyclawd Executor", version="0.1.0")


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
# Upstash helpers (SAFE URL ENCODING)
# ----------------------------
def _upstash_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _clean(s: str) -> str:
    # убираем переносы/пробелы/случайные кавычки
    if s is None:
        return ""
    return str(s).strip().strip('"').strip("'")


def _enc(s: str) -> str:
    # ВАЖНО: безопасно кодируем для path-сегментов
    return quote(_clean(s), safe="")


async def _upstash_request(method: str, path: str, token: str) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN is empty")

    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.request(method, url, headers=_upstash_headers(token))

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    # Чтобы видеть причину 400 в ответе сервера
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Upstash error {r.status_code}: {r.text}")

    return r.json()


async def upstash_get(key: str, token: str) -> Optional[str]:
    data = await _upstash_request("GET", f"get/{_enc(key)}", token)
    # {"result": "..."} or {"result": null}
    res = data.get("result", None)
    if res is None:
        return None
    return str(res)


async def upstash_set(key: str, value: str, token: str) -> None:
    # /set/<key>/<value>
    await _upstash_request("POST", f"set/{_enc(key)}/{_enc(value)}", token)


async def upstash_del(key: str, token: str) -> None:
    await _upstash_request("POST", f"del/{_enc(key)}", token)


async def upstash_setnx(key: str, value: str, token: str) -> int:
    data = await _upstash_request("POST", f"setnx/{_enc(key)}/{_enc(value)}", token)
    return int(data.get("result", 0))


async def upstash_expire(key: str, seconds: int, token: str) -> None:
    # /expire/<key>/<seconds>
    await _upstash_request("POST", f"expire/{_enc(key)}/{int(seconds)}", token)


# ----------------------------
# Lock
# ----------------------------
async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id, KV_REST_API_TOKEN)
    if ok == 1:
        # ставим TTL, чтобы не словить вечный лок
        try:
            await upstash_expire(LOCK_KEY, LOCK_TTL_SECONDS, KV_REST_API_TOKEN)
        except Exception:
            pass
        return lock_id
    return None


async def unlock() -> None:
    try:
        await upstash_del(LOCK_KEY, KV_REST_API_TOKEN)
    except Exception:
        pass


# ----------------------------
# Scenario storage
# ----------------------------
async def get_scenario() -> str:
    try:
        raw = await upstash_get(SCENARIO_KEY, KV_REST_API_TOKEN)
        if raw and raw in ALLOWED_SCENARIOS:
            return raw
    except Exception:
        pass
    return DEFAULT_SCENARIO


async def set_scenario(s: str) -> str:
    s = _clean(s).lower()
    if s not in ALLOWED_SCENARIOS:
        raise HTTPException(status_code=422, detail=f"scenario must be one of {ALLOWED_SCENARIOS}")
    await upstash_set(SCENARIO_KEY, s, KV_REST_API_TOKEN)
    return s


# ----------------------------
# Fake agents (15)
# ----------------------------
AGENTS: List[Dict[str, str]] = [
    {"agent_id": "agent_01", "name": "Market Microstructure"},
    {"agent_id": "agent_02", "name": "Orderbook Reader"},
    {"agent_id": "agent_03", "name": "News Synthesizer"},
    {"agent_id": "agent_04", "name": "Social Sentiment"},
    {"agent_id": "agent_05", "name": "Volatility Monitor"},
    {"agent_id": "agent_06", "name": "Flow Tracker"},
    {"agent_id": "agent_07", "name": "Liquidity Scout"},
    {"agent_id": "agent_08", "name": "Contrarian"},
    {"agent_id": "agent_09", "name": "Momentum"},
    {"agent_id": "agent_10", "name": "Mean Reversion"},
    {"agent_id": "agent_11", "name": "Risk Manager"},
    {"agent_id": "agent_12", "name": "Macro Filter"},
    {"agent_id": "agent_13", "name": "Regime Detector"},
    {"agent_id": "agent_14", "name": "Anomaly Watch"},
    {"agent_id": "agent_15", "name": "Edge Aggregator"},
]

BUY_REASONS = [
    "Edge detected; scenario tailwind supports entry.",
    "Liquidity acceptable; expected value positive this tick.",
    "Pricing looks favorable vs. implied probability.",
    "Momentum aligns; small probe position is justified.",
]
HOLD_REASONS = [
    "Signals conflict; avoid overtrading.",
    "Spread too wide; wait for better entry.",
    "No clear edge; preserve capital.",
    "Regime uncertain; hold until confirmation.",
]


def _rand01(seed: int) -> float:
    # детерминированный "random"
    x = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    return (x % 10000) / 10000.0


def fake_agent_vote(agent_idx: int, tick: int, scenario: str) -> Dict[str, Any]:
    base_seed = (tick + 1) * 1000 + agent_idx * 97
    r = _rand01(base_seed)

    # bias от сценария
    bias = 0.0
    if scenario == "bull":
        bias = 0.12
    elif scenario == "bear":
        bias = -0.12

    score = r + bias
    vote = "BUY" if score >= 0.52 else "HOLD"
    conf = round(min(max(0.45, 0.55 + (score - 0.5)), 0.95), 2)

    reason = BUY_REASONS[int(_rand01(base_seed + 7) * len(BUY_REASONS))] if vote == "BUY" else HOLD_REASONS[int(_rand01(base_seed + 9) * len(HOLD_REASONS))]

    signals = {
        "price": round(_rand01(base_seed + 11), 3),
        "implied_prob": round(_rand01(base_seed + 13), 3),
        "sentiment": round(_rand01(base_seed + 17) * 2 - 1, 3),
        "volatility": round(_rand01(base_seed + 19), 3),
    }

    return {
        "agent_id": AGENTS[agent_idx]["agent_id"],
        "name": AGENTS[agent_idx]["name"],
        "vote": vote,
        "confidence": conf,
        "reason": reason,
        "signals": signals,
    }


def summarize(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy = sum(1 for a in agents if a["vote"] == "BUY")
    hold = sum(1 for a in agents if a["vote"] == "HOLD")
    avg_conf = round(sum(float(a["confidence"]) for a in agents) / max(1, len(agents)), 3)

    decision = "BUY" if buy >= 9 else "HOLD"
    return {
        "decision": decision,
        "buy_count": buy,
        "hold_count": hold,
        "avg_confidence": avg_conf,
        "rule": "BUY if buy_count >= 9 else HOLD",
    }


def make_digest(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy_reasons = [a["reason"] for a in agents if a["vote"] == "BUY"][:3]
    return {
        "buy_count": sum(1 for a in agents if a["vote"] == "BUY"),
        "top_buy_reasons": buy_reasons,
    }


# ----------------------------
# Storage keys
# ----------------------------
def latest_key(market_id: str) -> str:
    return f"executor:latest:{_clean(market_id)}"


def history_key(market_id: str) -> str:
    return f"executor:history:{_clean(market_id)}"


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
async def root():
    return {"ok": True, "service": "executor", "docs": "/docs"}


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "has_executor_secret": bool(EXECUTOR_SECRET),
        "has_kv_url": bool(KV_REST_API_URL),
        "has_kv_token": bool(KV_REST_API_TOKEN),
        "default_market_id": DEFAULT_MARKET_ID,
    }


@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    s = await get_scenario()
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_post(body: Dict[str, Any], _: bool = Depends(require_auth)):
    s = body.get("scenario", None)
    if not s:
        raise HTTPException(status_code=422, detail="body must contain: {\"scenario\": \"neutral|bull|bear\"}")
    new_s = await set_scenario(str(s))
    return {"ok": True, "scenario": new_s}


@app.get("/public/latest/{market_id}")
async def public_latest(market_id: str):
    k = latest_key(market_id)
    raw = await upstash_get(k, KV_REST_API_TOKEN)
    if not raw:
        return {"ok": False, "market_id": market_id, "error": "no data yet"}
    try:
        return json.loads(raw)
    except Exception:
        return {"ok": False, "market_id": market_id, "error": "corrupt json", "raw": raw}


@app.get("/public/history/{market_id}")
async def public_history(market_id: str):
    k = history_key(market_id)
    raw = await upstash_get(k, KV_REST_API_TOKEN)
    if not raw:
        return {"ok": True, "market_id": market_id, "items": []}
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            items = []
        return {"ok": True, "market_id": market_id, "items": items}
    except Exception:
        return {"ok": True, "market_id": market_id, "items": []}


@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked", "message": "Another run is in progress"}

    try:
        scenario = await get_scenario()
        market_id = DEFAULT_MARKET_ID

        # tick делаем монотонным (секунды // 300)
        tick = int(time.time() // 300)

        agents = [fake_agent_vote(i, tick, scenario) for i in range(len(AGENTS))]
        summary = summarize(agents)
        digest = make_digest(agents)

        item = {
            "ts": int(time.time()),
            "tick": tick,
            "market_id": market_id,
            "scenario": scenario,
            "summary": summary,
            "digest": digest,
            "agents": agents,
        }

        latest_payload = {
            "ok": True,
            "market_id": market_id,
            "data": item,
        }

        # latest
        await upstash_set(latest_key(market_id), json.dumps(latest_payload, ensure_ascii=False), KV_REST_API_TOKEN)
        await upstash_expire(latest_key(market_id), HISTORY_TTL_SECONDS, KV_REST_API_TOKEN)

        # history
        hk = history_key(market_id)
        raw_hist = await upstash_get(hk, KV_REST_API_TOKEN)
        hist: List[Dict[str, Any]] = []
        if raw_hist:
            try:
                hist = json.loads(raw_hist)
                if not isinstance(hist, list):
                    hist = []
            except Exception:
                hist = []

        # добавляем новый item в конец (как у тебя сейчас)
        hist.append(item)

        # режем по времени (последний час)
        cutoff = int(time.time()) - HISTORY_TTL_SECONDS
        hist = [x for x in hist if int(x.get("ts", 0)) >= cutoff]

        # режем по количеству
        if len(hist) > HISTORY_MAX_ITEMS:
            hist = hist[-HISTORY_MAX_ITEMS:]

        await upstash_set(hk, json.dumps(hist, ensure_ascii=False), KV_REST_API_TOKEN)
        await upstash_expire(hk, HISTORY_TTL_SECONDS, KV_REST_API_TOKEN)

        duration = round(time.time() - started, 3)
        return {"ok": True, "skipped": False, "scenario": scenario, "message": "tick executed", "duration_sec": duration}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        await unlock()
