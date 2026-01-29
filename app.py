import os
import time
import uuid
import random
import json
import base64
from typing import Optional, Dict, Any, List

import httpx
from fastapi import Depends, FastAPI, HTTPException, Path, Query
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

PUBLIC_INDEX_KEY = os.getenv("PUBLIC_INDEX_KEY", "public:latest:index")
PUBLIC_MARKET_KEY_PREFIX = os.getenv("PUBLIC_MARKET_KEY_PREFIX", "public:latest:market:")
PUBLIC_HISTORY_KEY_PREFIX = os.getenv("PUBLIC_HISTORY_KEY_PREFIX", "public:history:market:")

# 5 hours history, tick every 5 min = 60 points
HISTORY_POINTS = int(os.getenv("HISTORY_POINTS", "60"))

ALLOWED_SCENARIOS = {"bull", "neutral", "bear"}

DEFAULT_MARKETS = [
    "POLY:DEMO_001",
    "POLY:DEMO_002",
    "POLY:DEMO_003",
    "POLY:DEMO_004",
    "POLY:DEMO_005",
    "POLY:DEMO_006",
    "POLY:DEMO_007",
    "POLY:DEMO_008",
    "POLY:DEMO_009",
    "POLY:DEMO_010",
    "POLY:DEMO_011",
    "POLY:DEMO_012",
    "POLY:DEMO_013",
    "POLY:DEMO_014",
    "POLY:DEMO_015",
]


def get_markets() -> List[str]:
    raw = (os.getenv("MARKETS") or "").strip()
    if not raw:
        return DEFAULT_MARKETS
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:15] if len(out) > 15 else out


# ============================================================
# Auth
# ============================================================
bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="EXECUTOR_SECRET missing")

    if creds is None:
        raise HTTPException(status_code=403, detail="Not authenticated")

    if creds.credentials != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return True


# ============================================================
# Upstash helpers
# ============================================================
def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


async def _get(path: str, token: str) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="KV misconfigured")
    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=_headers(token))
    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized")
    r.raise_for_status()
    return r.json()


async def _post(path: str, token: str) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="KV misconfigured")
    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_headers(token))
    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized")
    r.raise_for_status()
    return r.json()


async def kv_set(key: str, value: str, token: str) -> bool:
    data = await _post(f"set/{key}/{value}", token)
    return str(data.get("result", "")).upper() == "OK"


async def kv_get(key: str, token: str) -> Optional[str]:
    data = await _get(f"get/{key}", token)
    v = data.get("result", None)
    return None if v is None else str(v)


async def kv_setnx(key: str, value: str, token: str) -> int:
    data = await _post(f"setnx/{key}/{value}", token)
    return int(data.get("result", 0))


async def kv_del(key: str, token: str) -> int:
    try:
        data = await _post(f"del/{key}", token)
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
# Scenario
# ============================================================
async def get_scenario() -> str:
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        return "neutral"
    v = await kv_get(SCENARIO_KEY, KV_REST_API_READ_ONLY_TOKEN)
    return v if v in ALLOWED_SCENARIOS else "neutral"


async def set_scenario(value: str) -> str:
    value = (value or "").strip().lower()
    if value not in ALLOWED_SCENARIOS:
        raise HTTPException(status_code=422, detail=f"Invalid scenario: {sorted(ALLOWED_SCENARIOS)}")
    ok = await kv_set(SCENARIO_KEY, value, KV_REST_API_TOKEN)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save scenario")
    return value


class ScenarioBody(BaseModel):
    scenario: str = Field(..., description="bull | neutral | bear")


# ============================================================
# Fake agents (deterministic per market & tick)
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


def now_ts() -> int:
    return int(time.time())


def tick_index(ts: int) -> int:
    return int(ts / 300)  # 5 minutes


def market_key_latest(market_id: str) -> str:
    return f"{PUBLIC_MARKET_KEY_PREFIX}{market_id}"


def market_key_history(market_id: str) -> str:
    return f"{PUBLIC_HISTORY_KEY_PREFIX}{market_id}"


def _agent_decision(agent_idx: int, scenario: str, market_id: str, tick: int) -> Dict[str, Any]:
    base = {"bull": 1, "neutral": 2, "bear": 3}[scenario]
    seed = (agent_idx + 1) * 1000 + base * 100 + (hash(market_id) % 1000) + (tick % 1000) * 7
    rnd = random.Random(seed)

    if scenario == "bull":
        buy_prob = 0.70
    elif scenario == "bear":
        buy_prob = 0.20
    else:
        buy_prob = 0.45

    vote_buy = rnd.random() < buy_prob
    confidence = round(0.55 + rnd.random() * 0.4, 2)
    if not vote_buy:
        confidence = round(max(0.5, confidence - 0.15), 2)

    reason_buy = [
        "Edge detected; scenario tailwind supports entry.",
        "Momentum aligns; small probe position is justified.",
        "Liquidity acceptable; expected value positive this tick.",
        "Pricing looks favorable vs. implied probability.",
    ]
    reason_hold = [
        "Edge unclear this tick; wait for better confirmation.",
        "Signals conflict; avoid overtrading.",
        "Risk not attractive at current pricing; hold.",
        "Spread/liquidity not favorable; skip.",
    ]
    reason = rnd.choice(reason_buy if vote_buy else reason_hold)

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


def run_fake_agents(scenario: str, market_id: str, tick: int) -> List[Dict[str, Any]]:
    return [_agent_decision(i, scenario, market_id, tick) for i in range(15)]


def aggregate_decision(agent_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    buys = [a for a in agent_reports if a["vote"] == "BUY"]
    holds = [a for a in agent_reports if a["vote"] != "BUY"]
    buy_count = len(buys)
    hold_count = len(holds)

    decision = "BUY" if buy_count >= 9 else "HOLD"
    avg_conf = round(sum(a["confidence"] for a in agent_reports) / max(1, len(agent_reports)), 3)

    return {
        "decision": decision,
        "buy_count": buy_count,
        "hold_count": hold_count,
        "avg_confidence": avg_conf,
        "rule": "BUY if buy_count >= 9 else HOLD",
    }


# ============================================================
# Base64 JSON store
# ============================================================
def encode_json(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def decode_json(b64: str) -> Optional[Dict[str, Any]]:
    try:
        raw = base64.urlsafe_b64decode(b64.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


async def save_json(key: str, payload: Dict[str, Any]) -> None:
    b64 = encode_json(payload)
    ok = await kv_set(key, b64, KV_REST_API_TOKEN)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to persist key: {key}")


async def load_json(key: str) -> Optional[Dict[str, Any]]:
    b64 = await kv_get(key, KV_REST_API_READ_ONLY_TOKEN)
    if not b64:
        return None
    return decode_json(b64)


async def append_history(market_id: str, item: Dict[str, Any]) -> None:
    """
    Храним массив последних HISTORY_POINTS элементов.
    """
    key = market_key_history(market_id)
    existing = await load_json(key)
    if not existing or "items" not in existing:
        history = {"market_id": market_id, "items": []}
    else:
        history = existing

    items: List[Dict[str, Any]] = history.get("items", [])
    items.append(item)

    # ограничиваем размер
    if len(items) > HISTORY_POINTS:
        items = items[-HISTORY_POINTS:]

    history["items"] = items
    await save_json(key, history)


def digest_agents(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buys = [a for a in agents if a["vote"] == "BUY"]
    top_reasons = [a["reason"] for a in buys[:3]]
    return {"buy_count": len(buys), "top_buy_reasons": top_reasons}


# ============================================================
# App
# ============================================================
app = FastAPI(title="Polyclawd Executor", version="0.6.0")


@app.get("/healthz")
async def healthz():
    mkts = get_markets()
    return {
        "ok": True,
        "has_kv_url": bool(KV_REST_API_URL),
        "has_kv_token": bool(KV_REST_API_TOKEN),
        "markets_count": len(mkts),
        "history_points": HISTORY_POINTS,
        "public_index_key": PUBLIC_INDEX_KEY,
        "history_key_prefix": PUBLIC_HISTORY_KEY_PREFIX,
    }


# ---- Scenario (admin) ----
@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    s = await get_scenario()
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_post(body: ScenarioBody, _: bool = Depends(require_auth)):
    s = await set_scenario(body.scenario)
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


# ---- Public API (no auth) ----
@app.get("/public/latest")
async def public_latest():
    data = await load_json(PUBLIC_INDEX_KEY)
    if not data:
        return {"ok": True, "empty": True, "message": "No data yet. Run /run once."}
    return {"ok": True, "data": data}


@app.get("/public/latest/{market_id}")
async def public_latest_market(market_id: str = Path(...)):
    data = await load_json(market_key_latest(market_id))
    if not data:
        return {"ok": True, "empty": True, "market_id": market_id}
    return {"ok": True, "market_id": market_id, "data": data}


# ✅ ШАГ 8: удобная история с лимитом и правильным порядком
@app.get("/public/history/{market_id}")
async def public_history_market(
    market_id: str = Path(...),
    limit: int = Query(60, ge=1, le=200, description="How many last items to return (1..200)"),
):
    """
    Возвращает последние N элементов истории (старые -> новые).
    Не требует авторизации.
    """
    data = await load_json(market_key_history(market_id))
    if not data or "items" not in data:
        return {"ok": True, "empty": True, "market_id": market_id, "items": []}

    items = data.get("items", [])
    if not isinstance(items, list):
        return {"ok": True, "empty": True, "market_id": market_id, "items": []}

    # берём последние N и гарантируем порядок старые->новые
    sliced = items[-limit:]

    # иногда бывает дубль одного tick при ручных вызовах внутри 5 минут —
    # это норм, но чтобы UI не "дёргался", можно убрать подряд дубли по (tick, scenario).
    deduped: List[Dict[str, Any]] = []
    last_key = None
    for it in sliced:
        try:
            k = (it.get("tick"), it.get("scenario"))
        except Exception:
            k = None
        if k is not None and k == last_key:
            # оставим всё равно, если хочешь видеть каждое нажатие /run
            # (сейчас НЕ удаляем, просто обновим last_key)
            pass
        deduped.append(it)
        last_key = k

    return {
        "ok": True,
        "market_id": market_id,
        "count": len(deduped),
        "items": deduped,
    }


@app.get("/public/health")
async def public_health():
    idx = await load_json(PUBLIC_INDEX_KEY)
    return {
        "ok": True,
        "has_index": bool(idx),
        "index_ts": idx.get("ts") if idx else None,
        "index_tick": idx.get("tick") if idx else None,
    }


# ---- Executor tick ----
@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked"}

    try:
        scenario = await get_scenario()
        mkts = get_markets()

        ts = now_ts()
        tick = tick_index(ts)

        markets_index: List[Dict[str, Any]] = []

        for market_id in mkts:
            agents = run_fake_agents(scenario, market_id, tick)
            summary = aggregate_decision(agents)

            payload_market = {
                "ts": ts,
                "tick": tick,
                "market_id": market_id,
                "scenario": scenario,
                "summary": summary,
                "agents": agents,  # full agents only in latest
            }
            await save_json(market_key_latest(market_id), payload_market)

            hist_item = {
                "ts": ts,
                "tick": tick,
                "scenario": scenario,
                "summary": summary,
                "digest": digest_agents(agents),
            }
            await append_history(market_id, hist_item)

            markets_index.append({"market_id": market_id, "scenario": scenario, "summary": summary})

        payload_index = {
            "ts": ts,
            "tick": tick,
            "scenario": scenario,
            "markets_count": len(markets_index),
            "markets": markets_index,
        }
        await save_json(PUBLIC_INDEX_KEY, payload_index)

        duration = round(time.time() - started, 3)
        return {
            "ok": True,
            "skipped": False,
            "message": "tick executed",
            "scenario": scenario,
            "saved_markets": len(markets_index),
            "history_points": HISTORY_POINTS,
            "duration_sec": duration,
        }

    finally:
        await unlock(lock_id)
