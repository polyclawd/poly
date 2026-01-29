import os
import time
import uuid
import random
import json
import base64
from typing import Optional, Dict, Any, List

import httpx
from fastapi import Depends, FastAPI, HTTPException, Path
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

# base keys for public data
PUBLIC_INDEX_KEY = os.getenv("PUBLIC_INDEX_KEY", "public:latest:index")
PUBLIC_MARKET_KEY_PREFIX = os.getenv("PUBLIC_MARKET_KEY_PREFIX", "public:latest:market:")

LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))
RUN_TIMEOUT_SECONDS = float(os.getenv("EXECUTOR_RUN_TIMEOUT_SECONDS", "25"))

ALLOWED_SCENARIOS = {"bull", "neutral", "bear"}

# 15 markets (позиции) можно задать ENV: MARKETS="id1,id2,id3,..."
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
    # clean & unique
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # keep order, remove dups
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:15] if len(out) > 15 else out


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
# Fake Agents (per market)
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


def _agent_decision(agent_idx: int, scenario: str, market_id: str, tick: int) -> Dict[str, Any]:
    # детерминированный seed на (agent, scenario, market, tick)
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
# Public storage (base64 JSON)
# ============================================================
def _b64_encode_json(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def _b64_decode_json(b64: str) -> Optional[Dict[str, Any]]:
    try:
        raw = base64.urlsafe_b64decode(b64.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


async def save_json_key(key: str, payload: Dict[str, Any]) -> None:
    b64 = _b64_encode_json(payload)
    ok = await kv_set(key, b64, KV_REST_API_TOKEN)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to persist KV key: {key}")


async def load_json_key(key: str) -> Optional[Dict[str, Any]]:
    b64 = await kv_get(key, KV_REST_API_READ_ONLY_TOKEN)
    if not b64:
        return None
    return _b64_decode_json(b64)


def market_key(market_id: str) -> str:
    return f"{PUBLIC_MARKET_KEY_PREFIX}{market_id}"


def now_ts() -> int:
    return int(time.time())


def tick_index(ts: int) -> int:
    return int(ts / 300)  # 5 minutes


# ============================================================
# App
# ============================================================
app = FastAPI(title="Polyclawd Executor", version="0.4.0")


@app.get("/")
async def root():
    return {"ok": True, "service": "executor", "docs": "/docs", "health": "/healthz"}


@app.get("/healthz")
async def healthz():
    mkts = get_markets()
    return {
        "ok": True,
        "has_executor_secret": bool(EXECUTOR_SECRET),
        "has_kv_url": bool(KV_REST_API_URL),
        "has_kv_token": bool(KV_REST_API_TOKEN),
        "lock_key": LOCK_KEY,
        "scenario_key": SCENARIO_KEY,
        "public_index_key": PUBLIC_INDEX_KEY,
        "public_market_key_prefix": PUBLIC_MARKET_KEY_PREFIX,
        "markets_count": len(mkts),
        "markets_preview": mkts[:3],
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
@app.get("/public/latest")
async def public_latest():
    """
    Отдаёт все 15 карточек одной выдачей (для сайта).
    """
    data = await load_json_key(PUBLIC_INDEX_KEY)
    if not data:
        return {"ok": True, "empty": True, "message": "No data yet. Run /run at least once."}
    return {"ok": True, "data": data}


@app.get("/public/latest/{market_id}")
async def public_latest_market(market_id: str = Path(..., description="Market id from MARKETS list")):
    """
    Отдаёт одну позицию.
    """
    key = market_key(market_id)
    data = await load_json_key(key)
    if not data:
        return {"ok": True, "empty": True, "market_id": market_id, "message": "No data yet for this market."}
    return {"ok": True, "market_id": market_id, "data": data}


@app.get("/public/health")
async def public_health():
    idx = await load_json_key(PUBLIC_INDEX_KEY)
    return {
        "ok": True,
        "has_index": bool(idx),
        "index_ts": idx.get("ts") if idx else None,
        "index_tick": idx.get("tick") if idx else None,
    }


# ---------- Executor tick ----------
@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked", "message": "Another run is in progress"}

    try:
        scenario = await get_scenario()
        mkts = get_markets()

        ts = now_ts()
        tick = tick_index(ts)

        # Build results for 15 markets
        markets_out: List[Dict[str, Any]] = []
        for market_id in mkts:
            agents = run_fake_agents(scenario, market_id, tick)
            summary = aggregate_decision(agents)

            payload_market = {
                "ts": ts,
                "tick": tick,
                "market_id": market_id,
                "scenario": scenario,
                "summary": summary,
                "agents": agents,
            }

            # Save per-market latest
            await save_json_key(market_key(market_id), payload_market)
            markets_out.append(
                {
                    "market_id": market_id,
                    "scenario": scenario,
                    "summary": summary,
                    "key": market_key(market_id),
                }
            )

        # Save index: everything needed for front
        payload_index = {
            "ts": ts,
            "tick": tick,
            "scenario": scenario,
            "markets": markets_out,
            "markets_count": len(markets_out),
        }
        await save_json_key(PUBLIC_INDEX_KEY, payload_index)

        duration = round(time.time() - started, 3)
        return {
            "ok": True,
            "skipped": False,
            "message": "tick executed",
            "duration_sec": duration,
            "scenario": scenario,
            "saved_index_to": PUBLIC_INDEX_KEY,
            "saved_markets": len(markets_out),
            "markets": markets_out[:3],  # preview
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass
