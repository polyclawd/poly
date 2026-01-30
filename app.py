import os
import time
import uuid
import json
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# ------------------------ddd----
# Config
# ----------------------------
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")
KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))

#
#
# trading demo settings
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "6"))
MAX_TRADE_FRACTION = float(os.getenv("MAX_TRADE_FRACTION", "0.10"))  # 10% of portfolio value
EXIT_CONFIDENCE_THRESHOLD = float(os.getenv("EXIT_CONFIDENCE_THRESHOLD", "0.55"))
STARTING_CASH = float(os.getenv("STARTING_CASH", "250"))
# Minimum holding period (in ticks) before a position can be closed
MIN_HOLD_TICKS = int(os.getenv("MIN_HOLD_TICKS", "2"))  # minimum ticks to hold a position before allowing exit
EMERGENCY_EXIT_CONFIDENCE = float(os.getenv("EMERGENCY_EXIT_CONFIDENCE", "0.90"))  # allow early exit if SELL confidence >= this
# New settings for portfolio engine
MAX_POSITION_FRACTION = float(os.getenv("MAX_POSITION_FRACTION", "0.20"))  # max 20% of portfolio value in a single position
FULL_EXIT_CONFIDENCE = float(os.getenv("FULL_EXIT_CONFIDENCE", "0.75"))  # if SELL and avg_confidence >= this -> full close
REDUCE_FRACTION = float(os.getenv("REDUCE_FRACTION", "0.50"))  # fraction of position to sell on non-full exit
PRICE_VOLATILITY = float(os.getenv("PRICE_VOLATILITY", "0.06"))  # 0..1 swing amplitude per tick (demo)

# demo markets
MARKETS = os.getenv("MARKETS", "POLY:DEMO_001,POLY:DEMO_002,POLY:DEMO_003,POLY:DEMO_004,POLY:DEMO_005,POLY:DEMO_006").split(",")

# KV keys
SCENARIO_KEY = os.getenv("SCENARIO_KEY", "executor:scenario")
PORTFOLIO_KEY = os.getenv("PORTFOLIO_KEY", "executor:portfolio")
HISTORY_PREFIX = os.getenv("HISTORY_PREFIX", "executor:history:")
PRESETS_KEY = os.getenv("PRESETS_KEY", "executor:presets")
AGENTS_PRESET_JSON = os.getenv("AGENTS_PRESET_JSON", "")


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
def _upstash_headers() -> dict:
    return {
        "Authorization": f"Bearer {KV_REST_API_TOKEN}",
        "Accept": "application/json",
    }


async def upstash_get(key: str) -> Optional[str]:
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN is empty")

    url = f"{KV_REST_API_URL}/get/{quote(key)}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=_upstash_headers())

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()
    data = r.json()
    return data.get("result")


async def upstash_set(key: str, value: str) -> None:
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN is empty")

    # IMPORTANT: value must be url-encoded, иначе будут 400
    url = f"{KV_REST_API_URL}/set/{quote(key)}/{quote(value)}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers())

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()


async def upstash_del(key: str) -> int:
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        return 0

    url = f"{KV_REST_API_URL}/del/{quote(key)}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers())

    if r.status_code == 401:
        return 0

    r.raise_for_status()
    data = r.json()
    return int(data.get("result", 0))


async def upstash_setnx(key: str, value: str) -> int:
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN is empty")

    url = f"{KV_REST_API_URL}/setnx/{quote(key)}/{quote(value)}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers())

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()
    data = r.json()
    return int(data.get("result", 0))


async def upstash_expire(key: str, ttl_seconds: int) -> int:
    """Upstash Redis REST: EXPIRE key ttl_seconds"""
    if not KV_REST_API_URL or not KV_REST_API_TOKEN:
        return 0

    url = f"{KV_REST_API_URL}/expire/{quote(key)}/{int(ttl_seconds)}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers())

    if r.status_code == 401:
        return 0

    r.raise_for_status()
    data = r.json()
    return int(data.get("result", 0))


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id)
    if ok == 1:
        # best-effort TTL so a crashed worker doesn't block forever
        try:
            await upstash_expire(LOCK_KEY, LOCK_TTL_SECONDS)
        except Exception:
            pass
        return lock_id
    return None


async def unlock(_: str) -> None:
    await upstash_del(LOCK_KEY)


# ----------------------------
# Scenario
# ----------------------------
VALID_SCENARIOS = {"bull", "bear", "neutral"}


async def get_scenario() -> str:
    s = await upstash_get(SCENARIO_KEY)
    if not s:
        return "neutral"
    s = str(s).strip().lower()
    if s not in VALID_SCENARIOS:
        return "neutral"
    return s


async def set_scenario(s: str) -> str:
    s = (s or "").strip().lower()
    if s not in VALID_SCENARIOS:
        raise HTTPException(status_code=422, detail=f"Invalid scenario. Use one of: {sorted(VALID_SCENARIOS)}")
    await upstash_set(SCENARIO_KEY, s)
    return s


# ----------------------------
# Portfolio state
# ----------------------------
def _portfolio_default() -> Dict[str, Any]:
    return {
        "cash": STARTING_CASH,
        "positions": {},  # market_id -> position
        "last_tick": None,
        "created_ts": int(time.time()),
        "realized_pnl": 0.0,
        "equity_curve": [],  # list of {tick, value}
    }


async def load_portfolio() -> Dict[str, Any]:
    raw = await upstash_get(PORTFOLIO_KEY)
    if not raw:
        p = _portfolio_default()
        await upstash_set(PORTFOLIO_KEY, json.dumps(p))
        return p
    try:
        return json.loads(raw)
    except Exception:
        p = _portfolio_default()
        await upstash_set(PORTFOLIO_KEY, json.dumps(p))
        return p


async def save_portfolio(p: Dict[str, Any]) -> None:
    await upstash_set(PORTFOLIO_KEY, json.dumps(p))



def _clamp(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, x))


def market_price(market_id: str, scenario: str, tick: int) -> float:
    """Deterministic demo 'price/probability' in [0.01, 0.99].

    This replaces real Polymarket prices for now. It is stable, repeatable, and scenario-tilted.
    """
    base_seed = _seed_int(f"BASE|{market_id}")
    # base around 0.30..0.70 depending on market
    base = 0.30 + (base_seed % 400) / 1000.0

    t_seed = _seed_int(f"PRICE|{market_id}|{tick}")
    # deterministic pseudo-noise in [-1, +1]
    noise = ((t_seed % 2000) - 1000) / 1000.0
    drift = 0.0
    if scenario == "bull":
        drift = 0.015
    elif scenario == "bear":
        drift = -0.015

    px = base + drift + noise * PRICE_VOLATILITY
    return round(_clamp(px), 4)


def portfolio_value(p: Dict[str, Any]) -> float:
    cash = float(p.get("cash", 0) or 0)
    pos_total = 0.0
    for pos in (p.get("positions") or {}).values():
        try:
            shares = float(pos.get("shares", 0) or 0)
            last_price = float(pos.get("last_price", pos.get("entry_price", 0)) or 0)
            pos_total += shares * last_price
        except Exception:
            continue
    return round(cash + pos_total, 6)


# ----------------------------
# Presets (optional)
# Structure expected (JSON):
# {
#   "bull": {
#     "POLY:DEMO_001": [ {"agents": [...]} , {"agents": [...]} , ... ],
#     "POLY:DEMO_002": [ ... ]
#   },
#   "bear": { ... },
#   "neutral": { ... }
# }
# Each entry in the list is a "frame" that can be rotated by tick.
# If a scenario/market is missing, we fall back to deterministic fake agents.
# ----------------------------
AGENT_COUNT = 15

def _seed_int(s: str) -> int:
    # простой детерминированный seed
    x = 0
    for ch in s:
        x = (x * 131 + ord(ch)) % 1_000_000_007
    return x

def _agent_vote(seed: int) -> Tuple[str, float, str]:
    # vote, confidence, reason
    # deterministic-ish
    r = seed % 1000 / 1000.0
    if r > 0.62:
        return "BUY", min(0.95, 0.55 + r * 0.5), "Liquidity acceptable; expected value positive this tick."
    if r < 0.38:
        return "SELL", min(0.95, 0.55 + (1 - r) * 0.5), "Risk-off: expected value negative; reduce exposure."
    return "HOLD", 0.50 + abs(r - 0.5), "Signals conflict; avoid overtrading."


# ----------------------------
# Presets (optional)
# Structure expected (JSON):
# {
#   "bull": {
#     "POLY:DEMO_001": [ {"agents": [...]} , {"agents": [...]} , ... ],
#     "POLY:DEMO_002": [ ... ]
#   },
#   "bear": { ... },
#   "neutral": { ... }
# }
# Each entry in the list is a "frame" that can be rotated by tick.
# If a scenario/market is missing, we fall back to deterministic fake agents.
# ----------------------------

def _safe_json_loads(s: Optional[str]) -> Optional[Any]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


async def load_presets() -> Optional[Dict[str, Any]]:
    """Load presets from Upstash first; if empty, fall back to env var."""
    # 1) Upstash
    try:
        raw = await upstash_get(PRESETS_KEY)
    except Exception:
        raw = None
    presets = _safe_json_loads(raw)
    if isinstance(presets, dict) and presets:
        return presets

    # 2) Env var fallback
    presets = _safe_json_loads(AGENTS_PRESET_JSON)
    if isinstance(presets, dict) and presets:
        return presets

    return None


def _summarize_from_agents(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy = sum(1 for a in agents if str(a.get("vote", "")).upper() == "BUY")
    sell = sum(1 for a in agents if str(a.get("vote", "")).upper() == "SELL")
    hold = sum(1 for a in agents if str(a.get("vote", "")).upper() == "HOLD")

    confs = []
    for a in agents:
        try:
            confs.append(float(a.get("confidence", 0)))
        except Exception:
            pass
    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0

    decision = "HOLD"
    if buy >= 9:
        decision = "BUY"
    if sell >= 9:
        decision = "SELL"

    return {
        "decision": decision,
        "buy_count": buy,
        "hold_count": hold,
        "sell_count": sell,
        "avg_confidence": avg_conf,
        "rule": "BUY if buy_count >= 9; SELL if sell_count >= 9; else HOLD",
    }


def _digest_from_agents(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy_reasons = [a.get("reason") for a in agents if str(a.get("vote", "")).upper() == "BUY" and a.get("reason")]
    return {
        "buy_count": sum(1 for a in agents if str(a.get("vote", "")).upper() == "BUY"),
        "top_buy_reasons": buy_reasons[:3],
    }


def _normalize_agents(agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure each agent has required fields; keep any extra fields (signals, etc.)."""
    out = []
    for i, a in enumerate(agents, start=1):
        if not isinstance(a, dict):
            continue
        agent_id = a.get("agent_id") or f"agent_{i:02d}"
        name = a.get("name") or f"Agent {i:02d}"
        vote = str(a.get("vote", "HOLD")).upper()
        if vote not in {"BUY", "HOLD", "SELL"}:
            vote = "HOLD"
        try:
            confidence = float(a.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        reason = a.get("reason") or ""

        # Preserve any extra keys (e.g., signals)
        merged = dict(a)
        merged.update({
            "agent_id": agent_id,
            "name": name,
            "vote": vote,
            "confidence": round(confidence, 3),
            "reason": reason,
        })
        out.append(merged)

    # If fewer than AGENT_COUNT, pad deterministically
    if len(out) < AGENT_COUNT:
        for j in range(len(out) + 1, AGENT_COUNT + 1):
            out.append({
                "agent_id": f"agent_{j:02d}",
                "name": f"Agent {j:02d}",
                "vote": "HOLD",
                "confidence": 0.5,
                "reason": "(preset missing)"
            })

    # If more than AGENT_COUNT, trim
    return out[:AGENT_COUNT]


def run_agents_for_market(market_id: str, scenario: str, tick: int, presets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # 0) If presets available, try to use them
    if isinstance(presets, dict):
        scen = presets.get(scenario) or presets.get("default") or {}
        if isinstance(scen, dict):
            frames = scen.get(market_id)
            if isinstance(frames, list) and frames:
                frame = frames[tick % len(frames)]
                if isinstance(frame, dict) and isinstance(frame.get("agents"), list):
                    agents = _normalize_agents(frame["agents"])
                    summary = _summarize_from_agents(agents)
                    digest = _digest_from_agents(agents)
                    return {
                        "market_id": market_id,
                        "scenario": scenario,
                        "summary": summary,
                        "agents": agents,
                        "digest": digest,
                    }

    # 1) Fallback: deterministic fake agents (previous behavior)
    agents = []
    buy_count = hold_count = sell_count = 0
    confs = []

    for i in range(1, AGENT_COUNT + 1):
        seed = _seed_int(f"{scenario}|{market_id}|{tick}|agent_{i:02d}")
        vote, conf, reason = _agent_vote(seed)

        # scenario tilt
        if scenario == "bull" and vote == "HOLD" and (seed % 10) >= 7:
            vote = "BUY"
            reason = "Edge detected; scenario tailwind supports entry."
        if scenario == "bear" and vote == "HOLD" and (seed % 10) >= 7:
            vote = "SELL"
            reason = "Downside risk; scenario headwind supports reducing risk."

        if vote == "BUY":
            buy_count += 1
        elif vote == "SELL":
            sell_count += 1
        else:
            hold_count += 1

        confs.append(conf)
        agents.append({
            "agent_id": f"agent_{i:02d}",
            "name": f"Agent {i:02d}",
            "vote": vote,
            "confidence": round(conf, 3),
            "reason": reason,
        })

    avg_conf = round(sum(confs) / len(confs), 3)

    decision = "HOLD"
    if buy_count >= 9:
        decision = "BUY"
    if sell_count >= 9:
        decision = "SELL"

    return {
        "market_id": market_id,
        "scenario": scenario,
        "summary": {
            "decision": decision,
            "buy_count": buy_count,
            "hold_count": hold_count,
            "sell_count": sell_count,
            "avg_confidence": avg_conf,
            "rule": "BUY if buy_count >= 9; SELL if sell_count >= 9; else HOLD",
        },
        "agents": agents,
        "digest": {
            "buy_count": buy_count,
            "top_buy_reasons": [a["reason"] for a in agents if a["vote"] == "BUY"][:3],
        },
    }


# ----------------------------
# History per market
# ----------------------------
async def append_history(market_id: str, item: Dict[str, Any], limit: int = 48) -> None:
    # храним историю как JSON array в одном ключе, простой MVP
    key = f"{HISTORY_PREFIX}{market_id}"
    raw = await upstash_get(key)
    items = []
    if raw:
        try:
            items = json.loads(raw) or []
        except Exception:
            items = []
    items.append(item)
    if len(items) > limit:
        items = items[-limit:]
    await upstash_set(key, json.dumps(items))


async def get_latest(market_id: str) -> Dict[str, Any]:
    key = f"{HISTORY_PREFIX}{market_id}"
    raw = await upstash_get(key)
    if not raw:
        return {"ok": False, "market_id": market_id, "data": None}
    try:
        items = json.loads(raw) or []
    except Exception:
        items = []
    if not items:
        return {"ok": False, "market_id": market_id, "data": None}
    return {"ok": True, "market_id": market_id, "data": items[-1]}


async def get_history(market_id: str) -> Dict[str, Any]:
    key = f"{HISTORY_PREFIX}{market_id}"
    raw = await upstash_get(key)
    if not raw:
        return {"ok": True, "market_id": market_id, "items": []}
    try:
        items = json.loads(raw) or []
    except Exception:
        items = []
    return {"ok": True, "market_id": market_id, "items": items}


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
        "markets": MARKETS,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "max_trade_fraction": MAX_TRADE_FRACTION,
        "min_hold_ticks": MIN_HOLD_TICKS,
        "emergency_exit_confidence": EMERGENCY_EXIT_CONFIDENCE,
        "max_position_fraction": MAX_POSITION_FRACTION,
        "full_exit_confidence": FULL_EXIT_CONFIDENCE,
        "reduce_fraction": REDUCE_FRACTION,
        "price_volatility": PRICE_VOLATILITY,
    }


@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    s = await get_scenario()
    return {"ok": True, "scenario": s, "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_post(body: Dict[str, Any], _: bool = Depends(require_auth)):
    s = body.get("scenario")
    saved = await set_scenario(s)
    return {"ok": True, "scenario": saved}


@app.get("/public/latest/{market_id}")
async def public_latest(market_id: str):
    return await get_latest(market_id)


@app.get("/public/history/{market_id}")
async def public_history(market_id: str):
    return await get_history(market_id)


@app.get("/public/portfolio")
async def public_portfolio():
    p = await load_portfolio()
    return {"ok": True, "portfolio": p, "value": portfolio_value(p)}


# ---- Inserted endpoints ----

@app.get("/public/markets")
async def public_markets():
    """Small helper for the frontend so it doesn't hardcode markets."""
    return {"ok": True, "markets": MARKETS}


@app.get("/public/snapshot")
async def public_snapshot():
    """One-call API for the frontend: scenario + portfolio + latest per market."""
    scenario = await get_scenario()
    p = await load_portfolio()

    latest_by_market: Dict[str, Any] = {}
    for m in MARKETS:
        latest_by_market[m] = await get_latest(m)

    scenario_prices: Dict[str, float] = {m: market_price(m, scenario, int(time.time() // 900)) for m in MARKETS}

    return {
        "ok": True,
        "scenario": scenario,
        "markets": MARKETS,
        "portfolio": p,
        "value": portfolio_value(p),
        "latest": latest_by_market,
        "prices": scenario_prices,
        "realized_pnl": p.get("realized_pnl", 0),
        "equity_curve": p.get("equity_curve", []),
        "ts": int(time.time()),
        "ts_ms": int(time.time() * 1000),
    }


@app.post("/admin/reset")
async def admin_reset(_: bool = Depends(require_auth)):
    deleted = []
    deleted.append(await upstash_del(PORTFOLIO_KEY))
    deleted.append(await upstash_del(SCENARIO_KEY))
    deleted.append(await upstash_del(PRESETS_KEY))
    for m in MARKETS:
        deleted.append(await upstash_del(f"{HISTORY_PREFIX}{m}"))
    deleted.append(await upstash_del(LOCK_KEY))
    return {"ok": True, "deleted": deleted}



@app.get("/admin/presets")
async def admin_presets_get(_: bool = Depends(require_auth)):
    raw = await upstash_get(PRESETS_KEY)
    presets = _safe_json_loads(raw)
    return {"ok": True, "key": PRESETS_KEY, "has_presets": bool(presets), "presets": presets}


@app.post("/admin/presets")
async def admin_presets_post(body: Dict[str, Any], _: bool = Depends(require_auth)):
    # Accept either {"presets": {...}} or the presets object directly
    presets_obj = body.get("presets") if isinstance(body, dict) else None
    if presets_obj is None:
        presets_obj = body

    if not isinstance(presets_obj, dict):
        raise HTTPException(status_code=422, detail="Invalid presets payload: expected JSON object")

    await upstash_set(PRESETS_KEY, json.dumps(presets_obj))
    return {"ok": True, "key": PRESETS_KEY}



async def _execute_run(force: bool = False) -> Dict[str, Any]:
    """Shared run implementation used by authenticated /run and token-based cron endpoint."""
    started = time.time()
    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked", "message": "Another run is in progress"}

    try:
        scenario = await get_scenario()
        presets = await load_presets()
        p = await load_portfolio()

        # tick increments (demo): based on unix time / 900 sec (15 min)
        tick = int(time.time() // 900)
        run_id = str(uuid.uuid4())

        # Prevent accidental multiple executions within the same 15-min tick
        # (use ?force=true to override for manual testing)
        if not force and p.get("last_tick") == tick:
            duration = round(time.time() - started, 3)
            return {
                "ok": True,
                "skipped": True,
                "reason": "same_tick",
                "message": "Already executed this tick (use force=true to override)",
                "tick": tick,
                "scenario": scenario,
                "ts_ms": int(time.time() * 1000),
                "portfolio": {"cash": p.get("cash"), "open_positions": len(p.get("positions", {}) or {})},
                "duration_sec": duration,
            }

        # 1) analyze all markets
        analyses = []
        for market_id in MARKETS:
            analyses.append(run_agents_for_market(market_id, scenario, tick, presets=presets))

        # deterministic demo prices for this tick
        prices: Dict[str, float] = {m: market_price(m, scenario, tick) for m in MARKETS}

        # 2) close positions where SELL or confidence drops
        positions: Dict[str, Any] = p.get("positions", {}) or {}
        positions_before: Dict[str, Any] = dict(positions)
        cash = float(p.get("cash", 0.0))

        closed_positions: List[str] = []
        opened_positions: List[str] = []
        open_skip_reason: Dict[str, str] = {}
        reduced_positions: List[str] = []

        # map market -> analysis for quick access
        analysis_by_market = {a["market_id"]: a for a in analyses}

        for mid in list(positions.keys()):
            a = analysis_by_market.get(mid)
            if not a:
                continue

            decision = str(a["summary"].get("decision", "HOLD")).upper()
            avg_conf = float(a["summary"].get("avg_confidence", 0))

            pos = positions.get(mid) or {}
            # Backward compat: migrate old positions (qty_usd -> shares/cost_basis)
            last_price = float(prices.get(mid) or 0)
            if "shares" not in pos or "cost_basis_usd" not in pos or "entry_price" not in pos:
                qty_usd = float(pos.get("qty_usd", 0) or 0)
                entry_price = float(pos.get("entry_price") or last_price or 0.5)
                entry_price = _clamp(entry_price)
                shares = qty_usd / entry_price if entry_price > 0 else 0.0
                pos["shares"] = round(shares, 8)
                pos["cost_basis_usd"] = round(qty_usd, 6)
                pos["entry_price"] = round(entry_price, 6)
                pos["last_price"] = round(last_price, 6)
                positions[mid] = pos

            # update mark price
            pos["last_price"] = round(last_price, 6)

            opened_tick = pos.get("opened_tick")
            if opened_tick is None:
                try:
                    opened_ts = int(pos.get("opened_ts") or 0)
                except Exception:
                    opened_ts = 0
                opened_tick = int(opened_ts // 900) if opened_ts else tick
                pos["opened_tick"] = opened_tick

            age_ticks = max(0, int(tick) - int(opened_tick))
            too_young_to_exit = age_ticks < MIN_HOLD_TICKS
            emergency_exit = (decision == "SELL") and (avg_conf >= EMERGENCY_EXIT_CONFIDENCE)

            wants_exit = (decision == "SELL") or (avg_conf <= EXIT_CONFIDENCE_THRESHOLD)
            if not wants_exit:
                continue

            if too_young_to_exit and not emergency_exit:
                # blocked by min-hold (surfaced in execution)
                continue

            # determine full close vs reduce
            full_close = bool((decision == "SELL" and avg_conf >= FULL_EXIT_CONFIDENCE) or (avg_conf <= EXIT_CONFIDENCE_THRESHOLD))
            if full_close:
                sell_fraction = 1.0
            else:
                sell_fraction = float(REDUCE_FRACTION)
                sell_fraction = max(0.0, min(1.0, sell_fraction))

            shares = float(pos.get("shares", 0) or 0)
            if shares <= 0:
                # invalid position, remove
                del positions[mid]
                closed_positions.append(mid)
                continue

            shares_to_sell = shares * sell_fraction
            shares_to_sell = min(shares, shares_to_sell)

            # proceeds and cost basis (proportional)
            proceeds = shares_to_sell * last_price
            cost_basis = float(pos.get("cost_basis_usd", 0) or 0)
            cost_sold = cost_basis * (shares_to_sell / shares) if shares > 0 else 0.0
            pnl = proceeds - cost_sold

            cash += proceeds
            p["realized_pnl"] = round(float(p.get("realized_pnl", 0) or 0) + pnl, 6)

            # update remaining position
            remaining_shares = shares - shares_to_sell
            remaining_cost = cost_basis - cost_sold
            if remaining_shares <= 1e-9:
                del positions[mid]
                closed_positions.append(mid)
            else:
                pos["shares"] = round(remaining_shares, 8)
                pos["cost_basis_usd"] = round(remaining_cost, 6)
                pos["entry_price"] = round((remaining_cost / remaining_shares) if remaining_shares > 0 else pos.get("entry_price", 0.5), 6)
                positions[mid] = pos
                reduced_positions.append(mid)

        # 3) open/add positions where BUY, respecting caps
        pv = portfolio_value({"cash": cash, "positions": positions})
        max_trade = pv * MAX_TRADE_FRACTION
        max_pos_value = pv * MAX_POSITION_FRACTION

        # candidates BUY sorted by confidence
        buy_candidates: List[Tuple[str, float]] = []
        for a in analyses:
            if a["summary"]["decision"] == "BUY":
                buy_candidates.append((a["market_id"], float(a["summary"]["avg_confidence"])))
        buy_candidates.sort(key=lambda x: x[1], reverse=True)

        for mid, _conf in buy_candidates:
            px = float(prices.get(mid) or 0.5)
            px = _clamp(px)

            # compute current position value if exists
            if mid in positions:
                pos = positions.get(mid) or {}
                shares = float(pos.get("shares", 0) or 0)
                cur_value = shares * float(pos.get("last_price", px) or px)
                # ADD (scale-in) only if under per-position cap
                room = max(0.0, max_pos_value - cur_value)
                if room <= 1e-6:
                    open_skip_reason[mid] = f"max position cap reached ({MAX_POSITION_FRACTION:.2f})"
                    continue
                if cash <= 0:
                    open_skip_reason[mid] = "insufficient cash"
                    continue
                qty = min(max_trade, cash, room)
                if qty <= 0:
                    open_skip_reason[mid] = "trade size <= 0"
                    continue

                add_shares = qty / px
                pos["shares"] = round(float(pos.get("shares", 0) or 0) + add_shares, 8)
                pos["cost_basis_usd"] = round(float(pos.get("cost_basis_usd", 0) or 0) + qty, 6)
                pos["entry_price"] = round(float(pos["cost_basis_usd"]) / float(pos["shares"]) if float(pos["shares"]) > 0 else px, 6)
                pos["last_price"] = round(px, 6)
                positions[mid] = pos

                cash -= qty
                opened_positions.append(mid)  # reuse for UI: treated as "OPEN/ADD" bucket
                continue

            # OPEN new position
            if len(positions) >= MAX_OPEN_POSITIONS:
                open_skip_reason[mid] = f"max open positions reached ({MAX_OPEN_POSITIONS})"
                break
            if cash <= 0:
                open_skip_reason[mid] = "insufficient cash"
                break

            qty = min(max_trade, cash, max_pos_value)
            if qty <= 0:
                open_skip_reason[mid] = "trade size <= 0"
                break

            shares = qty / px
            positions[mid] = {
                "market_id": mid,
                "shares": round(shares, 8),
                "cost_basis_usd": round(qty, 6),
                "entry_price": round(px, 6),
                "last_price": round(px, 6),
                "opened_ts": int(time.time()),
                "opened_tick": tick,
            }
            cash -= qty
            opened_positions.append(mid)

        # 4) save portfolio
        p["cash"] = round(cash, 2)
        p["positions"] = positions
        p["last_tick"] = tick
        await save_portfolio(p)

        # append equity curve point
        eq = float(portfolio_value(p))
        curve = list(p.get("equity_curve") or [])
        curve.append({"tick": tick, "value": round(eq, 6)})
        if len(curve) > 96:
            curve = curve[-96:]
        p["equity_curve"] = curve
        await save_portfolio(p)

        positions_after: Dict[str, Any] = p.get("positions", {}) or {}

        def _execution_for_market(mid: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
            decision = str(analysis.get("summary", {}).get("decision", "HOLD")).upper()
            avg_conf = float(analysis.get("summary", {}).get("avg_confidence", 0))

            was_in_position = mid in positions_before
            is_in_position = mid in positions_after

            age_ticks = None
            age_minutes = None
            exit_blocked = False

            if was_in_position:
                pos = positions_before.get(mid) or {}
                opened_tick = pos.get("opened_tick")
                if opened_tick is None:
                    try:
                        opened_ts = int(pos.get("opened_ts") or 0)
                    except Exception:
                        opened_ts = 0
                    opened_tick = int(opened_ts // 900) if opened_ts else tick

                age_ticks = max(0, int(tick) - int(opened_tick))
                age_minutes = int(age_ticks * 15)

                # If we still have the position but the strategy wanted to exit, it may have been blocked
                wants_exit = (decision == "SELL") or (avg_conf <= EXIT_CONFIDENCE_THRESHOLD)
                too_young_to_exit = age_ticks < MIN_HOLD_TICKS
                emergency_exit = (decision == "SELL") and (avg_conf >= EMERGENCY_EXIT_CONFIDENCE)
                exit_blocked = bool(wants_exit and too_young_to_exit and not emergency_exit and is_in_position)

            # INTENT: what the strategy wanted to do (separate from what actually happened)
            if was_in_position:
                if decision == "SELL" or avg_conf <= EXIT_CONFIDENCE_THRESHOLD:
                    intent = "EXIT"
                elif decision == "BUY":
                    intent = "ADD"
                else:
                    intent = "HOLD"
            else:
                if decision == "BUY":
                    intent = "OPEN"
                elif decision == "SELL":
                    # can't sell if we have no position; surface this clearly to UI
                    intent = "IGNORE"
                else:
                    intent = "HOLD"

            # ACTION: what actually happened
            action = "NONE"
            reason = ""

            if mid in closed_positions:
                action = "CLOSE"
                reason = "closed due to SELL" if decision == "SELL" else "closed due to low confidence"
            elif mid in reduced_positions:
                action = "REDUCE"
                reason = "reduced position (partial exit)"
            elif mid in opened_positions:
                action = "OPEN" if not was_in_position else "ADD"
                reason = "opened due to BUY" if not was_in_position else "added to position due to BUY"
            else:
                # No portfolio change on this market
                if exit_blocked:
                    action = "NONE"
                    reason = f"exit blocked: position too young (min_hold_ticks={MIN_HOLD_TICKS})"
                elif intent == "OPEN" and not is_in_position:
                    action = "NONE"
                    reason = open_skip_reason.get(mid, "not opened")
                elif intent == "EXIT" and was_in_position and is_in_position:
                    action = "NONE"
                    reason = "exit conditions not met"

                # Ensure UI always gets a meaningful explanation when nothing happens
                if action == "NONE" and not reason:
                    if decision == "SELL" and not was_in_position:
                        reason = "not in position"
                    elif intent == "HOLD":
                        reason = "no action"
                    elif intent == "OPEN":
                        reason = open_skip_reason.get(mid, "not opened")
                    else:
                        reason = "no action"

            px = float(prices.get(mid) or 0)
            pos_snap = None
            if is_in_position:
                pos = positions_after.get(mid) or {}
                shares = float(pos.get("shares", 0) or 0)
                entry = float(pos.get("entry_price", 0) or 0)
                lastp = float(pos.get("last_price", px) or px)
                value = round(shares * lastp, 6)
                cost = float(pos.get("cost_basis_usd", 0) or 0)
                unreal = round(value - cost, 6)
                pos_snap = {
                    "shares": round(shares, 8),
                    "entry_price": round(entry, 6),
                    "last_price": round(lastp, 6),
                    "value": value,
                    "cost_basis_usd": round(cost, 6),
                    "unrealized_pnl": unreal,
                }

            return {
                "in_position": is_in_position,
                "was_in_position": was_in_position,
                "intent": intent,
                "action": action,
                "reason": reason,
                "decision": decision,
                "avg_confidence": avg_conf,
                "age_ticks": age_ticks,
                "age_minutes": age_minutes,
                "exit_blocked": exit_blocked,
                "price": px,
                "position": pos_snap,
            }

        # 5) write histories (per market)
        for a in analyses:
            mid = a["market_id"]
            item = {
                "ts": int(time.time()),
                "ts_ms": int(time.time() * 1000),
                "run_id": run_id,
                "tick": tick,
                "market_id": mid,
                "scenario": scenario,
                "summary": a["summary"],
                "digest": a["digest"],
                "execution": _execution_for_market(mid, a),
                "price": prices.get(mid),
                "portfolio": {
                    "value": round(portfolio_value(p), 2),
                    "cash": p["cash"],
                    "open_positions": len(p["positions"]),
                    "positions": list(p["positions"].keys()),
                    "realized_pnl": p.get("realized_pnl", 0),
                    "equity": portfolio_value(p),
                },
                "agents": a["agents"],
            }
            await append_history(mid, item)

        duration = round(time.time() - started, 3)
        return {
            "ok": True,
            "skipped": False,
            "message": "tick executed",
            "tick": tick,
            "scenario": scenario,
            "run_id": run_id,
            "ts_ms": int(time.time() * 1000),
            "portfolio": {"cash": p["cash"], "open_positions": len(p["positions"])},
            "execution_summary": {
                "opened": opened_positions,
                "closed": closed_positions,
                "open_skipped": open_skip_reason,
            },
            "duration_sec": duration,
        }

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass


@app.post("/run")
async def run(force: bool = False, _: bool = Depends(require_auth)):
    try:
        return await _execute_run(force=force)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e


# Cron-friendly endpoint for Render/other schedulers
@app.get("/cron/run")
async def cron_run(token: str, force: bool = False):
    """Cron endpoint for Render/other schedulers that may not support Swagger auth.

    Call: GET /cron/run?token=<EXECUTOR_SECRET>
    """
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured: EXECUTOR_SECRET is empty")
    if token != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        return await _execute_run(force=force)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e
