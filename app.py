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
KV_REST_API_URL = os.getenv("KV_REST_API_URL", "").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))

# trading demo settings
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "6"))
MAX_TRADE_FRACTION = float(os.getenv("MAX_TRADE_FRACTION", "0.10"))  # 10% of portfolio value
STARTING_CASH = float(os.getenv("STARTING_CASH", "250"))

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


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id)
    if ok == 1:
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
        "positions": {},  # market_id -> {market_id, qty_usd, opened_ts}
        "last_tick": None,
        "created_ts": int(time.time()),
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


def portfolio_value(p: Dict[str, Any]) -> float:
    # demo: считаем что позиции по номиналу (без price), для MVP
    pos_total = sum(float(v.get("qty_usd", 0)) for v in p.get("positions", {}).values())
    return float(p.get("cash", 0)) + pos_total


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


@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()
    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked", "message": "Another run is in progress"}

    try:
        scenario = await get_scenario()
        presets = await load_presets()
        p = await load_portfolio()
        pv = portfolio_value(p)

        # tick increments (demo): based on unix time / 900 sec (15 min)
        tick = int(time.time() // 900)

        # 1) analyze all markets
        analyses = []
        for market_id in MARKETS:
            analyses.append(run_agents_for_market(market_id, scenario, tick, presets=presets))

        # 2) close positions where SELL
        positions: Dict[str, Any] = p.get("positions", {}) or {}
        cash = float(p.get("cash", 0.0))

        for a in analyses:
            mid = a["market_id"]
            decision = a["summary"]["decision"]
            if mid in positions and decision == "SELL":
                qty = float(positions[mid].get("qty_usd", 0))
                cash += qty
                del positions[mid]

        # 3) open positions where BUY (but max 6)
        pv = cash + sum(float(v.get("qty_usd", 0)) for v in positions.values())
        max_trade = pv * MAX_TRADE_FRACTION

        # candidates BUY sorted by confidence
        buy_candidates: List[Tuple[str, float]] = []
        for a in analyses:
            if a["summary"]["decision"] == "BUY":
                buy_candidates.append((a["market_id"], float(a["summary"]["avg_confidence"])))
        buy_candidates.sort(key=lambda x: x[1], reverse=True)

        for mid, _conf in buy_candidates:
            if mid in positions:
                continue
            if len(positions) >= MAX_OPEN_POSITIONS:
                break
            if cash <= 0:
                break
            qty = min(max_trade, cash)
            if qty <= 0:
                break
            positions[mid] = {"market_id": mid, "qty_usd": round(qty, 2), "opened_ts": int(time.time())}
            cash -= qty

        # 4) save portfolio
        p["cash"] = round(cash, 2)
        p["positions"] = positions
        p["last_tick"] = tick
        await save_portfolio(p)

        # 5) write histories (per market)
        for a in analyses:
            mid = a["market_id"]
            item = {
                "ts": int(time.time()),
                "tick": tick,
                "market_id": mid,
                "scenario": scenario,
                "summary": a["summary"],
                "digest": a["digest"],
                "portfolio": {
                    "value": round(portfolio_value(p), 2),
                    "cash": p["cash"],
                    "open_positions": len(p["positions"]),
                    "positions": list(p["positions"].keys()),
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
            "portfolio": {"cash": p["cash"], "open_positions": len(p["positions"])},
            "duration_sec": duration,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass
