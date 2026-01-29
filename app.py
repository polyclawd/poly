import os
import time
import json
import uuid
import hashlib
from typing import Optional, Any, Dict, List

import httpx
from fastapi import Depends, FastAPI, HTTPException, Body
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# ----------------------------
# Config
# ----------------------------
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", "")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")

# keys
SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")
DEFAULT_SCENARIO = os.getenv("EXECUTOR_DEFAULT_SCENARIO", "neutral")

# public KV namespaces
PUBLIC_LATEST_PREFIX = os.getenv("EXECUTOR_PUBLIC_LATEST_PREFIX", "public:latest:")
PUBLIC_HISTORY_PREFIX = os.getenv("EXECUTOR_PUBLIC_HISTORY_PREFIX", "public:history:")

# behavior
TICK_SECONDS = int(os.getenv("EXECUTOR_TICK_SECONDS", "300"))  # 5 minutes
HISTORY_MAX = int(os.getenv("EXECUTOR_HISTORY_MAX", "50"))     # keep last N
HTTP_TIMEOUT = float(os.getenv("EXECUTOR_HTTP_TIMEOUT", "12"))

PRESET_PATH = os.getenv("EXECUTOR_PRESET_PATH", "agents_preset.json")


# ----------------------------
# Auth (Swagger "Authorize")
# ----------------------------
bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured: EXECUTOR_SECRET is empty")

    if creds is None:
        raise HTTPException(status_code=403, detail="Not authenticated")

    token = creds.credentials  # part after "Bearer "
    if token != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return True


# ----------------------------
# Upstash REST helpers
# ----------------------------
def _upstash_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


async def _upstash_post(path: str, token: str) -> dict:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN missing")

    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")
    r.raise_for_status()
    return r.json()


async def upstash_get(key: str, token: str) -> Optional[str]:
    data = await _upstash_post(f"get/{key}", token)
    # {"result": null} or {"result": "..."}
    res = data.get("result", None)
    return res


async def upstash_set(key: str, value: str, token: str) -> bool:
    data = await _upstash_post(f"set/{key}/{value}", token)
    # {"result":"OK"}
    return str(data.get("result", "")).upper() == "OK"


async def upstash_setnx(key: str, value: str, token: str) -> int:
    data = await _upstash_post(f"setnx/{key}/{value}", token)
    return int(data.get("result", 0))


async def upstash_del(key: str, token: str) -> int:
    data = await _upstash_post(f"del/{key}", token)
    return int(data.get("result", 0))


async def upstash_lpush(key: str, value: str, token: str) -> int:
    data = await _upstash_post(f"lpush/{key}/{value}", token)
    return int(data.get("result", 0))


async def upstash_ltrim(key: str, start: int, stop: int, token: str) -> str:
    data = await _upstash_post(f"ltrim/{key}/{start}/{stop}", token)
    return str(data.get("result", ""))


async def upstash_lrange(key: str, start: int, stop: int, token: str) -> List[str]:
    data = await _upstash_post(f"lrange/{key}/{start}/{stop}", token)
    res = data.get("result", [])
    if res is None:
        return []
    return list(res)


# ----------------------------
# Lock
# ----------------------------
async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id, KV_REST_API_TOKEN)
    return lock_id if ok == 1 else None


async def unlock(_: str) -> None:
    # simple unlock
    await upstash_del(LOCK_KEY, KV_REST_API_TOKEN)


# ----------------------------
# Presets (15 agents)
# ----------------------------
def load_preset() -> Dict[str, Any]:
    try:
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Preset file not found: {PRESET_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read preset file: {type(e).__name__}: {e}")


def stable_rand01(seed: str) -> float:
    # deterministic 0..1
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    # take first 12 hex chars -> int
    n = int(h[:12], 16)
    return (n % 10_000_000) / 10_000_000.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def scenario_bias(scenario: str) -> float:
    s = (scenario or "").lower()
    if s == "bull":
        return +0.12
    if s == "bear":
        return -0.12
    if s == "volatile":
        return 0.0
    return 0.0  # neutral/default


def pick_vote(score: float, scenario: str) -> str:
    # score in 0..1 around 0.5
    # volatile: more HOLDs
    s = (scenario or "").lower()
    if s == "volatile":
        if score >= 0.64:
            return "BUY"
        if score <= 0.36:
            return "SELL"
        return "HOLD"
    else:
        if score >= 0.58:
            return "BUY"
        if score <= 0.42:
            return "SELL"
        return "HOLD"


def reason_templates(vote: str, scenario: str) -> str:
    s = (scenario or "neutral").lower()
    if vote == "BUY":
        if s == "bull":
            return "Edge detected; scenario tailwind supports entry."
        if s == "bear":
            return "Contrarian value detected; risk acceptable for small entry."
        if s == "volatile":
            return "Volatility priced in; favorable risk/reward on this tick."
        return "Liquidity acceptable; expected value positive this tick."
    if vote == "SELL":
        if s == "bear":
            return "Downside skew; scenario headwind suggests exit."
        if s == "bull":
            return "Overheated pricing; trimming risk into strength."
        if s == "volatile":
            return "Risk-off in turbulence; preserving capital."
        return "Signals weakened; expected value negative on this tick."
    # HOLD
    if s == "volatile":
        return "Signals conflict; avoid overtrading in volatility."
    return "No clear edge; waiting for better setup."


def make_signals(seed: str) -> Dict[str, float]:
    # generate stable-ish signals
    r1 = stable_rand01(seed + ":p")
    r2 = stable_rand01(seed + ":ip")
    r3 = stable_rand01(seed + ":s")
    r4 = stable_rand01(seed + ":v")

    price = round(0.25 + 0.75 * r1, 3)
    implied_prob = round(0.25 + 0.75 * r2, 3)
    sentiment = round(-1.0 + 2.0 * r3, 3)      # -1..+1
    volatility = round(0.10 + 0.90 * r4, 3)    # 0.1..1.0

    return {
        "price": price,
        "implied_prob": implied_prob,
        "sentiment": sentiment,
        "volatility": volatility,
    }


def agent_decision(agent_id: str, agent_name: str, tick: int, scenario: str) -> Dict[str, Any]:
    seed = f"{agent_id}:{tick}:{scenario}"
    base = stable_rand01(seed + ":base")  # 0..1

    bias = scenario_bias(scenario)
    score = clamp(base + bias, 0.0, 1.0)

    vote = pick_vote(score, scenario)

    # confidence around distance from 0.5
    conf = clamp(0.50 + abs(score - 0.5) * 0.9, 0.50, 0.95)
    conf = round(conf, 2)

    return {
        "agent_id": agent_id,
        "name": agent_name,
        "vote": vote,
        "confidence": conf,
        "reason": reason_templates(vote, scenario),
        "signals": make_signals(seed),
    }


def summarize(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy = sum(1 for a in agents if a["vote"] == "BUY")
    sell = sum(1 for a in agents if a["vote"] == "SELL")
    hold = sum(1 for a in agents if a["vote"] == "HOLD")
    avg_conf = round(sum(a["confidence"] for a in agents) / max(len(agents), 1), 3)

    # simple decision rule (tweakable later)
    # BUY if buy_count >= 9 else HOLD (keep your current behavior)
    decision = "BUY" if buy >= 9 else "HOLD"

    rule = "BUY if buy_count >= 9 else HOLD"

    # top reasons (from BUY agents, highest confidence)
    buy_agents = sorted([a for a in agents if a["vote"] == "BUY"], key=lambda x: x["confidence"], reverse=True)
    top_buy_reasons = [a["reason"] for a in buy_agents[:3]]

    return {
        "decision": decision,
        "buy_count": buy,
        "sell_count": sell,
        "hold_count": hold,
        "avg_confidence": float(round(avg_conf, 3)),
        "rule": rule,
        "digest": {
            "buy_count": buy,
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
        "tick_seconds": TICK_SECONDS,
        "preset_path": PRESET_PATH,
    }


# ----------------------------
# Scenario API
# ----------------------------
@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    cur = await upstash_get(SCENARIO_KEY, KV_REST_API_TOKEN)
    if not cur:
        cur = DEFAULT_SCENARIO
    return {"ok": True, "scenario": cur, "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_post(
    payload: Dict[str, Any] = Body(default_factory=dict),
    _: bool = Depends(require_auth),
):
    scenario = str(payload.get("scenario", "")).strip().lower() if isinstance(payload, dict) else ""
    if scenario not in ("neutral", "bull", "bear", "volatile"):
        raise HTTPException(status_code=422, detail="scenario must be one of: neutral, bull, bear, volatile")

    await upstash_set(SCENARIO_KEY, scenario, KV_REST_API_TOKEN)
    return {"ok": True, "scenario": scenario, "key": SCENARIO_KEY}


# ----------------------------
# Public read API (no auth)
# ----------------------------
@app.get("/public/latest/{market_id}")
async def public_latest(market_id: str):
    key = f"{PUBLIC_LATEST_PREFIX}{market_id}"
    token = KV_REST_API_READ_ONLY_TOKEN or KV_REST_API_TOKEN
    raw = await upstash_get(key, token)
    if not raw:
        return {"ok": True, "market_id": market_id, "data": None}

    try:
        data = json.loads(raw)
    except Exception:
        data = raw

    return {"ok": True, "market_id": market_id, "data": data}


@app.get("/public/history/{market_id}")
async def public_history(market_id: str):
    key = f"{PUBLIC_HISTORY_PREFIX}{market_id}"
    token = KV_REST_API_READ_ONLY_TOKEN or KV_REST_API_TOKEN
    items = await upstash_lrange(key, 0, HISTORY_MAX - 1, token)

    parsed = []
    for it in items:
        try:
            parsed.append(json.loads(it))
        except Exception:
            parsed.append({"raw": it})

    return {"ok": True, "market_id": market_id, "items": parsed}


# ----------------------------
# Main tick (/run)
# ----------------------------
@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    # 1) lock
    lock_id = await try_lock()
    if not lock_id:
        return {
            "ok": True,
            "skipped": True,
            "reason": "locked",
            "message": "Another executor run is in progress",
        }

    try:
        preset = load_preset()
        market_id = str(preset.get("market_id", "POLY:DEMO_001"))
        agent_list = preset.get("agents", [])
        if not isinstance(agent_list, list) or len(agent_list) == 0:
            raise HTTPException(status_code=500, detail="Preset agents list is empty")

        scenario = await upstash_get(SCENARIO_KEY, KV_REST_API_TOKEN) or DEFAULT_SCENARIO

        ts = int(time.time())
        tick = int(ts // TICK_SECONDS)

        agents = []
        for a in agent_list:
            agent_id = str(a.get("agent_id", "")).strip()
            name = str(a.get("name", agent_id)).strip() or agent_id
            if not agent_id:
                continue
            agents.append(agent_decision(agent_id, name, tick, scenario))

        if len(agents) < 15:
            # not fatal, but you want 15
            pass

        summary = summarize(agents)

        record = {
            "ts": ts,
            "tick": tick,
            "market_id": market_id,
            "scenario": scenario,
            "summary": {
                "decision": summary["decision"],
                "buy_count": summary["buy_count"],
                "sell_count": summary["sell_count"],
                "hold_count": summary["hold_count"],
                "avg_confidence": summary["avg_confidence"],
                "rule": summary["rule"],
            },
            "digest": summary["digest"],
            "agents": agents,
        }

        # 2) write latest + history
        latest_key = f"{PUBLIC_LATEST_PREFIX}{market_id}"
        hist_key = f"{PUBLIC_HISTORY_PREFIX}{market_id}"

        await upstash_set(latest_key, json.dumps(record, ensure_ascii=False), KV_REST_API_TOKEN)
        await upstash_lpush(hist_key, json.dumps(record, ensure_ascii=False), KV_REST_API_TOKEN)
        await upstash_ltrim(hist_key, 0, HISTORY_MAX - 1, KV_REST_API_TOKEN)

        duration = round(time.time() - started, 3)
        return {"ok": True, "skipped": False, "scenario": scenario, "message": "tick executed", "duration_sec": duration}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass
