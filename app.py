import os
import time
import uuid
from typing import Optional, Literal, List, Dict, Any

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from prebaked_data import get_prebaked_snapshot

# ============================
# Config
# ============================
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")

# Scenario stored in KV
SCENARIO_KEY = os.getenv("SCENARIO_KEY", "public:scenario")

# ============================
# Auth (Swagger "Authorize")
# ============================
bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="EXECUTOR_SECRET missing")

    if creds is None:
        raise HTTPException(status_code=403, detail="Not authenticated")

    if creds.credentials != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return True


# ============================
# Upstash helpers
# ============================
def _headers():
    return {"Authorization": f"Bearer {KV_REST_API_TOKEN}", "Accept": "application/json"}


async def kv_get(key: str) -> Optional[str]:
    url = f"{KV_REST_API_URL}/get/{key}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_headers())
    r.raise_for_status()
    return r.json().get("result")


async def kv_set(key: str, value: str) -> bool:
    url = f"{KV_REST_API_URL}/set/{key}/{value}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=_headers())
    return r.status_code == 200


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    url = f"{KV_REST_API_URL}/setnx/{LOCK_KEY}/{lock_id}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=_headers())
    if r.status_code == 200 and r.json().get("result") == 1:
        return lock_id
    return None


async def unlock() -> None:
    url = f"{KV_REST_API_URL}/del/{LOCK_KEY}"
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(url, headers=_headers())


# ============================
# Agents (with weights + disagreements)
# ============================
Signal = Literal["BUY", "SELL", "HOLD"]

AGENTS: List[Dict[str, Any]] = [
    {"id": "A01", "role": "Market Analyst", "weight": 1.0},
    {"id": "A02", "role": "News Analyst", "weight": 1.0},
    {"id": "A03", "role": "Social Sentiment", "weight": 0.9},
    {"id": "A04", "role": "Orderbook/Microstructure", "weight": 1.2},
    {"id": "A05", "role": "Volatility/Risk", "weight": 1.3},
    {"id": "A06", "role": "Macro Context", "weight": 0.7},
    {"id": "A07", "role": "Contrarian", "weight": 0.8},
    {"id": "A08", "role": "Momentum", "weight": 0.9},
    {"id": "A09", "role": "Mean Reversion", "weight": 0.9},
    {"id": "A10", "role": "Liquidity", "weight": 1.1},
    {"id": "A11", "role": "Event Calendar", "weight": 0.6},
    {"id": "A12", "role": "Position Sizing", "weight": 1.4},
    {"id": "A13", "role": "Execution Safety", "weight": 1.3},
    {"id": "A14", "role": "Compliance/Guardrails", "weight": 1.2},
    {"id": "A15", "role": "Final Reviewer", "weight": 1.0},
]

# Blockers: if these agents say HOLD/SELL with high confidence, they can veto BUY
BLOCKER_IDS = {"A05", "A12", "A13", "A14"}

# Deterministic "personality" offsets (so they disagree in a stable way)
PERSONALITY = {
    "A01": 0.00,
    "A02": -0.05,
    "A03": 0.03,
    "A04": 0.04,
    "A05": -0.06,  # more conservative
    "A06": -0.02,
    "A07": -0.08,  # contrarian more likely to oppose
    "A08": 0.05,
    "A09": -0.04,
    "A10": -0.03,
    "A11": -0.01,
    "A12": -0.07,  # sizing conservative
    "A13": -0.06,  # safety conservative
    "A14": -0.05,  # guardrails conservative
    "A15": 0.00,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _agent_decide(agent: Dict[str, Any], snap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scenario gives a base direction, but each agent has a stable personality offset.
    Also low edge -> HOLD more often.
    """
    scenario = snap.get("scenario", "neutral")
    price = float(snap.get("market_price", 0.5) or 0.5)
    fv = float(snap.get("fair_value", 0.5) or 0.5)
    edge = fv - price  # positive edge => buy bias

    base = 0.0
    if scenario == "bull":
        base = 0.06
    elif scenario == "bear":
        base = -0.06

    pid = agent["id"]
    bias = base + PERSONALITY.get(pid, 0.0)

    # incorporate edge
    bias += _clamp(edge, -0.08, 0.08)

    # decide signal
    if bias >= 0.06:
        signal: Signal = "BUY"
    elif bias <= -0.06:
        signal = "SELL"
    else:
        signal = "HOLD"

    # confidence scales with |bias|
    conf = _clamp(0.5 + abs(bias) * 2.2, 0.45, 0.92)

    # conservative agents reduce confidence a bit
    if pid in BLOCKER_IDS:
        conf = _clamp(conf - 0.05, 0.45, 0.92)

    notes = [
        f"scenario={scenario}",
        f"market_price={round(price, 3)}",
        f"fair_value={round(fv, 3)}",
        f"edge={round(edge, 3)}",
        snap.get("headline", ""),
    ]

    evidence_items = []
    for item in (snap.get("news", [])[:1] + snap.get("social", [])[:1]):
        evidence_items.append(
            {
                "title": item.get("title", "evidence"),
                "snippet": item.get("snippet", ""),
                "url": item.get("url"),
                "score": float(item.get("score", 0.5)),
            }
        )

    return {
        "agent_id": pid,
        "role": agent["role"],
        "weight": agent["weight"],
        "signal": signal,
        "confidence": round(conf, 3),
        "notes": notes,
        "evidence": {"source": "prebaked", "items": evidence_items},
    }


def _aggregate(agents_out: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Weighted vote with blocker veto.
    BUY=+1, HOLD=0, SELL=-1.
    """
    wsum = 0.0
    score = 0.0
    conf_sum = 0.0

    veto_buy = False
    veto_reasons = []

    for a in agents_out:
        w = float(a.get("weight", 1.0))
        s = a["signal"]
        c = float(a.get("confidence", 0.0))

        wsum += w
        conf_sum += c

        v = 0.0
        if s == "BUY":
            v = 1.0
        elif s == "SELL":
            v = -1.0

        score += w * v * c

        # veto logic: conservative agents can block BUY if they're strongly against
        if a["agent_id"] in BLOCKER_IDS and s != "BUY" and c >= 0.70:
            veto_buy = True
            veto_reasons.append(f"{a['agent_id']}({a['role']}) blocks BUY: {s} conf={c}")

    avg_conf = conf_sum / max(len(agents_out), 1)
    norm_score = score / max(wsum, 1e-9)

    # base decision from score
    if norm_score >= 0.25:
        decision: Signal = "BUY"
    elif norm_score <= -0.25:
        decision = "SELL"
    else:
        decision = "HOLD"

    # apply veto
    if decision == "BUY" and veto_buy:
        decision = "HOLD"

    return {
        "decision": decision,
        "score": round(norm_score, 3),
        "avg_conf": round(avg_conf, 3),
        "veto_buy": veto_buy,
        "veto_reasons": veto_reasons[:3],
    }


# ============================
# App
# ============================
app = FastAPI(title="Polyclawd Executor", version="0.5.0")


@app.get("/")
async def root():
    return {"ok": True, "service": "executor", "docs": "/docs", "health": "/healthz"}


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    s = await kv_get(SCENARIO_KEY)
    return {"ok": True, "scenario": s or "neutral"}


@app.post("/scenario/{value}")
async def scenario_set(value: Literal["neutral", "bull", "bear"], _: bool = Depends(require_auth)):
    ok = await kv_set(SCENARIO_KEY, value)
    return {"ok": ok, "scenario": value}


@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked"}

    try:
        scenario = (await kv_get(SCENARIO_KEY)) or "neutral"
        snap = get_prebaked_snapshot(scenario)

        agents_out = [_agent_decide(agent, snap) for agent in AGENTS]
        agg = _aggregate(agents_out)

        duration = round(time.time() - started, 3)

        return {
            "ok": True,
            "skipped": False,
            "message": "tick executed",
            "ts": snap.get("ts"),
            "tick_index": snap.get("tick_index"),
            "scenario": scenario,
            "position": snap.get("position"),
            "market_price": snap.get("market_price"),
            "fair_value": snap.get("fair_value"),
            "decision": agg["decision"],
            "score": agg["score"],
            "avg_conf": agg["avg_conf"],
            "veto_buy": agg["veto_buy"],
            "veto_reasons": agg["veto_reasons"],
            "agents": agents_out,
            "duration_sec": duration,
        }

    finally:
        await unlock()
