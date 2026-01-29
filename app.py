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
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", KV_REST_API_TOKEN)

EXECUTOR_LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "360"))  # 6 минут
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "200"))

DEFAULT_MARKET_ID = os.getenv("DEFAULT_MARKET_ID", "POLY:DEMO_001")


def sanitize_market_id(x: str) -> str:
    s = (x or "").strip()
    # убираем внешние кавычки если кто-то вставил "POLY:DEMO_001"
    if (len(s) >= 2) and ((s[0] == s[-1]) and (s[0] in ("'", '"'))):
        s = s[1:-1].strip()
    return s or "POLY:DEMO_001"


DEFAULT_MARKET_ID = sanitize_market_id(DEFAULT_MARKET_ID)


# KV keys
def k_scenario() -> str:
    return "executor:scenario"


def k_latest(market_id: str) -> str:
    return f"executor:latest:{sanitize_market_id(market_id)}"


def k_history(market_id: str) -> str:
    return f"executor:history:{sanitize_market_id(market_id)}"


# ----------------------------
# Auth (Swagger "Authorize")
# ----------------------------
bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured: EXECUTOR_SECRET is empty")

    if creds is None:
        raise HTTPException(status_code=403, detail="Not authenticated")

    if creds.credentials != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return True


# ----------------------------
# Upstash REST helpers
# ----------------------------
def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _q(s: str) -> str:
    return quote(str(s), safe="")


async def _post(path: str, token: str, timeout: float = 10.0) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="KV_REST_API_URL / KV_REST_API_TOKEN missing")

    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=_h(token))

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")
    r.raise_for_status()
    return r.json()


async def kv_get(key: str, token: str) -> Optional[str]:
    # ВАЖНО: используем POST и для get (самый надежный вариант)
    data = await _post(f"get/{_q(key)}", token)
    return data.get("result", None)


async def kv_set(key: str, value: str, token: str) -> bool:
    data = await _post(f"set/{_q(key)}/{_q(value)}", token)
    return bool(data.get("result"))


async def kv_del(key: str, token: str) -> int:
    data = await _post(f"del/{_q(key)}", token)
    return int(data.get("result", 0) or 0)


async def kv_setnx(key: str, value: str, token: str) -> int:
    data = await _post(f"setnx/{_q(key)}/{_q(value)}", token)
    return int(data.get("result", 0) or 0)


async def kv_expire(key: str, seconds: int, token: str) -> int:
    data = await _post(f"expire/{_q(key)}/{int(seconds)}", token)
    return int(data.get("result", 0) or 0)


# ----------------------------
# Lock
# ----------------------------
async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await kv_setnx(EXECUTOR_LOCK_KEY, lock_id, KV_REST_API_TOKEN)
    if ok == 1:
        try:
            await kv_expire(EXECUTOR_LOCK_KEY, LOCK_TTL_SECONDS, KV_REST_API_TOKEN)
        except Exception:
            pass
        return lock_id
    return None


async def unlock(_: str) -> None:
    try:
        await kv_del(EXECUTOR_LOCK_KEY, KV_REST_API_TOKEN)
    except Exception:
        pass


# ----------------------------
# Fake agents preset
# ----------------------------
def load_agents_preset() -> List[Dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), "agents_preset.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("agents"), list):
                return data["agents"]
            if isinstance(data, list):
                return data
        except Exception:
            pass

    agents = []
    for i in range(1, 16):
        agents.append(
            {
                "agent_id": f"agent_{i:02d}",
                "name": f"Agent {i:02d}",
                "vote": "HOLD",
                "confidence": 0.55,
                "reason": "Preset missing; using default HOLD.",
                "signals": {
                    "price": round(0.3 + 0.4 * (i / 15), 3),
                    "implied_prob": round(0.5 + 0.2 * (i / 15), 3),
                    "sentiment": round(-0.2 + 0.4 * (i / 15), 3),
                    "volatility": round(0.2 + 0.3 * (i / 15), 3),
                },
            }
        )
    return agents


def summarize(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy = sum(1 for a in agents if a.get("vote") == "BUY")
    hold = sum(1 for a in agents if a.get("vote") == "HOLD")
    sell = sum(1 for a in agents if a.get("vote") == "SELL")
    confs = [float(a.get("confidence", 0) or 0) for a in agents]
    avg_conf = round(sum(confs) / max(1, len(confs)), 3)

    decision = "BUY" if buy >= 9 else "HOLD"
    rule = "BUY if buy_count >= 9 else HOLD"

    return {
        "decision": decision,
        "buy_count": buy,
        "hold_count": hold,
        "sell_count": sell,
        "avg_confidence": avg_conf,
        "rule": rule,
    }


def digest_from_agents(agents: List[Dict[str, Any]], buy_count: int) -> Dict[str, Any]:
    buy_reasons = []
    for a in agents:
        if a.get("vote") == "BUY":
            r = str(a.get("reason", "")).strip()
            if r:
                buy_reasons.append(r)
    return {"buy_count": buy_count, "top_buy_reasons": buy_reasons[:3]}


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
        "lock_key": EXECUTOR_LOCK_KEY,
        "default_market": DEFAULT_MARKET_ID,
    }


# --- Scenario (GET/POST) ---
@app.get("/scenario")
async def get_scenario(_: bool = Depends(require_auth)):
    s = await kv_get(k_scenario(), KV_REST_API_READ_ONLY_TOKEN)
    return {"ok": True, "scenario": s or "neutral", "key": k_scenario()}


@app.post("/scenario")
async def set_scenario(body: Dict[str, Any], _: bool = Depends(require_auth)):
    scenario = str(body.get("scenario", "neutral")).strip() or "neutral"
    await kv_set(k_scenario(), scenario, KV_REST_API_TOKEN)
    return {"ok": True, "scenario": scenario}


# --- Public endpoints for UI ---
@app.get("/public/latest/{market_id}")
async def public_latest(market_id: str):
    market_id = sanitize_market_id(market_id)
    raw = await kv_get(k_latest(market_id), KV_REST_API_READ_ONLY_TOKEN)
    if not raw:
        return {"ok": True, "market_id": market_id, "data": None}
    try:
        return {"ok": True, "market_id": market_id, "data": json.loads(raw)}
    except Exception:
        return {"ok": True, "market_id": market_id, "data": raw}


@app.get("/public/history/{market_id}")
async def public_history(market_id: str, minutes: int = 60):
    market_id = sanitize_market_id(market_id)
    raw = await kv_get(k_history(market_id), KV_REST_API_READ_ONLY_TOKEN)
    if not raw:
        return {"ok": True, "market_id": market_id, "items": []}

    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            items = []
    except Exception:
        items = []

    now = int(time.time())
    cutoff = now - max(1, int(minutes)) * 60
    items = [it for it in items if int(it.get("ts", 0) or 0) >= cutoff]

    return {"ok": True, "market_id": market_id, "items": items}


# --- Main tick endpoint ---
@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked", "message": "Another run in progress"}

    try:
        scenario = (await kv_get(k_scenario(), KV_REST_API_READ_ONLY_TOKEN)) or "neutral"

        tick = int(time.time())  # новый тик каждый запуск

        agents = load_agents_preset()

        tail = {
            "bull": "scenario tailwind supports entry.",
            "bear": "risk-off regime; be selective.",
            "neutral": "balanced regime; wait for edge.",
        }.get(scenario, "balanced regime; wait for edge.")

        for a in agents:
            r = str(a.get("reason", "")).strip()
            if r and tail not in r:
                a["reason"] = (r + " " + tail).strip()

        summary = summarize(agents)
        digest = digest_from_agents(agents, summary["buy_count"])

        payload = {
            "ts": int(time.time()),
            "tick": tick,
            "market_id": DEFAULT_MARKET_ID,
            "scenario": scenario,
            "summary": summary,
            "digest": digest,
            "agents": agents,
        }

        # latest
        await kv_set(k_latest(DEFAULT_MARKET_ID), json.dumps(payload, ensure_ascii=False), KV_REST_API_TOKEN)

        # history append
        raw_hist = await kv_get(k_history(DEFAULT_MARKET_ID), KV_REST_API_READ_ONLY_TOKEN)
        if raw_hist:
            try:
                hist = json.loads(raw_hist)
                if not isinstance(hist, list):
                    hist = []
            except Exception:
                hist = []
        else:
            hist = []

        hist.append(payload)
        if len(hist) > HISTORY_LIMIT:
            hist = hist[-HISTORY_LIMIT:]

        await kv_set(k_history(DEFAULT_MARKET_ID), json.dumps(hist, ensure_ascii=False), KV_REST_API_TOKEN)

        duration = round(time.time() - started, 3)
        return {"ok": True, "skipped": False, "scenario": scenario, "message": "tick executed", "duration_sec": duration}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        await unlock(lock_id)
