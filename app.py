import os
import time
import uuid 
import json
import asyncio
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

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))

DEFAULT_MARKET_ID = os.getenv("DEFAULT_MARKET_ID", "POLY:DEMO_001")

SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")
DEFAULT_SCENARIO = os.getenv("DEFAULT_SCENARIO", "neutral")
ALLOWED_SCENARIOS = ["neutral", "bull", "bear"]

HISTORY_TTL_SECONDS = int(os.getenv("EXECUTOR_HISTORY_TTL_SECONDS", "3600"))  # 1 час
HISTORY_MAX_ITEMS = int(os.getenv("EXECUTOR_HISTORY_MAX_ITEMS", "120"))

HTTP_TIMEOUT = float(os.getenv("EXECUTOR_HTTP_TIMEOUT", "12"))

AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "2.5"))

PRESET_FILE = os.getenv("AGENTS_PRESET_FILE", "agents_preset.json")
PRESET_INDEX_KEY = os.getenv("PRESET_INDEX_KEY", "executor:preset_index")


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
# Upstash helpers
# ----------------------------
def _upstash_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _clean(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().strip('"').strip("'")


def _enc(s: str) -> str:
    return quote(_clean(s), safe="")


class UpstashWrongType(Exception):
    pass


async def _upstash_request(method: str, path: str, token: str) -> Dict[str, Any]:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN is empty")

    url = f"{KV_REST_API_URL}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.request(method, url, headers=_upstash_headers(token))

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    if r.status_code >= 400:
        txt = r.text or ""
        if "WRONGTYPE" in txt:
            raise UpstashWrongType(txt)
        raise HTTPException(status_code=500, detail=f"Upstash error {r.status_code}: {txt}")

    return r.json()


async def upstash_get(key: str, token: str) -> Optional[str]:
    data = await _upstash_request("GET", f"get/{_enc(key)}", token)
    res = data.get("result", None)
    if res is None:
        return None
    return str(res)


async def upstash_set(key: str, value: str, token: str) -> None:
    await _upstash_request("POST", f"set/{_enc(key)}/{_enc(value)}", token)


async def upstash_del(key: str, token: str) -> None:
    await _upstash_request("POST", f"del/{_enc(key)}", token)


async def upstash_setnx(key: str, value: str, token: str) -> int:
    data = await _upstash_request("POST", f"setnx/{_enc(key)}/{_enc(value)}", token)
    return int(data.get("result", 0))


async def upstash_expire(key: str, seconds: int, token: str) -> None:
    await _upstash_request("POST", f"expire/{_enc(key)}/{int(seconds)}", token)


# ----------------------------
# Safe wrappers (auto-heal WRONGTYPE)
# ----------------------------
async def safe_get_string(key: str) -> Optional[str]:
    try:
        return await upstash_get(key, KV_REST_API_TOKEN)
    except UpstashWrongType:
        await upstash_del(key, KV_REST_API_TOKEN)
        return None


async def safe_set_string(key: str, value: str) -> None:
    try:
        await upstash_set(key, value, KV_REST_API_TOKEN)
    except UpstashWrongType:
        await upstash_del(key, KV_REST_API_TOKEN)
        await upstash_set(key, value, KV_REST_API_TOKEN)


async def safe_setnx_string(key: str, value: str) -> int:
    try:
        return await upstash_setnx(key, value, KV_REST_API_TOKEN)
    except UpstashWrongType:
        await upstash_del(key, KV_REST_API_TOKEN)
        return await upstash_setnx(key, value, KV_REST_API_TOKEN)


# ----------------------------
# Keys
# ----------------------------
def latest_key(market_id: str) -> str:
    return f"executor:latest:{_clean(market_id)}"


def history_key(market_id: str) -> str:
    return f"executor:history:{_clean(market_id)}"


# ----------------------------
# Lock
# ----------------------------
async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await safe_setnx_string(LOCK_KEY, lock_id)
    if ok == 1:
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
# Scenario
# ----------------------------
async def get_scenario() -> str:
    raw = await safe_get_string(SCENARIO_KEY)
    if raw and raw in ALLOWED_SCENARIOS:
        return raw
    return DEFAULT_SCENARIO


async def set_scenario(s: str) -> str:
    s = _clean(s).lower()
    if s not in ALLOWED_SCENARIOS:
        raise HTTPException(status_code=422, detail=f"scenario must be one of {ALLOWED_SCENARIOS}")
    await safe_set_string(SCENARIO_KEY, s)
    return s


# ----------------------------
# Presets loader
# ----------------------------
def load_presets() -> Dict[str, Any]:
    try:
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "snapshots" not in data or not isinstance(data["snapshots"], list) or len(data["snapshots"]) == 0:
            raise ValueError("agents_preset.json must contain non-empty snapshots[]")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load {PRESET_FILE}: {type(e).__name__}: {e}")


PRESETS = load_presets()


def apply_scenario_bias(agents: List[Dict[str, Any]], scenario: str, tick: int) -> List[Dict[str, Any]]:
    """
    Чуть “подкручиваем” пресет под сценарий:
    - bull: иногда переводим HOLD -> BUY (с высокой уверенностью)
    - bear: иногда переводим BUY -> HOLD
    Делаем детерминированно от tick, чтобы было воспроизводимо.
    """
    out = []
    for i, a in enumerate(agents):
        a2 = dict(a)
        # нормализуем поля
        a2.setdefault("status", "ok")
        a2.setdefault("signals", {})
        vote = str(a2.get("vote", "HOLD")).upper()
        conf = float(a2.get("confidence", 0.55))

        # детерминированная “монетка”
        r = ((tick * 1009 + i * 97) % 100) / 100.0

        if scenario == "bull":
            # 20% HOLD -> BUY
            if vote == "HOLD" and r < 0.20:
                a2["vote"] = "BUY"
                a2["confidence"] = round(min(conf + 0.10, 0.95), 2)
                a2["reason"] = (a2.get("reason", "") + " (bull bias)").strip()
        elif scenario == "bear":
            # 20% BUY -> HOLD
            if vote == "BUY" and r < 0.20:
                a2["vote"] = "HOLD"
                a2["confidence"] = round(max(conf - 0.10, 0.45), 2)
                a2["reason"] = (a2.get("reason", "") + " (bear caution)").strip()
        else:
            a2["vote"] = vote

        out.append(a2)
    return out


def summarize(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy = sum(1 for a in agents if str(a.get("vote", "")).upper() == "BUY")
    hold = sum(1 for a in agents if str(a.get("vote", "")).upper() == "HOLD")
    ok_cnt = sum(1 for a in agents if a.get("status") == "ok")
    avg_conf = round(
        sum(float(a.get("confidence", 0.0)) for a in agents if a.get("status") == "ok") / max(1, ok_cnt),
        3,
    )
    decision = "BUY" if buy >= 9 else "HOLD"
    return {
        "decision": decision,
        "buy_count": buy,
        "hold_count": hold,
        "avg_confidence": avg_conf,
        "rule": "BUY if buy_count >= 9 else HOLD",
    }


def make_digest(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy_reasons = [a.get("reason", "") for a in agents if str(a.get("vote", "")).upper() == "BUY"][:3]
    return {
        "buy_count": sum(1 for a in agents if str(a.get("vote", "")).upper() == "BUY"),
        "top_buy_reasons": buy_reasons,
    }


async def next_snapshot_index() -> int:
    raw = await safe_get_string(PRESET_INDEX_KEY)
    try:
        idx = int(raw) if raw is not None else 0
    except Exception:
        idx = 0
    snap_count = len(PRESETS["snapshots"])
    idx = idx % snap_count
    # сохраняем следующий (круговой)
    next_idx = (idx + 1) % snap_count
    await safe_set_string(PRESET_INDEX_KEY, str(next_idx))
    return idx


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
        "preset_file": PRESET_FILE,
        "preset_snapshots": len(PRESETS.get("snapshots", [])),
        "preset_index_key": PRESET_INDEX_KEY,
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
    raw = await safe_get_string(k)
    if not raw:
        return {"ok": False, "market_id": market_id, "error": "no data yet"}
    try:
        return json.loads(raw)
    except Exception:
        return {"ok": False, "market_id": market_id, "error": "corrupt json", "raw": raw}


@app.get("/public/history/{market_id}")
async def public_history(market_id: str):
    k = history_key(market_id)
    raw = await safe_get_string(k)
    if not raw:
        return {"ok": True, "market_id": market_id, "items": []}
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            items = []
        return {"ok": True, "market_id": market_id, "items": items}
    except Exception:
        return {"ok": True, "market_id": market_id, "items": []}


@app.post("/admin/reset")
async def admin_reset(_: bool = Depends(require_auth)):
    keys = [
        LOCK_KEY,
        SCENARIO_KEY,
        PRESET_INDEX_KEY,
        latest_key(DEFAULT_MARKET_ID),
        history_key(DEFAULT_MARKET_ID),
    ]
    for k in keys:
        try:
            await upstash_del(k, KV_REST_API_TOKEN)
        except Exception:
            pass
    return {"ok": True, "deleted": keys}


@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {"ok": True, "skipped": True, "reason": "locked", "message": "Another run is in progress"}

    try:
        scenario = await get_scenario()
        market_id = DEFAULT_MARKET_ID
        tick = int(time.time() // 300)

        # берём следующий пресет
        snap_idx = await next_snapshot_index()
        snap = PRESETS["snapshots"][snap_idx]
        agents_raw = snap.get("agents", [])
        if not isinstance(agents_raw, list) or len(agents_raw) != 15:
            raise HTTPException(status_code=500, detail="Preset snapshot must contain exactly 15 agents")

        # применяем сценарий
        agents = apply_scenario_bias(agents_raw, scenario, tick)

        # summary/digest
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

        latest_payload = {"ok": True, "market_id": market_id, "data": item}

        lk = latest_key(market_id)
        await safe_set_string(lk, json.dumps(latest_payload, ensure_ascii=False))
        try:
            await upstash_expire(lk, HISTORY_TTL_SECONDS, KV_REST_API_TOKEN)
        except Exception:
            pass

        hk = history_key(market_id)
        raw_hist = await safe_get_string(hk)
        hist: List[Dict[str, Any]] = []
        if raw_hist:
            try:
                hist = json.loads(raw_hist)
                if not isinstance(hist, list):
                    hist = []
            except Exception:
                hist = []

        hist.append(item)

        cutoff = int(time.time()) - HISTORY_TTL_SECONDS
        hist = [x for x in hist if int(x.get("ts", 0)) >= cutoff]

        if len(hist) > HISTORY_MAX_ITEMS:
            hist = hist[-HISTORY_MAX_ITEMS:]

        await safe_set_string(hk, json.dumps(hist, ensure_ascii=False))
        try:
            await upstash_expire(hk, HISTORY_TTL_SECONDS, KV_REST_API_TOKEN)
        except Exception:
            pass

        duration = round(time.time() - started, 3)
        return {
            "ok": True,
            "skipped": False,
            "scenario": scenario,
            "preset_snapshot": snap_idx,
            "message": "tick executed (preset)",
            "duration_sec": duration,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        await unlock()
