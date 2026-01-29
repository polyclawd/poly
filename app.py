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

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))

DEFAULT_MARKET_ID = os.getenv("DEFAULT_MARKET_ID", "POLY:DEMO_001")

SCENARIO_KEY = os.getenv("EXECUTOR_SCENARIO_KEY", "executor:scenario")
DEFAULT_SCENARIO = os.getenv("DEFAULT_SCENARIO", "neutral")
ALLOWED_SCENARIOS = ["neutral", "bull", "bear"]

HISTORY_TTL_SECONDS = int(os.getenv("EXECUTOR_HISTORY_TTL_SECONDS", "3600"))  # 1 час
HISTORY_MAX_ITEMS = int(os.getenv("EXECUTOR_HISTORY_MAX_ITEMS", "120"))

HTTP_TIMEOUT = float(os.getenv("EXECUTOR_HTTP_TIMEOUT", "12"))

# ---- Presets (ты сказал уже сделал) ----
PRESET_INDEX_KEY = os.getenv("PRESET_INDEX_KEY", "executor:preset_index")
PRESET_FILE = os.getenv("AGENTS_PRESET_FILE", "agents_preset.json")

# ---- Portfolio rules ----
PORTFOLIO_KEY = os.getenv("PORTFOLIO_KEY", "executor:portfolio")
START_CASH = float(os.getenv("START_CASH", "250"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "6"))
MAX_TRADE_PCT = float(os.getenv("MAX_TRADE_PCT", "0.10"))  # 10% equity per trade
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.40"))      # 40% equity per market
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "3600"))  # 60 минут

# tick = 15 минут
TICK_SECONDS = int(os.getenv("TICK_SECONDS", "900"))


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="Polyclawd Executor", version="0.1.0")


# ----------------------------
# Auth
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


async def upstash_get(key: str) -> Optional[str]:
    data = await _upstash_request("GET", f"get/{_enc(key)}", KV_REST_API_TOKEN)
    res = data.get("result", None)
    if res is None:
        return None
    return str(res)


async def upstash_set(key: str, value: str) -> None:
    await _upstash_request("POST", f"set/{_enc(key)}/{_enc(value)}", KV_REST_API_TOKEN)


async def upstash_del(key: str) -> None:
    await _upstash_request("POST", f"del/{_enc(key)}", KV_REST_API_TOKEN)


async def upstash_setnx(key: str, value: str) -> int:
    data = await _upstash_request("POST", f"setnx/{_enc(key)}/{_enc(value)}", KV_REST_API_TOKEN)
    return int(data.get("result", 0))


async def upstash_expire(key: str, seconds: int) -> None:
    await _upstash_request("POST", f"expire/{_enc(key)}/{int(seconds)}", KV_REST_API_TOKEN)


async def safe_get(key: str) -> Optional[str]:
    try:
        return await upstash_get(key)
    except UpstashWrongType:
        await upstash_del(key)
        return None


async def safe_set(key: str, value: str) -> None:
    try:
        await upstash_set(key, value)
    except UpstashWrongType:
        await upstash_del(key)
        await upstash_set(key, value)


async def safe_setnx(key: str, value: str) -> int:
    try:
        return await upstash_setnx(key, value)
    except UpstashWrongType:
        await upstash_del(key)
        return await upstash_setnx(key, value)


# ----------------------------
# Keys
# ----------------------------
def latest_key(market_id: str) -> str:
    return f"executor:latest:{_clean(market_id)}"


def history_key(market_id: str) -> str:
    return f"executor:history:{_clean(market_id)}"


def last_trade_key(market_id: str) -> str:
    return f"executor:last_trade_ts:{_clean(market_id)}"


# ----------------------------
# Lock
# ----------------------------
async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await safe_setnx(LOCK_KEY, lock_id)
    if ok == 1:
        try:
            await upstash_expire(LOCK_KEY, LOCK_TTL_SECONDS)
        except Exception:
            pass
        return lock_id
    return None


async def unlock() -> None:
    try:
        await upstash_del(LOCK_KEY)
    except Exception:
        pass


# ----------------------------
# Scenario
# ----------------------------
async def get_scenario() -> str:
    raw = await safe_get(SCENARIO_KEY)
    if raw and raw in ALLOWED_SCENARIOS:
        return raw
    return DEFAULT_SCENARIO


async def set_scenario(s: str) -> str:
    s = _clean(s).lower()
    if s not in ALLOWED_SCENARIOS:
        raise HTTPException(status_code=422, detail=f"scenario must be one of {ALLOWED_SCENARIOS}")
    await safe_set(SCENARIO_KEY, s)
    return s


# ----------------------------
# Presets
# ----------------------------
def load_presets() -> Dict[str, Any]:
    with open(PRESET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "snapshots" not in data or not isinstance(data["snapshots"], list) or len(data["snapshots"]) == 0:
        raise RuntimeError("agents_preset.json must contain snapshots[]")
    return data


PRESETS = load_presets()


async def next_snapshot_index() -> int:
    raw = await safe_get(PRESET_INDEX_KEY)
    try:
        idx = int(raw) if raw is not None else 0
    except Exception:
        idx = 0
    n = len(PRESETS["snapshots"])
    idx = idx % n
    await safe_set(PRESET_INDEX_KEY, str((idx + 1) % n))
    return idx


def apply_scenario_bias(agents: List[Dict[str, Any]], scenario: str, tick: int) -> List[Dict[str, Any]]:
    out = []
    for i, a in enumerate(agents):
        a2 = dict(a)
        a2.setdefault("status", "ok")
        a2.setdefault("signals", {})
        vote = str(a2.get("vote", "HOLD")).upper()
        conf = float(a2.get("confidence", 0.55))
        r = ((tick * 1009 + i * 97) % 100) / 100.0

        if scenario == "bull":
            if vote == "HOLD" and r < 0.20:
                a2["vote"] = "BUY"
                a2["confidence"] = round(min(conf + 0.10, 0.95), 2)
                a2["reason"] = (str(a2.get("reason", "")).strip() + " (bull bias)").strip()
        elif scenario == "bear":
            if vote == "BUY" and r < 0.20:
                a2["vote"] = "HOLD"
                a2["confidence"] = round(max(conf - 0.10, 0.45), 2)
                a2["reason"] = (str(a2.get("reason", "")).strip() + " (bear caution)").strip()
        else:
            a2["vote"] = vote

        out.append(a2)
    return out


def summarize_agents(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy = sum(1 for a in agents if str(a.get("vote", "")).upper() == "BUY")
    hold = sum(1 for a in agents if str(a.get("vote", "")).upper() == "HOLD")
    ok_cnt = sum(1 for a in agents if a.get("status") == "ok")
    avg_conf = round(sum(float(a.get("confidence", 0.0)) for a in agents if a.get("status") == "ok") / max(1, ok_cnt), 3)
    decision = "BUY" if buy >= 9 else "HOLD"
    return {"decision": decision, "buy_count": buy, "hold_count": hold, "avg_confidence": avg_conf, "rule": "BUY if buy_count >= 9 else HOLD"}


def digest_from_agents(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    buy_reasons = [a.get("reason", "") for a in agents if str(a.get("vote", "")).upper() == "BUY"][:3]
    return {"buy_count": sum(1 for a in agents if str(a.get("vote", "")).upper() == "BUY"), "top_buy_reasons": buy_reasons}


# ----------------------------
# Portfolio state + execution engine
# ----------------------------
def default_portfolio() -> Dict[str, Any]:
    return {
        "cash": START_CASH,
        "positions": {},  # market_id -> {"side":"YES", "usd": float, "avg_price": float}
        "created_ts": int(time.time()),
        "updated_ts": int(time.time()),
    }


def compute_equity(portfolio: Dict[str, Any]) -> float:
    # пока mark-to-market делаем упрощенно: позиция считается по вложенным usd
    # (позже заменим на реальную цену и PnL)
    cash = float(portfolio.get("cash", 0.0))
    pos_usd = 0.0
    for _, p in (portfolio.get("positions", {}) or {}).items():
        pos_usd += float(p.get("usd", 0.0))
    return round(cash + pos_usd, 2)


async def load_portfolio() -> Dict[str, Any]:
    raw = await safe_get(PORTFOLIO_KEY)
    if not raw:
        pf = default_portfolio()
        await safe_set(PORTFOLIO_KEY, json.dumps(pf))
        return pf
    try:
        return json.loads(raw)
    except Exception:
        pf = default_portfolio()
        await safe_set(PORTFOLIO_KEY, json.dumps(pf))
        return pf


async def save_portfolio(pf: Dict[str, Any]) -> None:
    pf["updated_ts"] = int(time.time())
    await safe_set(PORTFOLIO_KEY, json.dumps(pf))


async def last_trade_ts(market_id: str) -> int:
    raw = await safe_get(last_trade_key(market_id))
    try:
        return int(raw) if raw else 0
    except Exception:
        return 0


async def set_last_trade_ts(market_id: str, ts: int) -> None:
    await safe_set(last_trade_key(market_id), str(ts))


def pick_trade_size(equity: float, cash: float) -> float:
    size = equity * MAX_TRADE_PCT
    size = min(size, cash)
    return round(max(0.0, size), 2)


def execute_trade_decision(
    market_id: str,
    summary: Dict[str, Any],
    agents: List[Dict[str, Any]],
    pf: Dict[str, Any],
    now_ts: int,
    last_ts: int,
) -> Dict[str, Any]:
    """
    Симуляция:
    - decision BUY: открываем/увеличиваем (если можно)
    - decision HOLD: ничего
    - decision SELL (пока не генерим, но оставим): закрываем
    """
    positions = pf.get("positions", {}) or {}
    equity = compute_equity(pf)
    cash = float(pf.get("cash", 0.0))
    open_count = len(positions)

    decision = str(summary.get("decision", "HOLD")).upper()
    avg_conf = float(summary.get("avg_confidence", 0.0))

    # cooldown
    if last_ts and (now_ts - last_ts) < COOLDOWN_SECONDS:
        return {"action": "SKIP", "reason": "cooldown", "cooldown_left_sec": COOLDOWN_SECONDS - (now_ts - last_ts)}

    # confidence gates
    if decision == "BUY" and avg_conf < 0.70:
        return {"action": "SKIP", "reason": "confidence_too_low", "avg_confidence": avg_conf}
    if decision == "HOLD":
        return {"action": "SKIP", "reason": "decision_hold", "avg_confidence": avg_conf}

    # SELL пока не используем, но заложим
    if decision == "SELL" or avg_conf <= 0.55:
        if market_id in positions:
            usd = float(positions[market_id].get("usd", 0.0))
            pf["cash"] = round(cash + usd, 2)
            del positions[market_id]
            pf["positions"] = positions
            return {"action": "CLOSE", "reason": "exit_signal", "size_usd": usd}
        return {"action": "SKIP", "reason": "no_position_to_close"}

    # BUY path
    if market_id not in positions:
        if open_count >= MAX_OPEN_POSITIONS:
            return {"action": "SKIP", "reason": "max_positions_reached", "max_positions": MAX_OPEN_POSITIONS}

        size = pick_trade_size(equity, cash)
        if size <= 0:
            return {"action": "SKIP", "reason": "no_cash"}

        # open position
        pf["cash"] = round(cash - size, 2)
        positions[market_id] = {"side": "YES", "usd": size, "avg_price": None}
        pf["positions"] = positions
        return {"action": "OPEN", "reason": "entry_signal", "size_usd": size}

    # already in position: maybe add
    current_usd = float(positions[market_id].get("usd", 0.0))
    cap = equity * MAX_POS_PCT
    if current_usd >= cap:
        return {"action": "SKIP", "reason": "position_cap_reached", "cap_usd": round(cap, 2), "current_usd": current_usd}

    size = pick_trade_size(equity, cash)
    if size <= 0:
        return {"action": "SKIP", "reason": "no_cash"}

    # don't exceed cap
    size = min(size, cap - current_usd)
    size = round(max(0.0, size), 2)
    if size <= 0:
        return {"action": "SKIP", "reason": "cap_remaining_too_small"}

    pf["cash"] = round(cash - size, 2)
    positions[market_id]["usd"] = round(current_usd + size, 2)
    pf["positions"] = positions
    return {"action": "ADD", "reason": "add_signal", "size_usd": size}


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
        "default_market_id": DEFAULT_MARKET_ID,
        "tick_seconds": TICK_SECONDS,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "max_trade_pct": MAX_TRADE_PCT,
        "max_pos_pct": MAX_POS_PCT,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "preset_snapshots": len(PRESETS.get("snapshots", [])),
    }


@app.get("/scenario")
async def scenario_get(_: bool = Depends(require_auth)):
    return {"ok": True, "scenario": await get_scenario(), "key": SCENARIO_KEY}


@app.post("/scenario")
async def scenario_post(body: Dict[str, Any], _: bool = Depends(require_auth)):
    s = body.get("scenario", None)
    if not s:
        raise HTTPException(status_code=422, detail="body must contain: {\"scenario\": \"neutral|bull|bear\"}")
    return {"ok": True, "scenario": await set_scenario(str(s))}


@app.get("/public/latest/{market_id}")
async def public_latest(market_id: str):
    raw = await safe_get(latest_key(market_id))
    if not raw:
        return {"ok": False, "market_id": market_id, "error": "no data yet"}
    return json.loads(raw)


@app.get("/public/history/{market_id}")
async def public_history(market_id: str):
    raw = await safe_get(history_key(market_id))
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
        PORTFOLIO_KEY,
        latest_key(DEFAULT_MARKET_ID),
        history_key(DEFAULT_MARKET_ID),
        last_trade_key(DEFAULT_MARKET_ID),
    ]
    for k in keys:
        try:
            await upstash_del(k)
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
        now_ts = int(time.time())
        tick = int(now_ts // TICK_SECONDS)

        scenario = await get_scenario()
        market_id = DEFAULT_MARKET_ID

        # load presets snapshot
        snap_idx = await next_snapshot_index()
        snap = PRESETS["snapshots"][snap_idx]
        agents_raw = snap.get("agents", [])
        if not isinstance(agents_raw, list) or len(agents_raw) != 15:
            raise HTTPException(status_code=500, detail="Preset snapshot must contain exactly 15 agents")

        agents = apply_scenario_bias(agents_raw, scenario, tick)
        summary = summarize_agents(agents)
        digest = digest_from_agents(agents)

        # portfolio + execution
        pf = await load_portfolio()
        lt = await last_trade_ts(market_id)

        exec_res = execute_trade_decision(market_id, summary, agents, pf, now_ts, lt)

        if exec_res.get("action") in ("OPEN", "ADD", "CLOSE"):
            await set_last_trade_ts(market_id, now_ts)

        await save_portfolio(pf)
        equity = compute_equity(pf)

        item = {
            "ts": now_ts,
            "tick": tick,
            "market_id": market_id,
            "scenario": scenario,
            "preset_snapshot": snap_idx,
            "summary": summary,
            "digest": digest,
            "execution": exec_res,
            "portfolio": {
                "cash": round(float(pf.get("cash", 0.0)), 2),
                "equity": equity,
                "open_positions": len((pf.get("positions", {}) or {})),
                "positions": pf.get("positions", {}),
            },
            "agents": agents,
        }

        latest_payload = {"ok": True, "market_id": market_id, "data": item}

        await safe_set(latest_key(market_id), json.dumps(latest_payload, ensure_ascii=False))
        await upstash_expire(latest_key(market_id), HISTORY_TTL_SECONDS)

        raw_hist = await safe_get(history_key(market_id))
        hist: List[Dict[str, Any]] = []
        if raw_hist:
            try:
                hist = json.loads(raw_hist)
                if not isinstance(hist, list):
                    hist = []
            except Exception:
                hist = []

        hist.append(item)

        cutoff = now_ts - HISTORY_TTL_SECONDS
        hist = [x for x in hist if int(x.get("ts", 0)) >= cutoff]
        if len(hist) > HISTORY_MAX_ITEMS:
            hist = hist[-HISTORY_MAX_ITEMS:]

        await safe_set(history_key(market_id), json.dumps(hist, ensure_ascii=False))
        await upstash_expire(history_key(market_id), HISTORY_TTL_SECONDS)

        duration = round(time.time() - started, 3)
        return {"ok": True, "skipped": False, "scenario": scenario, "message": "tick executed + execution", "duration_sec": duration}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        await unlock()
