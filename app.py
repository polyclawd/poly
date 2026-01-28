import os
import time
import uuid
from typing import Optional, Literal, List, Dict, Any

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ============================
# Config (ENV ONLY)
# ============================
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "")

KV_REST_API_URL = os.getenv("KV_REST_API_URL", "").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "")
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", "")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))
RUN_TIMEOUT_SECONDS = float(os.getenv("EXECUTOR_RUN_TIMEOUT_SECONDS", "25"))

# ============================
# Auth (Swagger "Authorize")
# ============================
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


# ============================
# Upstash REST helpers
# ============================
def _upstash_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


async def upstash_setnx(key: str, value: str, token: str) -> int:
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN is empty")

    url = f"{KV_REST_API_URL}/setnx/{key}/{value}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()
    data = r.json()
    return int(data.get("result", 0))


async def upstash_del(key: str, token: str) -> int:
    if not KV_REST_API_URL or not token:
        return 0

    url = f"{KV_REST_API_URL}/del/{key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        return 0

    r.raise_for_status()
    data = r.json()
    return int(data.get("result", 0))


async def try_lock() -> Optional[str]:
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id, KV_REST_API_TOKEN)
    if ok == 1:
        return lock_id
    return None


async def unlock(lock_id: str) -> None:
    # упрощенный unlock
    await upstash_del(LOCK_KEY, KV_REST_API_TOKEN)


# ============================
# Fake agents (prebaked)
# ============================
Signal = Literal["BUY", "SELL", "HOLD"]

AGENTS: List[Dict[str, str]] = [
    {"id": "A01", "role": "Market Analyst"},
    {"id": "A02", "role": "News Analyst"},
    {"id": "A03", "role": "Social Sentiment"},
    {"id": "A04", "role": "Orderbook/Microstructure"},
    {"id": "A05", "role": "Volatility/Risk"},
    {"id": "A06", "role": "Macro Context"},
    {"id": "A07", "role": "Contrarian"},
    {"id": "A08", "role": "Momentum"},
    {"id": "A09", "role": "Mean Reversion"},
    {"id": "A10", "role": "Liquidity"},
    {"id": "A11", "role": "Event Calendar"},
    {"id": "A12", "role": "Position Sizing"},
    {"id": "A13", "role": "Execution Safety"},
    {"id": "A14", "role": "Compliance/Guardrails"},
    {"id": "A15", "role": "Final Reviewer"},
]


def _load_prebaked() -> Dict[str, Any]:
    """
    На следующем шаге мы вынесем это в отдельный файл prebaked_data.py.
    Пока держим дефолт прямо тут, чтобы всё работало сразу.
    """
    return {
        "scenario": "neutral",
        "position": "example-market-123",
        "market_price": 0.52,
        "fair_value": 0.52,
        "news": [
            {"title": "No major updates", "snippet": "Quiet market conditions", "score": 0.50},
        ],
        "social": [
            {"title": "Neutral sentiment", "snippet": "Mixed opinions", "score": 0.50},
        ],
    }


def _agent_decide(agent: Dict[str, str], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Фейковая логика: агент читает "scenario" и выдаёт сигнал.
    Мы специально делаем просто, чтобы быстро запустить 15 агентов.
    """
    scenario = data.get("scenario", "neutral")

    if scenario == "bull":
        signal: Signal = "BUY"
        conf = 0.70
    elif scenario == "bear":
        signal = "SELL"
        conf = 0.70
    else:
        signal = "HOLD"
        conf = 0.55

    notes = [f"prebaked scenario={scenario}"]
    evidence_items = []

    for item in (data.get("news", [])[:1] + data.get("social", [])[:1]):
        evidence_items.append(
            {
                "title": item.get("title", "evidence"),
                "snippet": item.get("snippet", ""),
                "score": float(item.get("score", 0.5)),
            }
        )

    return {
        "agent_id": agent["id"],
        "role": agent["role"],
        "signal": signal,
        "confidence": round(conf, 3),
        "notes": notes,
        "evidence": {
            "source": "prebaked",
            "items": evidence_items,
        },
    }


def _aggregate(agents_out: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Взвешенное голосование: BUY=+1, HOLD=0, SELL=-1,
    умножаем на confidence.
    """
    score = 0.0
    conf_sum = 0.0

    for a in agents_out:
        s = a["signal"]
        c = float(a.get("confidence", 0.0))
        conf_sum += c

        if s == "BUY":
            score += 1.0 * c
        elif s == "SELL":
            score += -1.0 * c

    avg_conf = (conf_sum / max(len(agents_out), 1)) if agents_out else 0.0

    # пороги можно потом тюнить
    if score >= 2.5:
        decision: Signal = "BUY"
    elif score <= -2.5:
        decision = "SELL"
    else:
        decision = "HOLD"

    return {
        "decision": decision,
        "score": round(score, 3),
        "avg_conf": round(avg_conf, 3),
    }


# ============================
# App
# ============================
app = FastAPI(title="Polyclawd Executor", version="0.2.0")


@app.get("/")
async def root():
    return {
        "ok": True,
        "service": "executor",
        "docs": "/docs",
        "health": "/healthz",
    }


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "has_executor_secret": bool(EXECUTOR_SECRET),
        "has_kv_url": bool(KV_REST_API_URL),
        "has_kv_token": bool(KV_REST_API_TOKEN),
        "lock_key": LOCK_KEY,
    }


@app.post("/run")
async def run(_: bool = Depends(require_auth)):
    """
    Запускается GitHub Actions cron-ом.
    Требует: Authorization: Bearer <EXECUTOR_SECRET>
    """
    started = time.time()

    lock_id = await try_lock()
    if not lock_id:
        return {
            "ok": True,
            "skipped": True,
            "reason": "locked",
            "message": "Another executor run is in progress",
        }

    try:
        prebaked = _load_prebaked()

        # 15 агентов
        agents_out = [_agent_decide(agent, prebaked) for agent in AGENTS]

        # агрегируем
        agg = _aggregate(agents_out)

        duration = round(time.time() - started, 3)

        return {
            "ok": True,
            "skipped": False,
            "message": "tick executed",
            "scenario": prebaked.get("scenario", "neutral"),
            "position": prebaked.get("position"),
            "decision": agg["decision"],
            "score": agg["score"],
            "avg_conf": agg["avg_conf"],
            "agents": agents_out,
            "duration_sec": duration,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        try:
            await unlock(lock_id)
        except Exception:
            pass
