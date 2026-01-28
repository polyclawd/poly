import os
import time
import uuid
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# ----------------------------
# Config
# ----------------------------
EXECUTOR_SECRET = os.getenv("EXECUTOR_SECRET", "executor-secret-polyclawd-2026")
KV_REST_API_URL = os.getenv("KV_REST_API_URL", "https://thankful-bluegill-32910.upstash.io").rstrip("/")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN", "AYCOAAIncDI5OTM2N2MzOTBiNGE0NzJjODhjZTZjZWQzNzBjMDI5MXAyMzI5MTA")
KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN", "AoCOAAIgcDLO-2qllWynMs01ogYqMD8JqSY3FvxGj_ywnYHKeHFZxg")

LOCK_KEY = os.getenv("EXECUTOR_LOCK_KEY", "executor:lock")
LOCK_TTL_SECONDS = int(os.getenv("EXECUTOR_LOCK_TTL_SECONDS", "60"))  # сколько держим lock
RUN_TIMEOUT_SECONDS = float(os.getenv("EXECUTOR_RUN_TIMEOUT_SECONDS", "25"))

if not EXECUTOR_SECRET:
    # Не валим импортом, но /run будет отдавать 500 с понятной ошибкой
    pass


# ----------------------------
# Auth (Swagger "Authorize")
# ----------------------------
bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if not EXECUTOR_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured: EXECUTOR_SECRET is empty")

    if creds is None:
        # Swagger / curl без хедера
        raise HTTPException(status_code=403, detail="Not authenticated")

    token = creds.credentials  # это часть после "Bearer "
    if token != EXECUTOR_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return True


# ----------------------------
# Upstash REST helpers
# ----------------------------
def _upstash_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


async def upstash_setnx(key: str, value: str, token: str) -> int:
    """
    Upstash Redis REST: SETNX key value
    Возвращает 1 если установили, 0 если ключ уже был.
    """
    if not KV_REST_API_URL or not token:
        raise HTTPException(status_code=500, detail="Server misconfigured: KV_REST_API_URL / KV_REST_API_TOKEN is empty")

    url = f"{KV_REST_API_URL}/setnx/{key}/{value}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    # если токен неверный → 401
    if r.status_code == 401:
        raise HTTPException(status_code=500, detail="Upstash 401 Unauthorized: check KV_REST_API_TOKEN")

    r.raise_for_status()
    data = r.json()
    # Upstash обычно возвращает {"result": 1} или {"result": 0}
    return int(data.get("result", 0))


async def upstash_del(key: str, token: str) -> int:
    """
    Upstash Redis REST: DEL key
    """
    if not KV_REST_API_URL or not token:
        return 0

    url = f"{KV_REST_API_URL}/del/{key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=_upstash_headers(token))

    if r.status_code == 401:
        # не падаем жёстко на unlock, просто сообщим
        return 0

    r.raise_for_status()
    data = r.json()
    return int(data.get("result", 0))


async def try_lock() -> Optional[str]:
    """
    Ставит lock через SETNX.
    Возвращает lock_id если получилось, иначе None.
    """
    lock_id = str(uuid.uuid4())
    ok = await upstash_setnx(LOCK_KEY, lock_id, KV_REST_API_TOKEN)
    if ok == 1:
        return lock_id
    return None


async def unlock(lock_id: str) -> None:
    """
    Упрощённый unlock: удаляем ключ.
    (Для строгого unlock можно хранить owner и проверять через GET, но сейчас это ок.)
    """
    await upstash_del(LOCK_KEY, KV_REST_API_TOKEN)


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Polyclawd Executor", version="0.1.0")


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

    # 1) lock, чтобы не было параллельных тиков
    lock_id = await try_lock()
    if not lock_id:
        return {
            "ok": True,
            "skipped": True,
            "reason": "locked",
            "message": "Another executor run is in progress",
        }

    try:
        # 2) тут будет твоя логика: прочитать state из KV, дернуть агентов, отправить ордера и т.д.
        # Пока просто заглушка:
        # ------------------------------------------------------------
        # TODO: implement actual trading logic here
        # ------------------------------------------------------------

        duration = round(time.time() - started, 3)
        return {
            "ok": True,
            "skipped": False,
            "message": "tick executed",
            "duration_sec": duration,
        }

    except Exception as e:
        # чтобы в GitHub Actions было видно что именно сломалось
        raise HTTPException(status_code=500, detail=f"Executor error: {type(e).__name__}: {e}") from e

    finally:
        # 3) unlock
        try:
            await unlock(lock_id)
        except Exception:
            # не валим ответ, просто молча
            pass
