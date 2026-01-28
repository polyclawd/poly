from typing import Dict, Any, List
import time


def _ts() -> int:
    return int(time.time())


# 5 часов вперёд: 60 тиков (каждые 5 минут)
PREBAKED_TIMELINE: Dict[str, List[Dict[str, Any]]] = {
    "neutral": [
        {
            "position": "poly-market-EXAMPLE",
            "scenario": "neutral",
            "market_price": 0.52,
            "fair_value": 0.52,
            "headline": "Quiet market, no strong catalyst",
            "news": [
                {"title": "No major updates", "snippet": "Market waiting for new info", "score": 0.50, "url": None},
            ],
            "social": [
                {"title": "Neutral sentiment", "snippet": "Mixed takes, no consensus", "score": 0.50, "url": None},
            ],
        }
        for _ in range(60)
    ],
    "bull": [
        {
            "position": "poly-market-EXAMPLE",
            "scenario": "bull",
            "market_price": 0.56,
            "fair_value": 0.60,
            "headline": "Positive momentum with supportive news flow",
            "news": [
                {"title": "Positive update", "snippet": "New development increases probability", "score": 0.78, "url": None},
            ],
            "social": [
                {"title": "Bullish chatter", "snippet": "More users expect YES outcome", "score": 0.70, "url": None},
            ],
        }
        for _ in range(60)
    ],
    "bear": [
        {
            "position": "poly-market-EXAMPLE",
            "scenario": "bear",
            "market_price": 0.48,
            "fair_value": 0.42,
            "headline": "Negative drift amid skeptical narrative",
            "news": [
                {"title": "Negative angle", "snippet": "Counter-signal reduces probability", "score": 0.76, "url": None},
            ],
            "social": [
                {"title": "Bearish sentiment", "snippet": "Users lean NO, confidence growing", "score": 0.72, "url": None},
            ],
        }
        for _ in range(60)
    ],
}


def get_prebaked_snapshot(scenario: str) -> Dict[str, Any]:
    bucket = int(_ts() / 300)  # 5 минут
    idx = bucket % 60

    if scenario not in PREBAKED_TIMELINE:
        scenario = "neutral"

    row = PREBAKED_TIMELINE[scenario][idx].copy()
    row["ts"] = _ts()
    row["tick_index"] = idx
    return row
