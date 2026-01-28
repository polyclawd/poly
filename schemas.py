from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


Signal = Literal["BUY", "SELL", "HOLD"]


class EvidenceItem(BaseModel):
    title: str
    snippet: str
    url: Optional[str] = None
    score: float = Field(ge=0.0, le=1.0)


class Evidence(BaseModel):
    source: Literal["prebaked"] = "prebaked"
    items: List[EvidenceItem] = []


class AgentOutput(BaseModel):
    agent_id: str
    role: str
    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0)
    notes: List[str] = []
    evidence: Evidence = Evidence()


class RunResult(BaseModel):
    ok: bool = True
    skipped: bool = False
    message: str = "tick executed"
    scenario: str = "neutral"
    decision: Signal = "HOLD"
    score: float = 0.0
    avg_conf: float = 0.0
    agents: List[AgentOutput] = []
    duration_sec: float = 0.0
