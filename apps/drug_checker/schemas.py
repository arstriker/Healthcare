from typing import List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_doc: str = Field(..., description="Document title/source identifier")
    section: str = Field(..., description="Section title or number")
    excerpt: str = Field(..., description="Relevant text quote from guideline")
    confidence: float = Field(..., description="Similarity confidence score")


class DrugInteractionDetail(BaseModel):
    drug_a: str = Field(..., description="First drug name")
    drug_b: str = Field(..., description="Second drug name")
    severity: str = Field(..., description="Severity level: minor, moderate, major, contraindicated")
    mechanism: Optional[str] = Field(None, description="Pharmacological mechanism")
    recommendation: str = Field(..., description="Actionable clinical recommendation")


class EscalationAlert(BaseModel):
    is_escalated: bool = Field(..., description="True if red-flag symptoms or contraindicated interactions require human clinical review")
    reasons: List[str] = Field(default_factory=list, description="Specific triggers causing escalation")


class TokenStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0


class PrescriptionAnalysisResponse(BaseModel):
    summary: str = Field(..., description="Executive summary for pharmacist/health worker")
    prescribed_drugs: List[str] = Field(default_factory=list, description="Identified drugs from prescription")
    interactions: List[DrugInteractionDetail] = Field(default_factory=list, description="Detected drug-drug interactions from database")
    guideline_citations: List[Citation] = Field(default_factory=list, description="ICMR/WHO guideline citations grounding the response")
    escalation: EscalationAlert = Field(..., description="Escalation status and human review triggers")
    token_stats: TokenStats = Field(default_factory=TokenStats, description="Token usage & SLA metrics")
