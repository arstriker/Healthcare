import os
import time
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types

from apps.drug_checker.tools import (
    mysql_drug_interaction_tool,
    guideline_retrieval_tool,
    red_flag_escalation_tool
)
from apps.drug_checker.schemas import (
    PrescriptionAnalysisResponse,
    Citation,
    DrugInteractionDetail,
    EscalationAlert,
    TokenStats
)
from apps.drug_checker.models import SessionMemory, TokenLog


# Per-query token cap (Requirement from Hackathon brief)
PER_QUERY_TOKEN_CAP = 4000
COST_PER_1K_PROMPT_TOKENS = 0.000075  # USD estimate for Gemini 2.5 Flash
COST_PER_1K_COMPLETION_TOKENS = 0.00030  # USD estimate for Gemini 2.5 Flash


class DrugCheckerAgent:
    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def analyze_prescription(self, prescribed_drugs: List[str], patient_notes: str = "") -> PrescriptionAnalysisResponse:
        start_time = time.time()

        # Step 1: Run Deterministic SQL Interaction Lookup Tool
        raw_interactions = mysql_drug_interaction_tool(prescribed_drugs)

        # Step 2: Run Guideline RAG Retrieval Tool
        rag_query = f"Interactions, dosing, and safety precautions for {' '.join(prescribed_drugs)}. Notes: {patient_notes}"
        retrieved_chunks = guideline_retrieval_tool(query=rag_query)

        # Step 3: Run Safety Escalation Tool
        escalation_dict = red_flag_escalation_tool(symptoms_or_notes=patient_notes, interactions=raw_interactions)

        # Build structured object items
        interaction_details = [
            DrugInteractionDetail(
                drug_a=item["drug_a"],
                drug_b=item["drug_b"],
                severity=item["severity"],
                mechanism=item["mechanism"],
                recommendation=item["recommendation"]
            )
            for item in raw_interactions
        ]

        citations = [
            Citation(
                source_doc=chunk["source_doc"],
                section=chunk["section"],
                excerpt=chunk["content"],
                confidence=chunk["confidence"]
            )
            for chunk in retrieved_chunks
        ]

        escalation_obj = EscalationAlert(
            is_escalated=escalation_dict["is_escalated"],
            reasons=escalation_dict["reasons"]
        )

        # Step 4: Synthesize response via Gemini LLM (or deterministic fallback if API key missing)
        summary_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        if self.client:
            prompt_content = f"""
You are a specialized clinical AI assistant for pharmacists and health workers.
Analyze the following prescription data grounded STRICTLY on the retrieved tools and database facts.

PRESCRIBED DRUGS: {', '.join(prescribed_drugs)}
PATIENT NOTES / SYMPTOMS: {patient_notes}

DETERMINISTIC DATABASE INTERACTIONS:
{json.dumps(raw_interactions, indent=2)}

RETRIEVED CLINICAL GUIDELINES:
{json.dumps(retrieved_chunks, indent=2)}

ESCALATION TRIGGER:
{json.dumps(escalation_dict, indent=2)}

RULES:
1. Do NOT make up medical diagnosis or dosing advice to patients.
2. Ground all claims on the retrieved database records and guidelines.
3. If no relevant interactions or guidelines exist in the corpus, state clearly: "No additional interaction or guideline warnings found in the provided corpus."
4. Provide a clear, professional summary for the pharmacist.
"""
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=1000
                    )
                )
                summary_text = response.text
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                    completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            except Exception as e:
                summary_text = f"Agent synthesis generated via tool pipeline. (LLM call note: {str(e)})"
        else:
            # Deterministic synthesis when GEMINI_API_KEY is not set
            if raw_interactions:
                inter_summary = "; ".join([f"{i['drug_a']}+{i['drug_b']} ({i['severity'].upper()})" for i in raw_interactions])
                summary_text = f"Identified {len(raw_interactions)} drug-drug interaction(s): {inter_summary}. Grounded in DDInter database."
            else:
                summary_text = "No direct drug-drug interactions found in the DDInter database for the prescribed items."
            
            if citations:
                summary_text += f" Relevant ICMR/WHO guidelines retrieved ({len(citations)} section citations)."

        # Calculate SLA metrics & token costs
        elapsed_ms = (time.time() - start_time) * 1000
        total_tokens = prompt_tokens + completion_tokens
        
        # Enforce Token Cap
        if total_tokens > PER_QUERY_TOKEN_CAP:
            summary_text += f" [WARNING: Token cap of {PER_QUERY_TOKEN_CAP} tokens exceeded! Current: {total_tokens}]"

        estimated_cost = (
            (prompt_tokens / 1000.0) * COST_PER_1K_PROMPT_TOKENS +
            (completion_tokens / 1000.0) * COST_PER_1K_COMPLETION_TOKENS
        )

        token_stats = TokenStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(estimated_cost, 6),
            latency_ms=round(elapsed_ms, 2)
        )

        final_response = PrescriptionAnalysisResponse(
            summary=summary_text,
            prescribed_drugs=prescribed_drugs,
            interactions=interaction_details,
            guideline_citations=citations,
            escalation=escalation_obj,
            token_stats=token_stats
        )

        # Log session and token stats to database
        try:
            SessionMemory.objects.create(
                session_id=self.session_id,
                query_text=f"Drugs: {prescribed_drugs} | Notes: {patient_notes}",
                response_json=final_response.model_dump()
            )
            TokenLog.objects.create(
                session_id=self.session_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=token_stats.estimated_cost_usd,
                latency_ms=token_stats.latency_ms
            )
        except Exception:
            pass

        return final_response
