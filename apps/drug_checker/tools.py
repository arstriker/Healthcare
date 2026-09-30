from typing import List, Dict, Any
from apps.drug_checker.models import DrugInteraction
from apps.drug_checker.rag.retriever import GuidelineRetriever

retriever_instance = GuidelineRetriever()


def mysql_drug_interaction_tool(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Deterministic SQL query tool against MySQL/SQLite DrugInteraction table.
    Checks all unique pairs of prescribed drugs.
    """
    cleaned_drugs = [d.strip().title() for d in drugs if d.strip()]
    found_interactions = []
    
    seen_pairs = set()
    for i in range(len(cleaned_drugs)):
        for j in range(i + 1, len(cleaned_drugs)):
            d1, d2 = cleaned_drugs[i], cleaned_drugs[j]
            pair_key = tuple(sorted([d1, d2]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            qs = DrugInteraction.objects.filter(drug_a__iexact=d1, drug_b__iexact=d2)
            if not qs.exists():
                qs = DrugInteraction.objects.filter(drug_a__iexact=d2, drug_b__iexact=d1)

            for item in qs:
                found_interactions.append({
                    "drug_a": item.drug_a,
                    "drug_b": item.drug_b,
                    "severity": item.severity,
                    "mechanism": item.mechanism or "Mechanism not explicitly detailed.",
                    "recommendation": item.recommendation
                })
                
    return found_interactions


def guideline_retrieval_tool(query: str) -> List[Dict[str, Any]]:
    """
    RAG Search tool querying FAISS vector store for ICMR/WHO clinical treatment guidelines.
    """
    return retriever_instance.retrieve(query=query, top_k=3, score_threshold=0.30)


def red_flag_escalation_tool(symptoms_or_notes: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates clinical red flags (severe symptoms or contraindicated interactions)
    requiring mandatory human doctor/pharmacist intervention.
    """
    triggers = []
    
    RED_FLAG_KEYWORDS = [
        "cyanosis", "silent chest", "altered mental state", "severe hypotension",
        "anaphylaxis", "respiratory distress", "bleeding", "hemorrhage", "104"
    ]
    
    lower_notes = (symptoms_or_notes or "").lower()
    for kw in RED_FLAG_KEYWORDS:
        if kw in lower_notes:
            triggers.append(f"Patient note contains high-risk symptom trigger: '{kw}'")
            
    for inter in interactions:
        sev = (inter.get("severity") or "").lower()
        if sev in ["contraindicated", "major"]:
            triggers.append(
                f"Severe drug interaction detected: {inter['drug_a']} + {inter['drug_b']} ({sev.upper()})"
            )
            
    return {
        "is_escalated": len(triggers) > 0,
        "reasons": triggers
    }
