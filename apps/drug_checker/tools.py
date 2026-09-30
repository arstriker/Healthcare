import re
from typing import List, Dict, Any
from django.db.models import Q
from apps.drug_checker.models import DrugInteraction
from apps.drug_checker.rag.retriever import GuidelineRetriever

retriever_instance = GuidelineRetriever()


def clean_drug_name(raw_name: str) -> str:
    """Extract clean base drug name by stripping dosages and salt suffixes."""
    name = re.sub(r'\d+(\.\d+)?\s*(mg|g|mcg|ml|iu|tablet|capsule|ampoule)', '', raw_name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(sodium|hydrochloride|citrate|sulfate|bisulfate|monohydrate|trihydrate|potassium)', '', name, flags=re.IGNORECASE)
    return name.strip().title()


def mysql_drug_interaction_tool(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Deterministic SQL query tool against MySQL/SQLite DrugInteraction table (320,000+ pairs).
    Checks all unique pairs of prescribed drugs with high-speed indexed lookups.
    """
    cleaned_drugs = []
    for d in drugs:
        if d and d.strip():
            c = clean_drug_name(d)
            if c:
                cleaned_drugs.append(c)

    found_interactions = []
    seen_pairs = set()

    SEVERITY_WEIGHT = {
        'contraindicated': 4,
        'major': 3,
        'moderate': 2,
        'minor': 1
    }

    for i in range(len(cleaned_drugs)):
        for j in range(i + 1, len(cleaned_drugs)):
            d1, d2 = cleaned_drugs[i], cleaned_drugs[j]
            pair_key = tuple(sorted([d1.lower(), d2.lower()]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Query database for exact or case-insensitive match
            qs = DrugInteraction.objects.filter(
                (Q(drug_a__iexact=d1) & Q(drug_b__iexact=d2)) |
                (Q(drug_a__iexact=d2) & Q(drug_b__iexact=d1))
            )

            # Fallback to contains search if exact match yields no rows
            if not qs.exists():
                qs = DrugInteraction.objects.filter(
                    (Q(drug_a__icontains=d1) & Q(drug_b__icontains=d2)) |
                    (Q(drug_a__icontains=d2) & Q(drug_b__icontains=d1))
                )

            pair_items = []
            for item in qs[:5]:  # Limit top matches per pair
                pair_items.append({
                    "drug_a": item.drug_a,
                    "drug_b": item.drug_b,
                    "severity": item.severity,
                    "mechanism": item.mechanism or "Mechanism recorded in DDInter dataset.",
                    "recommendation": item.recommendation,
                    "_weight": SEVERITY_WEIGHT.get(item.severity.lower(), 1)
                })

            # Sort by severity weight
            pair_items.sort(key=lambda x: x["_weight"], reverse=True)
            for pi in pair_items:
                del pi["_weight"]
                found_interactions.append(pi)

    return found_interactions


def guideline_retrieval_tool(query: str) -> List[Dict[str, Any]]:
    """
    RAG Search tool querying FAISS vector store for ICMR/WHO clinical treatment guidelines.
    """
    return retriever_instance.retrieve(query=query, top_k=3, score_threshold=0.40)


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
