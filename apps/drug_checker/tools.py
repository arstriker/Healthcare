import re
import difflib
from typing import List, Dict, Any, Optional
from django.db.models import Q
from apps.drug_checker.models import Drug, DrugInteraction
from apps.drug_checker.rag.hybrid_retriever import HybridRetriever

hybrid_retriever_instance = HybridRetriever()


def clean_drug_name(raw_name: str) -> str:
    """Extract clean base drug name by stripping dosages and salt suffixes."""
    name = re.sub(r'\d+(\.\d+)?\s*(mg|g|mcg|ml|iu|tablet|capsule|ampoule)', '', raw_name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(sodium|hydrochloride|citrate|sulfate|bisulfate|monohydrate|trihydrate|potassium)', '', name, flags=re.IGNORECASE)
    return name.strip().title()


def get_fuzzy_drug_suggestions(query_name: str) -> List[str]:
    """Find close fuzzy matches for misspelled drug names from 1,941 drug database."""
    all_drugs = list(Drug.objects.values_list('name', flat=True))
    matches = difflib.get_close_matches(query_name.title(), all_drugs, n=3, cutoff=0.6)
    return matches


def mysql_drug_interaction_tool(drugs: List[str]) -> Dict[str, Any]:
    """
    Deterministic SQL query tool against MySQL/SQLite DrugInteraction table (320,000+ pairs).
    Returns explicit status ('FOUND' vs 'NOT_FOUND') and fuzzy 'did_you_mean' suggestions for typos.
    """
    cleaned_drugs = []
    for d in drugs:
        if d and d.strip():
            c = clean_drug_name(d)
            if c:
                cleaned_drugs.append(c)

    found_interactions = []
    did_you_mean_dict = {}
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

            if not qs.exists():
                # Check if drug names exist in catalog or find fuzzy suggestions
                sug_1 = get_fuzzy_drug_suggestions(d1)
                sug_2 = get_fuzzy_drug_suggestions(d2)
                if sug_1:
                    did_you_mean_dict[d1] = sug_1
                if sug_2:
                    did_you_mean_dict[d2] = sug_2
            else:
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

                pair_items.sort(key=lambda x: x["_weight"], reverse=True)
                for pi in pair_items:
                    del pi["_weight"]
                    found_interactions.append(pi)

    status_str = "FOUND" if len(found_interactions) > 0 else "NOT_FOUND"
    
    # Flatten fuzzy suggestions into unique list
    did_you_mean_list = []
    for drug_q, sugs in did_you_mean_dict.items():
        for s in sugs:
            if s not in did_you_mean_list and s.lower() != drug_q.lower():
                did_you_mean_list.append(s)

    return {
        "status": status_str,
        "interactions": found_interactions,
        "did_you_mean": did_you_mean_list,
        "message": f"Found {len(found_interactions)} interaction pair(s) in DDInter database." if status_str == "FOUND" else "No direct interactions recorded for queried items."
    }


def guideline_retrieval_tool(query: str, filter_doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    RAG Search tool querying Hybrid BM25 + FAISS + Reranker engine over ICMR/WHO guidelines.
    Returns explicit tool status payload.
    """
    results = hybrid_retriever_instance.retrieve(
        query=query,
        top_k=3,
        score_threshold=0.35,
        filter_doc_id=filter_doc_id
    )
    status_str = "FOUND" if len(results) > 0 else "NO_RELEVANT_GUIDELINES"
    return {
        "status": status_str,
        "citations": results,
        "message": f"Retrieved {len(results)} guideline section(s) via BM25 + FAISS + CrossEncoder reranking." if status_str == "FOUND" else "No clinical guidelines met the confidence relevance threshold."
    }


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
