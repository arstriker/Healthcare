from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Avg, Sum

from apps.drug_checker.agent import DrugCheckerAgent
from apps.drug_checker.models import Drug, DrugInteraction, TokenLog


class HealthCheckView(APIView):
    def get(self, request):
        total_drugs = Drug.objects.count()
        total_interactions = DrugInteraction.objects.count()
        total_logs = TokenLog.objects.count()
        
        avg_latency = TokenLog.objects.aggregate(Avg('latency_ms'))['latency_ms__avg'] or 0.0
        total_cost = TokenLog.objects.aggregate(Sum('estimated_cost_usd'))['estimated_cost_usd__sum'] or 0.0

        return Response({
            "status": "healthy",
            "service": "Healthcare Pharmacist RAG Agent",
            "version": "1.0.0",
            "database_stats": {
                "nlem_drugs_indexed": total_drugs,
                "ddinter_interactions_indexed": total_interactions,
                "total_queries_logged": total_logs,
                "avg_latency_ms": round(avg_latency, 2),
                "total_cost_usd": round(total_cost, 6)
            }
        }, status=status.HTTP_200_OK)


class CheckPrescriptionView(APIView):
    def post(self, request):
        data = request.data
        prescribed_drugs = data.get('prescribed_drugs', [])
        patient_notes = data.get('patient_notes', '')
        session_id = data.get('session_id', 'session_default')

        if not isinstance(prescribed_drugs, list) or not prescribed_drugs:
            return Response(
                {"error": "prescribed_drugs must be a non-empty list of drug names."},
                status=status.HTTP_400_BAD_REQUEST
            )

        agent = DrugCheckerAgent(session_id=session_id)
        analysis_response = agent.analyze_prescription(
            prescribed_drugs=prescribed_drugs,
            patient_notes=patient_notes
        )

        return Response(analysis_response.model_dump(), status=status.HTTP_200_OK)
