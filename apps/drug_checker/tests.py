from django.test import TestCase, Client
from django.urls import reverse
from apps.drug_checker.models import Drug, DrugInteraction, TokenLog
from apps.drug_checker.tools import mysql_drug_interaction_tool, red_flag_escalation_tool
from apps.drug_checker.agent import DrugCheckerAgent


class DrugCheckerTestCase(TestCase):
    def setUp(self):
        Drug.objects.create(name="Warfarin", category="Cardiovascular")
        Drug.objects.create(name="Aspirin", category="NSAID")
        
        DrugInteraction.objects.create(
            drug_a="Warfarin",
            drug_b="Aspirin",
            severity="major",
            mechanism="Synergistic bleeding risk",
            recommendation="Avoid combination"
        )

    def test_mysql_interaction_tool(self):
        payload = mysql_drug_interaction_tool(["Warfarin", "Aspirin"])
        self.assertEqual(payload["status"], "FOUND")
        self.assertGreaterEqual(len(payload["interactions"]), 1)
        self.assertEqual(payload["interactions"][0]["severity"], "major")

    def test_red_flag_escalation_tool(self):
        payload = mysql_drug_interaction_tool(["Warfarin", "Aspirin"])
        escalation = red_flag_escalation_tool("Patient reports severe hemorrhage", payload["interactions"])
        self.assertTrue(escalation["is_escalated"])
        self.assertGreater(len(escalation["reasons"]), 0)

    def test_agent_prescription_analysis(self):
        agent = DrugCheckerAgent(session_id="test_session")
        response = agent.analyze_prescription(
            prescribed_drugs=["Warfarin", "Aspirin"],
            patient_notes="Check for bleeding risks"
        )
        self.assertIn("Warfarin", response.prescribed_drugs)
        self.assertGreaterEqual(len(response.interactions), 1)
        self.assertTrue(response.escalation.is_escalated)

    def test_health_check_endpoint(self):
        client = Client()
        response = client.get('/api/health/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_check_prescription_endpoint(self):
        client = Client()
        response = client.post(
            '/api/check-prescription/',
            data={"prescribed_drugs": ["Warfarin", "Aspirin"], "patient_notes": "None"},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("summary", data)
        self.assertIn("interactions", data)
        self.assertIn("guideline_citations", data)
        self.assertIn("token_stats", data)
