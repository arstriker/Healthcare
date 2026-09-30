from django.db import models


class Drug(models.Model):
    """NLEM 2022 / Essential Medicines list model."""
    name = models.CharField(max_length=255, unique=True, db_index=True)
    generic_name = models.CharField(max_length=255, blank=True, null=True)
    category = models.CharField(max_length=150, blank=True, null=True)
    dosage_forms = models.CharField(max_length=255, blank=True, null=True)
    therapeutic_class = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.name


class DrugInteraction(models.Model):
    """DDInter Drug-Drug interaction lookup table."""
    SEVERITY_CHOICES = [
        ('minor', 'Minor'),
        ('moderate', 'Moderate'),
        ('major', 'Major'),
        ('contraindicated', 'Contraindicated'),
    ]

    drug_a = models.CharField(max_length=255, db_index=True)
    drug_b = models.CharField(max_length=255, db_index=True)
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='moderate')
    mechanism = models.TextField(blank=True, null=True)
    recommendation = models.TextField()

    class Meta:
        unique_together = ('drug_a', 'drug_b')

    def __str__(self):
        return f"{self.drug_a} + {self.drug_b} ({self.severity})"


class SessionMemory(models.Model):
    """MySQL session memory for pharmacist conversation audit."""
    session_id = models.CharField(max_length=100, db_index=True)
    query_text = models.TextField()
    response_json = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.session_id} - {self.created_at}"


class TokenLog(models.Model):
    """Logs token usage, cost, and latency per query (Module 2 requirement)."""
    session_id = models.CharField(max_length=100)
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    estimated_cost_usd = models.FloatField(default=0.0)
    latency_ms = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Tokens: {self.total_tokens} (${self.estimated_cost_usd:.6f})"
