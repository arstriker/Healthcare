from django.apps import AppConfig


class DrugCheckerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.drug_checker'
    verbose_name = 'Drug Interaction & Guideline Checker'
