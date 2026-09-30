from django.urls import path
from .views import HealthCheckView, CheckPrescriptionView

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('check-prescription/', CheckPrescriptionView.as_view(), name='check-prescription'),
]
