from django.urls import path
from .views import IndexView, HealthCheckView, CheckPrescriptionView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('check-prescription/', CheckPrescriptionView.as_view(), name='check-prescription'),
]
