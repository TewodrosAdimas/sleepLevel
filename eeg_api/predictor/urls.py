from django.urls import path
from .views import predict_eeg

urlpatterns = [
    path('predict/', predict_eeg),
]