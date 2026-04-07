from django.urls import path
from . import views

urlpatterns = [
    path("", views.spa, name="index"),
    path("api/detect-corners/", views.detect_corners, name="detect_corners"),
    path("api/sift/", views.compute_sift, name="compute_sift"),
    path("api/match-features/", views.match_features, name="match_features"),
]