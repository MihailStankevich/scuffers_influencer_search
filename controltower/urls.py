from django.urls import path

from . import views


app_name = "controltower"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("match/", views.creator_match, name="match"),
    path("service-worker.js", views.service_worker, name="service_worker"),
    path("api/actions/", views.actions_api, name="actions_api"),
    path("api/risk/launch", views.launch_risk_api, name="launch_risk_api"),
]
