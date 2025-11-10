from django.urls import path
from . import views

app_name = "rpa_llm"

urlpatterns = [
    path("", views.rpa_page, name="rpa_page"),  # /llm_hub/rpa/
]