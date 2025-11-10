from django.shortcuts import render

def rpa_page(request):
    # Streamlit RPA 앱의 포트로 연결
    return render(request, "llm_hub/rpa.html", {
        "streamlit_url": "http://127.0.0.1:8501/?embed=true"
    })
