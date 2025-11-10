"""
파일명: /Users/airim/github/hari_fold_django/llm_hub/rpa_llm/streamlit_rpa.py
기능: 사이드바 기반 데이터 전처리(업로드→병합→다운로드) + 분석용 데이터 업로드 + 시각화 + LLM
"""

import streamlit as st
import pandas as pd

import plotly.graph_objects as go
import os, sys, io, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dotenv import load_dotenv

from llm_handler import run_rpa_llm, generate_answer_rag, post_process_answer, llm

from streamlit_extras.dataframe_explorer import dataframe_explorer
import warnings
warnings.filterwarnings("ignore", message="Could not infer format")

# .env 로드
dotenv_path = os.path.join(os.path.dirname(__file__), "../../.env")
load_dotenv(dotenv_path)

BASE_DIR = os.path.dirname(__file__)

# 페이지 기본 설정
st.set_page_config(page_title="RPA 근무시간 데이터 분석", page_icon="⚙️", layout="wide")

if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None
if "llm_ready" not in st.session_state:
    st.session_state["llm_ready"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# 기준 키 컬럼
KEY_COLS = ["사번", "EMPLID", "성명", "직군", "직무", "직급", "경력경로", "부서"]


def preprocess_and_merge(uploaded_files: list) -> pd.DataFrame:
    """
    업로드된 여러 개의 엑셀 파일을 병합하고, 각 월별 '연장합계' 컬럼을 생성하여 반환한다.
    
    Parameters
    ----------
    uploaded_files : list
        사용자가 업로드한 xlsx 파일 리스트. 예: [UploadedFile, UploadedFile, ...]
        각 파일은 공통 키(KEY_COLS)와 월별 지표 컬럼을 포함한다고 가정.
        예시 컬럼: '1월_기본근무', '1월_야간근무', '1월_휴일근무', '1월_연장근무' 등

    Returns
    -------
    pd.DataFrame
        병합/정리 완료된 데이터프레임.
        특징:
        - KEY_COLS 기준 outer join
        - 각 월별 '연장합계' = 기본근무 + 야간근무 + 휴일근무 + 연장근무
        - '연장합계' 컬럼은 해당 월 '기본근무' 바로 뒤에 배치
        예외 상황:
        - 일부 월 컬럼이 누락된 경우 해당 월의 연장합계는 생성하지 않음
    """
    if not uploaded_files:
        return pd.DataFrame()

    dfs = [pd.read_excel(f) for f in uploaded_files]

    # 기준 DF를 첫 번째 파일로 설정한 뒤, 나머지를 순차 병합
    df = dfs[0].copy()
    for d in dfs[1:]:
        df = pd.merge(df, d, on=KEY_COLS, how="outer")

    # 월별 연장합계 컬럼 생성 및 컬럼 순서 조정
    # 규칙: "{월}_기본근무"가 있으면 해당 월을 기준으로 '연장합계' 생성
    for col in list(df.columns):
        if "기본근무" in col and re.match(r"^\d{1,2}월_기본근무$", col):
            month = col.split("_")[0]  # '1월', '2월' 등
            base = f"{month}_기본근무"
            night = f"{month}_야간근무"
            holiday = f"{month}_휴일근무"
            extend = f"{month}_연장근무"
            new_col = f"{month}_연장합계"

            needed = [base, night, holiday, extend]
            if all(c in df.columns for c in needed):
                # 결측치는 0으로 처리
                df[new_col] = (
                    df[base].fillna(0)
                    + df[night].fillna(0)
                    + df[holiday].fillna(0)
                    + df[extend].fillna(0)
                )

                # 컬럼 순서 재배치: '기본근무' 바로 뒤에 '연장합계' 배치
                cols = list(df.columns)
                base_idx = cols.index(base)
                if new_col in cols:
                    cols.remove(new_col)
                cols.insert(base_idx + 1, new_col)
                df = df[cols]

    return df


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "clean_data") -> io.BytesIO:
    """
    DataFrame을 메모리상의 엑셀 바이너리(BytesIO)로 변환한다.

    Parameters
    ----------
    df : pd.DataFrame
        저장할 데이터프레임
    sheet_name : str
        엑셀 시트명. 기본값은 'clean_data'

    Returns
    -------
    io.BytesIO
        엑셀 파일 바이너리 버퍼. Streamlit download_button에 data로 전달 가능
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer



# --- 사이드바 ---
st.sidebar.markdown(
    """
    <div style='
        font-size:30px;
        font-weight:bold;
        color:#2563eb;
        text-align:center;
        padding:8px 0;
        border-bottom:2px solid #e5e7eb;
        '>
        RPA Control Panel
    </div>
    """,
    unsafe_allow_html=True
)


# Step 1. 데이터 전처리 (원천 업로드 → 병합/정리 → clean_data.xlsx 다운로드)
with st.sidebar.expander("**Step 1. 데이터 전처리**", expanded=False):
    uploaded_files_raw = st.file_uploader(
        "전처리가 필요한 Excel 파일을 업로드 해주세요.",
        type=["xlsx"],
        accept_multiple_files=True,
        key="raw_files_uploader",
    )

    if uploaded_files_raw:
        st.success(f"{len(uploaded_files_raw)}개 파일 업로드 완료")

        if st.button("전처리 및 병합 실행", use_container_width=True, key="btn_preprocess"):
            df_clean = preprocess_and_merge(uploaded_files_raw)

            if df_clean.empty:
                st.warning("전처리 결과가 비어 있습니다. 업로드한 파일과 컬럼 구성을 확인하세요.")
            else:
                st.success("전처리 완료. clean_data.xlsx로 다운로드할 수 있습니다.")

                # 다운로드
                clean_bytes = to_excel_bytes(df_clean, sheet_name="clean_data")
                st.download_button(
                    label="clean_data.xlsx 다운로드",
                    data=clean_bytes,
                    file_name="clean_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# Step 2. 분석용 데이터 업로드 (전처리된 clean_data.xlsx 업로드 또는 기본 데이터 사용)
with st.sidebar.expander("**Step 2. 분석용 데이터 업로드**", expanded=True):
    df = None
    uploaded_clean = st.file_uploader(
        "",
        type=["xlsx"],
        accept_multiple_files=False,
        key="clean_file_uploader"
    )

    # 업로드 우선, 없으면 기본 데이터 사용
    if uploaded_clean is not None:
        try:
            df = pd.read_excel(uploaded_clean)
            st.session_state["uploaded_df"] = df
            st.success("분석용 데이터 업로드가 완료되었습니다.")
        except Exception as e:
            st.error(f"분석용 데이터 로드 중 오류가 발생했습니다: {e}")
    else:
        st.caption("규격에 맞는 분석용 데이터 업로드를 부탁드립니다.")

# Step 3. 대시보드 탐색 (분석용 데이터가 준비된 경우에만)
selected_viz = None
with st.sidebar.expander("**Step 3. 대시보드 탐색**", expanded=True):
    if st.session_state.get("uploaded_df") is None:
        st.warning("데이터를 먼저 업로드하세요 (Step 2).")
    else:
        viz_options = ["데이터 미리보기", "전사 OT 현황"]
        selected_viz = st.selectbox("시각화 항목 선택", viz_options, key="viz_selector")

# Step 4. LLM 분석 (분석용 데이터가 준비된 경우에만)
with st.sidebar.expander("**Step 4. 데이터 분석**", expanded=False):
    if st.session_state.get("uploaded_df") is None:
        st.warning("데이터를 먼저 업로드하세요 (Step 2).")
    else:
        st.markdown("""
        <div style="color:#6e6e6e;">
        HARI 추천 프롬프트<br>
        1. 정열창303의 1월 연장합계에 대해 알려줘<br>
        2. 정열창303 과 정열창49의 2월 연장근무에 비교분석 후 알려줘<br>
        3. 건설/개발 직군에서 1월 기본근무 시간 상위 5명을 알려줘
        </div>
        """, unsafe_allow_html=True)
        if st.button("LLM 기반 데이터분석 시작", width='stretch', key="btn_llm_ready"):
            st.session_state["llm_ready"] = True
            st.success("데이터를 분석할 준비가 되었습니다.")


# --- 메인 영역 ---
df = st.session_state.get("uploaded_df")

# --- 메인 화면 구성 ---
if df is None:
    st.markdown(
        """
        <div style="text-align:center; margin-top:120px;">
            <h2 style="color:#2563eb;">RPA On-boarding</h2>
            <p style="font-size:1.5rem; color:#4b5563;">
                좌측 사이드바에서 데이터를 직접 업로드하거나,<br>
                기본 데이터를 선택하여 분석을 시작하세요.
            </p>
            <p style="font-size:1.0rem; color:#6b7280;">
                RPA LLM은 데이터 업로드 부터 대시보드 연결 그리고 LLM 기반 분석까지<br>
                Data 기반의 End-to-End 시스템을 제공합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


else:
    # 선택된 시각화 실행
    if selected_viz == "데이터 미리보기":
        st.markdown("<h3 style='text-align:center;'>EDA (Exploratory Data Analysis)</h3>", unsafe_allow_html=True)

        if df is not None:
            # 실제 데이터프레임에서 사용 가능한 월 컬럼만 추출
            month_cols = [c for c in df.columns if re.match(r"(\d{1,2})월_", c)]
            available_months = sorted(set([re.match(r"(\d{1,2})월_", c).group(1) for c in month_cols]))

            # 조회할 월 선택
            if available_months:
                month = st.selectbox(
                    "조회할 월 선택",
                    [f"{m}월" for m in available_months],
                    index=0
                )
            else:
                st.warning("업로드된 데이터에서 조회 가능한 월이 없습니다.")
        else:
            st.info("먼저 데이터를 업로드하세요.")

        # 기본 컬럼
        base_cols = ["사번", "성명", "직군", "직무", "부서", "직급", "경력경로"]

        # 동적 컬럼 후보
        candidate_cols = [
            f"{month}_연장합계",
            f"{month}_연장근무",
            f"{month}_휴일근무",
            f"{month}_야간근무",
            f"{month}_기본근무",
        ]
        dynamic_cols = [c for c in candidate_cols if c in df.columns]
        show_cols = [c for c in base_cols if c in df.columns] + dynamic_cols

        tab1, tab2 = st.tabs(["상위 10행 요약", "전체 데이터 탐색"])

        with tab1:
            if show_cols:
                st.dataframe(df[show_cols].head(10), use_container_width=True, height=350)
            else:
                st.info("표시할 컬럼이 없습니다. 데이터 컬럼 구성을 확인하세요.")

        with tab2:
            st.markdown("<p style='text-align:center; color:#4b5563;'>필요 시 조건으로 필터링해 확인하세요.</p>", unsafe_allow_html=True)
            try:
                filtered_df = dataframe_explorer(df[show_cols], case=False)
                st.dataframe(filtered_df, use_container_width=True, height=500)
            except Exception:
                st.dataframe(df[show_cols], use_container_width=True, height=500)
    # 여기 
    elif selected_viz == "전사 OT 현황":
        st.markdown("<h3 style='text-align:center;'>25년도 전사 OT 현황</h3>", unsafe_allow_html=True)

        # 연장근무, 수당 관련 컬럼
        overtime_cols = [col for col in df.columns if re.match(r"^\d{1,2}월_연장근무$", col)]
        pay_cols = [col for col in df.columns if "연장근무수당" in col]

        monthly_overtime_avg = df[overtime_cols].mean().mean() if overtime_cols else 0
        avg_ot_pay = df[pay_cols].mean().mean() if pay_cols else 123456  # 실제 값 없으면 가상값

        # 좌측(카드 2개) + 우측(라인 그래프)
        col1, col2 = st.columns([1, 2])

        with col1:
            # (전사) 연장근무 월 평균 카드
            st.markdown(f"""
                <div style="background-color:#f8fafc; border-radius:12px; padding:16px; text-align:center; margin-bottom:16px; margin-top:60px;">
                    <p style="color:#64748b;">(전사) 연장근무 월 평균</p>
                    <div style="display:flex; align-items:center; justify-content:center; gap:8px;">
                        <span style="font-size:1.8rem;">⏳</span>
                        <span style="font-size:2rem; font-weight:700; color:#2563eb;">{monthly_overtime_avg:.2f}</span>
                        <span style="font-size:1rem; color:#64748b;">시간/월</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # (전사) 평균 OT 수당 카드
            st.markdown(f"""
                <div style="background-color:#fffbea; border-radius:12px; padding:16px; text-align:center;">
                    <p style="color:#a16207;">(전사) 평균 OT 수당</p>
                    <div style="display:flex; align-items:center; justify-content:center; gap:8px;">
                        <span style="font-size:1.8rem;">💰</span>
                        <span style="font-size:2rem; font-weight:700; color:#facc15;">{"1,234,567"}</span>
                        <span style="font-size:1rem; color:#a16207;">원/인</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            # (전사) 연장근무 추이 (라인 그래프)
            months, monthly_avg = [], []
            for col in overtime_cols:
                try:
                    month = col.split("_")[0]
                    months.append(month)
                    monthly_avg.append(df[col].mean())
                except Exception:
                    continue

            chart_data = pd.DataFrame({"월": months, "연장근무 평균": monthly_avg}).sort_values(by="월")
            if not chart_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_data["월"], y=chart_data["연장근무 평균"],
                    mode="lines+markers+text", name="연장근무 평균",
                    text=[f"{v:.2f}" for v in chart_data["연장근무 평균"]],
                    textposition="top center",
                    line=dict(color="#2563eb", width=3),
                    marker=dict(size=6, color="#2563eb")
                ))
                fig.update_layout(
                    title="(전사) 연장근무 추이",
                    title_x=0.5,
                    title_font=dict(size=18, family="Arial", color="black"),
                    template="simple_white",
                    height=400, margin=dict(l=20, r=20, t=60, b=40),
                    yaxis=dict(title="평균 시간", showgrid=True, gridcolor="lightgrey"),
                    xaxis=dict(
                        title="월", showgrid=True, gridcolor="lightgrey",
                        tickfont=dict(size=12, family="Arial", color="black"),
                        tickvals=chart_data["월"],
                        ticktext=[f"<b>{m}</b>" for m in chart_data["월"]]
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

        # 구분선
        st.divider()

        # --- 전사 OT Stacked Bar + Line ---
        months, overtime_avg, night_avg, holiday_avg, total_ot_avg = [], [], [], [], []
        for m in range(1, 13):
            month = f"{m}월"
            o = df[f"{month}_연장근무"].mean() if f"{month}_연장근무" in df.columns else 0
            n = df[f"{month}_야간근무"].mean() if f"{month}_야간근무" in df.columns else 0
            h = df[f"{month}_휴일근무"].mean() if f"{month}_휴일근무" in df.columns else 0
            overtime_avg.append(o)
            night_avg.append(n)
            holiday_avg.append(h)
            total_ot_avg.append(o + n + h)
            months.append(month)

        stacked_df = pd.DataFrame({
            "월": months, "연장근무": overtime_avg, "휴일근무": holiday_avg, "야간근무": night_avg
        })

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=stacked_df["월"], y=stacked_df["연장근무"], name="연장근무", marker_color="#2563eb"))
        fig2.add_trace(go.Bar(x=stacked_df["월"], y=stacked_df["휴일근무"], name="휴일근무", marker_color="#facc15"))
        fig2.add_trace(go.Bar(x=stacked_df["월"], y=stacked_df["야간근무"], name="야간근무", marker_color="#ef4444"))
        fig2.add_trace(go.Scatter(
            x=stacked_df["월"], y=total_ot_avg,
            mode="lines+markers+text", name="월 연장근무 시간",
            line=dict(color="#1f2937", width=3, dash="dot"),
            marker=dict(size=7, color="#1f2937"),
            text=[f"{v:.1f}" for v in total_ot_avg],
            textposition="top center"
        ))
        fig2.update_layout(
            title="인당 월평균 OT 수당 및 시간 현황",
            title_x=0.4,
            title_y=0.95,  # 기본보다 조금 위쪽
            title_font=dict(size=22, family="Arial", color="black"),
            barmode="stack", height=450, template="simple_white",
            yaxis_title="근무시간 (시간)", xaxis_title="월",
            xaxis=dict(
                tickvals=stacked_df["월"],
                ticktext=[f"<b>{m}</b>" for m in stacked_df["월"]],
                tickfont=dict(size=12, family="Arial", color="black")
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=60, r=60, t=100, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)




    # LLM 분석 UI: llm_ready 가 True인 경우에만
    if st.session_state.get("llm_ready") and df is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <h3 style='text-align:center; margin-bottom:0;'>RPA Assistant</h3>
            <p style='text-align:center; font-size:14px; color:#7f8c8d; margin-top:4px;'>
                Robotic Process Automation 기반 데이터 질의응답
            </p>
            """,
            unsafe_allow_html=True
        )

        # 세션 상태 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 이전 대화 출력
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 사용자 입력
        user_input = st.chat_input("질문을 입력하세요 (예: 건설/개발 직군의 3월 OT 평균은?)")
        if user_input:
            avatar = os.path.join(BASE_DIR, "assets", "cj_company.png")
            with st.chat_message("user",
                avatar=avatar):
                st.markdown(user_input)
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "avatar": avatar
            })

            # LLM 응답
            session_id = "rpa_session"
            raw_answer = generate_answer_rag(user_input, session_id=session_id)
            final_answer = post_process_answer(user_input, raw_answer, llm)
            
            HARI_ICON = os.path.join(BASE_DIR, "assets", "HARI_ICON.png")
            
            with st.chat_message("assistant",
                avatar=HARI_ICON):
                st.markdown(final_answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "avatar": HARI_ICON
            })
