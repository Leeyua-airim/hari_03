# 파일명: llm_hub/rpa_llm/llm_handler.py
import os
import re
import unicodedata
import uuid
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# =========================
# 1. 환경 변수 및 API 키
# =========================
BASE_DIR = os.path.dirname(__file__)
load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error("환경 변수 OPENAI_API_KEY가 없습니다. .env 확인 요망.")
    st.stop()

# =========================
# 2. 데이터 로드
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    # path = "llm_hub/rpa_llm/data/암호화_시연용_OT_1to7.xlsx"
    path = os.path.join(BASE_DIR, "data", "암호화_시연용_OT_1to7.xlsx")
    return pd.read_excel(path)

df = load_data()

# =========================
# 3. 벡터스토어 구축
# =========================
@st.cache_resource(show_spinner=True)
def get_vectorstore(source_df: pd.DataFrame) -> FAISS:
    main_cols = ["성명", "사번", "직군", "직급", "경력경로", "부서"]
    used_cols = [c for c in main_cols if c in source_df.columns]
    num_cols = [c for c in source_df.columns if re.match(r"\d{1,2}월_.+", c)]

    chunks = []
    for start in range(0, len(source_df), 100):
        sub_df = source_df.iloc[start:start+100][used_cols + num_cols]
        text = sub_df.astype(str).to_csv(index=False)
        if len(text) > 1000:
            text = text[:1000] + "\n...(중략)"
        chunks.append(Document(page_content=text))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embeddings)

vectorstore = get_vectorstore(df)

def retrieve_context_limited(question: str, k: int = 5, max_chars: int = 3000) -> str:
    results = vectorstore.max_marginal_relevance_search(question, k=k, fetch_k=20)
    context_parts, total_len = [], 0
    for doc in results:
        content = doc.page_content
        if total_len + len(content) > max_chars:
            remain = max_chars - total_len
            if remain > 0:
                context_parts.append(content[:remain] + "\n...(생략)")
            break
        context_parts.append(content)
        total_len += len(content)
    return "\n\n".join(context_parts)

# =========================
# 4. 퍼지 매칭 기반 파서
# =========================
def _normalize_str(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).strip().lower().replace(" ", "")

def build_vocab(source_df: pd.DataFrame) -> dict:
    names = sorted(set(map(str, source_df["성명"].dropna().unique())))
    groups = sorted(set(map(str, source_df["직군"].dropna().unique()))) if "직군" in source_df.columns else []
    grades = sorted(set(map(str, source_df["직급"].dropna().unique()))) if "직급" in source_df.columns else []
    paths = sorted(set(map(str, source_df["경력경로"].dropna().unique()))) if "경력경로" in source_df.columns else []

    metrics_map = {
        "기본근무": "기본근무",
        "기본": "기본근무",
        "연장합계": "연장합계",
        "연장": "연장근무",
        "연장근무": "연장근무",
        "야간": "야간근무",
        "야간근무": "야간근무",
        "휴일": "휴일근무",
        "휴일근무": "휴일근무"
}


    def make_norm_table(values):
        return {_normalize_str(v): v for v in values}

    return {
        "names_table": make_norm_table(names),
        "groups_table": make_norm_table(groups),
        "grades_table": make_norm_table(grades),
        "paths_table": make_norm_table(paths),
        "metrics_map": metrics_map,
    }

VOCAB = build_vocab(df)

def fuzzy_pick(query_norm: str, table: dict, score_cutoff: int) -> tuple:
    if not table: return None, 0
    candidates = list(table.keys())
    result = process.extractOne(query_norm, candidates, scorer=fuzz.partial_ratio, score_cutoff=score_cutoff)
    if result:
        norm_key, score, _ = result
        return table[norm_key], score
    return None, 0

def parse_months(raw_q: str) -> list[int]:
    matches = re.findall(r"(\d{1,2})\s*월", raw_q)
    months = []
    for m in matches:
        m_int = int(m)
        if 1 <= m_int <= 12:
            months.append(m_int)
    return months


def parse_metric(raw_q: str, metrics_map: dict) -> str | None:
    for alias, canonical in metrics_map.items():
        if alias in raw_q: return canonical
    return None

def answer_top_n(parsed: dict, source_df: pd.DataFrame, n: int = 10) -> dict:
    """월, 지표 기준으로 상위 N명 추출"""
    month, metric = parsed["month"], parsed["metric"]
    if not month or not metric:
        return {"ok": False, "reason": "월/지표 식별 실패", "used_col": None, "values": []}

    col = f"{month}월_{metric}"
    if col not in source_df.columns:
        return {"ok": False, "reason": f"컬럼 없음: {col}", "used_col": col, "values": []}

    # 직군 필터링 (있으면 그 직군만, 없으면 전체)
    if parsed["group"]:
        subset = source_df[source_df["직군"] == parsed["group"]]
    else:
        subset = source_df

    # 상위 N명 정렬
    top_n = subset[["성명", col]].sort_values(by=col, ascending=False).head(n)

    values = [{"name": row["성명"], "value": row[col]} for _, row in top_n.iterrows()]
    return {"ok": True, "used_col": col, "values": values}


def extract_multiple_names(raw_q: str, names_table: dict) -> list:
    q_norm = _normalize_str(raw_q)
    return [orig for norm_key, orig in names_table.items() if norm_key in q_norm]

def extract_multiple_groups(raw_q: str, group_table: dict) -> list:
    q_norm = _normalize_str(raw_q)
    return [orig for norm_key, orig in group_table.items() if norm_key in q_norm]

# def extract_filters(question: str, vocab: dict) -> dict:
    # q_norm = _normalize_str(question)
    # name, ns = fuzzy_pick(q_norm, vocab["names_table"], 88)
    # group, gs = fuzzy_pick(q_norm, vocab["groups_table"], 80)
    # grade, ds = fuzzy_pick(q_norm, vocab["grades_table"], 80)
    # path, ps = fuzzy_pick(q_norm, vocab["paths_table"], 80)
    # months = parse_months(question)
    # metric = parse_metric(question, vocab["metrics_map"])
    # return {
        # "name": name, "group": group, "grade": grade, "path": path,
        # "months": months, "metric": metric, "raw_question": question,
        # "confidence": {"name": ns, "group": gs}
    # }


def extract_filters(question: str, vocab: dict) -> dict:
    q_norm = _normalize_str(question)
    name, ns = fuzzy_pick(q_norm, vocab["names_table"], 88)
    group, gs = fuzzy_pick(q_norm, vocab["groups_table"], 80)
    grade, ds = fuzzy_pick(q_norm, vocab["grades_table"], 80)
    path, ps = fuzzy_pick(q_norm, vocab["paths_table"], 80)
    months = parse_months(question)  # e.g. [1] or empty list
    metric = parse_metric(question, vocab["metrics_map"])
    return {
        "name": name,
        "group": group,
        "grade": grade,
        "path": path,
        "months": months,
        "month": months[0] if months else None,  # 추가된 부분
        "metric": metric,
        "raw_question": question,
        "confidence": {"name": ns, "group": gs}
    }




# =========================
# 5. 이름/직군 처리 분리
# =========================
def answer_multiple_names(parsed: dict, source_df: pd.DataFrame) -> dict:
    months, metric = parsed["months"], parsed["metric"]
    if not months or not metric:
        return {"ok": False, "reason": "월/지표 식별 실패", "values": []}

    values = []
    for m in months:
        col = f"{m}월_{metric}"
        if col not in source_df.columns:
            values.append({"month": m, "name": None, "value": None})
            continue

        names = extract_multiple_names(parsed["raw_question"], VOCAB["names_table"])
        if not names:
            if parsed["name"]: names = [parsed["name"]]
            else: continue

        for n in names:
            row = source_df[source_df["성명"] == n]
            val = float(row[col].iloc[0]) if not row.empty else None
            values.append({"month": m, "name": n, "value": val})

    return {"ok": True, "used_col": f"{metric}", "values": values}

def answer_multiple_groups(parsed: dict, source_df: pd.DataFrame) -> dict:
    """여러 직군의 월별 평균값 계산"""
    months, metric = parsed.get("months"), parsed.get("metric")

    if not months or not metric:
        return {"ok": False, "reason": "월/지표 식별 실패", "used_col": None, "values": []}

    groups = extract_multiple_groups(parsed["raw_question"], VOCAB["groups_table"])
    if not groups:
        if parsed.get("group"):
            groups = [parsed["group"]]
        else:
            return {"ok": False, "reason": "질문에 직군명이 없음", "used_col": None, "values": []}

    values = []
    for m in months:
        col = f"{m}월_{metric}"
        if col not in source_df.columns:
            # 해당 월의 컬럼이 없을 경우 None 처리
            for g in groups:
                values.append({"months": m, "group": g, "avg": None})
            continue

        for g in groups:
            subset = source_df[source_df["직군"] == g]
            avg_val = float(subset[col].mean()) if not subset.empty else None
            values.append({"months": m, "group": g, "avg": avg_val})

    return {"ok": True, "used_col": metric, "values": values}



# =========================
# 6. 프롬프트/LLM
# =========================
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 CJ대한통운 건설부문 데이터 분석 AI HARI다. "
               "2025년 통합 근무시간 데이터를 바탕으로 간결하고 명확하게 보고서를 작성한다."
               "숫자는 소수점 둘째 자리까지 표기한다."),
    ("user", "질문:\n{question}\n\n<context>\n{context}\n\n<근거>\n{found_row}\n\n<집계>\n{agg_value}")
])

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

_session_store: dict[str, InMemoryChatMessageHistory] = {}

def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]

# =========================
# 7. 응답 생성
# =========================
def generate_answer_rag(question: str, session_id: str) -> str:
    context = retrieve_context_limited(question)
    parsed = extract_filters(question, VOCAB)

    # "상위 N명" 패턴 탐지
    m = re.search(r"상위\s*(\d+)", question)
    if m:
        n = int(m.group(1))
        agg = answer_top_n(parsed, df, n=n)
    elif extract_multiple_names(parsed["raw_question"], VOCAB["names_table"]):
        agg = answer_multiple_names(parsed, df)
    else:
        agg = answer_multiple_groups(parsed, df)

    if agg["ok"] and agg["values"]:
        found_rows, parts = [], []
        for item in agg["values"]:
            if "name" in item:
                label, val = item["name"], item["value"]
                parts.append(f"{label}: {val:.2f}" if val is not None else f"{label}: 계산 불가")
                found_rows.append(f"<성명: {label}> {agg['used_col']} = {val}")
            else:
                label, val = item["group"], item["avg"]
                parts.append(f"{label}: {val:.2f}" if val is not None else f"{label}: 계산 불가")
                found_rows.append(f"<직군: {label}> 평균 {agg['used_col']} = {val}")
        found_row = "\n".join(found_rows)
        agg_value = f"{agg['used_col']} 값 = " + ", ".join(parts)
    else:
        found_row, agg_value = f"[집계 실패] {agg.get('reason','원인 미상')}", "데이터 없음"

    chain = RunnableWithMessageHistory(
        prompt | llm,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    result = chain.invoke(
        {"question": question, "context": context, "found_row": found_row, "agg_value": agg_value},
        config={"configurable": {"session_id": session_id}}
    )
    return getattr(result, "content", str(result))



def run_rpa_llm(user_input: str, session_id: str):
    """
    Streamlit 채팅 UI에서 사용할 수 있도록 generator 형태로 LLM 응답을 반환

    Parameters
    - user_input: 사용자 입력 질문
    - session_id: 세션 식별자

    Yields
    - str: 응답 텍스트 청크
    """
    response_text = generate_answer_rag(user_input, session_id)

    # 단순 generator로 감싸서 반환
    for chunk in response_text.split():
        yield chunk + " "


# ------------------------
# 1. 질문 정제 (Question Refiner)
# ------------------------
def refine_question(question: str, llm) -> str:
    """사용자 질문을 간결하고 보고서 제목으로 적합하게 다듬는다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 전문 보고서용 질문 리라이팅 어시스턴트다. "
                   "사용자의 질문을 간결하고 이해하기 쉬운 제목/문장으로 다듬어라."),
        ("user", "{question}")
    ])
    chain = prompt | llm
    result = chain.invoke({"question": question})
    return getattr(result, "content", str(result))


# ------------------------
# 2. 답변 후처리 (Report Formatter)
# ------------------------
def post_process_answer(question: str, raw_answer: str, llm) -> str:
    """정제된 질문과 원시 답변을 바탕으로 보고서 형식으로 꾸민다."""
    refined_q = refine_question(question, llm)

    post_process_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "너는 전문 보고서 작성 보조 AI다. "
         "아래의 '정제된 질문'과 '분석 답변'을 바탕으로 "
         "보고서 형식의 출력물을 작성한다.\n\n"
         "출력 형식은 반드시 다음과 같다:\n\n"
         "#### 질문\n{refined_question}\n\n"
         "#### 분석 결과\n(분석 답변 요약)\n\n"
         "#### 제언\n(보충 설명, 추가 분석 방향 제안)"),
        ("user", 
         "정제된 질문: {refined_question}\n\n"
         "분석 답변: {raw_answer}\n\n"
         "위 내용을 바탕으로 보고서를 작성하라.")
    ])

    chain = post_process_prompt | llm
    result = chain.invoke({
        "refined_question": refined_q,
        "raw_answer": raw_answer
    })
    return getattr(result, "content", str(result))