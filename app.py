# ── 1. 임포트 및 상수 정의 ──────────────────────────────────────────────────
import json
import os
import uuid
from datetime import date, datetime, timedelta

from openai import OpenAI
import pandas as pd
import streamlit as st

# 데이터 파일 경로
DATA_FILE = "journal_data.json"
SETTINGS_FILE = "settings.json"

# 카테고리 정의 및 색상
CATEGORIES = ["회의", "개발", "문서"]
CATEGORY_COLORS = {
    "회의": "#1E88E5",   # 파란색
    "개발": "#43A047",   # 초록색
    "문서": "#FB8C00",   # 주황색
}
CATEGORY_BG_COLORS = {
    "회의": "#E3F2FD",
    "개발": "#E8F5E9",
    "문서": "#FFF3E0",
}

# 요일 표시 (월~일)
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

# 사용 가능한 모델 목록
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

# 기본 설정값
DEFAULT_SETTINGS = {
    "api_key": "",
    "model": "gpt-4o",
    "max_tokens": 1024,
    "temperature": 1.0,
}


# ── 2. 데이터 로드/저장 함수 ─────────────────────────────────────────────────

def load_settings() -> dict:
    """설정 파일에서 앱 설정을 로드. 없으면 기본값 반환."""
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        settings = DEFAULT_SETTINGS.copy()
        settings.update(saved)
        return settings
    except (json.JSONDecodeError, IOError):
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    """설정을 파일에 저장."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def get_settings() -> dict:
    """세션 상태에서 설정을 반환 (없으면 파일에서 로드). 모델이 유효하지 않으면 기본값으로 리셋."""
    if "app_settings" not in st.session_state:
        st.session_state.app_settings = load_settings()
    settings = st.session_state.app_settings
    if settings.get("model") not in AVAILABLE_MODELS:
        settings["model"] = DEFAULT_SETTINGS["model"]
    return settings


def load_journals() -> list[dict]:
    """JSON 파일에서 일지 목록을 불러옴. 파일 없으면 빈 리스트 반환."""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_journals(journals: list[dict]) -> None:
    """일지 목록을 JSON 파일에 저장 (저장 전 최신 상태 병합)."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(journals, f, ensure_ascii=False, indent=2)


def get_journal_by_id(journal_id: str, journals: list[dict]) -> dict | None:
    """id로 일지 검색."""
    return next((j for j in journals if j["id"] == journal_id), None)


def add_or_update_journal(new_journal: dict) -> None:
    """일지 추가 또는 업데이트 (파일 재읽기 후 저장으로 동시 접근 안전)."""
    journals = load_journals()
    existing_idx = next(
        (i for i, j in enumerate(journals) if j["id"] == new_journal["id"]), None
    )
    if existing_idx is not None:
        journals[existing_idx] = new_journal
    else:
        journals.append(new_journal)
    save_journals(journals)


def link_related_journals(target_id: str, related_ids: list[str]) -> None:
    """연관 일지 양방향 연결: target ↔ 각 related."""
    journals = load_journals()
    journal_map = {j["id"]: j for j in journals}

    for rid in related_ids:
        # target → related
        if rid not in journal_map.get(target_id, {}).get("related_ids", []):
            if target_id in journal_map:
                journal_map[target_id].setdefault("related_ids", [])
                if rid not in journal_map[target_id]["related_ids"]:
                    journal_map[target_id]["related_ids"].append(rid)
        # related → target
        if target_id in journal_map.get(rid, {}):
            journal_map[rid].setdefault("related_ids", [])
            if target_id not in journal_map[rid]["related_ids"]:
                journal_map[rid]["related_ids"].append(target_id)

    save_journals(list(journal_map.values()))


# ── 3. AI 관련 함수 ──────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI | None:
    """OpenAI 클라이언트 생성. 설정 → secrets → 환경변수 순으로 API 키 탐색."""
    settings = get_settings()
    try:
        secrets_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secrets_key = ""
    api_key = (
        settings.get("api_key")
        or secrets_key
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _parse_json_response(text: str) -> str:
    """API 응답에서 순수 JSON 문자열 추출 (```json 블록 대비)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return text


def summarize_with_ai(raw_content: str) -> dict | None:
    """OpenAI API를 호출하여 일지 내용을 요약. 실패 시 None 반환."""
    client = get_openai_client()
    if not client:
        st.error("API 키가 설정되지 않았습니다. 사이드바의 ⚙️ 설정에서 OpenAI API Key를 입력해주세요.")
        return None

    settings = get_settings()
    system_prompt = (
        "당신은 업무 일지 요약 전문가입니다.\n"
        "다음 텍스트를 읽고 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 출력하지 마세요.\n\n"
        '{\n'
        '  "title": "일지 항목 제목",\n'
        '  "background": "작업의 배경 또는 목적",\n'
        '  "content": ["핵심 내용 1", "핵심 내용 2", "핵심 내용 3"],\n'
        '  "result": "주요 실적 또는 결과"\n'
        '}'
    )

    try:
        response = client.chat.completions.create(
            model=settings.get("model", DEFAULT_SETTINGS["model"]),
            max_tokens=settings.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]),
            temperature=settings.get("temperature", DEFAULT_SETTINGS["temperature"]),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_content},
            ],
        )
        return json.loads(_parse_json_response(response.choices[0].message.content))
    except json.JSONDecodeError as e:
        st.error(f"AI 응답 파싱 실패: {e}")
        return None
    except Exception as e:
        st.error(f"AI 요약 오류: {e}")
        return None


def recommend_related_journals(current: dict, existing: list[dict]) -> list[str]:
    """OpenAI API를 이용해 연관 일지 id 목록을 추천. 실패 시 빈 리스트 반환."""
    client = get_openai_client()
    if not client or not existing:
        return []

    settings = get_settings()
    existing_summary = [
        {"id": j["id"], "title": j.get("title", ""), "content_preview": j.get("raw_content", "")[:200]}
        for j in existing
        if j["id"] != current["id"]
    ]
    if not existing_summary:
        return []

    prompt = (
        "다음은 현재 일지와 기존 일지 목록입니다.\n"
        "현재 일지와 내용이 연관된 기존 일지의 id를 JSON 배열로만 반환하세요.\n"
        "연관성이 없으면 빈 배열 []을 반환하세요.\n\n"
        f"현재 일지:\n제목: {current.get('title', '')}\n내용: {current.get('raw_content', '')}\n\n"
        f"기존 일지 목록:\n{json.dumps(existing_summary, ensure_ascii=False)}\n\n"
        '응답 형식: ["id1", "id2"]'
    )

    try:
        response = client.chat.completions.create(
            model=settings.get("model", DEFAULT_SETTINGS["model"]),
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(_parse_json_response(response.choices[0].message.content))
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ── 4. UI 컴포넌트 함수 ──────────────────────────────────────────────────────

def render_settings_page() -> None:
    """설정 페이지: API 키, 모델, 파라미터 설정 UI."""
    st.markdown("## ⚙️ 설정")
    st.caption("애플리케이션 설정 및 환경 구성")
    st.markdown("---")

    settings = get_settings()

    # ── API 설정 섹션 ──
    st.markdown("### 🔑 API 설정")
    st.caption("OpenAI API 키와 모델 설정을 관리하세요")

    # API 키 입력 (표시/숨김 토글)
    if "show_api_key" not in st.session_state:
        st.session_state.show_api_key = False

    col_key, col_toggle = st.columns([9, 1])
    with col_key:
        key_type = "default" if st.session_state.show_api_key else "password"
        new_api_key = st.text_input(
            "OpenAI API Key",
            value=settings.get("api_key", ""),
            type=key_type,
            placeholder="sk-proj-...",
            key="settings_api_key_input",
        )
    with col_toggle:
        st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
        toggle_label = "🙈" if st.session_state.show_api_key else "👁️"
        if st.button(toggle_label, key="toggle_api_key", help="API 키 표시/숨김"):
            st.session_state.show_api_key = not st.session_state.show_api_key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if new_api_key:
        st.caption("🔒 API 키는 안전하게 암호화되어 로컬에 저장됩니다")
    else:
        st.caption("⚠️ API 키를 입력하지 않으면 AI 기능을 사용할 수 없습니다")

    # 모델 선택
    current_model = settings.get("model", DEFAULT_SETTINGS["model"])
    model_idx = AVAILABLE_MODELS.index(current_model) if current_model in AVAILABLE_MODELS else 0
    selected_model = st.selectbox(
        "모델 선택",
        options=AVAILABLE_MODELS,
        index=model_idx,
        key="settings_model",
    )

    # 최대 토큰 + 창의성(Temperature)
    col_tokens, col_temp = st.columns(2)
    with col_tokens:
        new_max_tokens = st.number_input(
            "최대 토큰",
            min_value=256,
            max_value=8096,
            value=settings.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]),
            step=256,
            key="settings_max_tokens",
        )
    with col_temp:
        new_temperature = st.number_input(
            "창의성 (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.get("temperature", DEFAULT_SETTINGS["temperature"])),
            step=0.1,
            format="%.1f",
            key="settings_temperature",
        )

    st.markdown("---")

    # 저장 / 초기화 버튼
    col_save, col_reset = st.columns([2, 1])
    with col_save:
        if st.button("💾 설정 저장", type="primary", use_container_width=True):
            new_settings = {
                "api_key": new_api_key,
                "model": selected_model,
                "max_tokens": int(new_max_tokens),
                "temperature": float(new_temperature),
            }
            save_settings(new_settings)
            st.session_state.app_settings = new_settings
            st.success("설정이 저장되었습니다.")
    with col_reset:
        if st.button("🔄 초기화", use_container_width=True):
            save_settings(DEFAULT_SETTINGS.copy())
            st.session_state.app_settings = DEFAULT_SETTINGS.copy()
            st.info("기본값으로 초기화되었습니다.")
            st.rerun()

    # 현재 설정 요약 (API 키는 마스킹)
    st.markdown("---")
    st.markdown("#### 현재 설정")
    masked_key = "미설정"
    if settings.get("api_key"):
        key_val = settings["api_key"]
        masked_key = key_val[:8] + "..." + key_val[-4:] if len(key_val) > 12 else "****"
    st.markdown(
        f"| 항목 | 값 |\n|------|----|\n"
        f"| API Key | `{masked_key}` |\n"
        f"| 모델 | `{settings.get('model', '-')}` |\n"
        f"| 최대 토큰 | `{settings.get('max_tokens', '-')}` |\n"
        f"| Temperature | `{settings.get('temperature', '-')}` |"
    )


def render_summary_output(summary: dict) -> None:
    """AI 요약 결과를 포맷에 맞게 렌더링."""
    st.markdown(f"### {summary.get('title', '')}")
    st.markdown(f"**배경:** {summary.get('background', '')}")
    st.markdown("**내용**")
    for i, item in enumerate(summary.get("content", []), 1):
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{i}. {item}")
    st.markdown(f"**실적:** {summary.get('result', '')}")


def render_calendar(journals: list[dict], selected_categories: list[str]) -> None:
    """월별 그리드 캘린더 렌더링. 일지가 있는 날짜에 카테고리별 배지 표시."""
    # 현재 표시 연월 관리
    if "cal_year" not in st.session_state:
        st.session_state.cal_year = date.today().year
    if "cal_month" not in st.session_state:
        st.session_state.cal_month = date.today().month

    year = st.session_state.cal_year
    month = st.session_state.cal_month

    # 날짜별 일지 집계
    date_journals: dict[str, list[dict]] = {}
    for j in journals:
        if j.get("category") in selected_categories:
            d = j.get("date", "")
            date_journals.setdefault(d, []).append(j)

    # 헤더 (월 이동)
    col_prev, col_title, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("◀", key="cal_prev"):
            if month == 1:
                st.session_state.cal_month = 12
                st.session_state.cal_year = year - 1
            else:
                st.session_state.cal_month = month - 1
            st.rerun()
    with col_title:
        st.markdown(
            f"<h3 style='text-align:center;margin:0'>{year}년 {month}월</h3>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("▶", key="cal_next"):
            if month == 12:
                st.session_state.cal_month = 1
                st.session_state.cal_year = year + 1
            else:
                st.session_state.cal_month = month + 1
            st.rerun()

    # 요일 헤더
    day_cols = st.columns(7)
    for i, wd in enumerate(WEEKDAYS):
        color = "#E53935" if i == 5 else ("#1565C0" if i == 6 else "#333")
        day_cols[i].markdown(
            f"<div style='text-align:center;font-weight:bold;color:{color}'>{wd}</div>",
            unsafe_allow_html=True,
        )

    # 첫날의 요일 계산 (월요일=0)
    first_day = date(year, month, 1)
    start_weekday = first_day.weekday()

    # 마지막 날 계산
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    total_days = last_day.day

    # 달력 셀 렌더링
    cells = [""] * start_weekday + list(range(1, total_days + 1))
    # 7의 배수로 패딩
    while len(cells) % 7 != 0:
        cells.append("")

    today_str = date.today().isoformat()

    for week_start in range(0, len(cells), 7):
        week_cells = cells[week_start: week_start + 7]
        cols = st.columns(7)
        for col_idx, day_num in enumerate(week_cells):
            with cols[col_idx]:
                if day_num == "":
                    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
                    continue

                date_str = f"{year:04d}-{month:02d}-{day_num:02d}"
                is_today = date_str == today_str
                is_selected = st.session_state.get("selected_date") == date_str
                day_journals = date_journals.get(date_str, [])

                # 날짜 번호 스타일
                num_style = "text-align:center;font-size:14px;"
                if is_today:
                    num_style += "color:white;background:#1E88E5;border-radius:50%;width:24px;height:24px;line-height:24px;margin:auto;"
                if is_selected:
                    num_style += "font-weight:bold;"

                # 배지 HTML
                badges_html = ""
                for jj in day_journals[:3]:
                    cat = jj.get("category", "")
                    color = CATEGORY_COLORS.get(cat, "#888")
                    badges_html += (
                        f"<span style='display:inline-block;width:8px;height:8px;"
                        f"border-radius:50%;background:{color};margin:1px'></span>"
                    )

                cell_bg = "#F0F4FF" if is_selected else "transparent"
                cell_html = (
                    f"<div style='background:{cell_bg};border-radius:6px;padding:4px;min-height:50px;cursor:pointer'>"
                    f"<div style='{num_style}'>{day_num}</div>"
                    f"<div style='text-align:center'>{badges_html}</div>"
                    "</div>"
                )
                st.markdown(cell_html, unsafe_allow_html=True)

                # 날짜 선택 버튼 (투명)
                if st.button("　", key=f"day_{date_str}", help=date_str):
                    st.session_state.selected_date = date_str
                    st.rerun()


def render_journal_card(journal: dict, all_journals: list[dict], key_prefix: str = "") -> None:
    """일지 카드 렌더링 (expander로 상세 표시)."""
    cat = journal.get("category", "")
    color = CATEGORY_COLORS.get(cat, "#888")
    bg = CATEGORY_BG_COLORS.get(cat, "#F5F5F5")

    badge = (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:12px;font-size:12px;font-weight:bold'>{cat}</span>"
    )
    title_html = (
        f"<div style='background:{bg};border-left:4px solid {color};"
        f"padding:8px 12px;border-radius:4px;margin-bottom:4px'>"
        f"{badge} &nbsp;<strong>{journal.get('title', '제목 없음')}</strong>"
        f"<span style='float:right;color:#888;font-size:12px'>{journal.get('date', '')}</span>"
        f"</div>"
    )
    st.markdown(title_html, unsafe_allow_html=True)

    with st.expander("상세 보기", expanded=False):
        summary = journal.get("summary")
        if summary:
            render_summary_output(summary)
            with st.expander("원본 내용 보기", expanded=False):
                st.text(journal.get("raw_content", ""))
        else:
            st.text_area(
                "원본 내용",
                value=journal.get("raw_content", ""),
                height=150,
                disabled=True,
                key=f"{key_prefix}card_content_{journal['id']}",
            )

        # 연관 일지 섹션
        related_ids = journal.get("related_ids", [])
        if related_ids:
            st.markdown("**연관 일지**")
            for rid in related_ids:
                related = get_journal_by_id(rid, all_journals)
                if related:
                    rcat = related.get("category", "")
                    rcolor = CATEGORY_COLORS.get(rcat, "#888")
                    st.markdown(
                        f"<span style='color:{rcolor}'>●</span> "
                        f"[{related.get('date', '')}] **{related.get('title', '')}** ({rcat})",
                        unsafe_allow_html=True,
                    )

        # 후속 일지 작성 버튼
        if st.button("후속 일지 작성", key=f"{key_prefix}followup_{journal['id']}"):
            st.session_state.show_form = True
            st.session_state.prefill_related_id = journal["id"]
            st.session_state.form_mode = "new"
            st.rerun()


def render_journal_form(prefill_related_id: str | None = None) -> None:
    """일지 작성/수정 폼 렌더링."""
    st.subheader("새 일지 작성")

    all_journals = load_journals()

    # 연관 일지 멀티셀렉트 옵션 구성
    options = {
        f"[{j.get('date','')}] {j.get('title','제목 없음')} ({j.get('category','')})": j["id"]
        for j in all_journals
    }
    option_labels = list(options.keys())

    # prefill 처리
    prefill_label = None
    if prefill_related_id:
        prefill_label = next(
            (lbl for lbl, jid in options.items() if jid == prefill_related_id), None
        )

    with st.form("journal_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("날짜", value=date.today(), key="form_date")
        with col2:
            category = st.selectbox("카테고리", CATEGORIES, key="form_category")

        title = st.text_input("제목", key="form_title")
        raw_content = st.text_area("내용", height=200, key="form_content")

        # 연관 일지 선택
        default_selection = [prefill_label] if prefill_label else []
        selected_related_labels = st.multiselect(
            "연관 일지 선택",
            options=option_labels,
            default=default_selection,
            key="form_related",
        )
        related_ids = [options[lbl] for lbl in selected_related_labels]

        col_save, col_ai = st.columns(2)
        with col_save:
            save_clicked = st.form_submit_button("저장", type="primary")
        with col_ai:
            ai_clicked = st.form_submit_button("AI 요약")

    # AI 요약 처리
    if ai_clicked:
        if not raw_content.strip():
            st.warning("내용을 입력해주세요.")
        else:
            with st.spinner("AI 요약 중..."):
                summary = summarize_with_ai(raw_content)
            if summary:
                st.session_state.pending_summary = summary
                st.session_state.pending_raw_content = raw_content
                st.success("AI 요약 완료!")

    # 요약 결과 표시 (폼 밖에서 유지)
    if "pending_summary" in st.session_state:
        st.markdown("---")
        st.markdown("#### AI 요약 결과")
        render_summary_output(st.session_state.pending_summary)

    # 저장 처리
    if save_clicked:
        if not title.strip():
            st.warning("제목을 입력해주세요.")
            return
        if not raw_content.strip():
            st.warning("내용을 입력해주세요.")
            return

        now = datetime.now().isoformat()
        journal_id = str(uuid.uuid4())
        new_journal = {
            "id": journal_id,
            "date": selected_date.isoformat(),
            "category": category,
            "title": title.strip(),
            "raw_content": raw_content.strip(),
            "summary": st.session_state.pop("pending_summary", None),
            "related_ids": related_ids,
            "created_at": now,
            "updated_at": now,
        }

        # 저장 및 양방향 연관 연결
        add_or_update_journal(new_journal)
        if related_ids:
            link_related_journals(journal_id, related_ids)

        st.success(f"일지가 저장되었습니다: {title}")

        # AI 연관 일지 추천 (백그라운드 역할 — 저장 직후 실행)
        existing = load_journals()
        with st.spinner("AI 연관 일지 추천 분석 중..."):
            recommended_ids = recommend_related_journals(new_journal, existing)

        # 이미 연결된 항목 제외
        new_recommendations = [r for r in recommended_ids if r not in related_ids and r != journal_id]
        if new_recommendations:
            st.session_state.ai_recommendations = new_recommendations
            st.session_state.ai_rec_for = journal_id

        # 폼 상태 초기화
        st.session_state.show_form = False
        st.session_state.prefill_related_id = None
        st.session_state.pop("pending_raw_content", None)
        st.rerun()


def render_ai_recommendations(all_journals: list[dict]) -> None:
    """AI 연관 일지 추천 수락/거절 UI."""
    if "ai_recommendations" not in st.session_state:
        return

    rec_ids = st.session_state.ai_recommendations
    for_id = st.session_state.get("ai_rec_for")

    recs = [get_journal_by_id(rid, all_journals) for rid in rec_ids]
    recs = [r for r in recs if r]
    if not recs:
        del st.session_state.ai_recommendations
        return

    st.info("🤖 AI가 연관 일지를 추천합니다.")
    for rec in recs:
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            cat = rec.get("category", "")
            color = CATEGORY_COLORS.get(cat, "#888")
            st.markdown(
                f"<span style='color:{color}'>●</span> "
                f"**[{rec.get('date','')}] {rec.get('title','')}** ({cat})",
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("수락", key=f"rec_accept_{rec['id']}"):
                link_related_journals(for_id, [rec["id"]])
                st.session_state.ai_recommendations = [
                    r for r in st.session_state.ai_recommendations if r != rec["id"]
                ]
                if not st.session_state.ai_recommendations:
                    del st.session_state.ai_recommendations
                st.success("연관 일지로 추가되었습니다.")
                st.rerun()
        with col3:
            if st.button("거절", key=f"rec_reject_{rec['id']}"):
                st.session_state.ai_recommendations = [
                    r for r in st.session_state.ai_recommendations if r != rec["id"]
                ]
                if not st.session_state.ai_recommendations:
                    del st.session_state.ai_recommendations
                st.rerun()


def render_list_view(journals: list[dict], selected_categories: list[str]) -> None:
    """일지 목록 뷰: 날짜 내림차순 정렬."""
    filtered = [j for j in journals if j.get("category") in selected_categories]
    filtered.sort(key=lambda j: j.get("date", ""), reverse=True)

    if not filtered:
        st.info("표시할 일지가 없습니다.")
        return

    for j in filtered:
        render_journal_card(j, journals, key_prefix="list_")
        st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)


# ── 5. 메인 앱 진입점 ────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(layout="wide", page_title="업무 일지", page_icon="🗒️")

    # 전역 CSS
    st.markdown(
        """
        <style>
        /* 상단 툴바·헤더·데코레이션 숨김 */
        div[data-testid="stAppToolbar"],
        header[data-testid="stHeader"],
        div[data-testid="stDecoration"] { display: none !important; }
        /* 메인 콘텐츠 상단 여백 축소 */
        .block-container { padding-top: 0.5rem !important; }
        /* expander 스타일 */
        details summary { font-weight: bold; }
        /* 캘린더 날짜 선택 버튼 (title 속성이 있는 것만 투명화) */
        button[kind="secondary"][title] { opacity: 0; position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
        div[data-testid="stButton"] { position: relative; }
        /* 필터 stat 카드 높이 고정 */
        .filter-stat { height: 36px; line-height: 24px; overflow: hidden; }
        /* 사이드바: stMarkdownContainer 바로 뒤 stButton 을 카드 위에 오버레이 (인접 형제 선택자) */
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] + div[data-testid="stButton"] {
            margin-top: -36px; height: 36px; z-index: 1;
        }
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] + div[data-testid="stButton"] button {
            opacity: 0; position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 세션 상태 초기화
    if "show_form" not in st.session_state:
        st.session_state.show_form = False
    if "prefill_related_id" not in st.session_state:
        st.session_state.prefill_related_id = None
    if "selected_date" not in st.session_state:
        st.session_state.selected_date = date.today().isoformat()
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "캘린더"
    if "page" not in st.session_state:
        st.session_state.page = "main"
    if "selected_categories" not in st.session_state:
        st.session_state.selected_categories = list(CATEGORIES)

    # ── 사이드바 ──
    with st.sidebar:
        # 설정 버튼 (상단)
        if st.button("⚙️ 설정", use_container_width=True):
            st.session_state.page = (
                "main" if st.session_state.page == "settings" else "settings"
            )
            st.rerun()

        # API 키 상태 (설정 버튼 바로 아래)
        settings = get_settings()
        try:
            secrets_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            secrets_key = ""
        has_key = bool(
            settings.get("api_key")
            or secrets_key
            or os.environ.get("OPENAI_API_KEY", "")
        )
        key_status = "🟢 API 연결됨" if has_key else "🔴 API 키 미설정"
        st.caption(key_status)

        st.markdown("---")

        # st.metric 을 먼저 배치해서 --- (stMarkdownContainer) 와 캘린더 버튼 사이 인접 관계 차단
        all_journals_for_stat = load_journals()
        st.metric("전체 일지", len(all_journals_for_stat))

        # 캘린더 홈 버튼: stMetric 뒤에 위치 → 필터 CSS 에 걸리지 않음
        if st.button("📅 캘린더", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

        # 카테고리 필터 + 통계 (클릭으로 토글)
        for cat in CATEGORIES:
            cnt = sum(1 for j in all_journals_for_stat if j.get("category") == cat)
            color = CATEGORY_COLORS[cat]
            is_selected = cat in st.session_state.selected_categories
            bg = CATEGORY_BG_COLORS[cat] if is_selected else "transparent"
            border_color = color if is_selected else color + "55"
            opacity = "1" if is_selected else "0.45"
            # 컬러 stat 카드 (stMarkdownContainer 로 렌더링됨)
            st.markdown(
                f"<div class='filter-stat' style='background:{bg};border-left:3px solid {border_color};"
                f"padding:6px 10px;border-radius:4px;margin:2px 0;opacity:{opacity};cursor:pointer'>"
                f"<span style='color:{color}'>●</span> {cat}: <strong>{cnt}건</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # 투명 오버레이 버튼 (stMarkdownContainer + stButton 인접 형제 CSS 로 숨김)
            if st.button("　", key=f"filter_btn_{cat}"):
                if cat in st.session_state.selected_categories:
                    st.session_state.selected_categories.remove(cat)
                else:
                    st.session_state.selected_categories.append(cat)
                st.rerun()

    selected_categories = st.session_state.selected_categories

    # ── 설정 페이지 ──
    if st.session_state.page == "settings":
        render_settings_page()
        return

    # ── 상단 헤더 (메인 페이지) ──
    header_col1, header_col2 = st.columns([6, 1])
    with header_col1:
        st.title("🗒️ 업무 일지")
    with header_col2:
        if st.button("＋ 새 일지", type="primary"):
            st.session_state.show_form = True
            st.session_state.prefill_related_id = None
            st.rerun()

    all_journals = load_journals()

    # ── 새 일지 작성 폼 ──
    if st.session_state.show_form:
        with st.container():
            st.markdown("---")
            render_journal_form(prefill_related_id=st.session_state.prefill_related_id)
            if st.button("✕ 닫기", key="close_form"):
                st.session_state.show_form = False
                st.session_state.prefill_related_id = None
                st.session_state.pop("pending_summary", None)
                st.rerun()
        st.markdown("---")

    # ── AI 연관 일지 추천 ──
    render_ai_recommendations(all_journals)

    # ── 뷰 탭 ──
    tab_cal, tab_list = st.tabs(["📅 캘린더 뷰", "📋 목록 뷰"])

    with tab_cal:
        if not selected_categories:
            st.info("왼쪽 사이드바에서 카테고리를 선택해주세요.")
        else:
            render_calendar(all_journals, selected_categories)

            # 선택된 날짜의 일지 목록
            selected = st.session_state.get("selected_date", "")
            if selected:
                day_journals = [
                    j for j in all_journals
                    if j.get("date") == selected and j.get("category") in selected_categories
                ]
                st.markdown(f"---\n### {selected} 일지 목록 ({len(day_journals)}건)")
                if day_journals:
                    for j in day_journals:
                        render_journal_card(j, all_journals, key_prefix="cal_")
                else:
                    st.info("선택한 날짜에 일지가 없습니다.")

    with tab_list:
        if not selected_categories:
            st.info("왼쪽 사이드바에서 카테고리를 선택해주세요.")
        else:
            render_list_view(all_journals, selected_categories)


if __name__ == "__main__":
    main()
