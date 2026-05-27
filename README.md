# 🗒️ 업무 일지 관리 앱

Streamlit 기반 업무 일지 관리 웹 애플리케이션입니다.

## 기술 스택

| 항목 | 라이브러리 |
|------|-----------|
| UI 프레임워크 | `streamlit >= 1.32.0` |
| AI 요약 | `anthropic >= 0.25.0` (claude-sonnet-4-20250514) |
| 데이터 처리 | `pandas >= 2.0.0` |
| 영속성 저장 | `json` (로컬 파일: `journal_data.json`) |

## 주요 기능

### 📅 캘린더 뷰
- 월별 그리드 캘린더로 일지 현황 한눈에 확인
- 카테고리별 색상 배지 표시 — 회의(파랑), 개발(초록), 문서(주황)
- 날짜 클릭 시 해당 날짜의 일지 목록 즉시 표시
- 이전/다음 월 이동 버튼

### 📋 목록 뷰
- 전체 일지를 날짜 내림차순으로 정렬하여 표시
- 카테고리 카드형 UI (컬러 보더 및 배지)

### ✏️ 일지 작성
- 날짜, 카테고리(회의/개발/문서), 제목, 내용 입력 폼
- 연관 일지 수동 선택 (multiselect)
- 저장 시 `journal_data.json`에 영속 보관

### 🤖 AI 요약 (Anthropic API)
- 작성 폼의 "AI 요약" 버튼 클릭으로 수동 트리거
- 입력 내용을 아래 구조화 포맷으로 자동 요약

  ```
  {제목}
  배경: {배경 내용}
  내용
    1. {핵심 내용 1}
    2. {핵심 내용 2}
    3. {핵심 내용 3}
  실적: {실적 내용}
  ```

- 저장 버튼 클릭 시 요약 결과가 일지의 `summary` 필드에 함께 저장

### 🔗 연관 일지 관리
- **수동 연결**: 작성 폼에서 multiselect로 기존 일지 선택 → 양방향 자동 연결
- **AI 자동 추천**: 저장 후 Anthropic API가 유사 일지를 분석·추천, 수락/거절 선택 가능
- **후속 일지 작성**: 상세 카드의 버튼 클릭 시 현재 일지가 연관 일지로 pre-select된 작성 폼으로 이동

### 🔍 필터 및 통계
- 사이드바에서 카테고리별 표시 여부 필터링
- 카테고리별 일지 건수 통계 표시

## 파일 구조

```
app.py              # 메인 Streamlit 앱 (단일 파일)
journal_data.json   # 데이터 영속 저장 파일 (자동 생성)
requirements.txt    # 의존성 목록
```

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. Anthropic API 키 설정

**.streamlit/secrets.toml** 파일 생성:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

또는 환경변수로 설정:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. 앱 실행

```bash
streamlit run app.py
```

## 데이터 구조

```json
{
  "id": "uuid4 문자열",
  "date": "YYYY-MM-DD",
  "category": "회의 | 개발 | 문서",
  "title": "제목",
  "raw_content": "원본 입력 텍스트",
  "summary": {
    "title": "요약 제목",
    "background": "배경",
    "content": ["내용1", "내용2", "내용3"],
    "result": "실적"
  },
  "related_ids": ["연관 일지 id 목록"],
  "created_at": "ISO 8601 datetime",
  "updated_at": "ISO 8601 datetime"
}
```
