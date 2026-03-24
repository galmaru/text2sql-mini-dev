"""
Text2SQL 웹 앱 (vanna.ai + GPT-4o + Bird-SQL mini_dev)
실행: OPENAI_API_KEY=sk-... .venv/bin/python3 app.py
"""

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from flask import Flask, jsonify, render_template, request
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore

# ─── 설정 ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_FILE   = SCRIPT_DIR / "finetuning/inference/mini_dev_prompt.jsonl"
DB_BASE_DIR = SCRIPT_DIR / "llm/mini_dev_data/minidev/MINIDEV/dev_databases"
_chroma_override = os.environ.get("CHROMA_ROOT_OVERRIDE")
CHROMA_ROOT = Path(_chroma_override) if _chroma_override else SCRIPT_DIR / ".vanna_chroma_per_db"

DB_LIST = [
    "california_schools",
    "debit_card_specializing",
    "financial",
    "formula_1",
    "student_club",
    "superhero",
    "thrombosis_prediction",
    "toxicology",
]

# ─── Vanna 클래스 ──────────────────────────────────────────────────────────────

class BirdVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self._last_rag = {}
        self._last_history = {}

    def _query_collection(self, collection, question):
        """ChromaDB 컬렉션 직접 쿼리 — documents + distances 동시 반환"""
        n = min(getattr(self, "n_results", 10), collection.count())
        if n == 0:
            return [], []
        results   = collection.query(query_texts=[question], n_results=n)
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances",  [[]])[0]
        return documents, distances

    def get_related_ddl(self, question, **kwargs):
        """관련 DDL 검색 결과 + 유사도 캡처"""
        try:
            docs, dists = self._query_collection(self.ddl_collection, question)
            self._last_rag["ddl"]        = docs
            self._last_rag["ddl_scores"] = dists
            return docs
        except Exception:
            result = super().get_related_ddl(question, **kwargs)
            self._last_rag["ddl"]        = result
            self._last_rag["ddl_scores"] = []
            return result

    def get_similar_question_sql(self, question, **kwargs):
        """유사 Q&A 검색 결과 + 유사도 캡처"""
        try:
            docs, dists = self._query_collection(self.sql_collection, question)
            # vanna가 저장한 형식: "question: ... sql: ..." 또는 dict
            parsed = []
            for d in docs:
                if isinstance(d, dict):
                    parsed.append(d)
                else:
                    # "question: ...\nsql: ..." 파싱 시도
                    q_part, s_part = "", d
                    if "\nsql:" in d.lower():
                        parts = d.split("\n", 1)
                        q_part = parts[0].replace("question:", "").strip()
                        s_part = parts[1].replace("sql:", "").strip() if len(parts) > 1 else d
                    parsed.append({"question": q_part, "sql": s_part})
            self._last_rag["similar_questions"]    = parsed
            self._last_rag["similar_questions_raw"] = docs
            self._last_rag["sq_scores"]            = dists
            return parsed
        except Exception:
            result = super().get_similar_question_sql(question, **kwargs)
            self._last_rag["similar_questions"] = result
            self._last_rag["sq_scores"]         = []
            return result

    def get_related_documentation(self, question, **kwargs):
        """관련 문서 검색 결과 + 유사도 캡처"""
        try:
            docs, dists = self._query_collection(self.documentation_collection, question)
            self._last_rag["documentation"]        = docs
            self._last_rag["documentation_scores"] = dists
            return docs
        except Exception:
            result = super().get_related_documentation(question, **kwargs)
            self._last_rag["documentation"]        = result
            self._last_rag["documentation_scores"] = []
            return result

    def submit_prompt(self, prompt, **kwargs) -> str:
        """프롬프트와 응답 캡처 (History용)"""
        response = super().submit_prompt(prompt, **kwargs)
        self._last_history = {"messages": prompt, "response": response}
        return response

    def get_sql_prompt(self, initial_prompt, question, question_sql_list, ddl_list, doc_list, **kwargs):
        """항상 SQL만 반환하도록 가이드라인 오버라이드"""
        if initial_prompt is None:
            initial_prompt = (
                "You are a SQLite expert. "
                "Generate a SQL query to answer the question. "
                "Your response should ONLY be based on the given context. "
            )

        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list, max_tokens=self.max_tokens)
        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=self.max_tokens)
        initial_prompt += (
            "===Response Guidelines\n"
            "1. Always generate a valid SQLite SQL query — never return explanations.\n"
            "2. If the table or column name doesn't exactly match, use the most semantically similar one.\n"
            "3. Use JOINs when multiple tables are needed.\n"
            "4. Output only the SQL query, no markdown, no explanation.\n"
        )

        message_log = [self.system_message(initial_prompt)]
        for q_sql in question_sql_list:
            message_log.append(self.user_message(q_sql["question"]))
            message_log.append(self.assistant_message(q_sql["sql"]))
        message_log.append(self.user_message(question))
        return message_log


# ─── DB별 Vanna 인스턴스 관리 ──────────────────────────────────────────────────

app = Flask(__name__)
vn_instances: dict[str, BirdVanna] = {}


def make_vanna(db_id: str, api_key: str) -> BirdVanna:
    chroma_path = CHROMA_ROOT / db_id
    return BirdVanna(config={
        "api_key": api_key,
        "model": "gpt-4o",
        "path": str(chroma_path),
    })


def init_vanna_per_db():
    """DB별로 독립된 ChromaDB에 스키마 학습"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY 환경변수가 필요합니다.")
        sys.exit(1)

    # JSONL에서 DB별 스키마 수집 (파일 없으면 건너뜀 — Vercel 환경 등)
    db_schemas: dict[str, str] = {}
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            for line in f:
                item = json.loads(line)
                db_id = item["db_id"]
                if db_id not in db_schemas:
                    db_schemas[db_id] = item["schema"]
        print(f"총 {len(db_schemas)}개 DB 초기화 중...\n")
    else:
        print("DATA_FILE 없음 — 사전 학습된 ChromaDB만 사용\n")

    for db_id in DB_LIST:
        chroma_path = CHROMA_ROOT / db_id
        trained_flag = chroma_path / ".trained"

        vn = make_vanna(db_id, api_key)
        vn_instances[db_id] = vn

        if trained_flag.exists():
            print(f"  [{db_id}] 기존 ChromaDB 재사용")
            continue

        schema = db_schemas.get(db_id)
        if not schema:
            print(f"  [{db_id}] 스키마 없음 — 건너뜀")
            continue

        print(f"  [{db_id}] 학습 중...")
        vn.train(ddl=schema)
        trained_flag.touch()
        print(f"  [{db_id}] 완료")

    print(f"\n모든 DB 초기화 완료\n")


# ─── SQLite 유틸 ───────────────────────────────────────────────────────────────

def find_db_path(db_id: str):
    for ext in ["sqlite", "db"]:
        for p in DB_BASE_DIR.rglob(f"{db_id}.{ext}"):
            return p
    return None


def execute_sql(db_id: str, sql: str):
    db_path = find_db_path(db_id)
    if db_path is None:
        return None, "DB 파일 없음"

    sql = re.sub(r'\bLEFT\s*\(',      'SUBSTR(', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bSUBSTRING\s*\(', 'SUBSTR(', sql, flags=re.IGNORECASE)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows    = cur.fetchmany(100)
        columns = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        return {"columns": columns, "rows": [list(r) for r in rows]}, None
    except Exception as e:
        return None, str(e)


# ─── 라우트 ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    db_status = {db_id: find_db_path(db_id) is not None for db_id in DB_LIST}
    return render_template("index.html", db_list=DB_LIST, db_status=db_status)


@app.route("/api/generate", methods=["POST"])
def generate():
    data     = request.json
    question = data.get("question", "").strip()
    db_id    = data.get("db_id", "")

    if not question:
        return jsonify({"error": "질문을 입력해주세요."}), 400
    if db_id not in vn_instances:
        return jsonify({"error": f"알 수 없는 DB: {db_id}"}), 400

    vn = vn_instances[db_id]   # DB 전용 RAG 인스턴스 사용
    vn._last_rag = {}          # 매 요청마다 초기화

    try:
        sql = vn.generate_sql(question)
    except Exception as e:
        import traceback
        return jsonify({"error": f"SQL 생성 실패: {e}", "traceback": traceback.format_exc()}), 500

    sql_keywords = ("select", "insert", "update", "delete", "with", "create", "drop", "pragma")
    if not sql or not sql.strip().lower().startswith(sql_keywords):
        return jsonify({"error": f"SQL을 생성할 수 없습니다: {sql}"}), 200

    result, exec_error = execute_sql(db_id, sql)

    # RAG 검색 결과 수집 (내용 + 유사도 점수)
    rag = {
        "ddl":                    vn._last_rag.get("ddl", []),
        "ddl_scores":             vn._last_rag.get("ddl_scores", []),
        "similar_questions":      vn._last_rag.get("similar_questions", []),
        "similar_questions_raw":  vn._last_rag.get("similar_questions_raw", []),
        "sq_scores":              vn._last_rag.get("sq_scores", []),
        "documentation":          vn._last_rag.get("documentation", []),
        "documentation_scores":   vn._last_rag.get("documentation_scores", []),
    }

    # LLM 대화 이력 수집
    history = []
    if hasattr(vn, "_last_history") and vn._last_history:
        for msg in vn._last_history["messages"]:
            history.append({"role": msg["role"], "content": msg["content"]})
        history.append({
            "role": "assistant",
            "content": vn._last_history["response"],
            "is_final": True,
        })

    return jsonify({"sql": sql, "result": result, "exec_error": exec_error, "history": history, "rag": rag})


@app.route("/api/execute", methods=["POST"])
def execute():
    """수정된 SQL 직접 실행"""
    data   = request.json
    sql    = data.get("sql", "").strip()
    db_id  = data.get("db_id", "")

    if not sql:
        return jsonify({"error": "SQL을 입력해주세요."}), 400

    result, exec_error = execute_sql(db_id, sql)
    return jsonify({"result": result, "exec_error": exec_error})


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def create_app():
    init_vanna_per_db()
    return app

if __name__ == "__main__":
    create_app()
    print("서버 시작: http://localhost:5001\n")
    app.run(debug=False, port=5001)
