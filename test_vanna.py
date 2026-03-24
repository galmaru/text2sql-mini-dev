"""
vanna.ai + GPT-4 + Bird-SQL mini_dev 테스트 스크립트

사전 준비:
  export OPENAI_API_KEY="sk-..."
  .venv/bin/python3 test_vanna.py

옵션:
  --limit N      : 테스트할 질문 수 (기본: 10)
  --db DB_ID     : 특정 DB만 테스트 (기본: 전체)
  --no-exec      : SQL 실행 없이 생성만 테스트
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# vanna legacy API 사용
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore


# ─── Vanna 클래스 정의 ────────────────────────────────────────────────────────

class BirdVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


# ─── 설정 ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "finetuning/inference/mini_dev_prompt.jsonl"
DB_DIR = SCRIPT_DIR / "llm/mini_dev_data"   # SQLite .sqlite 파일 위치
CHROMA_PATH = SCRIPT_DIR / ".vanna_chroma"  # ChromaDB 저장 경로


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data(db_filter=None):
    items = []
    with open(DATA_FILE) as f:
        for line in f:
            item = json.loads(line)
            if db_filter is None or item["db_id"] == db_filter:
                items.append(item)
    return items


# ─── DB별 SQLite 연결 ─────────────────────────────────────────────────────────

def get_sqlite_path(db_id: str):
    """DB_DIR 하위에서 db_id.sqlite 파일을 재귀 탐색"""
    for ext in ["sqlite", "db"]:
        for p in DB_DIR.rglob(f"{db_id}.{ext}"):
            return p
    return None


def execute_sql(db_id: str, sql: str):
    """SQLite에서 SQL 실행 후 결과 반환. DB 파일 없으면 None 반환."""
    db_path = get_sqlite_path(db_id)
    if db_path is None:
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return f"ERROR: {e}"


# ─── vanna 초기화 및 스키마 학습 ──────────────────────────────────────────────

def init_vanna(data: list, force_retrain: bool = False) -> BirdVanna:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    vn = BirdVanna(config={
        "api_key": api_key,
        "model": "gpt-4o",
        "path": str(CHROMA_PATH),
    })

    # ChromaDB에 이미 학습된 데이터가 있으면 재사용
    trained_flag = CHROMA_PATH / ".trained"
    if trained_flag.exists() and not force_retrain:
        print(f"기존 학습 데이터 재사용: {CHROMA_PATH}")
        return vn

    print("DB 스키마 학습 중...")
    trained_dbs = set()
    for item in data:
        db_id = item["db_id"]
        if db_id not in trained_dbs:
            # DDL 학습
            vn.train(ddl=item["schema"])
            # 정답 SQL + 질문을 Few-shot 예시로 추가 (선택)
            # vn.train(question=item["question"], sql=item["SQL"])
            trained_dbs.add(db_id)
            print(f"  학습 완료: {db_id}")

    trained_flag.touch()
    print(f"총 {len(trained_dbs)}개 DB 학습 완료\n")
    return vn


# ─── 테스트 실행 ──────────────────────────────────────────────────────────────

def run_test(vn: BirdVanna, data: list, limit: int, no_exec: bool):
    test_data = data[:limit]
    results = []
    correct = 0

    print(f"{'='*60}")
    print(f"테스트 시작: {len(test_data)}개 질문")
    print(f"{'='*60}\n")

    for i, item in enumerate(test_data, 1):
        qid = item["question_id"]
        db_id = item["db_id"]
        question = item["question"]
        evidence = item.get("evidence", "")
        gold_sql = item["SQL"]
        difficulty = item["difficulty"]

        # evidence가 있으면 질문에 포함
        full_question = f"{evidence}\n{question}" if evidence else question

        print(f"[{i}/{len(test_data)}] Q#{qid} ({db_id}, {difficulty})")
        print(f"  질문: {question}")

        try:
            generated_sql = vn.generate_sql(full_question)
        except Exception as e:
            generated_sql = f"ERROR: {e}"

        print(f"  생성 SQL: {generated_sql}")
        print(f"  정답 SQL: {gold_sql}")

        result = {
            "question_id": qid,
            "db_id": db_id,
            "question": question,
            "difficulty": difficulty,
            "generated_sql": generated_sql,
            "gold_sql": gold_sql,
        }

        # SQL 실행 비교 (DB 파일 있을 때만)
        if not no_exec and not generated_sql.startswith("ERROR"):
            gen_result = execute_sql(db_id, generated_sql)
            gold_result = execute_sql(db_id, gold_sql)

            if gen_result is None:
                print(f"  실행: DB 파일 없음 (SQL 생성만 테스트)")
                result["exec_match"] = None
            elif isinstance(gen_result, str) and gen_result.startswith("ERROR"):
                print(f"  실행 오류: {gen_result}")
                result["exec_match"] = False
            else:
                match = str(sorted(str(r) for r in gen_result)) == \
                        str(sorted(str(r) for r in (gold_result or [])))
                result["exec_match"] = match
                if match:
                    correct += 1
                    print(f"  실행 결과: 일치")
                else:
                    print(f"  실행 결과: 불일치")
                    print(f"    생성: {gen_result[:3] if gen_result else '[]'}")
                    print(f"    정답: {(gold_result or [])[:3]}")

        results.append(result)
        print()

    # 요약
    print(f"{'='*60}")
    print(f"테스트 완료: {len(results)}개")
    exec_results = [r for r in results if r.get("exec_match") is not None]
    if exec_results:
        acc = sum(1 for r in exec_results if r["exec_match"]) / len(exec_results)
        print(f"실행 정확도(EX): {acc:.1%} ({sum(1 for r in exec_results if r['exec_match'])}/{len(exec_results)})")
    else:
        print("SQL 생성만 테스트 (DB 파일 없음 — 실행 정확도 미측정)")

    # 결과 저장
    output_file = SCRIPT_DIR / "vanna_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {output_file}")

    return results


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="vanna.ai Bird-SQL mini_dev 테스트")
    parser.add_argument("--limit", type=int, default=10, help="테스트할 질문 수 (기본: 10)")
    parser.add_argument("--db", type=str, default=None, help="특정 DB만 테스트 (예: debit_card_specializing)")
    parser.add_argument("--no-exec", action="store_true", help="SQL 실행 없이 생성만 테스트")
    parser.add_argument("--retrain", action="store_true", help="ChromaDB 재학습")
    args = parser.parse_args()

    # 데이터 로드
    data = load_data(db_filter=args.db)
    if not data:
        print(f"ERROR: 데이터를 찾을 수 없습니다. db={args.db}")
        sys.exit(1)
    print(f"데이터 로드: {len(data)}개 질문")

    # DB 파일 존재 여부 안내
    sample_db = get_sqlite_path(data[0]["db_id"])
    if sample_db is None:
        print(f"SQLite DB 파일 없음 — SQL 생성만 테스트합니다.")
        print(f"DB 파일이 있으면 {DB_DIR}/<db_id>/<db_id>.sqlite 경로에 위치시키세요.\n")
        args.no_exec = True

    # vanna 초기화
    vn = init_vanna(data, force_retrain=args.retrain)

    # 테스트 실행
    run_test(vn, data, limit=args.limit, no_exec=args.no_exec)


if __name__ == "__main__":
    main()
