"""
한국어 질의로 SQL 생성 테스트
.venv/bin/python3 test_korean.py
"""

import os
import sys
from pathlib import Path
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore


class BirdVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


SCRIPT_DIR = Path(__file__).parent
CHROMA_PATH = SCRIPT_DIR / ".vanna_chroma"

# 한국어 테스트 질문 (Bird-SQL mini_dev 기반)
KOREAN_QUESTIONS = [
    {
        "db": "debit_card_specializing",
        "question": "EUR로 결제하는 고객과 CZK로 결제하는 고객의 비율은?",
        "gold_sql": "SELECT CAST(SUM(IIF(Currency = 'EUR', 1, 0)) AS FLOAT) / SUM(IIF(Currency = 'CZK', 1, 0)) AS ratio FROM customers",
    },
    {
        "db": "debit_card_specializing",
        "question": "2012년에 LAM 세그먼트에서 소비량이 가장 적은 고객은 누구인가요?",
        "gold_sql": "SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = 'LAM' AND SUBSTR(T2.Date, 1, 4) = '2012' GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) ASC LIMIT 1",
    },
    {
        "db": "debit_card_specializing",
        "question": "CZK로 결제한 고객들의 가스 소비량이 가장 많았던 연도는?",
        "gold_sql": "SELECT SUBSTR(T2.Date, 1, 4) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Currency = 'CZK' GROUP BY SUBSTR(T2.Date, 1, 4) ORDER BY SUM(T2.Consumption) DESC LIMIT 1",
    },
    {
        "db": "california_schools",
        "question": "SAT 점수가 가장 높은 학교의 이름은?",
        "gold_sql": None,  # 참고용 (정답 없음)
    },
    {
        "db": "superhero",
        "question": "Marvel Comics에서 출판된 슈퍼히어로는 몇 명인가요?",
        "gold_sql": None,
    },
]


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY 환경변수가 필요합니다.")
        sys.exit(1)

    if not CHROMA_PATH.exists():
        print("ERROR: 먼저 test_vanna.py를 실행해서 스키마를 학습시켜 주세요.")
        sys.exit(1)

    vn = BirdVanna(config={
        "api_key": api_key,
        "model": "gpt-4o",
        "path": str(CHROMA_PATH),
    })

    print("=" * 60)
    print("한국어 질의 → SQL 생성 테스트")
    print("=" * 60)

    for i, item in enumerate(KOREAN_QUESTIONS, 1):
        print(f"\n[{i}] DB: {item['db']}")
        print(f"  질문: {item['question']}")

        try:
            sql = vn.generate_sql(item["question"])
        except Exception as e:
            sql = f"ERROR: {e}"

        print(f"  생성 SQL:\n    {sql}")
        if item["gold_sql"]:
            print(f"  정답 SQL:\n    {item['gold_sql']}")
        print()


if __name__ == "__main__":
    main()
