"""
Vercel 서버리스 진입점
- ChromaDB 데이터를 읽기 가능한 /tmp로 복사 후 app 초기화
- SQLite 실행은 DB 파일 부재로 동작하지 않음 (SQL 생성만 지원)
"""
import os
import sys
import shutil
from pathlib import Path

# 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Vercel 환경: ChromaDB는 /tmp(쓰기 가능)에 복사해야 함
VERCEL = os.environ.get("VERCEL", "")
if VERCEL:
    # ChromaDB가 홈 디렉토리에 텔레메트리/설정 파일을 쓰려다 errno 30 발생 방지
    os.environ["HOME"] = "/tmp"
    os.environ["ANONYMIZED_TELEMETRY"] = "false"

    src = ROOT / ".vanna_chroma_per_db"
    dst = Path("/tmp/.vanna_chroma_per_db")
    if src.exists() and not dst.exists():
        shutil.copytree(str(src), str(dst))
    os.environ["CHROMA_ROOT_OVERRIDE"] = str(dst)

from app import create_app  # noqa: E402

app = create_app()
