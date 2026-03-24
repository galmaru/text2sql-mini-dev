"""
Vercel 서버리스 진입점
- ChromaDB 데이터를 읽기 가능한 /tmp로 복사 후 app 초기화
- SQLite DB 파일도 배포에 포함 (상위 3개 대용량 제외)
"""
import os
import sys
import shutil
from pathlib import Path

# 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Vercel 환경: 쓰기 가능한 경로는 /tmp뿐
# chromadb, openai, vanna 등이 홈 디렉토리에 파일을 쓰려다 errno 30 발생
VERCEL = os.environ.get("VERCEL", "")
if VERCEL:
    TMP = Path("/tmp")

    # 모든 라이브러리의 홈/캐시/설정 경로를 /tmp로 리디렉션
    os.environ["HOME"]             = str(TMP)
    os.environ["TMPDIR"]           = str(TMP)
    os.environ["XDG_CONFIG_HOME"]  = str(TMP / ".config")
    os.environ["XDG_CACHE_HOME"]   = str(TMP / ".cache")
    os.environ["XDG_DATA_HOME"]    = str(TMP / ".local" / "share")

    # ChromaDB 텔레메트리 비활성화
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    os.environ["CHROMA_TELEMETRY"]     = "false"

    # /tmp 하위 디렉토리 미리 생성
    for d in [".config", ".cache", ".local/share"]:
        (TMP / d).mkdir(parents=True, exist_ok=True)

    # ChromaDB 벡터 DB를 /tmp로 복사 (읽기 전용 배포 경로 → 쓰기 가능 /tmp)
    src = ROOT / ".vanna_chroma_per_db"
    dst = TMP / ".vanna_chroma_per_db"
    if src.exists() and not dst.exists():
        shutil.copytree(str(src), str(dst))
    os.environ["CHROMA_ROOT_OVERRIDE"] = str(dst)

from app import create_app  # noqa: E402

app = create_app()
