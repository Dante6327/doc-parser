import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from converter import DocumentParserService
from llm_router import get_llm_client
from models import ParseResponse

load_dotenv()

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx", ".ppt",
    ".xlsx", ".xls",
    ".md",
    ".txt",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 Docling 모델 사전 로딩 (최초 1회, 이후 재사용)"""
    print("[DocParser] Docling 모델 초기화 중...")
    app.state.parser = DocumentParserService()
    print("[DocParser] 초기화 완료. 서비스 시작.")
    yield
    print("[DocParser] 서비스 종료.")


app = FastAPI(
    title="Doc Parser Microservice",
    description="Docling + Gemini/Ollama 기반 문서 분석 마이크로서비스",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/parse", response_model=ParseResponse)
async def parse_document(
    file: UploadFile = File(..., description="분석할 문서 (PDF, XLSX, PPTX)"),
    x_llm_provider: Annotated[
        str | None,
        Header(description="사용할 LLM provider. 'gemini' 또는 'ollama'"),
    ] = None,
):
    # ── 1. Provider 헤더 검증 ──────────────────────────────────────────────
    if not x_llm_provider:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "MISSING_PROVIDER_HEADER",
                "message": "X-LLM-Provider 헤더가 필요합니다.",
                "allowed_values": ["gemini", "ollama"],
            },
        )

    provider = x_llm_provider.strip().lower()
    if provider not in ("gemini", "ollama"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_PROVIDER",
                "message": f"지원하지 않는 provider: '{x_llm_provider}'",
                "allowed_values": ["gemini", "ollama"],
            },
        )

    # ── 2. 파일 형식 검증 ─────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "UNSUPPORTED_FILE_TYPE",
                "message": f"지원하지 않는 파일 형식: '{suffix}'",
                "allowed_extensions": list(ALLOWED_EXTENSIONS),
            },
        )

    # ── 3. LLM 클라이언트 생성 (요청마다 provider에 맞게) ──────────────────
    try:
        llm_client = get_llm_client(provider)
    except ValueError as e:
        raise HTTPException(status_code=500, detail={"error": "LLM_INIT_FAILED", "message": str(e)})

    # ── 4. 임시 파일 저장 후 파싱 ─────────────────────────────────────────
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await app.state.parser.parse(
            file_path=tmp_path,
            filename=file.filename,
            llm_client=llm_client,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "PARSE_FAILED", "message": str(e)},
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development",
        workers=1,  # Docling 모델은 단일 프로세스에서 공유
    )
