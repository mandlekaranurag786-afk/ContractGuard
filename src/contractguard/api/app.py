"""Application factory: uvicorn --factory contractguard.api.app:create_app."""

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile

from contractguard.checker import ContractChecker
from contractguard.extraction.extractor import Extractor
from contractguard.extraction.prompt import PRODUCT_PROMPT_VERSION
from contractguard.ingestion.pdf import PDFIngestionError
from contractguard.llm.cache import CachedLLMClient
from contractguard.llm.errors import LLMQuotaError
from contractguard.llm.gemini import GeminiClient
from contractguard.pipeline import load_document
from contractguard.report import ContractReport
from contractguard.retrieval.vector import warm_up_model
from contractguard.rules.models import load_rules

logger = logging.getLogger("contractguard.api")


@dataclass(frozen=True)
class ApiSettings:
    api_key: str | None
    max_upload_bytes: int
    max_pages: int
    cache_dir: Path

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.environ.get("CONTRACTGUARD_API_KEY") or None,
            max_upload_bytes=int(
                float(os.environ.get("CONTRACTGUARD_MAX_UPLOAD_MB", "10")) * 1024**2
            ),
            max_pages=int(os.environ.get("CONTRACTGUARD_MAX_PAGES", "200")),
            cache_dir=Path(
                os.environ.get("CONTRACTGUARD_CACHE_DIR", ".cache/api_llm_cache")
            ),
        )


def api_error(status, code, message, headers=None):
    return HTTPException(
        status_code=status, detail={"code": code, "message": message}, headers=headers
    )


def create_app(
    checker=None, policy_path=None, settings=None, document_loader=load_document
) -> FastAPI:
    settings = settings if settings is not None else ApiSettings.from_env()
    if settings.api_key is None:
        logger.warning("API key not set; auth disabled")
    policy_path = Path(
        policy_path
        if policy_path is not None
        else os.environ.get("CONTRACTGUARD_POLICY", "policies/eval_policy.json")
    )
    rules = load_rules(policy_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if checker is None:
            client = GeminiClient()
            extractor = Extractor(
                CachedLLMClient(
                    client,
                    model=client.model,
                    cache_dir=settings.cache_dir,
                ),
                prompt_version=PRODUCT_PROMPT_VERSION,
            )
            app.state.checker = ContractChecker(extractor)
            warm_up_model()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.checker = checker
    # One uvicorn worker only: each worker has its own lock and model.
    app.state.check_lock = threading.Lock()

    def authorize(x_api_key: str | None = Header(default=None)):
        if settings.api_key is not None and (
            x_api_key is None
            or not secrets.compare_digest(
                x_api_key.encode("utf-8"), settings.api_key.encode("utf-8")
            )
        ):
            raise api_error(401, "unauthorized", "A valid API key is required.")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "prompt_version": PRODUCT_PROMPT_VERSION,
            "rules": len(rules),
            "auth": settings.api_key is not None,
            "max_upload_mb": settings.max_upload_bytes / 1024**2,
            "max_pages": settings.max_pages,
        }

    @app.get("/policy", dependencies=[Depends(authorize)])
    def policy():
        return {"policy": policy_path.name, "rules": [r.model_dump() for r in rules]}

    @app.post(
        "/check", response_model=ContractReport, dependencies=[Depends(authorize)]
    )
    def check(file: UploadFile, followup: bool = Form(False)):
        if not app.state.check_lock.acquire(blocking=False):
            raise api_error(
                429, "busy", "Another contract is being checked.", {"Retry-After": "10"}
            )
        tmp_path = None
        try:
            with NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
                tmp_path = Path(temporary.name)
                total = 0
                prefix = b""
                while chunk := file.file.read(1024**2):
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise api_error(413, "file_too_large", "The file is too large.")
                    prefix += chunk[: 1024 - len(prefix)]
                    temporary.write(chunk)
            if b"%PDF-" not in prefix:
                raise api_error(415, "not_a_pdf", "The file is not a PDF.")
            try:
                document = document_loader(tmp_path)
            except PDFIngestionError:
                raise api_error(
                    422, "unreadable_pdf",
                    "The PDF cannot be opened; "
                    "it may be corrupt or password-protected.",
                ) from None
            if document.page_count == 0:
                raise api_error(422, "empty_pdf", "The PDF has no pages.")
            if document.page_count > settings.max_pages:
                raise api_error(422, "too_many_pages", "The PDF has too many pages.")
            if all(page.needs_ocr for page in document.pages):
                raise api_error(
                    422, "scanned_pdf",
                    "The PDF has no text layer; OCR is not supported.",
                )
            if followup:
                report = app.state.checker.check(
                    tmp_path, rules, policy_path, followup=True, followup_version="v2"
                )
            else:
                report = app.state.checker.check(tmp_path, rules, policy_path)
            filename = Path((file.filename or "upload.pdf").replace("\\", "/")).name
            return report.model_copy(update={"contract": filename})
        except HTTPException:
            raise
        except LLMQuotaError:
            raise api_error(
                503, "llm_quota_exhausted",
                "The Gemini per-minute or daily limit was reached.",
                {"Retry-After": "60"},
            ) from None
        except Exception:
            logger.exception("Contract check failed")
            raise api_error(
                500, "internal_error", "The contract could not be checked."
            ) from None
        finally:
            try:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
            finally:
                app.state.check_lock.release()

    return app
