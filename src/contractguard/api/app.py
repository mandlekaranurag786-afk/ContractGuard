"""Application factory: uvicorn --factory contractguard.api.app:create_app."""

import hashlib
import json
import logging
import math
import os
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Query, UploadFile

from contractguard.api.history import (
    CheckRecord,
    HistoryError,
    HistoryStore,
    RuleRecord,
)
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
    db_path: Path | None = None
    daily_cap: int = 30
    history_mode: str = "metadata"

    def __post_init__(self):
        if self.daily_cap < 1:
            raise ValueError("daily_cap must be at least 1")
        if self.history_mode not in ("metadata", "full"):
            raise ValueError("history_mode must be metadata or full")

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
            db_path=Path(os.environ.get("CONTRACTGUARD_DB_PATH", ".cache/history.db")),
            daily_cap=int(os.environ.get("CONTRACTGUARD_DAILY_CAP", "30")),
            history_mode=os.environ.get("CONTRACTGUARD_HISTORY_MODE", "metadata"),
        )


def api_error(status, code, message, headers=None):
    return HTTPException(
        status_code=status, detail={"code": code, "message": message}, headers=headers
    )


def history_report_json(report: ContractReport, mode: str) -> str:
    if mode == "full":
        return report.model_dump_json()
    return report.model_dump_json(include={
        "contract": True,
        "policy_path": True,
        "trust_policy": True,
        "model": True,
        "prompt_version": True,
        "overall_status": True,
        "overall_reason": True,
        "counts": True,
        "cost": True,
        "rule_results": {"__all__": {
            "rule_id", "clause_type", "policy", "decision", "reason_code",
            "field", "field_status", "condition_field", "condition_status",
            "currency_status", "evidence_strength", "confidence_level",
            "field_evidence_source", "condition_evidence_source",
        }},
        "clauses": {"__all__": {"clause_type", "confidence_level"}},
    })


def create_app(
    checker=None, policy_path=None, settings=None, document_loader=load_document,
    *, history=None, clock=None,
) -> FastAPI:
    settings = settings if settings is not None else ApiSettings.from_env()
    clock = clock if clock is not None else lambda: datetime.now(timezone.utc)
    if history is None and settings.db_path is not None:
        history = HistoryStore(settings.db_path)
    if history is None:
        logger.warning("History disabled; no daily cap")
    if settings.api_key is None:
        logger.warning("API key not set; auth disabled")
    policy_path = Path(
        policy_path
        if policy_path is not None
        else os.environ.get("CONTRACTGUARD_POLICY", "policies/eval_policy.json")
    )
    rules = load_rules(policy_path)
    policy_hash = hashlib.sha256(json.dumps(
        [rule.model_dump(mode="json") for rule in rules],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

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
    app.state.history_failed = False

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
            "history": history is not None,
            "daily_cap": settings.daily_cap,
            "history_mode": settings.history_mode,
        }

    @app.get("/policy", dependencies=[Depends(authorize)])
    def policy():
        return {"policy": policy_path.name, "rules": [r.model_dump() for r in rules]}

    @app.get("/checks", dependencies=[Depends(authorize)])
    def list_checks(
        name: str | None = None,
        sha256: str | None = None,
        status: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        if history is None:
            raise api_error(
                503, "history_unavailable", "Check history is unavailable."
            )
        if date_from is not None and date_from.utcoffset() is None:
            date_from = date_from.replace(tzinfo=timezone.utc)
        if date_to is not None and date_to.utcoffset() is None:
            date_to = date_to.replace(tzinfo=timezone.utc)
        try:
            checks = history.list_checks(
                name=name, sha256=sha256, status=status,
                date_from=date_from, date_to=date_to, limit=limit, offset=offset,
            )
        except HistoryError:
            logger.exception("Check history could not be listed")
            raise api_error(
                503, "history_unavailable", "Check history is unavailable."
            ) from None
        return {"checks": checks, "limit": limit, "offset": offset}

    @app.get("/checks/{check_id}", dependencies=[Depends(authorize)])
    def get_check(check_id: int):
        if history is None:
            raise api_error(
                503, "history_unavailable", "Check history is unavailable."
            )
        try:
            check = history.get_check(check_id)
        except HistoryError:
            logger.exception("Check history could not be read")
            raise api_error(
                503, "history_unavailable", "Check history is unavailable."
            ) from None
        if check is None:
            raise api_error(404, "not_found", "No check with this id.")
        report_json = check.pop("report_json")
        check["report"] = json.loads(report_json) if report_json is not None else None
        return check

    @app.post(
        "/check", response_model=ContractReport, dependencies=[Depends(authorize)]
    )
    def check(file: UploadFile, followup: bool = Form(False)):
        started = perf_counter()
        created_at = clock() if history is not None else None
        filename = Path((file.filename or "upload.pdf").replace("\\", "/")).name
        file_hash = hashlib.sha256()
        file_sha256 = None
        tmp_path = None
        acquired = False
        counted = False
        record_allowed = True
        report = None
        http_status, error_code = 200, None
        try:
            acquired = app.state.check_lock.acquire(blocking=False)
            if not acquired:
                raise api_error(
                    429, "busy", "Another contract is being checked.",
                    {"Retry-After": "10"},
                )
            if history is not None:
                if app.state.history_failed:
                    raise api_error(
                        503, "history_unavailable", "Check history is unavailable."
                    )
                try:
                    window_start = created_at - timedelta(hours=24)
                    if history.counted_since(window_start) >= settings.daily_cap:
                        oldest = history.oldest_counted_since(window_start)
                        if oldest is None:
                            raise HistoryError("Counted window has no oldest row")
                        retry_after = max(1, math.ceil((
                            oldest + timedelta(hours=24) - created_at
                        ).total_seconds()))
                        raise api_error(
                            429, "daily_limit_reached",
                            "The daily check limit was reached.",
                            {"Retry-After": str(retry_after)},
                        )
                except HistoryError:
                    record_allowed = False
                    logger.exception("Check history could not be read")
                    raise api_error(
                        503, "history_unavailable", "Check history is unavailable."
                    ) from None
            with NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
                tmp_path = Path(temporary.name)
                total = 0
                prefix = b""
                while chunk := file.file.read(1024**2):
                    file_hash.update(chunk)
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise api_error(413, "file_too_large", "The file is too large.")
                    prefix += chunk[: 1024 - len(prefix)]
                    temporary.write(chunk)
                file_sha256 = file_hash.hexdigest()
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
            counted = True
            if followup:
                report = app.state.checker.check(
                    tmp_path, rules, policy_path, followup=True, followup_version="v2"
                )
            else:
                report = app.state.checker.check(tmp_path, rules, policy_path)
            report = report.model_copy(update={"contract": filename})
            return report
        except HTTPException as exc:
            http_status = exc.status_code
            error_code = (
                exc.detail.get("code") if isinstance(exc.detail, dict) else None
            )
            report = None
            raise
        except LLMQuotaError:
            http_status, error_code, report = 503, "llm_quota_exhausted", None
            raise api_error(
                503, "llm_quota_exhausted",
                "The Gemini per-minute or daily limit was reached.",
                {"Retry-After": "60"},
            ) from None
        except Exception:
            http_status, error_code, report = 500, "internal_error", None
            logger.exception("Contract check failed")
            raise api_error(
                500, "internal_error", "The contract could not be checked."
            ) from None
        finally:
            try:
                try:
                    if tmp_path is not None:
                        tmp_path.unlink(missing_ok=True)
                finally:
                    if history is not None and record_allowed:
                        try:
                            cost = report.cost if report is not None else None
                            record = CheckRecord(
                                created_at=created_at, file_name=filename,
                                file_sha256=file_sha256, policy_hash=policy_hash,
                                prompt_version=PRODUCT_PROMPT_VERSION,
                                followup=followup,
                                http_status=http_status, error_code=error_code,
                                overall_status=(
                                    report.overall_status.value
                                    if report is not None else None
                                ),
                                counted=counted,
                                llm_calls=cost.llm_calls if cost is not None else None,
                                cache_hits=(
                                    cost.cache_hits if cost is not None else None
                                ),
                                input_tokens=(
                                    cost.input_tokens if cost is not None else None
                                ),
                                output_tokens=(
                                    cost.output_tokens if cost is not None else None
                                ),
                                elapsed_ms=(perf_counter() - started) * 1000,
                                report_json=(
                                    history_report_json(report, settings.history_mode)
                                    if report is not None else None
                                ),
                            )
                            rule_rows = [
                                RuleRecord(r.rule_id, r.decision.value, r.reason_code)
                                for r in report.rule_results
                            ] if report is not None else []
                            history.record(record, rule_rows)
                        except Exception:
                            logger.exception("Check history could not be recorded")
                            if counted:
                                app.state.history_failed = True
            finally:
                if acquired:
                    app.state.check_lock.release()

    return app
