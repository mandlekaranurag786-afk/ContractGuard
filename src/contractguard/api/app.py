"""Application factory: uvicorn --factory contractguard.api.app:create_app."""

import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, Form, UploadFile

from contractguard.checker import ContractChecker
from contractguard.extraction.extractor import Extractor
from contractguard.extraction.prompt import PRODUCT_PROMPT_VERSION
from contractguard.llm.cache import CachedLLMClient
from contractguard.llm.gemini import GeminiClient
from contractguard.report import ContractReport
from contractguard.retrieval.vector import warm_up_model
from contractguard.rules.models import load_rules


def create_app(checker=None, policy_path=None) -> FastAPI:
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
                    cache_dir=Path(
                        os.environ.get(
                            "CONTRACTGUARD_CACHE_DIR", ".cache/api_llm_cache"
                        )
                    ),
                ),
                prompt_version=PRODUCT_PROMPT_VERSION,
            )
            app.state.checker = ContractChecker(extractor)
            warm_up_model()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.checker = checker

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "prompt_version": PRODUCT_PROMPT_VERSION,
            "rules": len(rules),
        }

    @app.get("/policy")
    def policy():
        return {"policy": policy_path.name, "rules": [r.model_dump() for r in rules]}

    @app.post("/check", response_model=ContractReport)
    def check(file: UploadFile, followup: bool = Form(False)):
        tmp_path = None
        try:
            with NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
                tmp_path = Path(temporary.name)
                shutil.copyfileobj(file.file, temporary)
            if followup:
                report = app.state.checker.check(
                    tmp_path, rules, policy_path, followup=True, followup_version="v2"
                )
            else:
                report = app.state.checker.check(tmp_path, rules, policy_path)
            filename = Path((file.filename or "upload.pdf").replace("\\", "/")).name
            return report.model_copy(update={"contract": filename})
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    return app
