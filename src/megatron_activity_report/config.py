"""Configuration for report collection, generation, and publication."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TEST_PREFIXES = ("tests/", "test/")
DEFAULT_CI_PREFIXES = (
    ".github/",
    "ci/",
    "tools/ci/",
    "tests/test_utils/python_scripts/",
)
DEFAULT_PRESERVE_TERMS = (
    "Megatron-LM",
    "Megatron Core",
    "Megatron-FSDP",
    "FSDP",
    "MFSDP",
    "CUDA Graph",
    "THD",
    "GTP",
    "MTP",
    "MoE",
    "RL",
    "GRPO",
    "MIMO",
    "DeepSeek-V4",
    "Qwen3.5-VL",
    "Gated Delta Product",
    "Gated DeltaNet",
    "HybridModel",
    "Muon",
    "NIXL",
    "NCCL",
    "SSM",
    "KV cache",
    "RoPE",
    "MRoPE",
    "MLA",
    "CSA",
    "DSA",
    "DTensor",
    "ZeRO",
    "MXFP8",
    "FP8",
    "BF16",
    "FP32",
    "cuDNN",
    "cuBLASLt",
    "TileLang",
    "Triton",
    "CuTe DSL",
    "PyTorch",
    "OpenAI",
    "vLLM",
    "Transformer Engine",
    "Energon",
    "Top-K",
    "LayerWise",
)


@dataclasses.dataclass(frozen=True)
class ReportConfig:
    source_repo: str
    destination_repo: str
    report_repo: str
    report_branch: str = "main"
    timezone: str = "Asia/Shanghai"
    database_path: Path = Path("runtime/activity.duckdb")
    artifacts_dir: Path = Path("runtime/runs")
    state_path: Path = Path("state/issues.json")
    reports_dir: Path = Path("reports")
    github_workers: int = 8
    request_timeout_seconds: int = 60
    max_themes_per_section: int = 10
    summarizer_provider: str = "openai"
    model: str = "gpt-5"
    reasoning_effort: str = "medium"
    batch_size: int = 60
    max_output_tokens: int = 24000
    codex_binary: str = "codex"
    test_prefixes: tuple[str, ...] = DEFAULT_TEST_PREFIXES
    ci_prefixes: tuple[str, ...] = DEFAULT_CI_PREFIXES
    translation_preserve_terms: tuple[str, ...] = DEFAULT_PRESERVE_TERMS
    project_root: Path = dataclasses.field(default=Path.cwd(), compare=False)
    config_path: Path | None = dataclasses.field(default=None, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> "ReportConfig":
        config_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("report config must be a YAML mapping")
        root = _find_project_root(config_path.parent)
        for field in ("source_repo", "destination_repo", "report_repo"):
            _validate_repo(str(payload.get(field, "")), field)

        filters = _mapping(payload.get("filters"), "filters")
        summarizer = _mapping(payload.get("summarizer"), "summarizer")
        translation = _mapping(payload.get("translation"), "translation")

        def root_path(key: str, default: str) -> Path:
            value = Path(str(payload.get(key, default))).expanduser()
            return value.resolve() if value.is_absolute() else (root / value).resolve()

        config = cls(
            source_repo=str(payload["source_repo"]),
            destination_repo=str(payload["destination_repo"]),
            report_repo=str(payload["report_repo"]),
            report_branch=str(payload.get("report_branch", "main")),
            timezone=str(payload.get("timezone", "Asia/Shanghai")),
            database_path=root_path("database_path", "runtime/activity.duckdb"),
            artifacts_dir=root_path("artifacts_dir", "runtime/runs"),
            state_path=root_path("state_path", "state/issues.json"),
            reports_dir=root_path("reports_dir", "reports"),
            github_workers=int(payload.get("github_workers", 8)),
            request_timeout_seconds=int(payload.get("request_timeout_seconds", 60)),
            max_themes_per_section=int(payload.get("max_themes_per_section", 10)),
            summarizer_provider=os.environ.get(
                "REPORT_SUMMARIZER_PROVIDER", str(summarizer.get("provider", "openai"))
            ),
            model=os.environ.get(
                "OPENAI_MODEL", str(summarizer.get("model", "gpt-5"))
            ),
            reasoning_effort=str(summarizer.get("reasoning_effort", "medium")),
            batch_size=int(summarizer.get("batch_size", 60)),
            max_output_tokens=int(summarizer.get("max_output_tokens", 24000)),
            codex_binary=str(summarizer.get("codex_binary", "codex")),
            test_prefixes=tuple(filters.get("test_prefixes", DEFAULT_TEST_PREFIXES)),
            ci_prefixes=tuple(filters.get("ci_prefixes", DEFAULT_CI_PREFIXES)),
            translation_preserve_terms=tuple(
                translation.get("preserve_terms", DEFAULT_PRESERVE_TERMS)
            ),
            project_root=root,
            config_path=config_path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        from zoneinfo import ZoneInfo

        ZoneInfo(self.timezone)
        if self.summarizer_provider not in {"openai", "codex"}:
            raise ValueError("summarizer.provider must be openai or codex")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported summarizer.reasoning_effort")
        if not 1 <= self.github_workers <= 32:
            raise ValueError("github_workers must be between 1 and 32")
        if not 1 <= self.batch_size <= 100:
            raise ValueError("summarizer.batch_size must be between 1 and 100")
        if not 1 <= self.max_themes_per_section <= 20:
            raise ValueError("max_themes_per_section must be between 1 and 20")
        if any(not str(term).strip() for term in self.translation_preserve_terms):
            raise ValueError("translation.preserve_terms cannot contain empty values")
        if len(set(self.translation_preserve_terms)) != len(
            self.translation_preserve_terms
        ):
            raise ValueError("translation.preserve_terms cannot contain duplicates")

    def fingerprint(self) -> str:
        payload = dataclasses.asdict(self)
        payload.pop("config_path", None)
        payload.pop("project_root", None)
        for key in ("database_path", "artifacts_dir", "state_path", "reports_dir"):
            payload[key] = str(payload[key])
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a YAML mapping")
    return value


def _validate_repo(value: str, field: str) -> None:
    if len(value.split("/")) != 2 or not all(value.split("/")):
        raise ValueError(f"{field} must use owner/name form")


def _find_project_root(start: Path) -> Path:
    current = start
    while current != current.parent:
        if (current / "pyproject.toml").is_file():
            return current
        current = current.parent
    raise ValueError(f"could not find project root above {start}")
