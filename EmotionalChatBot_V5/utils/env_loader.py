"""
Robust .env loader for local dev.

Why:
- In this repo, `main.py` expects `python-dotenv` to exist. If it isn't installed,
  `.env` won't be loaded and LangSmith/OpenAI env vars will be missing.
- Users sometimes accidentally paste a raw key as the first line (no KEY=VALUE).
  We try to detect common key prefixes and map them to the right env var.

This loader:
- Tries `python-dotenv` if available.
- Falls back to a tiny parser that supports KEY=VALUE lines.
- Also supports a *single raw key line*:
  - `sk-...`   -> OPENAI_API_KEY
  - `ls__...` / `lsv2_...` -> LANGCHAIN_API_KEY
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def _parse_env_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            if k:
                data[k] = v
            continue

        # raw key fallback (best-effort)
        if s.startswith("sk-") and "OPENAI_API_KEY" not in data:
            data["OPENAI_API_KEY"] = s
        elif (s.startswith("ls__") or s.startswith("lsv2_")) and "LANGCHAIN_API_KEY" not in data:
            data["LANGCHAIN_API_KEY"] = s
    return data


def load_env_from_file(env_path: Path) -> Dict[str, str]:
    """
    Load env vars from a given `.env` file into `os.environ` (does not override existing).
    Returns the key-values loaded (best-effort).
    """
    loaded: Dict[str, str] = {}

    # 1) Prefer python-dotenv when available
    try:
        from dotenv import dotenv_values  # type: ignore
    except Exception:
        dotenv_values = None  # type: ignore

    if dotenv_values is not None and env_path.exists():
        values = dotenv_values(str(env_path)) or {}
        for k, v in values.items():
            if not k or v is None:
                continue
            if k not in os.environ:
                os.environ[k] = str(v)
            loaded[k] = str(v)
    else:
        values = _parse_env_file(env_path)
        for k, v in values.items():
            if k not in os.environ:
                os.environ[k] = v
            loaded[k] = v

    # Convenience aliasing for this repo's multi-provider setup.
    # - Keep OPENAI_API_KEY_OPENAI in .env as a backup key.
    # - When using OpenAI official base_url (or a GPT model), prefer it automatically.
    try:
        base_url = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").strip()
        model = (os.environ.get("OPENAI_MODEL") or "").strip()
        openai_key = (os.environ.get("OPENAI_API_KEY_OPENAI") or loaded.get("OPENAI_API_KEY_OPENAI") or "").strip()
        if openai_key and (
            base_url.startswith("https://api.openai.com")
            or base_url.startswith("https://api.openai.com/")
            or model.startswith("gpt-")
        ):
            os.environ["OPENAI_API_KEY"] = openai_key
            loaded["OPENAI_API_KEY"] = openai_key
    except Exception:
        pass

    return loaded


def load_project_env(project_dir: Path) -> Dict[str, str]:
    """Load `{project_dir}/.env` if it exists."""
    return load_env_from_file(project_dir / ".env")


def mask_env_value(v: Optional[str]) -> str:
    if not v:
        return "<missing>"
    if len(v) <= 10:
        return v
    return v[:4] + "..." + v[-4:]

