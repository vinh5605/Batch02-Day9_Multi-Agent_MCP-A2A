"""Shared LLM factory for all agents.

Supports two backends via env vars:
  - Ollama (local):    LLM_BASE_URL=http://localhost:11434/v1, LLM_API_KEY=ollama
  - OpenRouter (cloud): LLM_BASE_URL=https://openrouter.ai/api/v1 (default)

Model is selected via LLM_MODEL (or the legacy OPENROUTER_MODEL alias).
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at the configured LLM backend."""
    model = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL", "qwen2.5:3b")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "ollama")
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.3,
    )
