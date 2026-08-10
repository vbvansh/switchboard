"""The no-remote-provider guarantee, as an executable test.

If someone later edits config.py and weakens `_require_local_host`, these fail.
That is the entire point: the promise that Switchboard holds no API keys and
contacts no paid endpoint is enforced mechanically, not by memory.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from switchboard.config import Settings

REMOTE_URLS = [
    "https://api.openai.com",
    "https://api.anthropic.com",
    "https://generativelanguage.googleapis.com",
    "http://192.168.1.50:11434",
    "http://example.com:11434",
]

LOCAL_URLS = [
    "http://localhost:11434",
    "http://127.0.0.1:11434",
    "http://[::1]:11434",
]


@pytest.mark.parametrize("url", REMOTE_URLS)
def test_remote_provider_hosts_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError, match="non-local provider host"):
        Settings(ollama_base_url=url)


@pytest.mark.parametrize("url", LOCAL_URLS)
def test_local_provider_hosts_are_accepted(url: str) -> None:
    assert Settings(ollama_base_url=url).ollama_base_url


def test_non_http_scheme_is_rejected() -> None:
    with pytest.raises(ValidationError, match="http"):
        Settings(ollama_base_url="ftp://localhost:11434")


def test_trailing_slash_is_normalised() -> None:
    settings = Settings(ollama_base_url="http://localhost:11434/")
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.openai_compat_url == "http://localhost:11434/v1"
