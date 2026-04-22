from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, urlunparse

from openhands.sdk import LLM


def load_llm_config(config_path: str | Path) -> LLM:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"LLM config file {config_path} does not exist")

    with config_path.open("r", encoding="utf-8") as f:
        llm_config = f.read()

    return LLM.model_validate_json(llm_config)


def normalize_llm_base_url_for_workspace(llm: LLM, workspace_type: str) -> LLM:
    """Return an LLM copy with a Docker-reachable base URL when needed.

    When evals run in Docker, the model endpoint configured as localhost on the
    host machine is not reachable from inside the container. In that case,
    rewrite localhost/127.0.0.1/::1 to host.docker.internal.
    """
    if workspace_type != "docker" or not llm.base_url:
        return llm

    parsed = urlparse(llm.base_url)
    host = parsed.hostname
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return llm

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    netloc = f"{userinfo}host.docker.internal"
    if parsed.port is not None:
        netloc += f":{parsed.port}"

    rewritten = urlunparse(parsed._replace(netloc=netloc))
    return llm.model_copy(deep=True, update={"base_url": rewritten})
