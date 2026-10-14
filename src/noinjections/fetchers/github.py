"""Fetch content from GitHub. So far: a single file, through its raw URL."""

import httpx

from noinjections.extractors.agent_files import is_agent_file
from noinjections.fetchers.content import (
    DEFAULT_MAX_AGENT_FILE_BYTES,
    DEFAULT_MAX_FILE_BYTES,
    ContentRejected,
    SkippedFile,
    decode_text,
    text_encoding_for,
)
from noinjections.fetchers.local import LocalFile
from noinjections.fetchers.web import DEFAULT_TIMEOUT_SECONDS, fetch_bytes
from noinjections.targets import GitHubFileTarget


def fetch_github_file(
    target: GitHubFileTarget,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_agent_file_bytes: int = DEFAULT_MAX_AGENT_FILE_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: httpx.Client | None = None,
) -> LocalFile | SkippedFile:
    """Download one file from its ``raw_url`` and check it exactly like a local file.

    The same size caps apply (``max_agent_file_bytes`` for agent files), and binary content is
    returned as a :class:`SkippedFile`. The returned ``path`` is the target's ``ref_path``. For a
    truncated file, ``size`` is the number of bytes received, a lower bound on the real size.
    Network and HTTP errors raise :class:`~noinjections.fetchers.web.WebFetchError`.
    """
    if max_file_bytes < 1:
        raise ValueError(f"max_file_bytes must be >= 1, got {max_file_bytes}")
    if max_agent_file_bytes < 1:
        raise ValueError(f"max_agent_file_bytes must be >= 1, got {max_agent_file_bytes}")
    path = target.ref_path
    # Agent files never get a lower cap than ordinary files.
    max_bytes = max(max_file_bytes, max_agent_file_bytes) if is_agent_file(path) else max_file_bytes
    response = fetch_bytes(target.raw_url, max_bytes=max_bytes, timeout=timeout, client=client)
    # The raw host's Content-Type says little about the file itself, so it is ignored and the
    # file is judged by its name and content, exactly as a local file is.
    encoding = text_encoding_for(path)
    try:
        text = decode_text(response.data, truncated=response.truncated, encoding=encoding)
    except ContentRejected as exc:
        return SkippedFile(path, exc.reason, exc.detail)
    return LocalFile(path, text, size=response.received, truncated=response.truncated)
