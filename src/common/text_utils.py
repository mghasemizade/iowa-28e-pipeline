"""Small shared helpers used across pipeline stages."""

from . import config


def truncate_text(text: str, max_chars: int | None = None) -> str:
    max_chars = config.MAX_INPUT_CHARS if max_chars is None else max_chars
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def iter_with_progress(items, desc: str = ""):
    """tqdm if available, otherwise a plain iterator — keeps tqdm out of
    requirements.txt as a hard dependency."""
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ImportError:
        return items
