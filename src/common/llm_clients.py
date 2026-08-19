"""
Unified chat-completion interface across the three providers used in the
paper: GPT (OpenAI API), Gemini (Google Gen AI SDK), and LLaMA (a local
OpenAI-compatible server — Ollama or vLLM — per LLAMA_BASE_URL).

Every call site in the pipeline (summarization, chain-of-thought
classification, financial extraction) goes through `chat_complete` so
provider-specific quirks live in exactly one place.
"""

import json
import time
from functools import lru_cache

from . import config

_RETRYABLE_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 5


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI()  # reads OPENAI_API_KEY from the environment


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    return genai.Client()  # reads GOOGLE_API_KEY from the environment


@lru_cache(maxsize=1)
def _llama_client():
    from openai import OpenAI

    return OpenAI(base_url=config.LLAMA_BASE_URL, api_key=config.LLAMA_API_KEY)


def chat_complete(
    prompt: str,
    model_key: str,
    *,
    system: str | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    model: str | None = None,
) -> str:
    """Send `prompt` to the given provider ("gpt", "gemini", or "llama") and
    return the raw text response, retrying transient errors.

    `model` overrides that provider's default CHAT_MODEL_IDS entry for this
    call only — e.g. financial extraction uses a different (faster/cheaper)
    GPT variant than classification/summarization."""
    temperature = config.TEMPERATURE if temperature is None else temperature
    dispatch = {"gpt": _chat_gpt, "gemini": _chat_gemini, "llama": _chat_llama}
    if model_key not in dispatch:
        raise ValueError(f"Unknown model_key: {model_key!r} (expected one of {list(dispatch)})")

    last_err = None
    for attempt in range(_RETRYABLE_ATTEMPTS):
        try:
            return dispatch[model_key](prompt, system, json_mode, temperature, model)
        except Exception as exc:  # noqa: BLE001 — broad on purpose, retried below
            last_err = exc
            if attempt < _RETRYABLE_ATTEMPTS - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(
        f"chat_complete failed for model_key={model_key!r} after {_RETRYABLE_ATTEMPTS} attempts"
    ) from last_err


def _chat_gpt(prompt, system, json_mode, temperature, model=None):
    client = _openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model or config.CHAT_MODEL_IDS["gpt"],
        messages=messages,
        temperature=temperature,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        **kwargs,
    )
    return resp.choices[0].message.content


def _chat_gemini(prompt, system, json_mode, temperature, model=None):
    client = _gemini_client()
    gen_config = {"temperature": temperature, "max_output_tokens": config.MAX_OUTPUT_TOKENS}
    if system:
        gen_config["system_instruction"] = system
    if json_mode:
        gen_config["response_mime_type"] = "application/json"
    resp = client.models.generate_content(
        model=model or config.CHAT_MODEL_IDS["gemini"], contents=prompt, config=gen_config
    )
    return resp.text


def _chat_llama(prompt, system, json_mode, temperature, model=None):
    client = _llama_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model or config.CHAT_MODEL_IDS["llama"],
        messages=messages,
        temperature=temperature,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        **kwargs,
    )
    return resp.choices[0].message.content


def extract_json_object(text: str) -> dict:
    """Best-effort JSON parse of a model response. Handles the common case
    where the model wraps the JSON in a markdown code fence despite being
    asked to return only JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response: {text!r}")
    return json.loads(text[start : end + 1])
