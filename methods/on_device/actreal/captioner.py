"""Caption icons through whichever vision API the operator actually has.

Mobile-Agent-E crops every icon it detects out of the screenshot and asks a
vision model to describe it in a sentence; the answer becomes the ``icon: ...``
text the planner reads.  Upstream that call is wired to DashScope through
``MultiModalConversation``, and the model name and key are module constants, so
an operator holding an OpenAI key and no Alibaba account cannot reach it by
configuration -- ``OPENAI_API_KEY`` has nowhere to go.

This rebinds that one function, the same way :mod:`actreal.agent_shim` rebinds
the controller's primitives and for the same reason: the vendored framework
stays byte-for-byte upstream, so what it plans and how it reasons is still
theirs, and only the plumbing underneath points somewhere else.  Nothing about
the prompt changes -- the caller still passes its own wording.

The upstream path is kept and preferred when a DashScope key exists, so this
changes nothing for anyone already set up for it.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Optional

# Icon captioning is a small, high-volume call -- one per icon per screenshot --
# so the default is the cheap vision model rather than the reasoning one.
DEFAULT_MODEL = "gpt-5.6-luna"
FALLBACK_TEXT = "This is an icon."


def _data_url(path: str) -> str:
    """Vision APIs take an inline image; the file never leaves as a path."""

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as handle:
        return f"data:{mime};base64," + base64.b64encode(handle.read()).decode()


def caption_with_openai(
    image_path: str,
    query: str,
    *,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
) -> str:
    """One icon, one sentence.

    Failures return the upstream fallback string rather than raising: a caption
    is an annotation on a detection that already has coordinates, and losing one
    should cost the agent a description, not the step.
    """

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            ],
        }
    ]
    # gpt-5.x renamed the cap and refuses a non-default temperature; asking for
    # no reasoning keeps a one-icon caption from costing a paragraph of thought.
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        extra = {"max_completion_tokens": 300, "reasoning_effort": "none"}
    else:
        extra = {"max_tokens": 100}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **extra,
        )
        return (response.choices[0].message.content or FALLBACK_TEXT).strip()
    except Exception:
        return FALLBACK_TEXT


def choose_provider(
    *,
    qwen_key: Optional[str] = None,
    openai_key: Optional[str] = None,
) -> dict[str, Any]:
    """Decide who captions, from what the environment actually holds.

    DashScope first, because that is what upstream does and a machine set up for
    it should behave exactly as before.  Only when there is no such key does the
    OpenAI path come into play, and if there is neither, that is reported rather
    than discovered later as every icon coming back "This is an icon."
    """

    qwen = qwen_key if qwen_key is not None else os.environ.get("QWEN_API_KEY")
    openai_key = openai_key if openai_key is not None else os.environ.get("OPENAI_API_KEY")

    if qwen:
        return {"provider": "dashscope", "reason": "QWEN_API_KEY is set; upstream path kept"}
    if openai_key:
        return {
            "provider": "openai",
            "model": os.environ.get("CAPTION_MODEL_OPENAI", DEFAULT_MODEL),
            "base_url": os.environ.get("OPENAI_BASE_URL") or None,
            "reason": "no QWEN_API_KEY; captioning rebound onto OPENAI_API_KEY",
        }
    return {
        "provider": "none",
        "reason": "neither QWEN_API_KEY nor OPENAI_API_KEY is set; every icon "
        "would come back as the fallback string",
    }


def install(module, **overrides) -> dict[str, Any]:
    """Point ``module.process_image`` at the chosen provider.

    ``generate_api`` submits ``process_image`` to a thread pool by name, so it
    resolves through the module's globals at call time and rebinding the
    attribute is enough.  The entry point is all definitions -- importing it
    starts no task -- so this can be installed after the import and before the
    framework's own ``run_single_task`` is ever called.
    """

    choice = choose_provider(
        qwen_key=overrides.get("qwen_key"), openai_key=overrides.get("openai_key")
    )
    if choice["provider"] != "openai":
        choice["installed"] = False
        return choice

    api_key = overrides.get("openai_key") or os.environ["OPENAI_API_KEY"]
    model = overrides.get("model") or choice["model"]
    base_url = overrides.get("base_url", choice["base_url"])
    original = getattr(module, "process_image", None)

    def process_image(image, query, caption_model=None):
        # Upstream's third argument names a Qwen model; it is deliberately
        # ignored here rather than passed on, because it would name a model this
        # provider does not have.
        return caption_with_openai(
            image, query, model=model, api_key=api_key, base_url=base_url
        )

    module.process_image = process_image
    choice.update({"installed": True, "model": model, "replaced": original is not None})
    return choice
