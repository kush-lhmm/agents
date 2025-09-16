import os
from utils.logger import logger

_USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() in ("1", "true", "yes")
_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL") or os.getenv("OPENAI_VISION_MODEL") or "gpt-4o-mini"

def maybe_summarize_with_openai(question: str, contexts: str) -> str:
    if not _USE_OPENAI or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI is disabled or API key not set.")

    try:
        from openai import OpenAI
        client = OpenAI()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise assistant. Only use the provided contexts. "
                    "If the answer is unclear, say 'Not enough information in CSV context.'"
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContexts:\n{contexts}\n\nAnswer using only the contexts.",
            },
        ]
        resp = client.chat.completions.create(
            model=_TEXT_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}")
        raise
