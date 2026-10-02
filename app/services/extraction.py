import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from app.prompts.templates import (
    SYSTEM_PROMPT,
    build_retry_message,
    build_user_message,
)
from app.schema.device import Device

load_dotenv()

MAX_RETRIES = 2  # 1 initial attempt + up to 2 retries = 3 calls maximum

_client = None


class ExtractionError(Exception):
    """Raised when the LLM fails to return valid data after all retries."""


def _get_client() -> OpenAI:
    """Create the OpenAI client once, on first use."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


def extract_device(text: str) -> Device:
    """Extract structured device specs from messy text using an LLM.

    Calls the LLM in JSON mode, parses and validates the result with Pydantic,
    and on failure sends the error back to the LLM, retrying up to MAX_RETRIES times.
    """
    client = _get_client()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(text)},
    ]

    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw_output = response.choices[0].message.content or ""

        try:
            data = json.loads(raw_output)          # parse the returned JSON
            return Device.model_validate(data)     # validate against the schema
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            # Show the LLM its own answer plus the error, then ask for a fix
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": build_retry_message(str(e))})

    raise ExtractionError(
        f"Could not extract a valid device after {MAX_RETRIES} retries: {last_error}"
    )