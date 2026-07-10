# -*- coding: utf-8 -*-
"""OpenAI-compatible chat completion backend for correction."""

from __future__ import annotations

import json
import os
from urllib import error, request


class OpenAICompatibleCorrectionBackend:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        timeout: int = 120,
        urlopen=None,
    ):
        if not api_base:
            raise ValueError("correction_api_base is required for openai_compatible correction backend")
        if not api_key:
            raise ValueError("API key is required for openai_compatible correction backend")
        if not model:
            raise ValueError("correction_api_model is required for openai_compatible correction backend")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.urlopen = urlopen or request.urlopen

    @classmethod
    def from_env(
        cls,
        api_base: str,
        api_key_env: str,
        model: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        timeout: int = 120,
        urlopen=None,
    ) -> "OpenAICompatibleCorrectionBackend":
        api_key = os.environ.get(api_key_env or "")
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is required for openai_compatible correction backend")
        return cls(
            api_base=api_base,
            api_key=api_key,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout=timeout,
            urlopen=urlopen,
        )

    def correct(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        req = request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.urlopen(req, timeout=self.timeout) as resp:
                status = int(getattr(resp, "status", 200))
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"correction API request failed with HTTP {exc.code}: {body[:500]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"correction API request failed: {exc.reason}") from exc

        if status < 200 or status >= 300:
            raise RuntimeError(f"correction API request failed with HTTP {status}: {body[:500]}")
        return self._extract_message_content(body)

    @staticmethod
    def _extract_message_content(body: str) -> str:
        try:
            data = json.loads(body)
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"correction API response missing choices[0].message.content: {body[:500]}") from exc
