"""
title: Web Search
id: web_search_toggle_filter
description: Instruct the model to search the web for the latest information.
required_open_webui_version: 0.6.10
version: 0.2.0
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel, Field
from loguru import logger


class Filter:
    XAI_PREFIX = "xai."
    ANTHROPIC_PREFIX = "anthropic."

    class Valves(BaseModel):
        MAX_SEARCH_RESULTS: int = 20
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    def _get_model_name(self, body: "Body") -> str:
        effective_model_name: str = body.get("model", "")

        if metadata := body.get("metadata"):
            base_model_name = (
                metadata.get("model", {}).get("info", {}).get("base_model_id", None)
            )
            if base_model_name:
                effective_model_name = base_model_name

        return effective_model_name

    def _get_provider(self, model_name: str) -> str | None:
        if model_name.startswith(self.XAI_PREFIX):
            return "xai"
        elif model_name.startswith(self.ANTHROPIC_PREFIX):
            return "anthropic"
        return None

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable] = None,
        __metadata__: Optional[dict] = None,
    ) -> Dict[str, Any]:
        model_name = self._get_model_name(body)
        provider = self._get_provider(model_name)
        logger.info(f"Web Search: Detected model={model_name!r}, provider={provider!r}")
        if not provider:
            return body

        if not __metadata__:
            return body

        features = __metadata__.setdefault("features", {})
        if not features.get("web_search"):
            return body

        # Disable WebUI's native search
        features["web_search"] = False

        if provider == "xai":
            features["grok_search"] = True
            features["grok_search_size"] = self.valves.MAX_SEARCH_RESULTS
        elif provider == "anthropic":
            features["claude_search"] = True

        logger.info(f"Web Search: Toggled web search features for provider={provider!r}")
        return body
