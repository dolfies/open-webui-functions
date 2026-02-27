"""
title: Web Search
id: web_search_toggle_filter
description: Instruct the model to search the web for the latest information.
required_open_webui_version: 0.6.10
version: 0.2.0

Note: Designed to work with the OpenAI Responses manifold
      https://github.com/jrkropp/open-webui-developer-toolkit/tree/main/functions/pipes/openai_responses_manifold
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import logging

# Models that already include the native web_search tool
WEB_SEARCH_MODELS = {
    "openai_responses.gpt-4.1",
    "openai_responses.gpt-4.1-mini",
    "openai_responses.gpt-4o",
    "openai_responses.gpt-4o-mini",
    "openai_responses.o3",
    "openai_responses.o1",
    "openai_responses.o3-deep-research",
    "openai_responses.o4-mini",
    "openai_responses.o4-mini-deep-research",
    "openai_responses.o4-mini-high",
    "openai_responses.o3-pro",
    "openai_responses.gpt-5",
    "openai_responses.gpt-5.1",
    "openai_responses.gpt-5.2",
    "openai_responses.gpt-5-chat-latest",
    "openai_responses.gpt-5.1-chat-latest",
    "openai_responses.gpt-5.2-chat-latest",
    "openai_responses.gpt-5-pro",
    "openai_responses.gpt-5-mini",
    "openai_responses.gpt-5-codex",
    "openai_responses.gpt-5.1-codex",
    "openai_responses.gpt-5.2-codex",
    "openai_responses.gpt-5.3-codex",
    "openai_responses.gpt-5.1-codex-mini",
    "openai_responses.gpt-5.1-codex-max",
    "openai_responses.gpt-5-codex-mini",
    "openai_responses.gpt-5-thinking",
    "openai_responses.gpt-5-thinking-high",
    "openai_responses.gpt-5-codex-thinking",
    "openai_responses.gpt-5-codex-thinking-high",
    "openai_responses.gpt-5-codex-thinking-minimal",
}

SUPPORT_TOOL_CHOICE_PARAMETER = {
    "openai_responses.gpt-4.1",
    "openai_responses.gpt-4.1-mini",
    "openai_responses.gpt-4o",
    "openai_responses.gpt-4o-mini",
}


class Filter:
    # ── User‑configurable knobs (valves) ──────────────────────────────
    class Valves(BaseModel):
        SEARCH_CONTEXT_SIZE: str = "medium"
        # DEFAULT_SEARCH_MODEL: str = "openai_responses.gpt-4o"
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.logger = logging.getLogger(__name__)

    def _get_model_name(self, body: "Body") -> tuple[str, bool]:
        effective_model_name: str = body.get("model", "")
        initial_model_name = effective_model_name
        base_model_name = None

        # Check for a base model ID in the metadata for custom models
        # If metadata exists, attempt to extract the base_model_id
        if metadata := body.get("metadata"):
            # Safely navigate the nested structure: metadata -> model -> info -> base_model_id
            base_model_name = (
                metadata.get("model", {}).get("info", {}).get("base_model_id", None)
            )
            # If a base model ID is found, it overrides the initially requested name
            if base_model_name:
                effective_model_name = base_model_name

        is_manifold_model = "openai_responses." in effective_model_name

        canonical_model_name = effective_model_name.replace("openai_responses.", "")
        return canonical_model_name, is_manifold_model

    # ─────────────────────────────────────────────────────────────────
    # 1.  INLET – choose the right model, disable WebUI’s own search
    # ─────────────────────────────────────────────────────────────────
    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[callable] = None,
        __metadata__: Optional[dict] = None,
    ) -> Dict[str, Any]:

        # 0) Turn off WebUI’s own (legacy) search toggle; we’ll manage tools ourselves.
        if __metadata__:
            __metadata__.setdefault("features", {}).update({"web_search": False})

        # 1) Ensure we’re on a search-capable model
        if body.get("model") not in WEB_SEARCH_MODELS:
            return body
            # body["model"] = self.valves.DEFAULT_SEARCH_MODEL

        # 2) Add OpenAI’s web-search tool via extra_tools (as-is; manifold will append & strip)
        body.setdefault("extra_tools", []).append(
            {
                "type": "web_search",
                "search_context_size": self.valves.SEARCH_CONTEXT_SIZE,
                # Optionally include user_location when you have one:
                # "user_location": {"type": "approximate", "country": "CA", "region": "BC", "city": "Langley"}
            }
        )

        # 3) (Optional) Nudge/force usage:
        #    If the model supports tool_choice for web_search, you can force it;
        #    otherwise add a gentle developer reminder.
        if body.get("model") in SUPPORT_TOOL_CHOICE_PARAMETER:
            body["tool_choice"] = {
                "type": "web_search"
            }  # keep if GA; otherwise leave unset
        else:
            body.setdefault("messages", []).append(
                {
                    "role": "developer",
                    "content": (
                        "Web search is enabled. Use the `web_search` tool whenever you need fresh information."
                    ),
                }
            )

        return body
