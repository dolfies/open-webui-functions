"""
title: Think Harder
id: reason_filter
description: Think before responding.
required_open_webui_version: 0.6.10
version: 0.2.0
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal
from pydantic import BaseModel, Field
from open_webui.models.models import Models

if TYPE_CHECKING:
    from open_webui.utils.manifold_types import *

REASONING_MODELS = {"gpt-5", "o1", "o3", "o4"}
REASONING_MODELS_BLACKLIST = {"gpt-5-chat", "gpt-5.1-chat", "gpt-5.2-chat"}


class Filter:
    OPENAI_PREFIX = "openai_responses."
    GEMINI_PREFIX = "gemini_manifold_google_genai."
    ANTHROPIC_PREFIX = "anthropic."
    XAI_PREFIX = "xai."

    class Valves(BaseModel):
        DEFAULT_MODEL: str = "openai_responses.gpt-5.2"
        REASONING_EFFORT: Literal["minimal", "low", "medium", "high", "not set"] = (
            "high"
        )
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB2aWV3Qm94PSIwIDAgMjQgMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgY2xhc3M9ImgtWzE4cHldIHctWzE4cHldIj48cGF0aCBkPSJtMTIgM2MtMy41ODUgMC02LjUgMi45MjI1LTYuNSA2LjUzODUgMCAyLjI4MjYgMS4xNjIgNC4yOTEzIDIuOTI0OCA1LjQ2MTVoNy4xNTA0YzEuNzYyOC0xLjE3MDIgMi45MjQ4LTMuMTc4OSAyLjkyNDgtNS40NjE1IDAtMy42MTU5LTIuOTE1LTYuNTM4NS02LjUtNi41Mzg1em0yLjg2NTMgMTRoLTUuNzMwNnYxaDUuNzMwNnYtMXptLTEuMTMyOSAzSC03LjQ2NDhjMC4zNDU4IDAuNTk3OCAwLjk5MjEgMSAxLjczMjQgMXMxLjM4NjYtMC40MDIyIDEuNzMyNC0xem0tNS42MDY0IDBjMC40NDQwMyAxLjcyNTIgMi4wMTAxIDMgMy44NzQgM3MzLjQzLTEuMjc0OCAzLjg3NC0zYzAuNTQ4My0wLjAwNDcgMC45OTEzLTAuNDUwNiAwLjk5MTMtMXYtMi40NTkzYzIuMTk2OS0xLjU0MzEgMy42MzQ3LTQuMTA0NSAzLjYzNDctNy4wMDIyIDAtNC43MTA4LTMuODAwOC04LjUzODUtOC41LTguNTM4NS00LjY5OTIgMC04LjUgMy44Mjc2LTguNSA4LjUzODUgMCAyLjg5NzcgMS40Mzc4IDUuNDU5MSAzLjYzNDcgNy4wMDIydjIuNDU5M2MwIDAuNTQ5NCAwLjQ0MzAxIDAuOTk1MyAwLjk5MTI4IDF6IiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGZpbGw9ImN1cnJlbnRDb2xvciIgZmlsbC1ydWxlPSJldmVub2RkIj48L3BhdGg+PC9zdmc+"
        self.logger = logging.getLogger(__name__)

    def get_model_name(self, body: dict) -> str:
        effective_model_name: str = body.get("model", "")
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

        return effective_model_name

    def get_type(self, body: dict) -> str | None:
        model_name = self.get_model_name(body)
        if model_name.startswith(self.OPENAI_PREFIX):
            return "openai"
        elif model_name.startswith(self.GEMINI_PREFIX):
            return "gemini"
        elif model_name.startswith(self.ANTHROPIC_PREFIX):
            return "anthropic"
        elif model_name.startswith(self.XAI_PREFIX):
            return "xai"

    def openai_can_reason(self, model_name: str) -> bool:
        return any(x in model_name for x in REASONING_MODELS) and not any(
            x in model_name for x in REASONING_MODELS_BLACKLIST
        )

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __metadata__: dict | None = None,
    ) -> dict:
        """
        Inlet: Modify the incoming request by setting the model, adding metadata, and optional reasoning effort.
        """
        model_type = self.get_type(body)
        if not model_type:
            # No handleable type detected
            return body

        model_name = self.get_model_name(body)

        # Handle Gemini
        if model_type in ("gemini", "anthropic", "xai"):
            self.logger.info("Think Deeper: %s detected", model_type.title())
            metadata = body.get("metadata", {})
            metadata_features = metadata.get("features")
            if metadata_features is None:
                metadata_features = {}
                metadata["features"] = metadata_features

            metadata_features["reason"] = True
            return body

        self.logger.info("Think Deeper: OpenAI detected")

        effort = self.valves.REASONING_EFFORT
        if effort != "not set":
            self.logger.info("Think Deeper: Setting reasoning effort to %s", effort)
            body["reasoning_effort"] = effort

        # Check if the model can reason
        if self.openai_can_reason(model_name):
            self.logger.info("Think Deeper: Model %s can already reason", model_name)
            return body

        # Update model in body so downstream pipe knows which model to use
        body["model"] = self.valves.DEFAULT_MODEL

        # Set __metadata__ for downstream pipes
        model_info = Models.get_model_by_id(self.valves.DEFAULT_MODEL)
        if __metadata__ and model_info:
            __metadata__["model"] = model_info.model_dump()

        self.logger.info("Think Deeper: Forced model to %s", self.valves.DEFAULT_MODEL)

        # Pass the updated request body downstream
        return body

    async def outlet(
        self,
        body: dict,
        __metadata__: dict | None = None,
    ) -> dict:
        """
        Outlet: Finalize the response by setting necessary UI-related fields.
        Note:
            1) event emitters are not available here.
            2) the body in the outlet is DIFFERENT from the inlet body.
            Read more here: https://github.com/jrkropp/open-webui-developer-toolkit/blob/development/functions/filters/README.md
        """
        model_type = self.get_type(body)
        if model_type != "openai":
            # No alternate shenanigans
            return body

        model_name = self.get_model_name(body)
        if self.openai_can_reason(model_name):
            # Model didn't change
            return body

        self.logger.info("Think Deeper: Output detected")

        # Ensure the final assistant message has correct model fields for frontend display
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                # Update model
                self.logger.info(
                    "Think Deeper: Updating output model to %s",
                    self.valves.DEFAULT_MODEL,
                )
                last_msg["model"] = self.valves.DEFAULT_MODEL
                last_msg.setdefault("modelName", self.valves.DEFAULT_MODEL)

        # Return the finalized response body ready for the UI
        return body
