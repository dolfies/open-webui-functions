"""
title: Anthropic Manifold
authors: Balaxxe, nbellochi, Bermont, Mark Kazakov, Christian Taillon, Dolfies
author_url: https://github.com/christian-taillon
funding_url: https://github.com/open-webui
version: 8.5
license: MIT
requirements: pydantic>=2.0.0, aiohttp>=3.8.0
"""

from __future__ import annotations

import os
import json
import re
import logging
import asyncio
import random
from typing import Union, Optional, AsyncGenerator, Any
import aiohttp
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    # Core API and Header Configuration
    API_VERSION = "2023-06-01"
    MESSAGES_URL = "https://api.anthropic.com/v1/messages"
    MODELS_URL = "https://api.anthropic.com/v1/models"
    REQUEST_TIMEOUT = 300
    BETA_HEADERS = {
        "CACHING": "prompt-caching-2024-07-31",
        "PDF": "pdfs-2024-09-25",
        "OUTPUT_128K": "output-128k-2025-02-19",
        "CONTEXT_1M": "context-1m-2025-08-07",
        "INTERLEAVED_THINKING": "interleaved-thinking-2025-05-14",
    }
    # Static capability metadata to enrich the dynamic model list

    MODEL_MAX_TOKENS = {
        # Claude 3 Series
        "claude-3-haiku": 4096,
        "claude-3-5-haiku": 8192,
        "claude-3-7-sonnet": 64000,
        # Claude 4 Series
        "claude-opus-4": 128000,
        "claude-sonnet-4": 64000,
        "claude-opus-4-1": 128000,
        # Claude 4.5 Series
        "claude-opus-4-5": 128000,
        "claude-sonnet-4-5": 64000,
        "claude-haiku-4-5": 64000,
        "claude-opus-4-6": 128000,
        "claude-sonnet-4-6": 64000,
        "claude-haiku-4-6": 64000,
    }

    THINKING_CAPABLE_MODELS = {
        # Claude 3.7+
        "claude-3-7-sonnet",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-haiku-4",
    }

    # Models that support adaptive thinking
    ADAPTIVE_THINKING_MODELS = {
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    }

    # Models that need the interleaved-thinking beta header for interleaved thinking
    # (4.6 models get it automatically via adaptive mode, 3.7 does not support it)
    INTERLEAVED_THINKING_BETA_MODELS = {
        "claude-opus-4",
        "claude-opus-4-1",
        "claude-opus-4-5",
        "claude-sonnet-4",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-haiku-4-6",
    }

    # Models that support web search and web fetch server tools
    WEB_SEARCH_CAPABLE_MODELS = {
        "claude-3-5-haiku",
        "claude-3-7-sonnet",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-haiku-4",
    }

    # Latest tool versions
    WEB_SEARCH_TOOL_TYPE = "web_search_20260209"
    WEB_FETCH_TOOL_TYPE = "web_fetch_20260209"

    # File and Content Constants
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    MAX_PDF_SIZE = 32 * 1024 * 1024

    class Valves(BaseModel):
        """Configurable settings for the Anthropic pipe."""

        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="Your Anthropic API key (get from console.anthropic.com)",
        )

        DISPLAY_THINKING: bool = Field(
            default=True,
            description="Display Claude's thinking process in chat (when thinking enabled)",
        )

        MAX_OUTPUT_TOKENS: bool = Field(
            default=True,
            description="Use maximum output tokens (128K for 3.7/4.x models)",
        )

        ENABLE_TOOL_CHOICE: bool = Field(default=True, description="Enable tool/function calling capabilities")

        ENABLE_CACHING: bool = Field(
            default=True,
            description="Enable prompt caching to reduce costs and latency",
        )

        SHOW_CACHE_INFO: bool = Field(default=True, description="Display cache hit statistics and cost savings")

        ENABLE_1M_CONTEXT: bool = Field(
            default=False,
            description="Enable 1M token context for Claude 4.x (beta feature)",
        )

        CLAUDE_45_USE_TEMPERATURE: bool = Field(
            default=True,
            description="Claude 4.5: Use temperature (True) or top_p (False) sampling",
        )

        REQUEST_TIMEOUT: int = Field(
            default=300,
            description="API request timeout in seconds (default: 5 minutes)",
        )

        MODEL_CACHE_TTL: int = Field(
            default=3600,
            description="Model list cache duration in seconds (default: 1 hour, 0 to disable)",
        )

        THINKING_BUDGET: int = Field(
            default=-1,
            ge=-1,
            le=128000,
            description="Thinking token budget when reasoning is enabled (-1 = model default: 16K for 3.7, 32K for 4.x). Min effective: 1024, must be < max_tokens. Ignored for adaptive thinking models (4.6).",
        )

    class UserValves(BaseModel):
        """User-overridable settings for the Anthropic pipe."""

        THINKING_BUDGET: Optional[int] = Field(
            default=None,
            ge=-1,
            le=128000,
            description="Override thinking budget (-1 = model default: 16K for 3.7, 32K for 4.x). Min effective: 1024, must be < max_tokens. Ignored for adaptive thinking models (4.6).",
        )

    def __init__(self):
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self.request_id = None
        self._models_list_cache = None
        self._models_cache_time = None

    def _get_model_base(self, model_id: str) -> str:
        """
        Extracts the base name of a model for capability lookups by stripping
        version metadata and matching against known MODEL_MAX_TOKENS keys.
        """
        # Normalize: Remove date suffixes (e.g., -20241022) or -latest
        base_id = re.sub(r"-(\d{8}|latest)$", "", model_id).lower()

        if base_id in self.MODEL_MAX_TOKENS:
            return base_id

        # 3Fallback logic: Find the best match among keys to handle variants
        # like 'claude-sonnet-4-5-v1' matching 'claude-sonnet-4-5'
        # Sorted by length descending to ensure the most specific match (e.g., 4-5 vs 4)
        sorted_keys = sorted(self.MODEL_MAX_TOKENS.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in base_id:
                return key

        return "claude-3-7-sonnet"

    def _supports_thinking(self, model_id: str) -> bool:
        """Determines if a model supports the extended thinking feature."""
        base_model = self._get_model_base(model_id)

        # Check for matches in the capability set
        if any(model in base_model for model in self.THINKING_CAPABLE_MODELS):
            return True
        return False

    def _supports_adaptive_thinking(self, model_id: str) -> bool:
        """Determines if a model supports adaptive thinking (4.6 models)."""
        base_model = self._get_model_base(model_id)
        return base_model in self.ADAPTIVE_THINKING_MODELS

    def _supports_web_search(self, model_id: str) -> bool:
        """Determines if a model supports web search and web fetch tools."""
        base_model = self._get_model_base(model_id)
        return any(model in base_model for model in self.WEB_SEARCH_CAPABLE_MODELS)

    def _build_web_search_tools(self) -> list[dict]:
        """Build web search and web fetch tool definitions."""
        return [
            {"type": self.WEB_SEARCH_TOOL_TYPE, "name": "web_search"},
            {
                "type": self.WEB_FETCH_TOOL_TYPE,
                "name": "web_fetch",
                "citations": {"enabled": True},
            },
        ]

    def _get_thinking_budget(self, base_model: str, max_tokens: int, __user__: Optional[dict] = None) -> int:
        """Resolve thinking budget from user valves, admin valves, or model defaults, with validation."""
        budget = None

        if __user__:
            user_valves = __user__.get("valves")
            if user_valves and hasattr(user_valves, "THINKING_BUDGET"):
                budget = user_valves.THINKING_BUDGET

        if budget is None:
            budget = self.valves.THINKING_BUDGET

        if budget is None or budget < 0:
            budget = 16000 if "3-7" in base_model else 32000

        # Clamp to valid range (Anthropic minimum: 1024, must be < max_tokens)
        budget = max(budget, 1024)
        if budget >= max_tokens:
            budget = max(max_tokens - 1, 1024)

        return budget

    async def _fetch_and_format_models_from_api(self) -> list[dict]:
        """Fetches all models from the Anthropic API and formats them simply."""
        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": self.API_VERSION,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(self.MODELS_URL, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                all_models_raw = data.get("data", [])

        # Simply map the API response to the required format
        formatted_models = [{"id": model["id"], "name": model["display_name"]} for model in all_models_raw]
        return sorted(formatted_models, key=lambda x: x["name"])

    async def pipes(self) -> list[dict]:
        """Provides the list of available models, fetched dynamically."""
        if self._models_list_cache is None:
            if not self.valves.ANTHROPIC_API_KEY:
                return []
            try:
                self._models_list_cache = await self._fetch_and_format_models_from_api()
            except Exception as e:
                logging.error(f"Failed to fetch models dynamically: {e}.")
        return self._models_list_cache or []

    async def _emit_status(
        self,
        __event_emitter__: Optional[Any],
        description: str,
        done: bool = False,
        hidden: bool = False,
    ) -> None:
        """
        Emit status event to UI.

        Args:
            __event_emitter__: Event emitter from OpenWebUI
            description: Status description to display
            done: Whether the status is complete
            hidden: Whether the status is hidden
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                        "hidden": hidden,
                    },
                }
            )

    async def _emit_error(
        self,
        __event_emitter__: Optional[Any],
        error_msg: str,
    ) -> None:
        """Emit a chat:completion error event to the OpenWebUI UI."""
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "chat:completion",
                    "data": {
                        "done": True,
                        "error": {"detail": error_msg},
                    },
                }
            )

    async def _emit_source(
        self,
        __event_emitter__: Optional[Any],
        url: str,
        title: str,
    ) -> None:
        """Emit a source/citation event to the OpenWebUI UI."""
        if __event_emitter__:
            host = url.split("//", 1)[-1].split("/", 1)[0].lower().lstrip("www.")
            await __event_emitter__(
                {
                    "type": "source",
                    "data": {
                        "source": {"name": host or "source", "url": url},
                        "document": [title],
                        "metadata": [{"source": url}],
                    },
                }
            )

    def _format_error(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        http_status: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Format error responses consistently, extracting clean messages from Anthropic JSON errors.

        Args:
            message: Error message or raw JSON response body from the API
            error_code: Error category (API_ERROR, VALIDATION_ERROR, TIMEOUT, etc.)
            http_status: HTTP status code if from API
            request_id: Anthropic request ID for debugging

        Returns:
            Formatted error string for display
        """
        # Try to parse Anthropic's JSON error body and extract the human-readable message.
        # Anthropic may repeat the error JSON on multiple lines in the response body,
        # so try each line until one parses successfully.
        clean_message = message
        for line in message.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "error" in data:
                    err = data["error"]
                    err_type = err.get("type", "")
                    err_msg = err.get("message", "")
                    if err_msg:
                        clean_message = f"{err_type}: {err_msg}" if err_type else err_msg
                        break
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

        error_parts = [f"❌ {error_code}"]

        if http_status:
            error_parts.append(f"HTTP {http_status}")

        if request_id:
            error_parts.append(f"[Request: {request_id}]")

        error_parts.append(f"\n{clean_message}")

        return " | ".join(error_parts[:3]) + error_parts[-1]

    def _get_cache_info(self, usage_data: dict, model_id: str) -> str:
        """Formats cache usage information for display."""
        if not self.valves.SHOW_CACHE_INFO or not usage_data:
            return ""
        input_tokens, output_tokens, cached_tokens = (
            usage_data.get("input_tokens", 0),
            usage_data.get("output_tokens", 0),
            usage_data.get("cache_read_input_tokens", 0),
        )
        if cached_tokens > 0:
            cache_percentage = (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0
            return f"```\n✅ CACHE HIT: {cache_percentage:.1f}% cached.\n   Tokens: {input_tokens:,} In / {output_tokens:,} Out\n```\n\n"
        else:
            return (
                f"```\n❌ CACHE MISS: No cache used.\n   Tokens: {input_tokens:,} In / {output_tokens:,} Out\n```\n\n"
            )

    def _normalize_content_blocks(
        self,
        raw_content: Union[list, dict, str],
        _depth: int = 0,
        _visited: Optional[set] = None,
    ) -> list[dict]:
        """
        Normalize content into proper block format with safety checks.

        Args:
            raw_content: Content to normalize (list, dict, or string)
            _depth: Current recursion depth (internal use)
            _visited: Set of visited object IDs to detect circular refs (internal use)

        Returns:
            List of normalized content blocks

        Raises:
            ValueError: If circular reference or max depth exceeded
        """
        # Initialize visited set on first call
        if _visited is None:
            _visited = set()

        # Check maximum nesting depth (prevent stack overflow)
        MAX_DEPTH = 10
        if _depth > MAX_DEPTH:
            logging.error(f"Content nesting exceeds maximum depth of {MAX_DEPTH}")
            raise ValueError(
                f"Content structure too deeply nested (max {MAX_DEPTH} levels). "
                "This may indicate a circular reference or invalid structure."
            )

        # Circular reference detection for mutable objects
        if isinstance(raw_content, (list, dict)):
            obj_id = id(raw_content)
            if obj_id in _visited:
                logging.error(f"Circular reference detected in content at depth {_depth}")
                raise ValueError("Circular reference detected in content structure. Content cannot reference itself.")
            _visited.add(obj_id)

        blocks = []

        # Convert to list if needed
        if isinstance(raw_content, list):
            items = raw_content
        else:
            items = [raw_content]

        # Process each item
        for idx, item in enumerate(items):
            try:
                if isinstance(item, dict) and item.get("type"):
                    # Direct block - copy and continue
                    blocks.append(dict(item))

                elif isinstance(item, dict) and "content" in item:
                    # Nested content - recurse with depth tracking
                    nested_blocks = self._normalize_content_blocks(
                        item["content"],
                        _depth=_depth + 1,
                        _visited=_visited.copy(),  # Copy to allow sibling references
                    )
                    blocks.extend(nested_blocks)

                elif item is not None:
                    # Fallback - convert to text
                    # Handle unexpected types gracefully
                    try:
                        text_value = str(item)
                    except Exception as e:
                        logging.warning(
                            f"Failed to convert content item to string at depth {_depth}, "
                            f"index {idx}: {type(item).__name__}. Error: {e}"
                        )
                        text_value = f"<unconvertible {type(item).__name__}>"

                    # Skip empty text blocks — Anthropic rejects them
                    if text_value.strip():
                        blocks.append({"type": "text", "text": text_value})

                # else: skip None items

            except ValueError:
                # Re-raise validation errors (circular ref, max depth)
                raise

            except Exception as e:
                # Log unexpected errors but continue processing
                logging.error(
                    f"Error processing content item at depth {_depth}, index {idx}: {e}",
                    exc_info=True,
                )
                # Add error indicator instead of crashing
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[Error processing content item {idx}: {str(e)[:100]}]",
                    }
                )

        return blocks

    def _attach_cache_control(self, block: dict) -> dict:
        """Attach cache control to a content block."""
        if not isinstance(block, dict):
            return block
        if block.get("type") in {"thinking", "redacted_thinking"}:
            return block
        if not block.get("type"):
            block["type"] = "text"
            if "text" not in block:
                block["text"] = ""

        block["cache_control"] = {"type": "ephemeral"}
        return block

    def _prepare_system_blocks(self, system_message: Union[dict, str, None]) -> Optional[list[dict]]:
        """Prepare system message with cache control."""
        if not system_message:
            return None
        content = (
            system_message.get("content")
            if isinstance(system_message, dict) and "content" in system_message
            else system_message
        )
        normalized_blocks = self._normalize_content_blocks(content)
        cached_blocks = [self._attach_cache_control(block) for block in normalized_blocks]
        return cached_blocks if cached_blocks else None

    def _apply_cache_control_to_last_message(self, messages: list[dict]) -> None:
        """Apply cache control to the last user message."""
        if not messages:
            return
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return
        for block in reversed(last_message.get("content", [])):
            if isinstance(block, dict) and block.get("type") not in {
                "thinking",
                "redacted_thinking",
            }:
                self._attach_cache_control(block)
                break

    def _process_content_item(self, item: dict) -> dict:
        """Process a content item with comprehensive validation."""

        # Validate item structure
        if not isinstance(item, dict):
            raise ValueError(f"Content item must be a dict, got {type(item).__name__}")

        if "type" not in item:
            raise ValueError(f"Content item missing required 'type' field: {item}")

        item_type = item["type"]

        # Process image_url type
        if item_type == "image_url":
            if "image_url" not in item:
                raise ValueError("image_url type requires 'image_url' field")

            if not isinstance(item["image_url"], dict) or "url" not in item["image_url"]:
                raise ValueError("image_url must contain {'url': '...'}")

            url = item["image_url"]["url"]

            if url.startswith("data:image"):
                # Existing base64 processing with validation
                try:
                    mime_type, base64_data = url.split(",", 1)
                    media_type = mime_type.split(":", 1)[1].split(";", 1)[0]
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid data URL format: {e}")

                if media_type not in self.SUPPORTED_IMAGE_TYPES:
                    raise ValueError(
                        f"Unsupported image type: {media_type}. "
                        f"Supported types: {', '.join(self.SUPPORTED_IMAGE_TYPES)}"
                    )

                # Validate size
                estimated_size = len(base64_data) * 3 / 4
                if estimated_size > self.MAX_IMAGE_SIZE:
                    raise ValueError(
                        f"Image size ({estimated_size / 1024 / 1024:.2f}MB) exceeds "
                        f"maximum {self.MAX_IMAGE_SIZE / 1024 / 1024}MB"
                    )

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }

            # URL-based image
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Image URL must start with http:// or https://, got: {url[:50]}...")

            return {"type": "image", "source": {"type": "url", "url": url}}

        # Process pdf_url type
        if item_type == "pdf_url":
            if "pdf_url" not in item:
                raise ValueError("pdf_url type requires 'pdf_url' field")

            if not isinstance(item["pdf_url"], dict) or "url" not in item["pdf_url"]:
                raise ValueError("pdf_url must contain {'url': '...'}")

            url = item["pdf_url"]["url"]

            if url.startswith("data:application/pdf"):
                try:
                    _, base64_data = url.split(",", 1)
                except ValueError as e:
                    raise ValueError(f"Invalid PDF data URL format: {e}")

                estimated_size = len(base64_data) * 3 / 4
                if estimated_size > self.MAX_PDF_SIZE:
                    raise ValueError(
                        f"PDF size ({estimated_size / 1024 / 1024:.2f}MB) exceeds "
                        f"maximum {self.MAX_PDF_SIZE / 1024 / 1024}MB"
                    )

                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_data,
                    },
                }

            # URL-based PDF
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"PDF URL must start with http:// or https://, got: {url[:50]}...")

            return {"type": "document", "source": {"type": "url", "url": url}}

        # Pass through other types unchanged
        return item

    async def pipe(
        self,
        body: dict,
        __metadata__: dict[str, Any],
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
    ) -> Union[str, AsyncGenerator[str, None]]:
        if not self.valves.ANTHROPIC_API_KEY:
            return "Error: ANTHROPIC_API_KEY is not set."

        try:
            model_id = body["model"].split(".")[-1]
            base_model = self._get_model_base(model_id)

            features = __metadata__.get("features", {})
            can_think = self._supports_thinking(base_model)
            reasoning_enabled = features.get("reason")
            will_enable_thinking = can_think and reasoning_enabled

            system_message, messages = pop_system_message(body["messages"])
            processed_messages, beta_headers_needed = [], set()
            for msg in messages:
                content_list = (
                    msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
                )
                processed_content = []
                for item in content_list:
                    if item.get("type") == "pdf_url":
                        beta_headers_needed.add(self.BETA_HEADERS["PDF"])
                    processed_item = self._process_content_item(item)
                    if self.valves.ENABLE_CACHING and item.get("type") in [
                        "tool_calls",
                        "tool_results",
                    ]:
                        processed_item["cache_control"] = {"type": "ephemeral"}
                        beta_headers_needed.add(self.BETA_HEADERS["CACHING"])
                    processed_content.append(processed_item)
                # Anthropic rejects empty text content blocks
                processed_content = [
                    item
                    for item in processed_content
                    if not (item.get("type") == "text" and not item.get("text", "").strip())
                ]
                if not processed_content:
                    continue
                processed_messages.append({"role": msg["role"], "content": processed_content})

            # Retrieve the model-specific limit from our dictionary
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(base_model, 4096)

            # 128K Override: Only for specific models that support this beta feature
            # Currently only 3.7 Sonnet is confirmed for 128k via the beta header.
            if "3-7-sonnet" in base_model and self.valves.MAX_OUTPUT_TOKENS:
                max_tokens_limit = 128000
                beta_headers_needed.add(self.BETA_HEADERS["OUTPUT_128K"])

            # Respect user-requested tokens, but clamp to the absolute model limit
            # This ensures we don't send 128000 to Opus-4 which only supports 64000
            requested_max = body.get("max_tokens")
            if requested_max:
                out_cap = min(requested_max, max_tokens_limit)
            else:
                out_cap = max_tokens_limit

            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": out_cap,
                "stream": body.get("stream", False),
            }

            is_new_gen = any(v in base_model for v in ["-4", "3-7"])

            if will_enable_thinking:
                use_adaptive = self._supports_adaptive_thinking(base_model)

                if use_adaptive:
                    # Adaptive thinking: model decides when/how much to think
                    reasoning_effort = __metadata__.get("reasoning_effort", "high")
                    payload["thinking"] = {"type": "adaptive"}
                    payload["output_config"] = {"effort": reasoning_effort}
                    logging.info(f"Enabled adaptive thinking, effort={reasoning_effort!r}")
                else:
                    # Manual thinking with budget
                    thinking_budget = self._get_thinking_budget(base_model, out_cap, __user__)
                    payload["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    }

                    # Enable interleaved thinking for supported models
                    interleaved = base_model in self.INTERLEAVED_THINKING_BETA_MODELS
                    if interleaved:
                        beta_headers_needed.add(self.BETA_HEADERS["INTERLEAVED_THINKING"])

                    logging.info(f"Enabled manual thinking with budget={thinking_budget}, interleaved={interleaved}")

                # Thinking requirement: temperature 1.0, no other samplers
                payload["temperature"] = 1.0
                payload.pop("top_p", None)
                payload.pop("top_k", None)
            elif is_new_gen:
                # Claude 4.5/3.7 Single-Sampler Enforcement
                if self.valves.CLAUDE_45_USE_TEMPERATURE:
                    payload["temperature"] = body.get("temperature", 1.0)
                    payload.pop("top_p", None)
                else:
                    payload["top_p"] = body.get("top_p", 1.0)
                    payload.pop("temperature", None)
                payload.pop("top_k", None)
            else:
                # Legacy Support (Claude 3 / 3.5)
                payload["temperature"] = body.get("temperature")
                payload["top_p"] = body.get("top_p")
                payload["top_k"] = body.get("top_k", 40)

            payload = {k: v for k, v in payload.items() if v is not None}

            if self.valves.ENABLE_CACHING:
                system_blocks = self._prepare_system_blocks(system_message)
                self._apply_cache_control_to_last_message(processed_messages)
                if system_blocks:
                    payload["system"] = system_blocks
                    beta_headers_needed.add(self.BETA_HEADERS["CACHING"])
            elif system_message:
                payload["system"] = str(system_message)

            if "tools" in body and self.valves.ENABLE_TOOL_CHOICE:
                payload["tools"], payload["tool_choice"] = body["tools"], body.get("tool_choice")

            # Web search tools (injected by web_search_toggle_filter)
            if features.get("claude_search") and self._supports_web_search(base_model):
                logging.info("Enabling web search")
                payload.setdefault("tools", []).extend(self._build_web_search_tools())

            if self.valves.ENABLE_1M_CONTEXT and ("-4" in base_model):
                beta_headers_needed.add(self.BETA_HEADERS["CONTEXT_1M"])

            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            }
            if beta_headers_needed:
                headers["anthropic-beta"] = ",".join(sorted(list(beta_headers_needed)))

            if payload.get("stream"):
                return self._stream_response(headers, payload, __event_emitter__, body)

            await self._emit_status(__event_emitter__, "Sending request...")
            response_data = await self._send_request(headers, payload)
            if isinstance(response_data, str):
                await self._emit_error(__event_emitter__, response_data)
                return response_data

            await self._emit_status(__event_emitter__, "Response received", done=True)
            content = response_data.get("content", [])
            if any(c.get("type") == "tool_use" for c in content):
                tool_calls = [
                    {
                        "id": c["tool_use"]["id"],
                        "type": "function",
                        "function": {
                            "name": c["tool_use"]["name"],
                            "arguments": json.dumps(c["tool_use"]["input"]),
                        },
                    }
                    for c in content
                    if c.get("type") == "tool_use"
                ]
                return json.dumps({"type": "tool_calls", "tool_calls": tool_calls})

            return "".join(c.get("text", "") for c in content if c.get("type") == "text")

        except Exception as e:
            logging.error(f"Error in pipe: {e}", exc_info=True)
            return f"An unexpected error occurred: {e}"

    async def _stream_response(
        self,
        headers: dict,
        payload: dict,
        __event_emitter__: Optional[Any] = None,
        body: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        is_thinking, is_tool_use = False, False
        is_server_tool, is_web_result = False, False
        tool_call_chunks = {}
        emitted_urls: set[str] = set()
        usage_data = None
        thinking_start_time = None  # Track start of reasoning
        last_thinking_title = None
        received_first = False

        try:
            # Emit connection status
            await self._emit_status(__event_emitter__, "Sending request...", done=False)

            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
                async with session.post(self.MESSAGES_URL, headers=headers, json=payload, timeout=timeout) as response:
                    self.request_id = response.headers.get("x-request-id")
                    logging.info(f"Streaming request initiated [Request ID: {self.request_id}]")
                    if response.status != 200:
                        error_msg = self._format_error(
                            message=await response.text(),
                            error_code="API_ERROR",
                            http_status=response.status,
                            request_id=self.request_id,
                        )
                        await self._emit_status(__event_emitter__, "Request failed", done=True)
                        await self._emit_error(__event_emitter__, error_msg)
                        return

                    # Emit streaming started
                    if not received_first:
                        await self._emit_status(__event_emitter__, "Response received", done=True)
                        received_first = True

                    async for line in response.content:
                        if not line.startswith(b"data: "):
                            continue
                        data = json.loads(line[6:])
                        event_type = data.get("type")
                        if event_type == "content_block_start":
                            block = data.get("content_block", {})
                            block_type = block.get("type")
                            if block_type == "thinking" or block_type == "redacted_thinking":
                                is_thinking = True
                                # Emit thinking started event
                                # await self._emit_status(
                                #     __event_emitter__,
                                #     "Claude is thinking...",
                                #     done=False,
                                # )
                                thinking_start_time = asyncio.get_event_loop().time()  # START CLOCK
                                if self.valves.DISPLAY_THINKING:
                                    yield "<thinking>"
                            elif block_type == "tool_use":
                                is_tool_use = True
                                tool_use = block.get("tool_use", {})
                                tool_name = tool_use.get("name", "unknown")
                                # Emit tool use detected
                                await self._emit_status(
                                    __event_emitter__,
                                    f"Using tool: {tool_name}",
                                    done=False,
                                )
                                tool_call_chunks[data["index"]] = {
                                    "id": tool_use.get("id"),
                                    "name": tool_name,
                                    "input_chunks": [],
                                }
                            elif block_type == "server_tool_use":
                                is_server_tool = True
                                server_tool_name = block.get("name", "")
                                if server_tool_name == "web_search":
                                    await self._emit_status(
                                        __event_emitter__,
                                        "Searching the web...",
                                        done=False,
                                    )
                                elif server_tool_name == "web_fetch":
                                    await self._emit_status(
                                        __event_emitter__,
                                        "Fetching web content...",
                                        done=False,
                                    )
                            elif block_type in (
                                "web_search_tool_result",
                                "web_fetch_tool_result",
                            ):
                                is_web_result = True
                                result_content = block.get("content")
                                if isinstance(result_content, list):
                                    for result in result_content:
                                        if result.get("type") == "web_search_result" and result.get("url"):
                                            url = result["url"]
                                            title = result.get("title", url)
                                            if url not in emitted_urls:
                                                emitted_urls.add(url)
                                                await self._emit_source(
                                                    __event_emitter__,
                                                    url,
                                                    title,
                                                )
                                elif isinstance(result_content, dict):
                                    if result_content.get("type") == "web_fetch_result" and result_content.get("url"):
                                        url = result_content["url"]
                                        title = url
                                        inner = result_content.get("content", {})
                                        if isinstance(inner, dict) and inner.get("title"):
                                            title = inner["title"]
                                        if url not in emitted_urls:
                                            emitted_urls.add(url)
                                            await self._emit_source(
                                                __event_emitter__,
                                                url,
                                                title,
                                            )
                            else:
                                is_thinking = is_tool_use = False
                                is_server_tool = is_web_result = False
                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if is_thinking and delta.get("type") == "thinking_delta":
                                text = delta.get("thinking", "")
                                if self.valves.DISPLAY_THINKING:
                                    yield text
                                # Extract heading titles for status updates
                                for m in re.finditer(r"(^|\n)###\s+(.+?)(?=\n|$)", text):
                                    title = m.group(2).strip().strip('""\u201c\u201d\u2018\u2019')
                                    if title and title != last_thinking_title:
                                        last_thinking_title = title
                                        await self._emit_status(__event_emitter__, title, done=False)
                                if last_thinking_title is None:
                                    for m in re.finditer(r"(^|\n)\s*\*\*(.+?)\*\*\s*(?=\n|$)", text):
                                        title = m.group(2).strip().strip('""\u201c\u201d\u2018\u2019')
                                        if title and title != last_thinking_title:
                                            last_thinking_title = title
                                            await self._emit_status(__event_emitter__, title, done=False)
                            elif is_tool_use and delta.get("type") == "input_json_delta":
                                tool_call_chunks[data["index"]]["input_chunks"].append(delta.get("partial_json", ""))
                            elif is_server_tool:
                                pass  # Skip server tool input deltas
                            elif is_web_result:
                                pass  # Skip web result content deltas
                            elif not is_thinking and not is_tool_use and delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                        elif event_type == "content_block_stop":
                            if is_thinking:
                                duration = 0.0
                                if thinking_start_time:
                                    duration = asyncio.get_event_loop().time() - thinking_start_time

                                # await self._emit_status(
                                #     __event_emitter__,
                                #     f"Thought for {duration:.1f} seconds",
                                #     done=False,
                                # )

                                if self.valves.DISPLAY_THINKING:
                                    yield "</thinking>"
                            if is_tool_use:
                                tool = tool_call_chunks.get(data["index"])
                                if tool:
                                    full_input = "".join(tool["input_chunks"])
                                    tool_call = {
                                        "id": tool["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tool["name"],
                                            "arguments": full_input,
                                        },
                                    }
                                    yield json.dumps(
                                        {
                                            "type": "tool_calls",
                                            "tool_calls": [tool_call],
                                        }
                                    )
                            is_thinking = is_tool_use = False
                            is_server_tool = is_web_result = False
                        elif event_type == "message_delta":
                            if "usage" in data:
                                usage_data = data["usage"]
                        elif event_type == "message_stop":
                            # Capture usage data
                            usage_data = data.get("usage")
                            usage_info = self._get_cache_info(usage_data, payload["model"])
                            await self._emit_status(
                                __event_emitter__,
                                "Request finished",
                                done=True,
                                hidden=True,
                            )
                            logging.info(f"Stream finished [Request ID: {self.request_id}]. {usage_info}")
                            break

                    # Yield cache info at end of stream
                    if self.valves.SHOW_CACHE_INFO and usage_data:
                        cache_info = self._get_cache_info(usage_data, payload["model"])
                        if cache_info:
                            yield cache_info
        except asyncio.TimeoutError as e:
            logging.error(f"Streaming timeout: {e}", exc_info=True)
            error_msg = self._format_error(
                message=f"Request timed out after {self.valves.REQUEST_TIMEOUT}s",
                error_code="TIMEOUT",
                request_id=self.request_id,
            )
            await self._emit_status(__event_emitter__, "Request timed out", done=True)
            await self._emit_error(__event_emitter__, error_msg)
        except aiohttp.ClientError as e:
            logging.error(f"Streaming network error: {e}", exc_info=True)
            error_msg = self._format_error(message=str(e), error_code="NETWORK_ERROR", request_id=self.request_id)
            await self._emit_status(__event_emitter__, "Network error occurred", done=True)
            await self._emit_error(__event_emitter__, error_msg)
        except Exception as e:
            logging.error(f"Streaming error: {e}", exc_info=True)
            error_msg = self._format_error(message=str(e), error_code="STREAM_ERROR", request_id=self.request_id)
            await self._emit_status(__event_emitter__, "Streaming error occurred", done=True)
            await self._emit_error(__event_emitter__, error_msg)

    async def _send_request(self, headers: dict, payload: dict) -> Union[dict, str]:
        for attempt in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
                    async with session.post(
                        self.MESSAGES_URL,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    ) as response:
                        self.request_id = response.headers.get("x-request-id")
                        logging.info(f"Non-streaming request sent [Request ID: {self.request_id}]")
                        if response.status == 200:
                            logging.info(f"Request successful [Request ID: {self.request_id}]")
                            return await response.json()
                        if response.status in [429, 500, 502, 503, 504] and attempt < 4:
                            delay = int(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                            await asyncio.sleep(delay + random.uniform(0, 1))
                            continue
                        return self._format_error(
                            message=await response.text(),
                            error_code="API_ERROR",
                            http_status=response.status,
                            request_id=self.request_id,
                        )
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep((2 ** (attempt + 1)) + random.uniform(0, 1))
                    continue
                return self._format_error(
                    message=f"Request timed out after {self.valves.REQUEST_TIMEOUT}s and multiple retries",
                    error_code="TIMEOUT",
                    request_id=self.request_id,
                )
            except aiohttp.ClientError as e:
                return self._format_error(
                    message=str(e),
                    error_code="NETWORK_ERROR",
                    request_id=self.request_id,
                )
        return self._format_error(
            message="Max retries exceeded",
            error_code="MAX_RETRIES",
            request_id=self.request_id,
        )
