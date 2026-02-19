"""
title: Grok Manifold
author: Siraj Florida
author_url: https://github.com/sirajperson
funding_url: https://www.kinothen.com
version: 0.1.0
license: GNUv3
"""

import os
import json
import logging
import re
import requests
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        """Configuration for the Grok API interactions."""

        GROK_API_KEY: str = Field(default="", description="API key for Grok services.")
        GROK_API_BASE_URL: str = Field(
            default="https://api.x.ai/v1",
            description="Base URL for Grok API endpoints.",
        )
        MAX_TOKENS: int = Field(
            default=4096, description="Maximum number of tokens to generate."
        )
        TEMPERATURE: float = Field(default=0.8, description="Sampling temperature.")
        TOP_P: float = Field(default=0.9, description="Nucleus sampling top_p value.")
        STREAM: bool = Field(default=True, description="Whether to stream responses.")

    def __init__(self):
        self.valves = self.Valves(
            GROK_API_KEY=os.getenv("GROK_API_KEY", ""),
            GROK_API_BASE_URL=os.getenv("GROK_API_BASE_URL", "https://api.x.ai/v1"),
        )
        self.type = "manifold"
        self.id = "grok"
        self.logger = logging.getLogger(__name__)

    def get_model_id(self, model_name: str) -> str:
        """Extract just the base model name from any format"""
        # Split on both / and . to handle any format
        parts = model_name.replace(".", "/").split("/")
        # Return only the actual model name (e.g. "grok-beta")
        return parts[-1]

    def get_model_name(self, model_id: str) -> str:
        stripped_id = re.sub(r"-\d{4}$", "", model_id)
        return stripped_id.replace("-", " ").title()

    def get_grok_models(self):
        headers = {
            "Authorization": f"Bearer {self.valves.GROK_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.get(
                f"{self.valves.GROK_API_BASE_URL}/models", headers=headers
            )
            response.raise_for_status()
            models_data = response.json()
            return [
                {"id": model["id"], "name": self.get_model_name(model["id"])}
                for model in models_data.get("data", [])
                if not "-image" in model["id"]
            ]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def pipes(self) -> List[dict]:
        return self.get_grok_models()

    def process_image(self, image_data):
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["image_url"]["url"]},
            }

    def pipe(self, body: dict, __metadata__: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body.get("messages", []))

        processed_messages = []
        for message in messages:
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_messages.append(
                            {"role": message["role"], "content": item["text"]}
                        )
                    elif item["type"] == "image_url":
                        processed_image = self.process_image(item)
                        processed_messages.append(processed_image)
            else:
                processed_messages.append(
                    {"role": message["role"], "content": message.get("content", "")}
                )

        # Include system message if present
        if system_message:
            processed_messages.insert(
                0, {"role": "system", "content": str(system_message)}
            )

        # Extract just the base model name
        model_id = self.get_model_id(body["model"])

        # Structure payload according to API spec
        payload = {
            "model": model_id,
            "messages": processed_messages,
            "stream": body.get("stream", self.valves.STREAM),
            "temperature": body.get("temperature", self.valves.TEMPERATURE),
            "max_tokens": body.get("max_tokens", self.valves.MAX_TOKENS),
            "top_p": body.get("top_p", self.valves.TOP_P),
            "frequency_penalty": body.get("frequency_penalty", 0),
            "presence_penalty": body.get("presence_penalty", 0),
            "stop": body.get("stop", []),
            "user": body.get("user", ""),
            "n": body.get("n", 1),
            "logprobs": body.get("logprobs", False),
            "top_logprobs": body.get("top_logprobs", 0),
            "search_parameters": {"mode": "auto", "return_citations": True},
        }

        features = __metadata__.get("features", {})

        # Force search
        if features.get("grok_search"):
            payload["search_parameters"]["mode"] = "on"
        if features.get("grok_search_size") is not None:
            payload["search_parameters"]["max_search_results"] = features[
                "grok_search_size"
            ]

        # Higher reasoning
        if features.get("reason") and model_id.startswith("grok-3-mini"):
            self.logger.info("Setting reasoning effort to HIGH")
            payload["reasoning_effort"] = "high"
        if (
            model_id.startswith("grok-4")
            or model_id.startswith("grok-code")
            or model_id.startswith("grok-3-mini")
        ):
            del payload["presence_penalty"]
            del payload["frequency_penalty"]
            del payload["stop"]

        headers = {
            "Authorization": f"Bearer {self.valves.GROK_API_KEY}",
            "Content-Type": "application/json",
        }

        url = f"{self.valves.GROK_API_BASE_URL}/chat/completions"

        try:
            if payload["stream"]:
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON: {line}")
                        except KeyError as e:
                            print(f"Unexpected data structure: {e}")
                            print(f"Full data: {data}")

    def non_stream_response(self, url, headers, payload):
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"HTTP Error {response.status_code}: {response.text}")

        res = response.json()
        return res["choices"][0]["message"]["content"] if res.get("choices") else ""
