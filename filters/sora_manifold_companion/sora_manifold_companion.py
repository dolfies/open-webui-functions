"""
title: Sora 2 Companion
id: openai_sora_2_companion
description: Companion filter for OpenAI Sora 2 Manifold to persist video metadata.
author: Dolfies
version: 0.1.0
license: MIT
requirements: loguru
"""

from pydantic import BaseModel
from fastapi import Request
from loguru import logger


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __request__: Request, __metadata__: dict) -> dict:
        return body

    def outlet(self, body: dict, __request__: Request, __metadata__: dict) -> dict:
        """
        Outlet runs after the Pipe has finished generating.
        It retrieves the Video ID stored in app.state by the Pipe.
        It uses the Chats model to find the User message in the DB and persist the ID.
        """
        chat_id = __metadata__.get("chat_id")
        message_id = __metadata__.get("message_id")

        # Skip temporary chats or missing context
        if not chat_id or chat_id == "local" or not message_id:
            return body

        # Retrieve video ID from app.state
        key = f"sora_video_id_{chat_id}_{message_id}"
        logger.debug(f"Trying to find Sora video ID in key {key}...")
        video_id = getattr(__request__.app.state, key, None)

        if video_id:
            logger.info(f"Filter: Found video ID {video_id} in state.")

            # The last message in body['messages'] is the Assistant response we just generated
            # Open WebUI saves this message to the DB after outlet returns
            if body.get("messages"):
                last_msg = body["messages"][-1]

                if last_msg.get("role") != "user":
                    if "info" not in last_msg or not isinstance(last_msg["info"], dict):
                        last_msg["info"] = {}

                    last_msg["info"]["sora_video_id"] = video_id
                    logger.info(
                        "Filter: Injected sora_video_id into Assistant message payload."
                    )
                else:
                    logger.warning(
                        "Filter: Last message was not an assistant message, skipping persistence."
                    )

            # Cleanup state
            try:
                delattr(__request__.app.state, key)
            except AttributeError:
                pass

        return body
