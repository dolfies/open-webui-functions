"""
title: Generate Image
author: @G30
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui/open-webui
version: 0.1.5
license: MIT
required_open_webui_version: 0.6.41
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIyLjMiIHN0cm9rZT0iY3VycmVudENvbG9yIj48cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Im0yLjI1IDE1Ljc1IDUuMTU5LTUuMTU5YTIuMjUgMi4yNSAwIDAgMSAzLjE4MiAwbDUuMTU5IDUuMTU5bS0xLjUtMS41IDEuNDA5LTEuNDA5YTIuMjUgMi4yNSAwIDAgMSAzLjE4MiAwbDIuOTA5IDIuOTA5bS0xOCAzLjc1aDE2LjVhMS41IDEuNSAwIDAgMCAxLjUtMS41VjZhMS41IDEuNSAwIDAgMC0xLjUtMS41SDMuNzVBMS41IDEuNSAwIDAgMCAyLjI1IDZ2MTJhMS41IDEuNSAwIDAgMCAxLjUgMS41Wm0xMC41LTExLjI1aC4wMDh2LjAwOGgtLjAwOFY4LjI1Wm0uMzc1IDBhLjM3NS4zNzUgMCAxIDEtLjc1IDAgLjM3NS4zNzUgMCAwIDEgLjc1IDBaIiAvPjwvc3ZnPg==
"""

from pydantic import BaseModel, Field
from typing import Optional

from open_webui.routers.images import image_generations, CreateImageForm
from open_webui.models.users import UserModel


class Action:
    class Valves(BaseModel):
        emit_status_events: bool = Field(
            default=True,
            description="Enable or disable status updates during generation.",
        )
        emit_images_incrementally: bool = Field(
            default=False,
            description="Emit images one by one as they complete instead of waiting for all to finish.",
        )
        image_count: int = Field(
            default=1, description="Number of images to generate (n)."
        )
        image_size: str = Field(
            default="1024x1024", description="Image dimensions (e.g. 512x512)."
        )
        image_steps: int = Field(
            default=20, description="Number of denoising steps (e.g. 20)."
        )
        bulk_image_steps: int = Field(
            default=10,
            description="Steps to use when generating multiple images (if count > 1).",
        )

    class UserValves(BaseModel):
        emit_status_events: bool = Field(
            default=True,
            description="Enable or disable status updates during generation.",
        )
        emit_images_incrementally: bool = Field(
            default=False,
            description="Emit images one by one as they complete instead of waiting for all to finish.",
        )
        image_count: int = Field(
            default=1, description="Number of images to generate (n)."
        )
        image_size: str = Field(
            default="1024x1024", description="Image dimensions (e.g. 512x512)."
        )
        image_steps: int = Field(
            default=20, description="Number of denoising steps (e.g. 20)."
        )
        bulk_image_steps: int = Field(
            default=10,
            description="Steps to use when generating multiple images (if count > 1).",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
        __request__=None,
        __model__=None,
    ) -> Optional[dict]:

        # Determine effective valves (UserValves overrides global Valves)
        valves = self.valves
        if __user__ and "valves" in __user__:
            valves = __user__["valves"]

        if __event_emitter__ and valves.emit_status_events:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Generating {valves.image_count} Image(s)...",
                        "done": False,
                    },
                }
            )

        try:
            # Check if 'messages' exists in body and get the last message content as prompt
            messages = body.get("messages", [])
            if messages:
                prompt = messages[-1]["content"]
            else:
                # Fallback if no messages found
                prompt = "Generate an image"

            if __request__ and __user__:
                # Convert __user__ dict to UserModel check if it's already a model or dict
                if isinstance(__user__, dict):
                    user_obj = UserModel(**__user__)
                else:
                    user_obj = __user__

                # Determine steps to use
                steps_to_use = valves.image_steps
                if valves.image_count > 1 and valves.bulk_image_steps > 0:
                    steps_to_use = valves.bulk_image_steps

                images = []

                # SEQUENTIAL LOOP: Run requests one by one to prevent database locking or list aggregation issues
                for i in range(valves.image_count):
                    try:
                        single_image_form = CreateImageForm(
                            prompt=prompt,
                            n=1,
                            size=valves.image_size,
                            steps=steps_to_use,
                        )

                        # Await each call individually
                        res = await image_generations(
                            request=__request__,
                            form_data=single_image_form,
                            user=user_obj,
                        )

                        if isinstance(res, list):
                            images.extend(res)

                            # Emit incrementally if enabled
                            if __event_emitter__ and valves.emit_images_incrementally:
                                for image in res:
                                    await __event_emitter__(
                                        {
                                            "type": "message",
                                            "data": {
                                                "content": f"![Generated Image]({image['url']})",
                                            },
                                        }
                                    )

                                # Update progress status
                                if valves.emit_status_events:
                                    await __event_emitter__(
                                        {
                                            "type": "status",
                                            "data": {
                                                "description": f"Generated {len(images)}/{valves.image_count} Image(s)...",
                                                "done": False,
                                            },
                                        }
                                    )

                    except Exception as inner_e:
                        print(f"Generation {i+1} failed: {inner_e}")

                if not images and valves.image_count > 0:
                    raise Exception("All image generation requests failed.")

                # Aggregated Emit: Only emit if incremental mode is disabled
                if __event_emitter__ and not valves.emit_images_incrementally:
                    content_list = []
                    for image in images:
                        content_list.append(f"![Generated Image]({image['url']})")

                    full_content = "\n".join(content_list)

                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": full_content,
                            },
                        }
                    )

                # Final status update
                if __event_emitter__ and valves.emit_status_events:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Generated {len(images)} Image(s)",
                                "done": True,
                            },
                        }
                    )

        except Exception as e:
            if __event_emitter__ and valves.emit_status_events:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {e}", "done": True},
                    }
                )
            print(f"Error generating image: {e}")

        return None
