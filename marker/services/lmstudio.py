import json
import time
from typing import Annotated

import openai
import PIL
from openai import APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel

from marker.logger import get_logger
from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class LMStudioService(BaseService):
  lmstudio_base_url: Annotated[str, "The base url for the LM Studio server.  No trailing slash."] = (
    "http://localhost:1234/v1"
  )
  lmstudio_model: Annotated[str, "The model name to use for LM Studio."] = None
  lmstudio_api_key: Annotated[str, "The API key for LM Studio. Any string works when auth is disabled."] = "lm-studio"
  lmstudio_image_format: Annotated[
    str,
    "The image format to use for LM Studio. Use 'png' for better local model compatability.",
  ] = "png"
  timeout: Annotated[int, "The timeout to use for the service. Local models may need longer."] = 120

  def process_images(self, images: list[Image.Image]) -> list[dict]:
    if isinstance(images, Image.Image):
      images = [images]

    img_fmt = self.lmstudio_image_format
    return [
      {
        "type": "image_url",
        "image_url": {
          "url": f"data:image/{img_fmt};base64,{self.img_to_base64(img, format=img_fmt)}",
        },
      }
      for img in images
    ]

  def _build_response_format(self, response_schema: type[BaseModel]) -> dict:
    schema = response_schema.model_json_schema()
    name = schema.get("title", response_schema.__name__)
    return {
      "type": "json_schema",
      "json_schema": {
        "name": name,
        "schema": schema,
      },
    }

  def __call__(
    self,
    prompt: str,
    image: PIL.Image.Image | list[PIL.Image.Image] | None,
    block: Block | None,
    response_schema: type[BaseModel],
    max_retries: int | None = None,
    timeout: int | None = None,
  ):
    if max_retries is None:
      max_retries = self.max_retries

    if timeout is None:
      timeout = self.timeout

    client = self.get_client()
    image_data = self.format_image_for_llm(image)
    response_format = self._build_response_format(response_schema)

    messages = [
      {
        "role": "user",
        "content": [
          *image_data,
          {"type": "text", "text": prompt},
        ],
      }
    ]

    total_tries = max_retries + 1
    for tries in range(1, total_tries + 1):
      try:
        response = client.chat.completions.create(
          model=self.lmstudio_model,
          messages=messages,
          timeout=timeout,
          response_format=response_format,
        )
        message = response.choices[0].message
        response_text = self.strip_thinking_tags(message.content or "")
        if not response_text:
          # Reasoning models (Qwen3.5, DeepSeek-R1) may put output in reasoning_content
          response_text = self.strip_thinking_tags(getattr(message, "reasoning_content", None) or "")
        total_tokens = response.usage.total_tokens
        if block:
          block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
        return json.loads(response_text)
      except (APITimeoutError, RateLimitError) as e:
        if tries == total_tries:
          logger.error(
            f"LM Studio timeout: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
          )
          break
        wait_time = tries * self.retry_wait_time
        logger.warning(
          f"LM Studio timeout: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})",
        )
        time.sleep(wait_time)
      except json.JSONDecodeError as e:
        logger.warning(f"LM Studio returned invalid JSON: {e}")
        break
      except Exception as e:
        logger.error(f"LM Studio inference failed: {e}")
        break

    return {}

  def get_client(self) -> openai.OpenAI:
    return openai.OpenAI(api_key=self.lmstudio_api_key, base_url=self.lmstudio_base_url)
