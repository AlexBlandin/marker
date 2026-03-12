# LM Studio API Reference

LM Studio is a local model server for running LLMs on your own hardware. Default base URL: `http://localhost:1234`.

Source: [LM Studio Developer Docs](https://lmstudio.ai/docs/developer) and local clone in `references/LMStudio/`.

## API Surfaces

LM Studio exposes two HTTP API surfaces:

| Surface | Base path | Structured output | Vision | Notes |
|---------|-----------|-------------------|--------|-------|
| Native v1 REST | `/api/v1/*` | No | Yes | Stateful chats, MCP integration |
| OpenAI-compatible | `/v1/*` | Yes (`response_format`) | Yes | Drop-in OpenAI replacement |

**For marker, use the OpenAI-compatible endpoint** — it's the only one that supports structured JSON output via `response_format` with `json_schema`, which marker requires for Pydantic schema-driven responses.

## OpenAI-Compatible Chat Completions

### Endpoint

`POST /v1/chat/completions`

### Supported parameters

```
model              messages           temperature        top_p
top_k              max_tokens         stream             stop
presence_penalty   frequency_penalty  logit_bias         repeat_penalty
seed               response_format
```

All parameters follow OpenAI semantics. See [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create).

### Python client setup

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # Any string works when auth is disabled
)
```

## Vision / Image Input

Images are sent as base64 data URLs in the message content array, using the standard OpenAI multimodal format:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/webp;base64,<base64-encoded-data>",
                },
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```

Requires a vision-capable model (e.g. `qwen/qwen3-vl-4b`).

## Structured Output

Enforce JSON output matching a schema via `response_format`:

```python
response = client.chat.completions.create(
    model="your-model",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"}
                },
                "required": ["field"],
            },
        },
    },
)

# Response is a JSON string that must be parsed
result = json.loads(response.choices[0].message.content)
```

The `openai` library's `client.beta.chat.completions.parse()` with `response_format=PydanticModel` should also work, since LM Studio follows the OpenAI structured output format.

### Limitations

- Models below 7B parameters may not support structured output.
- GGUF models use llama.cpp grammar-based sampling; MLX models use [Outlines](https://github.com/dottxt-ai/outlines).

## Authentication

Authentication is **disabled by default**. When enabled in LM Studio's server settings:

- Header: `Authorization: Bearer <token>`
- Tokens are created in Developer Page > Server Settings > Manage Tokens
- Tokens are shown once at creation and cannot be retrieved later

When auth is disabled, the `api_key` parameter can be any non-empty string (e.g. `"lm-studio"`).

## Token Usage

The OpenAI-compatible endpoint returns token usage in the standard format:

```json
{
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 45,
    "total_tokens": 168
  }
}
```

Accessible via `response.usage.total_tokens` in the Python client.

## Native v1 REST API (not used by marker)

Documented here for completeness. The native endpoint at `POST /api/v1/chat` uses a different format:

- Input: `"input"` field (string or array of `{"type": "text"|"image", ...}` objects)
- Images: `{"type": "image", "data_url": "data:image/png;base64,..."}`
- Response: `"output"` array of `{"type": "message"|"tool_call"|"reasoning", ...}` items
- Token stats: `"stats": {"input_tokens": N, "total_output_tokens": N, ...}`
- Unique features: stateful chats (`store`, `previous_response_id`), MCP integrations

This endpoint does **not** support `response_format` / structured output.
