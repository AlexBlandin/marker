# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Marker converts documents (PDF, images, DOCX, PPTX, XLSX, HTML, EPUB) to markdown, JSON, HTML, and chunked output formats. It uses ML models (via surya-ocr) for layout detection, text recognition, table recognition, and OCR, with optional LLM enhancement (Gemini, Claude, OpenAI, Azure, Ollama, Vertex).

## Commands

```bash
# Install (development)
poetry install --extras "full"

# Run all tests (requires GPU and HF_TOKEN env var for dataset access)
poetry run pytest

# Run a single test
poetry run pytest tests/processors/test_table.py::test_table_processor

# Lint and format (via pre-commit)
pre-commit run --all-files

# Convert a single file
poetry run marker_single /path/to/file.pdf --output_format markdown

# Batch convert
poetry run marker /path/to/input/dir --output_dir /path/to/output

# Start FastAPI server
poetry run marker_server --port 8001
```

## Architecture

### Pipeline: Provider → Builders → Processors → Renderer

The conversion pipeline in `PdfConverter.__call__` (`marker/converters/pdf.py`) follows four stages:

1. **Providers** (`marker/providers/`) — Extract raw data from input files. `provider_from_filepath()` in `marker/providers/registry.py` maps file extensions to provider classes.
2. **Builders** (`marker/builders/`) — Construct a `Document` with layout, text lines, OCR, and hierarchical structure. Runs: `DocumentBuilder` → `LayoutBuilder` → `LineBuilder` → `OcrBuilder` → `StructureBuilder`.
3. **Processors** (`marker/processors/`) — Transform document blocks in sequence. `PdfConverter.default_processors` defines the 27-processor chain. LLM processors (in `marker/processors/llm/`) only activate when `use_llm=True`.
4. **Renderers** (`marker/renderers/`) — Convert the processed `Document` to an output format (Markdown, JSON, HTML, chunks, extraction).

### Dependency Injection

`BaseConverter.resolve_dependencies()` (`marker/converters/__init__.py`) inspects class `__init__` signatures and auto-injects matching objects from `artifact_dict` (models, config, services). This is how ML models and LLM services flow into builders and processors.

### Configuration System

- **Settings** (`marker/settings.py`) — Pydantic `BaseSettings`, reads from env vars and `local.env`. Key: `TORCH_DEVICE`, `GOOGLE_API_KEY`.
- **ConfigParser** (`marker/config/parser.py`) — Parses CLI args, generates config dicts passed to all components.
- **Annotated types** — All configurable attributes on processors/builders/renderers use `Annotated[Type, "description"]`. `ConfigCrawler` (`marker/config/crawler.py`) introspects these for auto-generated help and JSON schemas.
- **assign_config()** — Utility that maps config dict values to class instance attributes; called in all component `__init__` methods.

### Document Schema

- **Document** (`marker/schema/document.py`) — Top-level container holding pages and metadata.
- **Block** (`marker/schema/blocks/base.py`) — Base class for all content elements. Has `polygon` (bounding box), `block_type` enum, and `structure_blocks()` for child management.
- **BlockTypes** (`marker/schema/__init__.py`) — Enum of ~29 block types (Text, Table, Figure, Equation, Code, etc.).
- **Block Registry** (`marker/schema/registry.py`) — `register_block_class()` allows overriding default block implementations via `PdfConverter.override_map`.

### Converter Variants

All inherit from `BaseConverter` (`marker/converters/__init__.py`):

| Converter | Purpose |
|-----------|---------|
| `PdfConverter` | Full document conversion (default) |
| `TableConverter` | Table-only extraction |
| `OCRConverter` | OCR-only, returns character-level bounding boxes |
| `ExtractionConverter` | Schema-driven structured extraction via LLM |

### ML Models

Loaded once via `create_model_dict()` in `marker/models.py`. All from `surya-ocr`:

- `layout_model` — Page layout detection
- `recognition_model` — Text recognition
- `table_rec_model` — Table structure recognition
- `detection_model` — Text detection
- `ocr_error_model` — OCR error detection

## Testing

- Framework: pytest, config in `pytest.ini`
- Test fixtures download PDFs from the `datalab-to/pdfs` HuggingFace dataset (requires `HF_TOKEN`)
- Models are **session-scoped** fixtures (loaded once per test session) — see `tests/conftest.py`
- Use `@pytest.mark.filename("name.pdf")` to select a test PDF from the dataset
- Use `@pytest.mark.config({...})` to pass config to test fixtures
- Use `@pytest.mark.output_format("markdown"|"json"|"html"|"chunks")` to select renderer
- CI runs on a `t4_gpu` runner — tests require GPU access

## Code Style

- Ruff for linting and formatting (defaults, no custom config)
- Pre-commit hooks configured in `.pre-commit-config.yaml`
