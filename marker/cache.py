import json
import os

from marker.logger import get_logger
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.registry import get_block_class

logger = get_logger()

CACHE_DIR_NAME = ".marker_cache"
CACHE_FILENAME = "document.json"


def save_document_cache(document: Document, cache_dir: str):
  """Save processed document state to a JSON cache file.

  Excludes images (lowres_image, highres_image) from all blocks.
  Images will be re-rendered from the PDF on the LLM enhancement pass.
  """
  os.makedirs(cache_dir, exist_ok=True)

  pages_data = []
  for page in document.pages:
    page_data = page.model_dump(
      exclude={
        "lowres_image": True,
        "highres_image": True,
        "children": {"__all__": {"lowres_image": True, "highres_image": True}},
      }
    )
    pages_data.append(page_data)

  cache_data = {
    "filepath": document.filepath,
    "table_of_contents": [toc.model_dump() for toc in document.table_of_contents]
    if document.table_of_contents
    else None,
    "pages": pages_data,
  }

  cache_path = os.path.join(cache_dir, CACHE_FILENAME)
  with open(cache_path, "w", encoding="utf-8") as f:
    json.dump(cache_data, f)

  logger.info(f"Saved document cache to {cache_path}")


def _deserialize_block(child_dict: dict):
  """Deserialize a single block dict using the block registry."""
  block_type = BlockTypes(child_dict["block_type"])
  block_cls = get_block_class(block_type)
  return block_cls.model_validate(child_dict)


def _deserialize_children(children_data: list):
  """Deserialize all children in a page, preserving list order."""
  if children_data is None:
    return None
  return [_deserialize_block(child) for child in children_data]


def load_document_cache(cache_dir: str) -> Document:
  """Load a cached document from JSON.

  Returns a Document with all blocks deserialized but images set to None.
  The caller must re-render page images from the PDF.
  """
  cache_path = os.path.join(cache_dir, CACHE_FILENAME)
  with open(cache_path, "r", encoding="utf-8") as f:
    cache_data = json.load(f)

  pages = []
  for page_data in cache_data["pages"]:
    children_data = page_data.pop("children", None)
    deserialized_children = _deserialize_children(children_data)

    page = PageGroup.model_validate(page_data)
    page.children = deserialized_children
    pages.append(page)

  from marker.schema.document import TocItem

  toc = None
  if cache_data.get("table_of_contents"):
    toc = [TocItem.model_validate(item) for item in cache_data["table_of_contents"]]

  document = Document(
    filepath=cache_data["filepath"],
    pages=pages,
    table_of_contents=toc,
  )

  logger.info(f"Loaded document cache from {cache_path}")
  return document
