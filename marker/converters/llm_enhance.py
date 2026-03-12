from collections import defaultdict
from typing import Annotated, Any, Dict, List, Optional, Tuple

from marker.cache import load_document_cache
from marker.converters import BaseConverter
from marker.logger import get_logger
from marker.processors import BaseProcessor
from marker.processors.blank_page import BlankPageProcessor
from marker.processors.debug import DebugProcessor
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_equation import LLMEquationProcessor
from marker.processors.llm.llm_form import LLMFormProcessor
from marker.processors.llm.llm_handwriting import LLMHandwritingProcessor
from marker.processors.llm.llm_image_description import LLMImageDescriptionProcessor
from marker.processors.llm.llm_mathblock import LLMMathBlockProcessor
from marker.processors.llm.llm_page_correction import LLMPageCorrectionProcessor
from marker.processors.llm.llm_sectionheader import LLMSectionHeaderProcessor
from marker.processors.llm.llm_table import LLMTableProcessor
from marker.processors.llm.llm_table_merge import LLMTableMergeProcessor
from marker.processors.reference import ReferenceProcessor
from marker.processors.text import TextProcessor
from marker.providers.pdf import PdfProvider
from marker.renderers.markdown import MarkdownRenderer
from marker.schema.document import Document
from marker.util import strings_to_classes


class LLMEnhanceConverter(BaseConverter):
  """
  A converter that loads a cached document and runs only LLM processors.

  Avoids loading ML models (surya-ocr). Re-renders page images from the
  PDF using pypdfium2 (CPU only).
  """

  use_llm: Annotated[
    bool,
    "Enable higher quality processing with LLMs.",
  ] = True
  lowres_image_dpi: Annotated[
    int,
    "DPI setting for low-resolution page images.",
  ] = 96
  highres_image_dpi: Annotated[
    int,
    "DPI setting for high-resolution page images.",
  ] = 192
  flatten_pdf: Annotated[
    bool,
    "Whether to flatten the PDF structure when rendering images.",
  ] = True
  default_processors: Tuple[BaseProcessor, ...] = (
    LLMTableProcessor,
    LLMTableMergeProcessor,
    LLMFormProcessor,
    TextProcessor,
    LLMComplexRegionProcessor,
    LLMImageDescriptionProcessor,
    LLMEquationProcessor,
    LLMHandwritingProcessor,
    LLMMathBlockProcessor,
    LLMSectionHeaderProcessor,
    LLMPageCorrectionProcessor,
    ReferenceProcessor,
    BlankPageProcessor,
    DebugProcessor,
  )

  def __init__(
    self,
    artifact_dict: Optional[Dict[str, Any]] = None,
    processor_list: Optional[List[str]] = None,
    renderer: str | None = None,
    llm_service: str | None = None,
    config=None,
  ):
    super().__init__(config)

    if config is None:
      config = {}

    if artifact_dict is None:
      artifact_dict = {}

    self.artifact_dict = artifact_dict

    if llm_service:
      llm_service_cls = strings_to_classes([llm_service])[0]
      llm_service = self.resolve_dependencies(llm_service_cls)
    else:
      raise ValueError("LLMEnhanceConverter requires an llm_service to be specified.")

    self.artifact_dict["llm_service"] = llm_service
    self.llm_service = llm_service

    if processor_list is not None:
      processor_list = strings_to_classes(processor_list)
    else:
      processor_list = self.default_processors

    if renderer:
      renderer = strings_to_classes([renderer])[0]
    else:
      renderer = MarkdownRenderer

    self.renderer = renderer
    processor_list = self.initialize_processors(processor_list)
    self.processor_list = processor_list

  def _render_page_images(self, document: Document, pdf_path: str):
    """Re-render page images from the PDF using pypdfium2 (no ML models)."""
    page_idxs = [page.page_id for page in document.pages]
    provider = PdfProvider.__new__(PdfProvider)
    provider.filepath = pdf_path
    provider.flatten_pdf = self.flatten_pdf

    lowres_images = provider.get_images(page_idxs, self.lowres_image_dpi)
    highres_images = provider.get_images(page_idxs, self.highres_image_dpi)

    for i, page in enumerate(document.pages):
      page.lowres_image = lowres_images[i]
      page.highres_image = highres_images[i]

  def __call__(self, cache_dir: str, pdf_path: str):
    document = load_document_cache(cache_dir)
    self._render_page_images(document, pdf_path)

    # Snapshot block HTML before LLM processing
    snapshot = {}
    for block in document.contained_blocks():
      if hasattr(block, "html"):
        snapshot[block.id] = block.html

    for processor in self.processor_list:
      processor(document)

    self._log_change_summary(document, snapshot)

    self.document = document
    self.page_count = len(document.pages)

    renderer = self.resolve_dependencies(self.renderer)
    return renderer(document)

  def _log_change_summary(self, document, snapshot: dict):
    logger = get_logger()
    changes_by_type = defaultdict(list)

    for block in document.contained_blocks():
      if not hasattr(block, "html"):
        continue
      block_id = block.id
      old_html = snapshot.get(block_id)
      new_html = block.html
      if old_html == new_html:
        continue
      block_type = block_id.block_type.name if block_id.block_type else "Unknown"
      old_preview = (old_html or "")[:60].replace("\n", " ")
      new_preview = (new_html or "")[:60].replace("\n", " ")
      changes_by_type[block_type].append((str(block_id), old_preview, new_preview))

    if not changes_by_type:
      logger.info("LLM enhancement summary: no blocks modified")
      return

    total = sum(len(v) for v in changes_by_type.values())
    lines = ["LLM enhancement summary:"]
    for block_type, changes in sorted(changes_by_type.items()):
      lines.append(f"  {block_type}: {len(changes)} blocks modified")
      for block_id_str, old_preview, new_preview in changes[:5]:
        lines.append(f"    {block_id_str}: {old_preview}... -> {new_preview}...")
      if len(changes) > 5:
        lines.append(f"    ... and {len(changes) - 5} more")
    lines.append(f"  Total: {total} blocks modified across {len(changes_by_type)} block types")
    logger.info("\n".join(lines))
