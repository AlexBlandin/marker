import os
import time

import click

from marker.cache import CACHE_DIR_NAME, CACHE_FILENAME
from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.converters.llm_enhance import LLMEnhanceConverter
from marker.logger import configure_logging, get_logger
from marker.output import save_output

configure_logging()
logger = get_logger()


@click.command(
  cls=CustomClickPrinter, help="Run LLM enhancement on a single previously-converted PDF using cached document state."
)
@click.argument("fpath", type=str)
@click.option(
  "--cache_dir",
  type=str,
  default=None,
  help="Path to the .marker_cache directory. Defaults to <output_dir>/<basename>/.marker_cache",
)
@ConfigParser.common_options
def llm_enhance_single_cli(fpath: str, cache_dir: str, **kwargs):
  start = time.time()

  # Force use_llm so LLM processors activate
  kwargs["use_llm"] = True
  config_parser = ConfigParser(kwargs)

  config_dict = config_parser.generate_config_dict()

  llm_service = kwargs.get("llm_service")
  if not llm_service:
    raise click.UsageError("--llm_service is required for LLM enhancement.")

  converter = LLMEnhanceConverter(
    config=config_dict,
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=llm_service,
  )

  out_folder = config_parser.get_output_folder(fpath)
  if cache_dir is None:
    cache_dir = os.path.join(out_folder, CACHE_DIR_NAME)

  if not os.path.exists(os.path.join(cache_dir, CACHE_FILENAME)):
    raise click.UsageError(
      f"No document cache found at {cache_dir}. Run the first pass with --save_document_state to create it."
    )

  rendered = converter(cache_dir, fpath)
  base_name = config_parser.get_base_filename(fpath)
  save_output(rendered, out_folder, base_name)

  logger.info(f"Saved LLM-enhanced output to {out_folder}")
  logger.info(f"Total time: {time.time() - start}")


@click.command(
  cls=CustomClickPrinter,
  help="Run LLM enhancement on all previously-converted PDFs in a folder using cached document state.",
)
@click.argument("in_folder", type=str)
@click.option("--skip_existing", is_flag=True, default=False, help="Skip files that already have LLM-enhanced output.")
@ConfigParser.common_options
def llm_enhance_cli(in_folder: str, **kwargs):
  start = time.time()
  in_folder = os.path.abspath(in_folder)

  # Force use_llm so LLM processors activate
  kwargs["use_llm"] = True
  config_parser = ConfigParser(kwargs)

  config_dict = config_parser.generate_config_dict()

  llm_service = kwargs.get("llm_service")
  if not llm_service:
    raise click.UsageError("--llm_service is required for LLM enhancement.")

  # Find all PDF files
  files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
  files = [f for f in files if os.path.isfile(f) and f.lower().endswith(".pdf")]

  if not files:
    logger.warning(f"No PDF files found in {in_folder}")
    return

  converter = LLMEnhanceConverter(
    config=config_dict,
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=llm_service,
  )

  total_pages = 0
  for fpath in files:
    out_folder = config_parser.get_output_folder(fpath)
    cache_dir = os.path.join(out_folder, CACHE_DIR_NAME)
    cache_path = os.path.join(cache_dir, CACHE_FILENAME)

    if not os.path.exists(cache_path):
      logger.warning(f"No cache found for {fpath}, skipping. Run first pass with --save_document_state.")
      continue

    try:
      logger.info(f"LLM enhancing {fpath}")
      rendered = converter(cache_dir, fpath)
      base_name = config_parser.get_base_filename(fpath)
      save_output(rendered, out_folder, base_name)
      total_pages += converter.page_count or 0
      logger.info(f"Saved LLM-enhanced output to {out_folder}")
    except Exception as e:
      logger.error(f"Error enhancing {fpath}: {e}")

  total_time = time.time() - start
  logger.info(f"LLM-enhanced {total_pages} pages in {total_time:.2f} seconds")
