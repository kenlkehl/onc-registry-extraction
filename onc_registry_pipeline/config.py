"""Pipeline configuration for NAACCR v26 extraction."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


class PipelineConfig(BaseModel):
    """Configuration for the NAACCR extraction pipeline."""

    # vLLM server settings
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "auto"  # discovered at runtime
    vllm_max_tokens: int = 16384
    vllm_temperature: float = 0.0
    vllm_timeout: int = 300
    vllm_reasoning_parser: Optional[str] = "auto"

    # Context / chunking settings
    model_context_window: int = 131072  # discovered at runtime
    chunk_target_tokens: int = 50000
    chunk_overlap_tokens: int = 500

    # Extraction batching
    items_per_call: int = 50  # NAACCR items per LLM call

    # NAACCR data-dictionary CSV paths
    data_items_csv: Path = Path("NAACCRDataItems/DataItems.csv")
    code_list_csv: Path = Path("NAACCRDataItems/CodeList.csv")
    alternate_names_csv: Path = Path("NAACCRDataItems/AlternateNames.csv")

    # Vendored SEER/NAACCR registry manuals for prompt retrieval
    seer_manuals_dir: Path = Path("SEERManuals")
    seer_context_max_chars: int = 12000

    # Retry / concurrency
    max_retries: int = 3
    max_concurrent_patients: int = 16

    # Validation
    confidence_threshold: float = 0.7

    # Output
    output_format: Literal["naaccr_xml", "naaccr_flat", "csv"] = "naaccr_xml"

    # Checkpointing
    checkpoint_dir: Optional[Path] = None
