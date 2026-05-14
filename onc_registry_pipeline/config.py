"""Pipeline configuration for NAACCR v26 extraction."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


class PipelineConfig(BaseModel):
    """Configuration for the NAACCR extraction pipeline."""

    # LLM endpoint settings
    llm_provider: Literal["vllm", "azure-openai", "anthropic-vertex"] = "vllm"
    llm_model: str = "auto"
    llm_base_url: Optional[str] = None

    # vLLM server settings (kept for backwards-compatible CLI/config names)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "auto"  # discovered at runtime
    vllm_max_tokens: int = 16384
    vllm_temperature: float = 0.0
    vllm_timeout: int = 300
    vllm_reasoning_parser: Optional[str] = "auto"

    # Azure OpenAI v1 settings
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key_env: str = "AZURE_OPENAI_API_KEY"
    azure_openai_auth_mode: Literal["bearer", "api-key"] = "bearer"
    azure_openai_token_refresh_command: Optional[str] = (
        "az account get-access-token "
        "--resource=https://cognitiveservices.azure.com/ "
        "--query accessToken --output tsv"
    )

    # Anthropic Claude on Vertex AI settings
    anthropic_vertex_project_id: Optional[str] = None
    anthropic_vertex_region: Optional[str] = None
    anthropic_vertex_token_env: str = "ANTHROPIC_VERTEX_ACCESS_TOKEN"
    anthropic_vertex_token_refresh_command: Optional[str] = (
        "gcloud auth application-default print-access-token"
    )

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
