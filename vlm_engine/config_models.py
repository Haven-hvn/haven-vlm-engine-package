from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Tuple, Any

class ModelConfig(BaseModel):
    type: str
    max_queue_size: Optional[int] = None
    max_batch_size: int = 1
    instance_count: int = 1
    model_threshold: Optional[float] = 0.5
    model_return_tags: bool = True
    model_return_confidence: bool = True
    model_category: Optional[Union[str, List[str]]] = None
    model_version: Optional[str] = None
    model_identifier: Optional[Any] = None
    model_image_size: Optional[Union[int, Tuple[int, int]]] = None
    model_info: Optional[str] = None
    tag_list: Optional[List[str]] = None
    vlm_model_name: Optional[str] = None
    use_quantization: bool = False
    model_id: Optional[str] = None
    api_base_url: Optional[str] = None
    fill_to_batch_size: bool = True
    device: Optional[str] = None
    category_mappings: Optional[Dict[int, int]] = None
    normalization_config: Union[int, Dict[str, List[float]]] = 1
    model_file_name: Optional[str] = None
    model_license_name: Optional[str] = None
    # Multiplexer configuration for multiple endpoints
    multiplexer_endpoints: Optional[List[Dict[str, Any]]] = None
    use_multiplexer: bool = False

class PipelineModelConfig(BaseModel):
    name: str
    inputs: List[str]
    outputs: Union[str, List[str]]

class PipelineConfig(BaseModel):
    inputs: List[str]
    output: str
    short_name: str
    version: float
    models: List[PipelineModelConfig]

class EngineConfig(BaseModel):
    loglevel: str = "INFO"
    pipelines: Dict[str, PipelineConfig]
    models: Dict[str, ModelConfig]
    category_config: Dict[str, Any]
    active_ai_models: Optional[List[str]] = None
