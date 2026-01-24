from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Tuple, Any

class ModelConfig(BaseModel):
    type: str
    max_queue_size: Optional[int] = None
    max_batch_size: int = 5
    instance_count: int = 5
    model_return_tags: bool = True
    model_return_confidence: bool = True
    model_category: Optional[Union[str, List[str]]] = None
    model_image_size: Optional[Union[int, Tuple[int, int]]] = None
    tag_list: Optional[List[str]] = None
    model_id: Optional[str] = None
    api_base_url: Optional[str] = None
    fill_to_batch_size: bool = True
    category_mappings: Optional[Dict[int, int]] = None
    normalization_config: Union[int, Dict[str, List[float]]] = 1
    # Multiplexer configuration for multiple endpoints
    multiplexer_endpoints: Optional[List[Dict[str, Any]]] = None
    use_multiplexer: bool = False
    # Python model function name
    function_name: Optional[str] = None

class PipelineModelConfig(BaseModel):
    name: str
    inputs: List[str]
    outputs: Union[str, List[str]]

class PipelineConfig(BaseModel):
    inputs: List[str]
    output: str
    version: float
    models: List[PipelineModelConfig]

class EngineConfig(BaseModel):
    pipelines: Dict[str, PipelineConfig]
    models: Dict[str, ModelConfig]
    category_config: Dict[str, Any]
    active_ai_models: Optional[List[str]] = None
