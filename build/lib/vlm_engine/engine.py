import logging
import asyncio
from .config_models import EngineConfig
from .pipeline import PipelineManager
from .models import ModelManager
from .dynamic_ai import DynamicAIManager
from typing import Any, Dict

logger = logging.getLogger(__name__)

class VLMEngine:
    def __init__(self, config: EngineConfig):
        """
        Initialize the VLMEngine with the provided configuration object.
        
        Args:
            config: EngineConfig object containing the configuration
        """
        self.config = config
        self.model_manager = ModelManager(self.config.models)
        # Load active AI models from config
        active_ai_models = getattr(self.config, 'active_ai_models', ['vlm_nsfw_model'])
        self.dynamic_ai_manager = DynamicAIManager(self.model_manager, active_ai_models)
        self.pipeline_manager = PipelineManager(self.config.pipelines, self.model_manager, self.config.category_config, self.dynamic_ai_manager)
        
    async def initialize(self):
        """Initializes the pipelines."""
        await self.pipeline_manager.load_pipelines()
        
    async def process_video(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process a video and return tagging information.
        
        Args:
            video_path: Path to the video file
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary containing tagging information
        """
        pipeline_name = kwargs.get("pipeline_name", "video_pipeline_dynamic")
        data = [
            video_path,
            kwargs.get("return_timestamps", True),
            kwargs.get("frame_interval", 0.5),
            kwargs.get("threshold", 0.5),
            kwargs.get("return_confidence", True),
            kwargs.get("vr_video", False),
            kwargs.get("existing_json_data", None),
            kwargs.get("skipped_categories", None),
        ]
        
        future = await self.pipeline_manager.get_request_future(data, pipeline_name)
        result = await future
        return result
