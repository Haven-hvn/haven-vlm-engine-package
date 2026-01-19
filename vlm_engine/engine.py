import logging
import asyncio
import time
import traceback
from .config_models import EngineConfig
from .pipeline import PipelineManager
from .models import ModelManager
from .dynamic_ai import DynamicAIManager
from typing import Any, Dict, Optional, Callable

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
        active_ai_models = getattr(self.config, 'active_ai_models',['llm_vlm_model'])
        self.dynamic_ai_manager = DynamicAIManager(self.model_manager, active_ai_models)
        self.pipeline_manager = PipelineManager(self.config.pipelines, self.model_manager, self.config.category_config, self.dynamic_ai_manager)
        
        # Debug instrumentation for exit status 120 investigation
        logger.debug("[DEBUG_VLM_ENGINE] VLMEngine initialized")
        
    async def initialize(self):
        """Initializes the pipelines."""
        try:
            logger.debug("[DEBUG_VLM_ENGINE] Starting pipeline initialization")
            await self.pipeline_manager.load_pipelines()
            logger.debug("[DEBUG_VLM_ENGINE] Pipeline initialization completed successfully")
        except Exception as e:
            logger.error(f"[DEBUG_VLM_ENGINE] Pipeline initialization failed: {e}")
            logger.debug(f"[DEBUG_VLM_ENGINE] Exception traceback: {traceback.format_exc()}")
            raise
        
    async def process_video(self, video_path: str, progress_callback: Optional[Callable[[int], None]] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a video and return tagging information.
        
        Args:
            video_path: Path to the video file
            progress_callback: Optional callback for progress updates (progress 0-100)
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary containing tagging information
        """
        start_time = time.time()
        logger.debug(f"[DEBUG_VLM_ENGINE] Starting video processing: {video_path}")
        logger.debug(f"[DEBUG_VLM_ENGINE] Processing parameters: {kwargs}")
        
        try:
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
            
            logger.debug(f"[DEBUG_VLM_ENGINE] Getting request future for pipeline: {pipeline_name}")
            future = await self.pipeline_manager.get_request_future(data, pipeline_name, callback=progress_callback)
            
            logger.debug(f"[DEBUG_VLM_ENGINE] Waiting for pipeline completion")
            result = await future
            
            duration = time.time() - start_time
            logger.debug(f"[DEBUG_VLM_ENGINE] Video processing completed in {duration:.2f}s")
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[DEBUG_VLM_ENGINE] Video processing failed after {duration:.2f}s: {e}")
            logger.debug(f"[DEBUG_VLM_ENGINE] Exception traceback: {traceback.format_exc()}")
            raise
