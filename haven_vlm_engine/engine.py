import logging
from typing import Dict, Any, Optional, List

from .models.model_manager import ModelManager
from .pipeline.dynamic_ai_manager import DynamicAIManager
from .pipeline.pipeline_manager import PipelineManager
from .utils.exceptions import ConfigurationException, EngineException

logger = logging.getLogger("logger")

class VLMEngine:
    def __init__(self, 
                 model_configs: Dict[str, Dict[str, Any]],
                 pipeline_configs: Dict[str, Dict[str, Any]],
                 dynamic_ai_active_model_names: Optional[List[str]] = None):
        """
        Initializes the VLM Engine.

        Args:
            model_configs: A dictionary where keys are unique model instance names
                           and values are their configuration dictionaries.
            pipeline_configs: A dictionary where keys are pipeline names and values
                              are their configuration dictionaries.
            dynamic_ai_active_model_names: A list of model instance names that should be
                                           part of the dynamic AI group. These names must
                                           also be present as keys in `model_configs`.
        """
        if not model_configs:
            raise ConfigurationException("Model configurations must be provided.")
        if not pipeline_configs:
            raise ConfigurationException("Pipeline configurations must be provided.")

        self.model_configs = model_configs
        self.pipeline_configs = pipeline_configs
        self.dynamic_ai_active_model_names = dynamic_ai_active_model_names or []

        self.model_manager = ModelManager(model_configurations=self.model_configs)
        
        # DynamicAIManager needs the list of *names* of models that are considered active for dynamic groups
        self.dynamic_ai_manager = DynamicAIManager(
            model_manager=self.model_manager,
            active_ai_model_names=self.dynamic_ai_active_model_names
        )
        
        self.pipeline_manager = PipelineManager(
            model_manager=self.model_manager,
            dynamic_ai_manager=self.dynamic_ai_manager
        )
        
        self._initialized = False
        logger.info("VLMEngine initialized. Call 'await engine.setup_engine()' before processing.")

    async def setup_engine(self) -> None:
        """
        Loads all models and pipelines, and starts model processing workers.
        This must be called after __init__ and before any processing.
        """
        if self._initialized:
            logger.info("Engine already initialized.")
            return

        logger.info("Setting up VLM Engine...")
        try:
            # 1. Load and verify dynamic AI models (if any are configured)
            #    This populates image_size and normalization_config in dynamic_ai_manager
            if self.dynamic_ai_active_model_names:
                logger.info("Loading and verifying dynamic AI models...")
                self.dynamic_ai_manager.load_and_verify_models() # This is synchronous
            
            # 2. Load all pipeline configurations
            #    This will create Pipeline instances which in turn will use ModelManager
            #    and DynamicAIManager to get required model processors.
            logger.info("Loading pipeline configurations...")
            await self.pipeline_manager.load_pipelines(self.pipeline_configs) # Async

            # 3. Start workers for all models managed by ModelManager (via Pipeline instances)
            #    ModelManager itself doesn't start workers; PipelineManager tells each Pipeline
            #    to start its models' workers.
            logger.info("Starting model workers for all pipelines...")
            await self.pipeline_manager.start_all_pipeline_model_workers() # Async
            
            self._initialized = True
            logger.info("VLMEngine setup complete and ready for processing.")
        except Exception as e:
            logger.error(f"Error during VLMEngine setup: {e}", exc_info=True)
            self._initialized = False # Ensure it's marked as not ready
            raise EngineException(f"Engine setup failed: {e}") from e


    async def process_image(self, 
                            pipeline_name: str, 
                            image_path: Optional[str] = None, 
                            image_data: Optional[Any] = None, # e.g., PIL Image or Tensor
                            **kwargs: Any) -> Any:
        """
        Processes a single image through the specified pipeline.

        Args:
            pipeline_name: The name of the image processing pipeline to use.
            image_path: Path to the image file (if providing path).
            image_data: Image data directly (e.g., PIL Image, torch.Tensor) (if not providing path).
            **kwargs: Additional parameters required by the pipeline's inputs.
                      For example, threshold, return_confidence, etc.

        Returns:
            The result from the pipeline.
        """
        if not self._initialized:
            raise EngineException("Engine not initialized. Call 'await engine.setup_engine()' first.")
        if not image_path and image_data is None:
            raise ValueError("Either 'image_path' or 'image_data' must be provided.")
        if image_path and image_data is not None:
            raise ValueError("Provide either 'image_path' or 'image_data', not both.")

        pipeline = self.pipeline_manager.get_pipeline(pipeline_name)
        
        # Construct initial data based on pipeline's defined inputs
        initial_data: Dict[str, Any] = {}
        
        # The primary image input name needs to be known or inferred.
        # Assuming the first input of the pipeline is for the image path/data.
        # This might need to be more robust or configurable.
        if not pipeline.inputs:
            raise ConfigurationException(f"Pipeline '{pipeline_name}' has no defined inputs.")
        
        image_input_key = pipeline.inputs[0] # Assume first input is image path/data
        initial_data[image_input_key] = image_path if image_path else image_data

        # Add other kwargs that match pipeline inputs
        for key, value in kwargs.items():
            if key in pipeline.inputs:
                initial_data[key] = value
            # else: logger.warning(f"Provided kwarg '{key}' is not a defined input for pipeline '{pipeline_name}'.")

        logger.info(f"Processing image with pipeline '{pipeline_name}' and inputs: {list(initial_data.keys())}")
        return await self.pipeline_manager.process_request(pipeline_name, initial_data)

    async def process_video(self, 
                            pipeline_name: str, 
                            video_path: str, 
                            **kwargs: Any) -> Any:
        """
        Processes a video through the specified pipeline.

        Args:
            pipeline_name: The name of the video processing pipeline to use.
            video_path: Path to the video file.
            **kwargs: Additional parameters required by the pipeline's inputs.
                      For example, frame_interval, threshold, return_confidence, etc.
        
        Returns:
            The result from the pipeline.
        """
        if not self._initialized:
            raise EngineException("Engine not initialized. Call 'await engine.setup_engine()' first.")
        if not video_path: # Basic check
            raise ValueError("'video_path' must be provided.")

        pipeline = self.pipeline_manager.get_pipeline(pipeline_name)
        initial_data: Dict[str, Any] = {}

        if not pipeline.inputs:
            raise ConfigurationException(f"Pipeline '{pipeline_name}' has no defined inputs.")

        video_path_input_key = pipeline.inputs[0] # Assume first input is video_path
        initial_data[video_path_input_key] = video_path
        
        for key, value in kwargs.items():
            if key in pipeline.inputs:
                initial_data[key] = value
        
        logger.info(f"Processing video '{video_path}' with pipeline '{pipeline_name}' and inputs: {list(initial_data.keys())}")
        return await self.pipeline_manager.process_request(pipeline_name, initial_data)

    async def shutdown(self) -> None:
        """Placeholder for any cleanup logic if needed (e.g., explicitly stopping workers)."""
        logger.info("VLMEngine shutting down...")
        # ModelProcessor workers are daemon tasks, they should exit when main program exits.
        # If explicit cleanup is needed (e.g. closing resources in models), it would go here.
        self._initialized = False
        logger.info("VLMEngine shutdown complete.")
