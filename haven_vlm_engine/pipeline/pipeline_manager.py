import logging
from typing import Dict, List, Any

from ..async_processing.item_future import ItemFuture
from .dynamic_ai_manager import DynamicAIManager
from .pipeline import Pipeline
from ..models.model_manager import ModelManager
from ..utils.exceptions import NoActiveModelsException, PipelineException, ConfigurationException

class PipelineManager:
    def __init__(self, model_manager: ModelManager, dynamic_ai_manager: DynamicAIManager):
        """
        Initializes the PipelineManager.

        Args:
            model_manager: An instance of ModelManager.
            dynamic_ai_manager: An instance of DynamicAIManager.
        """
        self.pipelines: Dict[str, Pipeline] = {}
        self.logger: logging.Logger = logging.getLogger("logger")
        self.model_manager: ModelManager = model_manager
        self.dynamic_ai_manager: DynamicAIManager = dynamic_ai_manager
    
    async def load_pipelines(self, pipeline_configs: Dict[str, Dict[str, Any]]):
        """
        Loads and initializes pipelines from their configuration dictionaries.

        Args:
            pipeline_configs: A dictionary where keys are pipeline names and values 
                              are their configuration dictionaries.
        """
        if not pipeline_configs:
            self.logger.warning("No pipeline configurations provided to load.")
            return

        for pipeline_name, config in pipeline_configs.items():
            self.logger.info(f"Loading pipeline: {pipeline_name}")
            if not isinstance(pipeline_name, str) or not pipeline_name:
                raise ConfigurationException("Pipeline name must be a non-empty string.")
            if not isinstance(config, dict):
                raise ConfigurationException(f"Configuration for pipeline '{pipeline_name}' must be a dictionary.")

            try:
                # Pass pipeline_name to Pipeline constructor for context
                new_pipeline = Pipeline(pipeline_name, config, self.model_manager, self.dynamic_ai_manager)
                self.pipelines[pipeline_name] = new_pipeline
                # Starting model processing workers is now typically handled by the main engine after all setup.
                # await new_pipeline.start_model_processing_workers() 
                self.logger.info(f"Pipeline '{pipeline_name}' V{new_pipeline.version} configured successfully.")
            except NoActiveModelsException as e_no_models:
                # This exception might be raised during DynamicAIManager's model verification
                self.logger.error(f"Error loading pipeline '{pipeline_name}' due to no active models: {e_no_models}")
                raise # Re-raise to be handled by the caller
            except Exception as e_general:
                if pipeline_name in self.pipelines:
                    del self.pipelines[pipeline_name] # Remove partially loaded pipeline
                self.logger.error(f"Error loading pipeline '{pipeline_name}': {e_general}", exc_info=True)
                # Optionally, wrap in a PipelineException or re-raise
                raise PipelineException(f"Failed to load pipeline '{pipeline_name}': {e_general}") from e_general
            
        if not self.pipelines:
            # This state implies all provided configs failed to load.
            self.logger.error("No valid pipelines were loaded from the provided configurations.")
            # Depending on desired strictness, could raise ConfigurationException here.
            # For now, allows proceeding if some pipelines failed but others might be usable if loaded incrementally.
            # However, if load_pipelines is called once with all configs, this means total failure.
            # raise ConfigurationException("Error: No valid pipelines loaded from the provided configurations!")

    async def start_all_pipeline_model_workers(self) -> None:
        """Starts model processing workers for all loaded pipelines."""
        if not self.pipelines:
            self.logger.info("No pipelines loaded, so no model workers to start.")
            return
        
        self.logger.info("Starting model processing workers for all loaded pipelines...")
        for pipeline_name, pipeline_instance in self.pipelines.items():
            try:
                self.logger.debug(f"Starting workers for pipeline: {pipeline_name}")
                await pipeline_instance.start_model_processing_workers()
            except Exception as e:
                self.logger.error(f"Failed to start model workers for pipeline '{pipeline_name}': {e}", exc_info=True)
        self.logger.info("Finished attempting to start model workers for all pipelines.")


    def get_pipeline(self, pipeline_name: str) -> Pipeline:
        if pipeline_name not in self.pipelines:
            self.logger.error(f"Error: Pipeline '{pipeline_name}' not found.")
            raise PipelineException(f"Pipeline '{pipeline_name}' not found. Available: {list(self.pipelines.keys())}")
        return self.pipelines[pipeline_name]

    async def process_request(self, pipeline_name: str, initial_data: Dict[str, Any]) -> Any:
        """
        Processes a request through the specified pipeline.

        Args:
            pipeline_name: The name of the pipeline to use.
            initial_data: A dictionary containing the initial input data for the pipeline,
                          where keys match the pipeline's defined input names.

        Returns:
            The final result from the pipeline.
        """
        pipeline = self.get_pipeline(pipeline_name)
        
        # Validate initial_data keys against pipeline inputs
        for required_input in pipeline.inputs:
            if required_input not in initial_data:
                raise PipelineException(
                    f"Missing required input '{required_input}' for pipeline '{pipeline_name}'. "
                    f"Provided data: {list(initial_data.keys())}"
                )
        
        # Create the initial ItemFuture. The event_handler is part of the Pipeline instance.
        # The ItemFuture will manage the data flow through the pipeline.
        item_future: ItemFuture = await ItemFuture.create(
            parent=None, 
            data=initial_data, 
            event_handler=pipeline.event_handler
        )
        
        # Await the ItemFuture itself, which completes when its underlying asyncio.Future is done.
        # The pipeline's event_handler is responsible for eventually calling item_future.close_future().
        try:
            result = await item_future
            return result
        except Exception as e:
            self.logger.error(f"Error during pipeline '{pipeline_name}' execution: {e}", exc_info=True)
            raise PipelineException(f"Execution error in pipeline '{pipeline_name}': {e}") from e
