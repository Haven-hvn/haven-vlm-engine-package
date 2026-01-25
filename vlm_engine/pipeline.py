import logging
from .async_utils import ItemFuture, QueueItem
from .models import ModelManager, VLMAIModel, VideoPreprocessorModel
from .config_models import PipelineConfig, PipelineModelConfig
from .dynamic_ai import DynamicAIManager
from .model_wrapper import ModelWrapper
from typing import List, Dict, Any, Optional, Set, Union, Tuple, Callable

logger: logging.Logger = logging.getLogger("logger")

class Pipeline:
    def __init__(self, config: PipelineConfig, model_manager: ModelManager, category_config: Dict, dynamic_ai_manager: DynamicAIManager):
        self.version: float = config.version
        self.inputs: List[str] = config.inputs
        self.output: str = config.output
        self.models: List[ModelWrapper] = []
        self.category_config = category_config

        for model_config in config.models:
            modelName: str = model_config.name
            model_inputs: List[str] = model_config.inputs
            model_outputs: Union[str, List[str]] = model_config.outputs

            if modelName == "video_analysis_pipeline":
                dynamic_models: List[ModelWrapper] = dynamic_ai_manager.get_dynamic_video_ai_models(model_inputs, model_outputs if isinstance(model_outputs, list) else [model_outputs] if model_outputs else [])
                self.models.extend(dynamic_models)
                continue

            returned_model: Any = model_manager.get_or_create_model(modelName)
            self.models.append(ModelWrapper(returned_model, model_inputs, model_outputs, model_name_for_logging=modelName))

        categories_set: Set[str] = set()
        for wrapper_model in self.models:
            if hasattr(wrapper_model.model, 'model') and isinstance(wrapper_model.model.model, VLMAIModel):
                current_categories: Union[str, List[str], None] = wrapper_model.model.model.model_category
                if isinstance(current_categories, str):
                    if current_categories in categories_set:
                        raise ValueError(f"Error: AI models must not have overlapping categories! Category: {current_categories}")
                    categories_set.add(current_categories)
                elif isinstance(current_categories, list):
                    for cat in current_categories:
                        if cat in categories_set:
                            raise ValueError(f"Error: AI models must not have overlapping categories! Category: {cat}")
                        categories_set.add(cat)

        is_vlm_pipeline: bool = any(isinstance(mw.model.model, VLMAIModel) for mw in self.models if hasattr(mw.model, 'model'))
        if is_vlm_pipeline:
            for model_wrapper in self.models:
                if hasattr(model_wrapper.model, 'model') and isinstance(model_wrapper.model.model, VideoPreprocessorModel):
                    model_wrapper.model.model.set_vlm_pipeline_mode(True)
    
    async def event_handler(self, itemFuture: ItemFuture, key: str) -> None:
        logger.debug(f"[DEBUG_PIPELINE] event_handler called with key='{key}', ItemFuture.data keys: {list(itemFuture.data.keys()) if itemFuture.data else 'None'}")
        if key == self.output:
            if key in itemFuture:
                itemFuture.close_future(itemFuture[key])
            else:
                pass

        for current_model_wrapper in self.models:
            if key in current_model_wrapper.inputs:
                allOtherInputsPresent: bool = True
                for inputName in current_model_wrapper.inputs:
                    if inputName != key:
                        is_present = (itemFuture.data is not None and inputName in itemFuture.data)
                        logger.debug(f"[DEBUG_PIPELINE] Model '{current_model_wrapper.model_name_for_logging}' needs input '{inputName}': present={is_present}")
                        if not is_present:
                            allOtherInputsPresent = False
                            break

                if allOtherInputsPresent:
                    logger.info(f"[DEBUG_PIPELINE] All inputs present for '{current_model_wrapper.model_name_for_logging}', triggering model")
                    await current_model_wrapper.model.add_to_queue(QueueItem(itemFuture, current_model_wrapper.inputs, current_model_wrapper.outputs))
                else:
                    logger.debug(f"[DEBUG_PIPELINE] Not all inputs present for '{current_model_wrapper.model_name_for_logging}', skipping")

    async def start_model_processing(self) -> None:
        for model_wrapper in self.models:
            if hasattr(model_wrapper.model, 'start_workers') and callable(model_wrapper.model.start_workers):
                 await model_wrapper.model.start_workers()

    def get_ai_models_info(self) -> List[Tuple[Union[str, float, None], Optional[str], Optional[str], Optional[Union[str, List[str]]]]]:
        ai_version_and_ids: List[Tuple[Union[str, float, None], Optional[str], Optional[str], Optional[Union[str, List[str]]]]] = []
        for model_wrapper in self.models:
            if hasattr(model_wrapper.model, 'model') and isinstance(model_wrapper.model.model, VLMAIModel):
                inner_ai_model: VLMAIModel = model_wrapper.model.model
                model_id = inner_ai_model.client_config.model_id
                category = inner_ai_model.model_category
                ai_version_and_ids.append((model_id, category))
        return ai_version_and_ids

class PipelineManager:
    def __init__(self, pipelines_config: Dict[str, PipelineConfig], model_manager: ModelManager, category_config: Dict, dynamic_ai_manager: DynamicAIManager):
        self.pipelines: Dict[str, Pipeline] = {}
        self.logger: logging.Logger = logging.getLogger("logger")
        self.model_manager = model_manager
        self.pipelines_config = pipelines_config
        self.category_config = category_config
        self.dynamic_ai_manager = dynamic_ai_manager
    
    async def load_pipelines(self):
        self.logger.debug(f"[DEBUG_PIPELINE_LOAD] Starting to load pipelines from config: {list(self.pipelines_config.keys())}")
        for pipeline_name, pipeline_config in self.pipelines_config.items():
            self.logger.info(f"Loading pipeline: {pipeline_name}")
            self.logger.debug(f"[DEBUG_PIPELINE_LOAD] Pipeline config for {pipeline_name}: inputs={pipeline_config.inputs}, output={pipeline_config.output}, version={pipeline_config.version}")
            
            try:
                self.logger.debug(f"[DEBUG_PIPELINE_LOAD] Creating Pipeline instance for {pipeline_name}")
                new_pipeline = Pipeline(pipeline_config, self.model_manager, self.category_config, self.dynamic_ai_manager)
                self.pipelines[pipeline_name] = new_pipeline
                self.logger.debug(f"[DEBUG_PIPELINE_LOAD] Starting model processing for pipeline {pipeline_name}")
                await new_pipeline.start_model_processing()
                self.logger.info(f"Pipeline {pipeline_name} loaded successfully!")
                self.logger.debug(f"[DEBUG_PIPELINE_LOAD] Pipeline {pipeline_name} loaded successfully!")
            except Exception as e:
                self.logger.error(f"[DEBUG_PIPELINE_LOAD] ERROR loading pipeline {pipeline_name}: {e}")
                if pipeline_name in self.pipelines:
                    del self.pipelines[pipeline_name]
                self.logger.error(f"Error loading pipeline {pipeline_name}: {e}")
                self.logger.debug(f"[DEBUG_PIPELINE_LOAD] Exception details:", exc_info=True)
            
        if not self.pipelines:
            self.logger.error(f"[DEBUG_PIPELINE_LOAD] CRITICAL: No valid pipelines loaded after processing all pipelines!")
            raise Exception("Error: No valid pipelines loaded!")
        else:
            self.logger.debug(f"[DEBUG_PIPELINE_LOAD] Successfully loaded {len(self.pipelines)} pipelines: {list(self.pipelines.keys())}")
            
    def get_pipeline(self, pipeline_name: str) -> Pipeline:
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found.")
        return self.pipelines[pipeline_name]

    async def get_request_future(self, data: List[Any], pipeline_name: str, callback: Optional[Callable[[int], None]] = None) -> ItemFuture:
        pipeline = self.get_pipeline(pipeline_name)
        futureData: Dict[str, Any] = {}
        if len(data) != len(pipeline.inputs):
            raise ValueError(f"Error: Data length does not match pipeline inputs length for pipeline {pipeline_name}!")
        
        for inputName, inputData in zip(pipeline.inputs, data):
            futureData[inputName] = inputData
        futureData["pipeline"] = pipeline
        futureData["category_config"] = self.category_config
        itemFuture: ItemFuture = await ItemFuture.create(None, futureData, pipeline.event_handler)
        
        if callback:
            await itemFuture.set_data("callback", callback)
        
        return itemFuture
