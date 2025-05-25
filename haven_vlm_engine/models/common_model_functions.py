import asyncio
import logging
from typing import List, Dict, Any, Optional

from ..async_processing.queue_item import QueueItem
from ..async_processing.item_future import ItemFuture
from ..utils.skip_input import Skip, SKIP_INSTANCE # Use SKIP_INSTANCE
from ..postprocessing.video_result import AIVideoResult # For video_result_postprocessor
from ..postprocessing import timeframe_processing # For video_result_postprocessor
from ..preprocessing.image_preprocessing import get_video_duration_decord # For video_result_postprocessor

logger: logging.Logger = logging.getLogger("logger")

async def result_coalescer(queue_items: List[QueueItem]) -> None:
    """
    Coalesces results from multiple inputs into a single dictionary output.
    Input QueueItems should have item_future.input_names populated with keys
    to retrieve from the item_future.data.
    The coalesced result is set to item_future.output_names[0].
    """
    for item_q in queue_items:
        item_future: ItemFuture = item_q.item_future
        coalesced_result: Dict[str, Any] = {}
        
        for input_name in item_q.input_names:
            if input_name in item_future: # Check if key exists in ItemFuture's data
                value = item_future[input_name]
                if not isinstance(value, Skip): # Do not include Skip instances
                    coalesced_result[input_name] = value
            else:
                logger.debug(f"ResultCoalescer: Input '{input_name}' not found in ItemFuture data for item {id(item_future)}.")
        
        output_target_name = item_q.output_names[0] if isinstance(item_q.output_names, list) and item_q.output_names else \
                             item_q.output_names if isinstance(item_q.output_names, str) else \
                             "coalesced_output" # Fallback output name

        logger.debug(f"ResultCoalescer: Item {id(item_future)} setting output '{output_target_name}' with keys: {list(coalesced_result.keys())}")
        await item_future.set_data(output_target_name, coalesced_result)

async def result_finisher(queue_items: List[QueueItem]) -> None:
    """
    Finalizes a pipeline result by closing the ItemFuture.
    Expects item_future.input_names[0] to be the key for the final result in item_future.data.
    """
    for item in queue_items:
        item_future: ItemFuture = item.item_future
        final_result_key = item.input_names[0]
        
        if final_result_key in item_future:
            final_result_data = item_future[final_result_key]
            logger.debug(f"ResultFinisher: Finalizing ItemFuture {id(item_future)} with result from key '{final_result_key}'.")
            item_future.close_future(final_result_data)
        else:
            logger.error(f"ResultFinisher: Final result key '{final_result_key}' not found in ItemFuture {id(item_future)}. Setting exception.")
            item_future.set_exception(KeyError(f"Final result key '{final_result_key}' not found in ItemFuture data."))

async def batch_awaiter(queue_items: List[QueueItem]) -> None:
    """
    Awaits a list of child ItemFutures.
    Expects item_future.input_names[0] to be the key for a list of child ItemFutures
    in item_future.data.
    The collected results from child futures are set to item_future.output_names[0].
    """
    for item in queue_items:
        item_future: ItemFuture = item.item_future # This is the parent ItemFuture
        child_futures_key = item.input_names[0]
        
        if child_futures_key in item_future:
            child_futures_list_any: Any = item_future[child_futures_key]
            
            if isinstance(child_futures_list_any, list):
                # Ensure all items in the list are awaitable (ItemFuture instances)
                child_item_futures: List[ItemFuture] = []
                valid_futures = True
                for child_future_obj in child_futures_list_any:
                    if isinstance(child_future_obj, ItemFuture): # Or check for __await__
                        child_item_futures.append(child_future_obj)
                    else:
                        logger.error(f"BatchAwaiter: Item in child futures list is not an ItemFuture. Type: {type(child_future_obj)}")
                        item_future.set_exception(TypeError("BatchAwaiter received non-ItemFuture in child list."))
                        valid_futures = False
                        break
                if not valid_futures: continue

                if not child_item_futures:
                    logger.debug(f"BatchAwaiter: Received an empty list of child futures for key '{child_futures_key}'.")
                    results = []
                else:
                    logger.debug(f"BatchAwaiter: Awaiting {len(child_item_futures)} child futures from key '{child_futures_key}'.")
                    results = await asyncio.gather(*child_item_futures, return_exceptions=True)
                
                # Log exceptions from gathered results
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        logger.error(f"BatchAwaiter: Child future {i} resulted in exception: {res}", exc_info=res)
                
                output_target_name = item.output_names[0] if isinstance(item.output_names, list) and item.output_names else \
                                     item.output_names if isinstance(item.output_names, str) else \
                                     "awaited_batch_results"
                await item_future.set_data(output_target_name, results)
            else:
                logger.error(f"BatchAwaiter: Input for key '{child_futures_key}' is not a list. Type: {type(child_futures_list_any)}")
                item_future.set_exception(TypeError(f"Input for BatchAwaiter ('{child_futures_key}') must be a list of ItemFutures."))
        else:
            logger.error(f"BatchAwaiter: Child futures key '{child_futures_key}' not found in ItemFuture {id(item_future)}.")
            item_future.set_exception(KeyError(f"Child futures key '{child_futures_key}' not found."))

async def video_result_postprocessor(queue_items: List[QueueItem]) -> None:
    """
    Post-processes video results.
    Expects specific inputs in item_future.data based on pipeline configuration.
    The configuration for this model should include 'category_settings' and 'post_processing_settings'.
    """
    for item in queue_items:
        item_future: ItemFuture = item.item_future
        try:
            # Retrieve necessary data from ItemFuture
            # These keys must match what the pipeline provides to this model
            frame_results = item_future[item.input_names[0]] # List of per-frame results
            video_path = item_future[item.input_names[1]]
            frame_interval = float(item_future[item.input_names[2]])
            threshold = float(item_future[item.input_names[3]])
            
            # 'pipeline_ref' was used to call get_ai_models_info().
            # This info should now be passed directly if needed, or reconstructed.
            # For simplicity, let's assume ai_models_info is passed if required.
            # If 'pipeline' object is in item_future, use it, otherwise expect 'ai_models_info'
            ai_models_info: Optional[List[Tuple]] = None
            if 'pipeline' in item_future and hasattr(item_future['pipeline'], 'get_ai_models_info'):
                ai_models_info = item_future['pipeline'].get_ai_models_info()
            elif item.input_names[5] in item_future : # Assuming 6th input is ai_models_info
                 ai_models_info = item_future[item.input_names[5]]


            if ai_models_info is None:
                logger.warning("video_result_postprocessor: 'ai_models_info' not found. Postprocessed result might be incomplete.")
                ai_models_info = []


            # Existing AIVideoResult object might be passed for updates
            existing_ai_video_result_obj: Optional[AIVideoResult] = None
            if len(item.input_names) > 4 and item.input_names[4] in item_future:
                existing_ai_video_result_obj = item_future[item.input_names[4]]


            duration: float = get_video_duration_decord(video_path) # Using decord by default
            
            pipeline_output_data = {
                "frames": frame_results, 
                "video_duration": duration, 
                "frame_interval": frame_interval, 
                "threshold": threshold, 
                "ai_models_info": ai_models_info 
            }

            final_ai_video_result: AIVideoResult
            if existing_ai_video_result_obj is not None and isinstance(existing_ai_video_result_obj, AIVideoResult):
                existing_ai_video_result_obj.add_server_result(pipeline_output_data) # Method name from original
                final_ai_video_result = existing_ai_video_result_obj
            else:
                if existing_ai_video_result_obj is not None:
                    logger.warning(f"Provided existing video result is not of type AIVideoResult, type: {type(existing_ai_video_result_obj)}. Creating new.")
                final_ai_video_result = AIVideoResult.from_pipeline_output(pipeline_output_data)

            # The user of the package will provide category_settings and post_processing_settings (for timeframe method)
            # These should be passed into this function, perhaps via model_config of PythonCallableModel
            # For now, this part is non-functional without those configs.
            # We'll create a basic result structure.
            # video_tag_info = timeframe_processing.compute_full_video_tag_info(final_ai_video_result, category_settings, ...)
            
            # Simplified output for now, as timeframe_processing requires external configs
            # In a real scenario, the PythonCallableModel's config would provide these.
            output_payload = {
                "json_result": final_ai_video_result.to_json_str(),
                # "video_tag_info": video_tag_info # This would be the full processed info
                "processed_timespans": final_ai_video_result.timespans # Return raw processed timespans
            }
            
            output_target_name = item.output_names[0] if isinstance(item.output_names, list) and item.output_names else \
                                 item.output_names if isinstance(item.output_names, str) else \
                                 "video_postprocessed_output"
            await item_future.set_data(output_target_name, output_payload)

        except Exception as e:
            logger.error(f"Error in video_result_postprocessor: {e}", exc_info=True)
            if item_future and not item_future.done():
                item_future.set_exception(e)


async def image_result_postprocessor(queue_items: List[QueueItem]) -> None:
    """
    Post-processes image results.
    Relies on 'category_settings' and 'post_processing_settings' being passed via model_config
    for the PythonCallableModel that wraps this function.
    """
    for item in queue_items:
        item_future: ItemFuture = item.item_future
        try:
            # Expected inputs:
            # item.input_names[0]: raw_image_results (Dict[category, List[tags/tag_tuples]])
            # item.input_names[1]: category_settings (Dict)
            # item.input_names[2]: post_processing_settings (Dict) - specifically for use_category_image_thresholds

            raw_results_dict: Dict[str, Any] = item_future[item.input_names[0]]
            
            # These settings would ideally come from the PythonCallableModel's own config,
            # passed into its __init__ and then accessible here.
            # For this example, we assume they are passed in the ItemFuture for simplicity of refactoring.
            category_settings: Dict[str, Dict[str, Any]] = item_future[item.input_names[1]]
            post_proc_settings: Dict[str, Any] = item_future[item.input_names[2]]

            processed_results: Dict[str, List[Union[Tuple[str, float], str]]] = {}

            for category, tags_in_category_list in raw_results_dict.items():
                if category not in category_settings:
                    logger.debug(f"ImagePostProcessor: Category '{category}' not in category_settings. Skipping.")
                    continue
                
                processed_results[category] = []
                category_specific_settings = category_settings[category]

                for tag_item in tags_in_category_list:
                    tag_name_original: str
                    confidence_original: Optional[float] = None

                    if isinstance(tag_item, tuple) and len(tag_item) == 2:
                        tag_name_original, confidence_original = str(tag_item[0]), float(tag_item[1])
                    elif isinstance(tag_item, str):
                        tag_name_original = tag_item
                    else:
                        logger.warning(f"ImagePostProcessor: Malformed tag item in category '{category}': {tag_item}. Skipping.")
                        continue
                    
                    if tag_name_original not in category_specific_settings:
                        logger.debug(f"ImagePostProcessor: Tag '{tag_name_original}' not in settings for category '{category}'. Skipping.")
                        continue
                    
                    tag_specific_config = category_specific_settings[tag_name_original]
                    tag_threshold: float = float(timeframe_processing.get_or_default(tag_specific_config, 'TagThreshold', 0.5))
                    renamed_tag: str = timeframe_processing.get_or_default(tag_specific_config, 'RenamedTag', tag_name_original)
                    
                    # Apply threshold only if confidence is available and global setting allows
                    passes_threshold = True
                    if confidence_original is not None and post_proc_settings.get("use_category_image_thresholds", False):
                        if confidence_original < tag_threshold:
                            passes_threshold = False
                    
                    if passes_threshold:
                        if confidence_original is not None: # If original had confidence, include it with renamed tag
                            processed_results[category].append((renamed_tag, confidence_original))
                        else: # Original was just a string tag
                            processed_results[category].append(renamed_tag)
            
            output_target_name = item.output_names[0] if isinstance(item.output_names, list) and item.output_names else \
                                 item.output_names if isinstance(item.output_names, str) else \
                                 "image_postprocessed_output"
            await item_future.set_data(output_target_name, processed_results)

        except Exception as e:
            logger.error(f"Error in image_result_postprocessor: {e}", exc_info=True)
            if item_future and not item_future.done():
                item_future.set_exception(e)
