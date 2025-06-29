import asyncio
import logging
from PIL import Image
from .preprocessing import get_video_duration_decord
from .postprocessing import AIVideoResult, compute_video_tag_info
from .vlm_client import OpenAICompatibleVLMClient
from .multiplexer_vlm_client import MultiplexerVLMClient
from typing import List, Dict, Any, Optional, Union, Tuple
from .async_utils import ItemFuture, QueueItem
import numpy as np
import json

logger: logging.Logger = logging.getLogger("logger")

_vlm_model: Optional[Union[OpenAICompatibleVLMClient, MultiplexerVLMClient]] = None

def get_vlm_model(config: Dict[str, Any]) -> Union[OpenAICompatibleVLMClient, MultiplexerVLMClient]:
    global _vlm_model
    if _vlm_model is None:
        # Check if multiplexer mode is enabled
        if config.get("use_multiplexer", False):
            _vlm_model = MultiplexerVLMClient(config=config)
        else:
            _vlm_model = OpenAICompatibleVLMClient(config=config)
    return _vlm_model

async def vlm_frame_analyzer(data: List[QueueItem]) -> None:
    """
    Analyze VLM frames with concurrent processing for improved performance.
    When multiple frames are present, they are processed concurrently rather than sequentially.
    """
    if not data:
        return
    
    # If only one item, process it directly to avoid overhead
    if len(data) == 1:
        item = data[0]
        item_future: ItemFuture = item.item_future
        try:
            frame_input: Any = item_future[item.input_names[0]]
            client_config: Dict[str, Any] = item_future[item.input_names[1]]
            
            frame_pil: Image.Image = _convert_frame_to_pil(frame_input)
            vlm: Union[OpenAICompatibleVLMClient, MultiplexerVLMClient] = get_vlm_model(client_config)
            scores: Dict[str, float] = await vlm.analyze_frame(frame_pil)
            
            await item_future.set_data(item.output_names[0] if isinstance(item.output_names, list) else item.output_names, scores)
        except Exception as e:
            item_future.set_exception(e)
        return
    
    # For multiple items, process them concurrently
    logger.debug(f"Processing {len(data)} frames concurrently")
    
    async def process_single_frame(item: QueueItem) -> None:
        item_future: ItemFuture = item.item_future
        try:
            frame_input: Any = item_future[item.input_names[0]]
            client_config: Dict[str, Any] = item_future[item.input_names[1]]
            
            frame_pil: Image.Image = _convert_frame_to_pil(frame_input)
            vlm: Union[OpenAICompatibleVLMClient, MultiplexerVLMClient] = get_vlm_model(client_config)
            scores: Dict[str, float] = await vlm.analyze_frame(frame_pil)
            
            await item_future.set_data(item.output_names[0] if isinstance(item.output_names, list) else item.output_names, scores)
        except Exception as e:
            item_future.set_exception(e)
    
    # Process all frames concurrently
    tasks = [process_single_frame(item) for item in data]
    await asyncio.gather(*tasks, return_exceptions=True)


def _convert_frame_to_pil(frame_input: Any) -> Image.Image:
    """Convert various frame input types to PIL Image."""
    if isinstance(frame_input, Image.Image):
        return frame_input
    elif hasattr(frame_input, 'cpu') and hasattr(frame_input, 'numpy'):
        img_np = frame_input.cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1,2,0))
        if img_np.dtype != np.uint8:
            if (img_np.dtype == np.float32 or img_np.dtype == np.float64) and img_np.max() <=1.0 and img_np.min() >=0:
                img_np = (img_np * 255)
            img_np = img_np.astype(np.uint8)
        return Image.fromarray(img_np)
    else:
        raise TypeError(f"Unsupported frame_input type: {type(frame_input)}")

async def result_coalescer(data: List[QueueItem]) -> None:
    for item_q in data:
        itemFuture: ItemFuture = item_q.item_future
        logger.debug(f"ResultCoalescer: Input names: {item_q.input_names}, Available data keys: {list(itemFuture.data.keys()) if itemFuture.data else 'None'}")
        result: Dict[str, Any] = {}
        for input_name in item_q.input_names:
            if itemFuture.data is not None and input_name in itemFuture.data:
                ai_result: Any = itemFuture[input_name] 
                result[input_name] = ai_result
                logger.debug(f"ResultCoalescer: Added {input_name} to result")
            else:
                logger.debug(f"ResultCoalescer: {input_name} not found in data")
        output_target = item_q.output_names[0] if isinstance(item_q.output_names, list) else item_q.output_names
        logger.debug(f"ResultCoalescer: Setting output {output_target} with result keys: {list(result.keys())}")
        await itemFuture.set_data(output_target, result)
        
async def result_finisher(data: List[QueueItem]) -> None:
    for item in data:
        itemFuture: ItemFuture = item.item_future
        if item.input_names[0] in itemFuture:
            future_results: Any = itemFuture[item.input_names[0]]
            itemFuture.close_future(future_results)
        else:
            itemFuture.set_exception(KeyError(f"Input {item.input_names[0]} not found"))

async def batch_awaiter(data: List[QueueItem]) -> None:
    for item in data:
        itemFuture: ItemFuture = item.item_future
        input_key = item.input_names[0]
        if input_key in itemFuture:
            child_futures_val: Any = itemFuture[input_key]
            if isinstance(child_futures_val, list):
                child_futures: List[ItemFuture] = child_futures_val
                results: List[Any] = await asyncio.gather(*child_futures, return_exceptions=True)
                output_target = item.output_names[0] if isinstance(item.output_names, list) else item.output_names
                await itemFuture.set_data(output_target, results)
            else:
                itemFuture.set_exception(TypeError(f"Input for batch_awaiter ('{input_key}') must be a list of futures."))
        else:
            itemFuture.set_exception(KeyError(f"Input {input_key} not found"))

async def video_result_postprocessor(data: List[QueueItem]) -> None:
    for item in data:
        itemFuture: ItemFuture = item.item_future
        try:
            duration: float = get_video_duration_decord(itemFuture[item.input_names[1]])
            
            frames = []
            raw_frames = itemFuture[item.input_names[0]]
            if isinstance(raw_frames, list):
                for frame_result in raw_frames:
                    if isinstance(frame_result, dict):
                        frames.append(frame_result)

            category_config = itemFuture["category_config"]

            videoResult = AIVideoResult.from_server_result({
                "frames": frames,
                "video_duration": duration,
                "frame_interval": float(itemFuture[item.input_names[2]]),
                "threshold": float(itemFuture[item.input_names[3]]),
                "ai_models_info": itemFuture['pipeline'].get_ai_models_info()
            })
            
            video_tag_info = compute_video_tag_info(videoResult, category_config)
            
            toReturn: Dict[str, Any] = {"json_result": json.loads(videoResult.to_json()), "video_tag_info": video_tag_info.model_dump()}
            
            output_target = item.output_names[0] if isinstance(item.output_names, list) else item.output_names
            await itemFuture.set_data(output_target, toReturn)
        except Exception as e:
            itemFuture.set_exception(e)
