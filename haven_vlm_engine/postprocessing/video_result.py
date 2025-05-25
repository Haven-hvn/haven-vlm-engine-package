import gzip
import json # Added for load_gzip_json
import logging
from typing import Dict, List, Optional, Tuple, Set, Any, Union, TYPE_CHECKING
from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo
import math

if TYPE_CHECKING:
    from .video_result_v0 import AIVideoResultV0 # Updated import

logger: logging.Logger = logging.getLogger("logger")

class TagTimeFrame(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    start: int # Frame index or timestamp (ms)
    end: Optional[int] = None # Frame index or timestamp (ms), None if single frame/event
    confidence: Optional[float] = None # Confidence score for this specific timeframe/event

    @field_validator('start', 'end', mode='before')
    @classmethod
    def _floor_and_convert_to_int(cls, v: Any, info: ValidationInfo) -> Optional[int]:
        if v is None and info.field_name == 'end': # 'end' can be None
            return None
        if v is None: # 'start' cannot be None
            raise ValueError(f"TagTimeFrame field '{info.field_name}' received None but is not Optional.")

        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(math.floor(v))
        try:
            return int(math.floor(float(str(v))))
        except (ValueError, TypeError) as e:
            logger.error(f"TagTimeFrame validator for '{info.field_name}': Could not convert '{v}' to int: {e}")
            raise ValueError(f"Invalid value '{v}' for TagTimeFrame field '{info.field_name}'.") from e

    def __str__(self) -> str:
        return f"TagTimeFrame(start={self.start}, end={self.end}, confidence={self.confidence})"

class ModelInfo(BaseModel): # Information about the model version and settings used for a category
    model_config = ConfigDict(extra='forbid')
    frame_interval: float # Frame interval used for this model's processing
    threshold: float      # Confidence threshold used
    version: Union[str, float] # Version of the AI model logic/weights
    ai_model_id: Union[str, int] # Unique identifier for the AI model (e.g. specific weights file or API model ID)
    file_name: Optional[str] = None # Original filename of the model, if applicable

    def needs_reprocessed(self, 
                          new_frame_interval: float, 
                          new_threshold: float, 
                          new_version: Union[str, float],
                          new_ai_model_id: Union[str, int], 
                          new_file_name: Optional[str]
                         ) -> int:
        """
        Determines if reprocessing is needed for this category.
        Returns:
            0: No reprocessing needed.
            1: Reprocessing beneficial (e.g., better model ID, same version).
            2: Reprocessing required (e.g., different version, incompatible settings).
        """
        # Convert versions to string for consistent comparison if one is float and other is str
        current_version_str = str(self.version)
        new_version_str = str(new_version)

        model_toReturn: int = -1

        # Exact same model and version
        if new_file_name == self.file_name and new_version_str == current_version_str and new_ai_model_id == self.ai_model_id:
            model_toReturn = 0
        # Same version, but new model ID is an improvement (heuristic: numerically smaller for local, or different for API)
        elif new_version_str == current_version_str:
            if isinstance(new_ai_model_id, int) and isinstance(self.ai_model_id, int):
                if new_ai_model_id < self.ai_model_id and self.ai_model_id >= 950: # Original heuristic
                     model_toReturn = 2 # Treat as major change if ID is significantly older
                elif new_ai_model_id < self.ai_model_id :
                    model_toReturn = 1 # Beneficial
                elif new_ai_model_id > self.ai_model_id:
                     model_toReturn = 0 # Current is newer or same
                else: # Same ID
                    model_toReturn = 0
            elif new_ai_model_id != self.ai_model_id: # Different string IDs (e.g. API model names)
                model_toReturn = 1 # Assume different ID is potentially better or different enough
            else: # Same string ID
                model_toReturn = 0
        else: # Different versions
            model_toReturn = 2


        # Check processing parameters (frame interval, threshold)
        # Reprocessing is required if new settings are more granular or sensitive.
        config_requires_reprocessing: bool = False
        if self.frame_interval == 0: # If original was image (no interval)
            if new_frame_interval != 0: # New is video interval
                config_requires_reprocessing = True
        # If new interval is smaller (more frames) or not a multiple of old (different frames)
        elif new_frame_interval < self.frame_interval or (self.frame_interval != 0 and new_frame_interval % self.frame_interval != 0) :
            config_requires_reprocessing = True
        
        # If new threshold is lower (more sensitive)
        if new_threshold < self.threshold:
            config_requires_reprocessing = True

        if config_requires_reprocessing:
            return 2 # Required due to config change
        else: # Config is compatible or less granular
            return model_toReturn # Decision based on model version/ID

    def __str__(self) -> str:
        return f"ModelInfo(version={self.version}, id={self.ai_model_id}, file={self.file_name}, interval={self.frame_interval}, thresh={self.threshold})"

class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid')
    duration: float # Video duration in seconds
    models: Dict[str, ModelInfo] # Category name -> ModelInfo used for that category

    def __str__(self) -> str:
        return f"VideoMetadata(duration={self.duration}, models_count={len(self.models)})"

class AIVideoResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: int = 1
    metadata: VideoMetadata
    timespans: Dict[str, Dict[str, List[TagTimeFrame]]] # Category -> Tag Name -> List of TagTimeFrame

    def to_json_str(self) -> str: # Renamed to avoid conflict with Pydantic's model_dump_json
        return self.model_dump_json(exclude_none=True)

    def add_server_result(self, server_pipeline_output: Dict[str, Any]) -> None:
        """
        Updates the AIVideoResult with new data from a pipeline run.
        'server_pipeline_output' is expected to contain keys like:
        'ai_models_info': List of tuples (version, id, filename, categories_list)
        'frame_interval': The frame interval used for this run.
        'threshold': The threshold used for this run.
        'frames': List of frame data, where each frame is a dict:
                  {'frame_index': float, 'category_name': List[Union[str, Tuple[str, float]]]}
        """
        ai_models_info_list: List[Tuple[Any, Any, Optional[str], List[str]]] = server_pipeline_output['ai_models_info']
        updated_categories: Set[str] = set()
        
        run_frame_interval: float = float(server_pipeline_output['frame_interval'])
        run_threshold: float = float(server_pipeline_output['threshold'])
        
        for model_ver, model_id, model_fname, model_cats in ai_models_info_list:
            for category in model_cats:
                new_model_info = ModelInfo(
                    frame_interval=run_frame_interval, 
                    threshold=run_threshold, 
                    version=model_ver, 
                    ai_model_id=model_id, 
                    file_name=model_fname
                )
                if category in self.metadata.models:
                    # Check if reprocessing was "better" or "required"
                    if self.metadata.models[category].needs_reprocessed(run_frame_interval, run_threshold, model_ver, model_id, model_fname) > 0:
                        self.metadata.models[category] = new_model_info
                        updated_categories.add(category)
                else: # New category
                    self.metadata.models[category] = new_model_info
                    updated_categories.add(category)

        # Process new frame data
        frames_data: List[Dict[str, Any]] = server_pipeline_output['frames']
        newly_processed_timespans = self._parse_frames_to_timespans(frames_data, run_frame_interval)
        
        for category in updated_categories:
            if category in newly_processed_timespans:
                self.timespans[category] = newly_processed_timespans[category]
            else: # Category was updated in models, but no new timespan data for it from this run
                self.timespans[category] = {} # Clear old data for this category
    
    @classmethod
    def from_pipeline_output(cls, pipeline_output: Dict[str, Any]) -> 'AIVideoResult':
        """
        Creates a new AIVideoResult instance from a fresh server/pipeline output.
        """
        frames: List[Dict[str, Any]] = pipeline_output['frames']
        video_duration: float = float(pipeline_output['video_duration'])
        frame_interval: float = float(pipeline_output['frame_interval'])
        threshold: float = float(pipeline_output['threshold'])
        
        timespans = cls._parse_frames_to_timespans(frames, frame_interval)
        
        model_infos_v1: Dict[str, ModelInfo] = {}
        ai_models_info_list: List[Tuple[Any, Any, Optional[str], List[str]]] = pipeline_output['ai_models_info']
        
        for model_ver, model_id, model_fname, model_cats in ai_models_info_list:
            model_info_instance = ModelInfo(
                frame_interval=frame_interval,
                threshold=threshold,
                version=model_ver,
                ai_model_id=model_id,
                file_name=model_fname
            )
            for category in model_cats:
                if category in model_infos_v1:
                    logger.warning(f"Duplicate category '{category}' in ai_models_info. Overwriting ModelInfo.")
                model_infos_v1[category] = model_info_instance
                
        metadata = VideoMetadata(duration=video_duration, models=model_infos_v1)
        return cls(schema_version=1, metadata=metadata, timespans=timespans)

    @classmethod
    def from_json_data(cls, json_data: Optional[Dict[str, Any]]) -> Tuple[Optional['AIVideoResult'], bool]:
        """
        Loads AIVideoResult from a JSON dictionary. Handles V0 to V1 migration.
        Returns (AIVideoResult instance or None, needs_resave_as_v1_flag).
        """
        if json_data is None:
            return None, False # Nothing to load, no resave needed
        
        if "schema_version" not in json_data or json_data["schema_version"] != 1:
            # Attempt to load as V0 and convert
            from .video_result_v0 import AIVideoResultV0 # Local import to avoid circularity at module level
            logger.info("Attempting to load AIVideoResult as V0 and convert to V1.")
            try:
                v0_instance = AIVideoResultV0(**json_data)
                v1_instance = v0_instance.to_V1()
                logger.info("Successfully converted V0 AIVideoResult to V1.")
                return v1_instance, True # Conversion happened, so resave is needed
            except Exception as e:
                logger.error(f"Failed to load and convert V0 AIVideoResult: {e}", exc_info=True)
                return None, False # Failed to load/convert
        else: # Already V1
            try:
                return cls(**json_data), False # No conversion, no resave needed for versioning
            except Exception as e:
                logger.error(f"Failed to load V1 AIVideoResult from JSON: {e}", exc_info=True)
                return None, False


    @classmethod
    def _parse_frames_to_timespans(cls, frames: List[Dict[str, Any]], frame_interval: float) -> Dict[str, Dict[str, List[TagTimeFrame]]]:
        """Helper to convert raw frame-based detections into consolidated timespans."""
        parsed_timespans: Dict[str, Dict[str, List[TagTimeFrame]]] = {}
        
        for frame_data in frames:
            # frame_index could be actual index or timestamp (if use_timestamps was true for VideoPreprocessor)
            # For TagTimeFrame, we expect integer frame indices or millisecond timestamps.
            # The original code used float(frame_data['frame_index']).
            # Let's assume it's a numerical value that can be floored to int for TagTimeFrame.start
            frame_start_val = int(math.floor(float(frame_data['frame_index'])))
            
            for category_key, tags_in_category in frame_data.items():
                if category_key == "frame_index":
                    continue
                
                category_map = parsed_timespans.setdefault(category_key, {})
                
                if not isinstance(tags_in_category, list):
                    logger.warning(f"Category data for '{category_key}' in a frame is not a list, skipping. Got: {type(tags_in_category)}")
                    continue

                for tag_item_data in tags_in_category:
                    tag_name: str
                    confidence_score: Optional[float]
                    
                    if isinstance(tag_item_data, tuple) and len(tag_item_data) == 2:
                        tag_name = str(tag_item_data[0])
                        confidence_score = float(tag_item_data[1]) if tag_item_data[1] is not None else None
                    elif isinstance(tag_item_data, str):
                        tag_name = tag_item_data
                        confidence_score = None # No confidence if only tag name string
                    else:
                        logger.warning(f"Skipping unrecognized tag item format in category '{category_key}': {tag_item_data}")
                        continue

                    tag_list = category_map.setdefault(tag_name, [])
                    
                    if not tag_list: # First time seeing this tag in this category
                        tag_list.append(TagTimeFrame(start=frame_start_val, end=None, confidence=confidence_score))
                    else:
                        last_ttf = tag_list[-1]
                        # Try to extend the last timeframe if contiguous and same confidence
                        # Assuming frame_start_val is an index, and frame_interval is 1 for index-based logic
                        # If frame_start_val is timestamp, frame_interval is duration between frames in same unit
                        is_contiguous = (last_ttf.end is None and (frame_start_val - last_ttf.start) == frame_interval) or \
                                        (last_ttf.end is not None and (frame_start_val - last_ttf.end) == frame_interval)
                        
                        if is_contiguous and last_ttf.confidence == confidence_score:
                            last_ttf.end = frame_start_val
                        else: # Start a new timeframe
                            tag_list.append(TagTimeFrame(start=frame_start_val, end=None, confidence=confidence_score))
        return parsed_timespans

def save_gzipped_json(data: AIVideoResult, file_path: str) -> bool:
    try:
        json_str = data.to_json_str()
        gzipped_json = gzip.compress(json_str.encode('utf-8'))
        with open(file_path, 'wb') as f:
            f.write(gzipped_json)
        return True
    except Exception as e:
        logger.error(f"Error saving gzipped JSON to {file_path}: {e}", exc_info=True)
        return False

def load_gzipped_json(file_path: str) -> Optional[AIVideoResult]:
    try:
        with open(file_path, 'rb') as f:
            gzipped_json = f.read()
        json_str = gzip.decompress(gzipped_json).decode('utf-8')
        json_dict_data = json.loads(json_str) # Use json.loads
        result_tuple = AIVideoResult.from_json_data(json_dict_data)
        return result_tuple[0] # Return the AIVideoResult instance
    except FileNotFoundError:
        logger.debug(f"Gzipped JSON file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading gzipped JSON from {file_path}: {e}", exc_info=True)
        return None
