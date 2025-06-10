import logging
import math
from copy import deepcopy
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Callable
from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo

logger: logging.Logger = logging.getLogger("logger")

class TagTimeFrame(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    start: int
    end: Optional[int] = None
    confidence: Optional[float] = None

class ModelInfo(BaseModel):
    frame_interval: float
    threshold: float
    version: float
    ai_model_id: int
    file_name: Optional[str] = None

class VideoMetadata(BaseModel):
    duration: float
    models: Dict[str, ModelInfo]

class AIVideoResult(BaseModel):
    schema_version: int
    metadata: VideoMetadata
    timespans: Dict[str, Dict[str, List[TagTimeFrame]]]

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_server_result(cls, server_result: Dict[str, Any]) -> 'AIVideoResult':
        frames: List[Dict[str, Any]] = server_result['frames']
        video_duration: float = float(server_result['video_duration'])
        frame_interval: float = float(server_result['frame_interval'])
        
        timespans: Dict[str, Dict[str, List[TagTimeFrame]]] = cls.__mutate_server_result_tags(frames, frame_interval)
        
        ai_version_and_ids: List[Tuple[float, int, Optional[str], List[str]]] = server_result['ai_models_info']
        modelinfos: Dict[str, ModelInfo] = {}
        
        for version, identifier, file_name, categories in ai_version_and_ids:
            model_info_params = {
                "frame_interval": frame_interval,
                "threshold": float(server_result['threshold']),
                "version": float(version) if version is not None else 1.0,
                "ai_model_id": int(identifier) if identifier is not None else 0,
                "file_name": file_name
            }
            model_info: ModelInfo = ModelInfo(**model_info_params)
            if isinstance(categories, list):
                for category in categories:
                    if category in modelinfos:
                        logger.error(f"Category {category} already exists in modelinfos. Models may have overlapping categories. Overwriting.")
                    modelinfos[category] = model_info
            elif isinstance(categories, str):
                if categories in modelinfos:
                    logger.error(f"Category {categories} already exists in modelinfos. Models may have overlapping categories. Overwriting.")
                modelinfos[categories] = model_info
                
        metadata: VideoMetadata = VideoMetadata(duration=video_duration, models=modelinfos)
        schema_version: int = 1
        return cls(schema_version=schema_version, metadata=metadata, timespans=timespans)

    @classmethod
    def __mutate_server_result_tags(cls, frames: List[Dict[str, Any]], frame_interval: float) -> Dict[str, Dict[str, List[TagTimeFrame]]]:
        toReturn: Dict[str, Dict[str, List[TagTimeFrame]]] = {}
        
        for frame_data in frames:
            frame_index: float = float(frame_data['frame_index'])
            
            for key, value in frame_data.items():
                if key != "frame_index":
                    currentCategoryDict: Dict[str, List[TagTimeFrame]]
                    if key in toReturn:
                        currentCategoryDict = toReturn[key]
                    else:
                        currentCategoryDict = {}
                        toReturn[key] = currentCategoryDict
                    
                    if not isinstance(value, list):
                        logger.warning(f"Category data for '{key}' is not a list, skipping. Got: {type(value)}")
                        continue

                    for item_data in value:
                        tag_name: str
                        confidence: Optional[float]
                        
                        if isinstance(item_data, tuple) and len(item_data) == 2:
                            tag_name = str(item_data[0])
                            confidence = float(item_data[1]) if item_data[1] is not None else None
                        elif isinstance(item_data, str):
                            tag_name = item_data
                            confidence = None
                        else:
                            logger.warning(f"Skipping unrecognized item format in category '{key}': {item_data}")
                            continue

                        if tag_name not in currentCategoryDict:
                            currentCategoryDict[tag_name] = [TagTimeFrame(start=int(frame_index), end=None, confidence=confidence)]
                        else:
                            last_time_frame: TagTimeFrame = currentCategoryDict[tag_name][-1]

                            if last_time_frame.end is None:
                                if (frame_index - float(last_time_frame.start)) == frame_interval and last_time_frame.confidence == confidence:
                                    last_time_frame.end = int(frame_index)
                                else:
                                    currentCategoryDict[tag_name].append(TagTimeFrame(start=int(frame_index), end=None, confidence=confidence))
                            elif last_time_frame.confidence == confidence and (frame_index - float(last_time_frame.end)) == frame_interval:
                                last_time_frame.end = int(frame_index)
                            else:
                                currentCategoryDict[tag_name].append(TagTimeFrame(start=int(frame_index), end=None, confidence=confidence))
        return toReturn

class TimeFrame(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    start: int
    end: int
    totalConfidence: Optional[float]

class VideoTagInfo(BaseModel):
    video_duration: float
    video_tags: Dict[str, List[str]]
    tag_totals: Dict[str, Dict[str, float]]
    tag_timespans: Dict[str, Dict[str, List[TimeFrame]]]

def compute_video_tag_info(video_result: AIVideoResult, category_config: Dict) -> VideoTagInfo:
    video_timespans = compute_video_timespans(video_result, category_config)
    video_tags, tag_totals = compute_video_tags(video_result, category_config)
    
    return VideoTagInfo(
        video_duration=video_result.metadata.duration,
        video_tags=video_tags,
        tag_totals=tag_totals,
        tag_timespans=video_timespans,
    )

def compute_video_timespans(video_result: AIVideoResult, category_config: Dict) -> Dict[str, Dict[str, List[TimeFrame]]]:
    toReturn: Dict[str, Dict[str, List[TimeFrame]]] = {}
    video_duration = video_result.metadata.duration
    
    for category, tag_to_raw_timespans_map in video_result.timespans.items():
        if category not in category_config:
            continue
        
        frame_interval: float = 0.5
        if hasattr(video_result.metadata, 'models') and category in video_result.metadata.models and hasattr(video_result.metadata.models[category], 'frame_interval'):
            frame_interval = float(video_result.metadata.models[category].frame_interval)

        toReturn[category] = {}
        for tag, raw_timespans_list in tag_to_raw_timespans_map.items():
            if tag not in category_config[category]:
                continue
            
            tag_min_duration: float = format_duration_or_percent(category_config[category][tag].get('MinMarkerDuration', "12s"), video_duration)
            if tag_min_duration <= 0:
                continue
            tag_threshold: float = float(category_config[category][tag].get('TagThreshold', 0.5))
            tag_max_gap: float = format_duration_or_percent(category_config[category][tag].get('MaxGap', "6s"), video_duration)
            renamed_tag: str = category_config[category][tag]['RenamedTag']

            # Process and merge timeframes with confidence aggregation
            processed_timeframes: List[Dict] = []
            for raw_timespan_obj in raw_timespans_list:
                if raw_timespan_obj.confidence is None or raw_timespan_obj.confidence < tag_threshold:
                    continue
                
                if not processed_timeframes:
                    # First timeframe - store with confidence info
                    processed_timeframes.append({
                        'obj': deepcopy(raw_timespan_obj),
                        'confidence_sum': raw_timespan_obj.confidence,
                        'confidence_count': 1
                    })
                    continue
                else:
                    previous_tf_data = processed_timeframes[-1]
                    previous_tf_obj = previous_tf_data['obj']
                    if not hasattr(raw_timespan_obj, 'start') or not hasattr(previous_tf_obj, 'start'): continue
                    
                    current_previous_end: float = previous_tf_obj.end if hasattr(previous_tf_obj, 'end') and previous_tf_obj.end is not None else previous_tf_obj.start
                    current_raw_start: float = raw_timespan_obj.start
                    current_raw_end: float = raw_timespan_obj.end if hasattr(raw_timespan_obj, 'end') and raw_timespan_obj.end is not None else raw_timespan_obj.start

                    if current_raw_start - current_previous_end - frame_interval <= tag_max_gap:
                        # Merge timeframes and aggregate confidence
                        previous_tf_obj.end = current_raw_end
                        previous_tf_data['confidence_sum'] += raw_timespan_obj.confidence
                        previous_tf_data['confidence_count'] += 1
                    else:
                        # New separate timeframe
                        processed_timeframes.append({
                            'obj': deepcopy(raw_timespan_obj),
                            'confidence_sum': raw_timespan_obj.confidence,
                            'confidence_count': 1
                        })
            
            # Create final timeframes with calculated totalConfidence
            final_tag_timeframes: List[TimeFrame] = []
            for tf_data in processed_timeframes:
                tf = tf_data['obj']
                if hasattr(tf, 'start') and (
                    (hasattr(tf, 'end') and tf.end is not None and (tf.end - tf.start >= tag_min_duration)) or 
                    ((not hasattr(tf, 'end') or tf.end is None) and tag_min_duration <= frame_interval)
                ):
                    # Calculate average confidence for merged timeframes
                    total_confidence = tf_data['confidence_sum'] / tf_data['confidence_count'] if tf_data['confidence_count'] > 0 else tf.confidence
                    final_tag_timeframes.append(
                        TimeFrame(
                            start=tf.start, 
                            end=(tf.end if hasattr(tf, 'end') and tf.end is not None else tf.start), 
                            totalConfidence=total_confidence
                        )
                    )
            
            if final_tag_timeframes:
                toReturn[category][renamed_tag] = final_tag_timeframes
    return toReturn

def compute_video_tags(video_result: AIVideoResult, category_config: Dict) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    video_tags: Dict[str, List[str]] = {}
    tag_totals: Dict[str, Dict[str, float]] = {}
    video_duration: float = video_result.metadata.duration
    
    for category, tag_raw_timespans_map in video_result.timespans.items():
        if category not in category_config:
            continue
        video_tags[category] = []
        tag_totals[category] = {}
        
        frame_interval: float = 0.5
        if hasattr(video_result.metadata, 'models') and category in video_result.metadata.models:
            frame_interval = float(video_result.metadata.models[category].frame_interval)

        for tag, raw_timespans_list in tag_raw_timespans_map.items():
            if tag not in category_config[category]:
                continue
            
            required_duration: float = format_duration_or_percent(category_config[category][tag].get('RequiredDuration', "20s"), video_duration)
            tag_threshold: float = float(category_config[category][tag].get('TagThreshold', 0.5))
            
            totalDuration: float = 0.0
            for raw_timespan in raw_timespans_list:
                if raw_timespan.confidence is not None and raw_timespan.confidence < tag_threshold:
                    continue
                if raw_timespan.end is None:
                    totalDuration += frame_interval
                else:
                    totalDuration += (raw_timespan.end - raw_timespan.start) + frame_interval
            
            renamed_tag: str = category_config[category][tag]['RenamedTag']
            tag_totals[category][renamed_tag] = totalDuration
            if required_duration > 0 and totalDuration >= required_duration:
                # Use list instead of set, and avoid duplicates
                if renamed_tag not in video_tags[category]:
                    video_tags[category].append(renamed_tag)
    return video_tags, tag_totals

def format_duration_or_percent(value: Union[str, float, int], video_duration: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.endswith('s'):
            return float(value[:-1])
        if value.endswith('%'):
            return video_duration * (float(value[:-1]) / 100.0)
    return 0.0
