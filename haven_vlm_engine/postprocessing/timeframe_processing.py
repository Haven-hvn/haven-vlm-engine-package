from copy import deepcopy
import logging
import math
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Callable
import itertools

from . import tag_models # Updated import
from .video_result import AIVideoResult, TagTimeFrame # Updated import

logger: logging.Logger = logging.getLogger("logger")

# Helper function, can be kept.
def get_or_default(config_dict: Dict[str, Any], key: str, default: Any) -> Any:
    return config_dict.get(key, default)

# Helper function, can be kept.
def format_duration_or_percent(value: Union[str, float, int], video_duration: float) -> float:
    try:
        if isinstance(value, (float, int)): # Simplified check
            return float(value)
        elif isinstance(value, str):
            if value.endswith('%'):
                return (float(value[:-1]) / 100.0) * video_duration
            elif value.endswith('s'):
                return float(value[:-1])
            else:
                return float(value) # Attempt to convert string number
        logger.warning(f"Unsupported type for format_duration_or_percent: {type(value)}, value: {value}")
        return 0.0
    except ValueError as e:
        logger.error(f"Error in format_duration_or_percent converting value '{value}': {e}")
        return 0.0
    except Exception as e: # Catch any other unexpected error
        logger.error(f"Unexpected error in format_duration_or_percent: {e} for value '{value}'", exc_info=True)
        return 0.0

# --- Refactored Timespan Computation Methods ---
# These methods now take category_specific_configs and tag_specific_configs

def compute_video_timespans_og(
    video_result: AIVideoResult,
    category_settings: Dict[str, Dict[str, Any]] # Category -> TagName -> SettingsDict
) -> Dict[str, Dict[str, List[tag_models.TimeFrame]]]:
    video_duration: float = video_result.metadata.duration
    toReturn: Dict[str, Dict[str, List[tag_models.TimeFrame]]] = {}
    
    for category, tag_to_raw_timespans_map in video_result.timespans.items():
        if category not in category_settings:
            logger.debug(f"Category {category} not found in provided category_settings for OG timespan computation.")
            continue
        
        category_tag_settings = category_settings[category]
        toReturn[category] = {}
        
        frame_interval: float = 0.5 # Default
        if video_result.metadata.models and category in video_result.metadata.models:
            frame_interval = float(video_result.metadata.models[category].frame_interval)
        else:
            logger.warning(f"Frame interval for category {category} not found in video_result metadata. Using default {frame_interval}s.")

        for tag, raw_timespans_list in tag_to_raw_timespans_map.items():
            if tag not in category_tag_settings:
                logger.debug(f"Tag {tag} not found in settings for category {category}.")
                continue
            
            tag_config = category_tag_settings[tag]
            tag_min_duration: float = format_duration_or_percent(get_or_default(tag_config, 'MinMarkerDuration', "12s"), video_duration)
            if tag_min_duration <= 0: continue

            tag_threshold: float = float(get_or_default(tag_config, 'TagThreshold', 0.5))
            tag_max_gap: float = format_duration_or_percent(get_or_default(tag_config, 'MaxGap', "6s"), video_duration)
            renamed_tag: str = get_or_default(tag_config, 'RenamedTag', tag) # Default to original tag name

            processed_timeframes: List[TagTimeFrame] = []
            for raw_timespan_obj in raw_timespans_list:
                if raw_timespan_obj.confidence is None or raw_timespan_obj.confidence < tag_threshold:
                    continue
                
                if not processed_timeframes:
                    processed_timeframes.append(deepcopy(raw_timespan_obj))
                else:
                    previous_tf_obj = processed_timeframes[-1]
                    current_previous_end = previous_tf_obj.end if previous_tf_obj.end is not None else previous_tf_obj.start
                    current_raw_start = raw_timespan_obj.start
                    current_raw_end = raw_timespan_obj.end if raw_timespan_obj.end is not None else raw_timespan_obj.start

                    if (current_raw_start - current_previous_end - frame_interval) <= tag_max_gap:
                        previous_tf_obj.end = current_raw_end
                    else:
                        processed_timeframes.append(deepcopy(raw_timespan_obj))
            
            final_tag_timeframes: List[tag_models.TimeFrame] = [
                tag_models.TimeFrame(start=tf.start, end=(tf.end if tf.end is not None else tf.start), totalConfidence=None)
                for tf in processed_timeframes 
                if ((tf.end is not None and ((tf.end - tf.start) + frame_interval >= tag_min_duration)) or 
                    (tf.end is None and frame_interval >= tag_min_duration))
            ]
            if final_tag_timeframes:
                toReturn[category][renamed_tag] = final_tag_timeframes
    return toReturn

def compute_video_timespans_clustering(
    video_result: AIVideoResult,
    category_settings: Dict[str, Dict[str, Any]], # Category -> TagName -> SettingsDict
    density_weight: float, 
    gap_factor: float, 
    average_factor: float, 
    min_gap: float
) -> Dict[str, Dict[str, List[tag_models.TimeFrame]]]:
    video_duration: float = video_result.metadata.duration
    toReturn: Dict[str, Dict[str, List[tag_models.TimeFrame]]] = {}

    for category, tag_to_raw_timespans_map in video_result.timespans.items():
        if category not in category_settings: continue
        category_tag_settings = category_settings[category]
        toReturn[category] = {}
        
        frame_interval: float = 0.5
        if video_result.metadata.models and category in video_result.metadata.models:
            frame_interval = float(video_result.metadata.models[category].frame_interval)

        for tag, raw_timespans_list in tag_to_raw_timespans_map.items():
            if tag not in category_tag_settings: continue
            tag_config = category_tag_settings[tag]
            tag_threshold: float = float(get_or_default(tag_config, 'TagThreshold', 0.5))
            renamed_tag: str = get_or_default(tag_config, 'RenamedTag', tag)
            tag_min_duration: float = format_duration_or_percent(get_or_default(tag_config, 'MinMarkerDuration', "12s"), video_duration)
            if tag_min_duration <= 0: continue
            
            initial_buckets: List[tag_models.TimeFrame] = []
            current_bucket: Optional[tag_models.TimeFrame] = None
            
            for raw_timespan_obj in raw_timespans_list:
                if raw_timespan_obj.confidence is None or raw_timespan_obj.confidence < tag_threshold: continue
                
                start_time = raw_timespan_obj.start
                end_time = raw_timespan_obj.end if raw_timespan_obj.end is not None else start_time
                duration_val = (end_time - start_time) + frame_interval
                
                if current_bucket is None:
                    current_bucket = tag_models.TimeFrame(start=start_time, end=end_time, totalConfidence=raw_timespan_obj.confidence * duration_val)
                else:
                    if start_time - current_bucket.end == frame_interval: # Contiguous
                        current_bucket.merge(start_time, end_time, raw_timespan_obj.confidence, frame_interval)
                    else:
                        initial_buckets.append(current_bucket)
                        current_bucket = tag_models.TimeFrame(start=start_time, end=end_time, totalConfidence=raw_timespan_obj.confidence * duration_val)
            if current_bucket: initial_buckets.append(current_bucket)
            
            def should_merge_buckets(b1, b2, fi, dw, gf, af, mg): # Simplified params
                gap = b2.start - b1.end - fi
                den_b1, den_b2 = b1.get_density(fi), b2.get_density(fi)
                wd_b1, wd_b2 = b1.get_duration(fi) * (1 + dw * den_b1), b2.get_duration(fi) * (1 + dw * den_b2)
                return gap <= mg + (min(wd_b1, wd_b2) + abs(wd_b1 - wd_b2) * af) * gf

            merged_buckets = initial_buckets
            for _ in range(10): # Max iterations
                new_buckets_list, merging_occurred, idx = [], False, 0
                while idx < len(merged_buckets):
                    if idx < len(merged_buckets) - 1 and should_merge_buckets(merged_buckets[idx], merged_buckets[idx+1], frame_interval, density_weight, gap_factor, average_factor, min_gap):
                        b1, b2 = merged_buckets[idx], merged_buckets[idx+1]
                        tc1, tc2 = b1.totalConfidence or 0.0, b2.totalConfidence or 0.0
                        new_buckets_list.append(tag_models.TimeFrame(start=b1.start, end=b2.end, totalConfidence=tc1 + tc2))
                        idx += 2; merging_occurred = True
                    else:
                        new_buckets_list.append(merged_buckets[idx]); idx += 1
                merged_buckets = new_buckets_list
                if not merging_occurred: break
            
            final_buckets = [b for b in merged_buckets if b.get_duration(frame_interval) >= tag_min_duration]
            if final_buckets: toReturn[category][renamed_tag] = final_buckets
    return toReturn

def compute_video_timespans_proportional_merge(
    video_result: AIVideoResult,
    category_settings: Dict[str, Dict[str, Any]], # Category -> TagName -> SettingsDict
    prop: float = 0.5
) -> Dict[str, Dict[str, List[tag_models.TimeFrame]]]:
    video_duration: float = video_result.metadata.duration
    toReturn: Dict[str, Dict[str, List[tag_models.TimeFrame]]] = {}
    
    for category, tag_to_raw_timespans_map in video_result.timespans.items():
        if category not in category_settings: continue
        category_tag_settings = category_settings[category]
        toReturn[category] = {}

        frame_interval: float = 0.5
        if video_result.metadata.models and category in video_result.metadata.models:
            frame_interval = float(video_result.metadata.models[category].frame_interval)

        for tag, raw_timespans_list in tag_to_raw_timespans_map.items():
            if tag not in category_tag_settings: continue
            tag_config = category_tag_settings[tag]
            tag_threshold: float = float(get_or_default(tag_config, 'TagThreshold', 0.5))
            tag_max_gap_config: float = format_duration_or_percent(get_or_default(tag_config, 'MaxGap', "6s"), video_duration)
            tag_min_duration: float = format_duration_or_percent(get_or_default(tag_config, 'MinMarkerDuration', "12s"), video_duration)
            renamed_tag: str = get_or_default(tag_config, 'RenamedTag', tag)
            if tag_min_duration <= 0: continue

            processed_tag_timeframes: List[TagTimeFrame] = []
            for raw_timespan_obj in raw_timespans_list:
                if raw_timespan_obj.confidence is None or raw_timespan_obj.confidence < tag_threshold: continue
                
                if not processed_tag_timeframes:
                    processed_tag_timeframes.append(deepcopy(raw_timespan_obj))
                else:
                    previous_tf_obj = processed_tag_timeframes[-1]
                    current_raw_start = raw_timespan_obj.start
                    current_raw_end = raw_timespan_obj.end if raw_timespan_obj.end is not None else current_raw_start
                    prev_end = previous_tf_obj.end if previous_tf_obj.end is not None else previous_tf_obj.start
                    
                    gap_with_prev = (current_raw_start - prev_end) - frame_interval
                    last_segment_duration = (prev_end - previous_tf_obj.start) + frame_interval

                    if gap_with_prev <= tag_max_gap_config or (last_segment_duration > 0 and gap_with_prev <= prop * last_segment_duration):
                        previous_tf_obj.end = current_raw_end
                    else:
                        processed_tag_timeframes.append(deepcopy(raw_timespan_obj))

            final_tag_timeframes_prop = [
                tag_models.TimeFrame(start=tf.start, end=(tf.end if tf.end is not None else tf.start), totalConfidence=None)
                for tf in processed_tag_timeframes 
                if ((tf.end is not None and ((tf.end - tf.start) + frame_interval >= tag_min_duration)) or
                    (tf.end is None and frame_interval >= tag_min_duration))
            ]
            if final_tag_timeframes_prop: toReturn[category][renamed_tag] = final_tag_timeframes_prop
    return toReturn


# Main function to be called by package users, allowing method and params selection
def compute_final_video_timespans(
    video_result: AIVideoResult,
    category_settings: Dict[str, Dict[str, Any]], # User provides this
    method_name: str = "Clustering", # Default method
    method_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, List[tag_models.TimeFrame]]]:
    
    timespan_methods_registry: Dict[str, Callable] = {
        "OG": compute_video_timespans_og,
        "Clustering": compute_video_timespans_clustering,
        "Proportional_Merge": compute_video_timespans_proportional_merge
    }

    method_func = timespan_methods_registry.get(method_name)
    if not method_func:
        logger.error(f"Timespan method '{method_name}' not found. Falling back to OG.")
        return compute_video_timespans_og(video_result, category_settings)

    # Prepare parameters for the chosen method
    params_to_pass = {}
    if method_name == "Clustering":
        # Default clustering params if not provided by user
        default_clustering_params = {
            "density_weight": 0.2, "gap_factor": 0.75, 
            "average_factor": 0.5, "min_gap": 1.0
        }
        params_to_pass = {**default_clustering_params, **(method_params or {})}
    elif method_name == "Proportional_Merge":
        default_prop_params = {"prop": 0.5}
        params_to_pass = {**default_prop_params, **(method_params or {})}
    
    try:
        if method_name == "OG":
            return method_func(video_result, category_settings)
        else:
            return method_func(video_result, category_settings, **params_to_pass)
    except Exception as e:
        logger.error(f"Error calling timespan method {method_name} with params {params_to_pass}: {e}", exc_info=True)
        logger.warning("Falling back to OG timespan computation.")
        return compute_video_timespans_og(video_result, category_settings)


def compute_video_tags_summary(
    video_result: AIVideoResult,
    category_settings: Dict[str, Dict[str, Any]] # User provides this
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
    """
    Computes a summary of tags present in the video and their total durations/scores.
    This is based on the raw timespans in AIVideoResult, filtered by TagThreshold.
    """
    video_tags: Dict[str, Set[str]] = {} # Category -> Set of RenamedTags
    tag_totals: Dict[str, Dict[str, float]] = {} # Category -> RenamedTag -> Total Duration
    video_duration: float = video_result.metadata.duration
    
    for category, tag_to_raw_timespans_map in video_result.timespans.items():
        if category not in category_settings:
            logger.debug(f"Category {category} not found in category_settings for tag summary.")
            continue
        
        category_tag_settings = category_settings[category]
        video_tags[category] = set()
        tag_totals[category] = {}
        
        frame_interval: float = 0.5 # Default
        if video_result.metadata.models and category in video_result.metadata.models:
            frame_interval = float(video_result.metadata.models[category].frame_interval)

        for tag, raw_timespans_list in tag_to_raw_timespans_map.items():
            if tag not in category_tag_settings: continue
            
            tag_config = category_tag_settings[tag]
            required_duration: float = format_duration_or_percent(get_or_default(tag_config, 'RequiredDuration', "20s"), video_duration)
            tag_threshold: float = float(get_or_default(tag_config, 'TagThreshold', 0.5))
            renamed_tag: str = get_or_default(tag_config, 'RenamedTag', tag)
            
            current_tag_total_duration: float = 0.0
            for raw_timespan in raw_timespans_list:
                if raw_timespan.confidence is not None and raw_timespan.confidence < tag_threshold:
                    continue
                
                start_time = raw_timespan.start
                end_time = raw_timespan.end if raw_timespan.end is not None else start_time
                current_tag_total_duration += (end_time - start_time) + frame_interval
            
            tag_totals[category][renamed_tag] = current_tag_total_duration
            if required_duration > 0 and current_tag_total_duration >= required_duration:
                video_tags[category].add(renamed_tag)
    return video_tags, tag_totals


def compute_full_video_tag_info(
    video_result: AIVideoResult,
    category_settings: Dict[str, Dict[str, Any]], # User provides this
    timespan_processing_method: str = "Clustering", # Default method
    timespan_method_params: Optional[Dict[str, Any]] = None
) -> tag_models.VideoTagInfo:
    """
    Top-level function to compute all processed tag information for a video.
    """
    processed_timespans = compute_final_video_timespans(
        video_result, 
        category_settings, 
        timespan_processing_method, 
        timespan_method_params
    )
    
    # compute_video_tags_summary uses the raw timespans from video_result,
    # not the processed_timespans. This matches original logic.
    summary_tags, summary_totals = compute_video_tags_summary(video_result, category_settings)
    
    return tag_models.VideoTagInfo(
        video_duration=video_result.metadata.duration, 
        video_tags=summary_tags, 
        tag_totals=summary_totals, 
        tag_timespans=processed_timespans # Use the processed timespans here
    )

# The determine_optimal_timespan_settings function is highly specific to an optimization workflow
# that involves sweeping parameters and comparing to "desired_timespan_data".
# This is more of an analysis/tuning script utility than a core package function.
# It will be omitted from the package version of this file for now.
# If needed, it can be provided as an example script or utility outside the core library.
