from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo
# Import from .video_result which will be the new AI_VideoResult.py
# We need AIVideoResult class and its inner Pydantic models like VideoMetadata, ModelInfo, TagTimeFrame
from .video_result import AIVideoResult, VideoMetadata, ModelInfo as ModelInfoV1, TagTimeFrame as TagTimeFrameV1
import math
import logging

logger: logging.Logger = logging.getLogger("logger")

# Removed category_config import and tag_to_category_dict global, as it's not used in to_V1()

class ModelConfigV0(BaseModel):
    frame_interval: float
    threshold: float
    def __str__(self) -> str:
        return f"ModelConfigV0(frame_interval={self.frame_interval}, threshold={self.threshold})"
    
class ModelInfoV0(BaseModel): # This is for the 'models' dict within VideoMetadataV0
    version: float
    ai_model_config: ModelConfigV0 # This structure is different from V1's ModelInfo
    def __str__(self) -> str:
        return f"ModelInfoV0(version={self.version}, ai_model_config={self.ai_model_config})"
    
class VideoMetadataV0(BaseModel):
    video_id: Optional[int] = None # Made optional as not used in to_V1 and might be missing
    duration: float
    phash: Optional[str] = None # Made optional
    models: Dict[str, ModelInfoV0] # Category name to ModelInfoV0
    # These top-level fields were used to construct ModelInfo for V1
    frame_interval: float 
    threshold: float
    ai_version: float # Corresponds to ModelInfo.version in V1
    ai_model_id: int  # Corresponds to ModelInfo.ai_model_id in V1
    ai_model_filename: Optional[str] = None # Corresponds to ModelInfo.file_name in V1

    def __str__(self) -> str:
        return f"VideoMetadataV0(duration={self.duration}, models_count={len(self.models)})"
    
class TagTimeFrameV0(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    start: int
    end: Optional[int] = None
    confidence: float # V0 had confidence as required float

    @field_validator('start', 'end', mode='before')
    @classmethod
    def _floor_and_convert_to_int_v0(cls, v: Any, info: ValidationInfo) -> Optional[int]:
        if v is None and info.field_name == 'end': # 'end' can be None
            return None
        if v is None: # 'start' cannot be None
            raise ValueError(f"TagTimeFrameV0 field '{info.field_name}' received None but is not Optional.")

        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(math.floor(v))
        try:
            return int(math.floor(float(str(v))))
        except (ValueError, TypeError) as e:
            logger.error(f"TagTimeFrameV0 validator for '{info.field_name}': Could not convert '{v}' to int: {e}")
            raise ValueError(f"Invalid value '{v}' for TagTimeFrameV0 field '{info.field_name}'.") from e

    def __str__(self) -> str:
        return f"TagTimeFrameV0(start={self.start}, end={self.end}, confidence={self.confidence})"
    
class TagDataV0(BaseModel): # This structure seems to be from an even older format, not directly used by AIVideoResultV0.timespans
    ai_model_name: str
    time_frames: List[TagTimeFrameV0]
    def __str__(self) -> str:
        return f"TagDataV0(model_name={self.ai_model_name}, time_frames_count={len(self.time_frames)})"

class AIVideoResultV0(BaseModel):
    video_metadata: VideoMetadataV0
    tags: Optional[Dict[str, TagDataV0]] = None # This 'tags' field is not used in to_V1 conversion from 'timespans'
    timespans: Dict[str, Dict[str, List[TagTimeFrameV0]]] # Category -> Tag -> List[TagTimeFrameV0]

    def to_V1(self) -> AIVideoResult:
        # Create V1 ModelInfo instances for each category based on top-level V0 metadata
        model_infos_v1: Dict[str, ModelInfoV1] = {}
        
        # The V0 format had a single set of frame_interval, threshold, etc., in video_metadata,
        # which applied to all categories/models implicitly.
        # V1's ModelInfo is per-category.
        shared_model_info_v1 = ModelInfoV1(
            frame_interval=self.video_metadata.frame_interval,
            threshold=self.video_metadata.threshold,
            version=self.video_metadata.ai_version, # V0 ai_version maps to V1 ModelInfo.version
            ai_model_id=self.video_metadata.ai_model_id,
            file_name=self.video_metadata.ai_model_filename
        )
        
        # Populate model_infos_v1 for each category found in timespans
        # If self.video_metadata.models (V0 format) existed and was per category, we'd use that.
        # But it seems the top-level metadata was global for all V0 results.
        for category_name in self.timespans.keys():
            model_infos_v1[category_name] = shared_model_info_v1.model_copy()


        metadata_v1 = VideoMetadata(
            duration=self.video_metadata.duration, 
            models=model_infos_v1
        )
        
        v1_timespans: Dict[str, Dict[str, List[TagTimeFrameV1]]] = {}
        for category, tags_map_v0 in self.timespans.items():
            v1_tags_map: Dict[str, List[TagTimeFrameV1]] = {}
            for tag_name, ttf_v0_list in tags_map_v0.items():
                v1_tags_map[tag_name] = [
                    TagTimeFrameV1(start=ttf_v0.start, end=ttf_v0.end, confidence=ttf_v0.confidence) 
                    for ttf_v0 in ttf_v0_list
                ]
            v1_timespans[category] = v1_tags_map

        return AIVideoResult(
            schema_version=1,
            metadata=metadata_v1,
            timespans=v1_timespans
        )
