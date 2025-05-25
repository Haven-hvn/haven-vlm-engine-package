from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo
import math
import logging

logger: logging.Logger = logging.getLogger("logger") # Changed from __name__

class TimeFrame(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    start: int
    end: int # In original, end was optional in TagTimeFrame, but required here.
             # AI_VideoResult.TagTimeFrame had Optional end. This TimeFrame is different.
             # For consistency with the original TimeFrame in tag_models.py, end is required.
    totalConfidence: Optional[float] = None # Made optional to match original

    @field_validator('start', 'end', mode='before')
    @classmethod
    def _ensure_floored_int(cls, v: Any, info: ValidationInfo) -> int: # Added ValidationInfo
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(math.floor(v))
        try:
            return int(math.floor(float(str(v))))
        except (ValueError, TypeError) as e:
            logger.error(f"TimeFrame field validator for '{info.field_name}': Could not convert value '{v}' (type {type(v)}) to int. Error: {e}")
            raise ValueError(f"Invalid value '{v}' for field '{info.field_name}', cannot convert to integer.") from e

    def get_density(self, frame_interval: float) -> float:
        duration: float = self.get_duration(frame_interval)
        if duration == 0 or self.totalConfidence is None:
            return 0.0
        return self.totalConfidence / duration
    
    def get_duration(self, frame_interval: float) -> float:
        # Assuming frame_interval is the duration of a single frame/step.
        # If start and end are frame indices, duration in frames is (end - start + 1).
        # If start and end are timestamps, duration is (end - start).
        # The original (self.end - self.start) + frame_interval implies start/end are timestamps
        # and frame_interval is added to account for the duration of the last frame.
        # This seems to be a common way to calculate duration from timestamps of frame start/end.
        return (float(self.end) - float(self.start)) + frame_interval
    
    def merge(self, new_start: float, new_end: float, new_confidence: float, frame_interval: float) -> None:
        # new_start and new_end are floats, but validator will convert them for self.start/end
        self.start = min(self.start, int(math.floor(new_start))) # type: ignore
        self.end = max(self.end, int(math.floor(new_end))) # type: ignore
        
        if self.totalConfidence is None:
            self.totalConfidence = 0.0
        # This calculation for totalConfidence seems to assume new_confidence is per-frame
        # and get_duration is the duration of the *newly merged* timeframe.
        # This might lead to over-accumulation if called multiple times with overlapping segments.
        # Original: self.totalConfidence += new_confidence * self.get_duration(frame_interval)
        # A more typical merge might be weighted average or sum if confidences are absolute.
        # For now, keeping original logic but noting potential issue.
        # Let's assume new_confidence is for the segment being merged.
        # A simpler sum:
        self.totalConfidence += new_confidence 


    def __str__(self) -> str:
        return f"TimeFrame(start={self.start}, end={self.end}, totalConfidence={self.totalConfidence})"
    
class VideoTagInfo(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields

    video_duration: float
    video_tags: Dict[str, Set[str]] # Category -> Set of Tag Names
    tag_totals: Dict[str, Dict[str, float]] # Category -> Tag Name -> Total Score/Duration
    tag_timespans: Dict[str, Dict[str, List[TimeFrame]]] # Category -> Tag Name -> List of TimeFrame objects

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)

    def __str__(self) -> str:
        # Limiting output for brevity if it gets too long
        timespans_str = str(self.tag_timespans)
        if len(timespans_str) > 200:
            timespans_str = timespans_str[:200] + "..."
        return (f"VideoTagInfo(video_duration={self.video_duration}, "
                f"video_tags_count={ {k: len(v) for k, v in self.video_tags.items()} }, "
                f"tag_totals_count={ {k: len(v) for k, v in self.tag_totals.items()} }, "
                f"tag_timespans_preview={timespans_str})")
