"""
Detects action boundaries using binary search logic
"""

from .action_range import ActionRange
from .adaptive_midpoint_collector import AdaptiveMidpointCollector
import logging
from typing import Dict, List, Optional

class ActionBoundaryDetector:
    """Detects action boundaries using binary search logic"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.logger = logging.getLogger("logger")
    
    def update_action_boundaries(
        self, 
        action_ranges: List['ActionRange'], 
        frame_idx: int, 
        action_results: Dict[str, float],
        total_frames: int
    ) -> None:
        """Update all action search boundaries based on frame analysis results"""
        
        for action_range in action_ranges:
            if action_range.is_resolved():
                continue
                
            action_confidence = action_results.get(action_range.action_tag, 0.0)
            action_detected = action_confidence >= self.threshold
            
            # Check if this frame is relevant to current search
            start_midpoint = action_range.get_start_midpoint()
            end_midpoint = action_range.get_end_midpoint()
            
            if frame_idx == start_midpoint:
                # Processing start boundary search
                self._update_start_boundary(action_range, frame_idx, action_detected, total_frames)
            elif frame_idx == end_midpoint:
                # Processing end boundary search
                self._update_end_boundary(action_range, frame_idx, action_detected)
    
    def _update_start_boundary(
        self, 
        action_range: 'ActionRange', 
        frame_idx: int, 
        action_detected: bool,
        total_frames: int
    ) -> None:
        """Update start boundary search based on detection result"""
        
        if action_detected:
            # Action found at midpoint - this could be the start frame
            if frame_idx > action_range.start_frame:
                # Found action at the very start of search range
                action_range.start_found = frame_idx
                action_range.confirmed_present = True
                # Initiate end search
                action_range.initiate_end_search(total_frames)
                self.logger.debug(f"Action '{action_range.action_tag}' start found at frame {frame_idx}, initiating end search")
            else:
                # Action detected, search earlier for actual start
                action_range.end_frame = frame_idx
                self.logger.debug(f"Action '{action_range.action_tag}' detected at {frame_idx}, searching earlier: [{action_range.start_frame}, {action_range.end_frame}]")
        else:
            # Action not found at midpoint - search later
            if action_range.end_frame == frame_idx:
                # Reached end of search range without finding action
                action_range.confirmed_absent = True
                self.logger.debug(f"Action '{action_range.action_tag}' confirmed absent in range [{action_range.start_frame}, {action_range.end_frame}]")
            else:
                # Search later in the range
                if frame_idx > action_range.start_frame:
                    action_range.start_frame = frame_idx + 1  # Only search later if midpoint > start
                self.logger.debug(f"Action '{action_range.action_tag}' not detected at {frame_idx}, searching later: [{action_range.start_frame}, {action_range.end_frame}]")
    
    def _update_end_boundary(
        self, 
        action_range: 'ActionRange', 
        frame_idx: int, 
        action_detected: bool
    ) -> None:
        """Update end boundary search based on detection result"""
        
        if action_detected:
            # Action still present - search later for end
            if action_range.end_search_end == frame_idx:
                # Action continues to the end of video
                action_range.end_found = frame_idx
                self.logger.debug(f"Action '{action_range.action_tag}' continues to end of video at frame {frame_idx}")
            else:
                # Action still present, search later
                action_range.end_search_start = frame_idx + 1
                self.logger.debug(f"Action '{action_range.action_tag}' still present at {frame_idx}, searching later: [{action_range.end_search_start}, {action_range.end_search_end}]")
                self.logger.debug(f'Midpoint: {frame_idx}, End Search Start: {action_range.end_search_start}, End Search End: {action_range.end_search_end}')
        else:
            # Action ended - this is past the end frame
            if action_range.end_search_start == frame_idx:
                # Action ended exactly at start of search range
                action_range.end_found = frame_idx - 1
                self.logger.debug(f"Action '{action_range.action_tag}' ended at frame {action_range.end_found}")
            else:
                # Action ended somewhere before this frame, search earlier
                action_range.end_search_end = frame_idx - 1
                self.logger.debug(f"Action '{action_range.action_tag}' ended before {frame_idx}, searching earlier: [{action_range.end_search_start}, {action_range.end_search_end}]")
