"""
Simplified action boundary detector focused on end-of-action binary search.
Used in Phase 2 of the hybrid linear scan + binary search algorithm.
"""

import logging
from typing import Dict, List, Any
from .action_range import ActionRange

class ActionBoundaryDetector:
    """
    Simplified action boundary detector focused on end-of-action binary search.
    Complex start boundary logic has been removed in favor of Phase 1 linear scan.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.logger = logging.getLogger("logger")

    def update_action_boundaries(
        self,
        action_ranges: List[ActionRange],
        frame_idx: int,
        action_results: Dict[str, float],
        total_frames: int
    ) -> None:
        """
        Update action boundaries based on VLM results for a single frame.
        
        Simplified to handle only end boundary detection since start boundaries
        are found in Phase 1 linear scan.
        """
        for action_range in action_ranges:
            if action_range.is_resolved():
                continue

            confidence = action_results.get(action_range.action_tag, 0.0)
            is_present = confidence >= self.threshold
            action_range.last_midpoint = frame_idx

            # Only handle end boundary search since starts are found in Phase 1
            if action_range.searching_end:
                if action_range.end_search_start is None or action_range.end_search_end is None:
                    continue

                if is_present:
                    # This frame has the action present. Record it as potential end frame
                    # and search for a later end frame.
                    action_range.end_found = frame_idx
                    action_range.end_search_start = frame_idx + 1
                else:
                    # Action is absent, so the actual end must be before this frame.
                    action_range.end_search_end = frame_idx - 1
                
                # If the search range has crossed, the end boundary is found.
                # The last recorded end_found is the correct end frame.
                if action_range.end_search_start > action_range.end_search_end:
                    self.logger.debug(f"End boundary found for {action_range.action_tag} at frame {action_range.end_found}")
            else:
                # This shouldn't happen in the hybrid approach since Phase 1 handles start detection
                self.logger.warning(f"Unexpected start boundary search for {action_range.action_tag} at frame {frame_idx}")
