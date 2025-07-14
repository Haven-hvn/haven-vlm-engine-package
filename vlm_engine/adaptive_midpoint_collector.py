"""
Collects unique frame indices from all active action searches
"""

from .action_range import ActionRange
import logging
from typing import List, Set

class AdaptiveMidpointCollector:
    """Collects unique frame indices from all active action searches"""
    
    def __init__(self):
        self.logger = logging.getLogger("logger")
    
    def collect_unique_midpoints(self, action_ranges: List['ActionRange']) -> Set[int]:
        """Collect all unique midpoint frames from active searches (prioritizes end searches)"""
        if all(ar.is_resolved() for ar in action_ranges):
            self.logger.debug("All action searches are already resolved - no midpoints to collect")
            return set()

        midpoints = set()
        start_searches = 0
        end_searches = 0
        
        for action_range in action_ranges:
            if action_range.is_resolved():
                continue
                
            # Prioritize end searches over start searches
            end_midpoint = action_range.get_end_midpoint()
            if end_midpoint is not None:
                midpoints.add(end_midpoint)
                end_searches += 1
                continue
                
            # Add start search midpoints
            start_midpoint = action_range.get_start_midpoint()
            if start_midpoint is not None:
                midpoints.add(start_midpoint)
                start_searches += 1
        
        self.logger.debug(f"Collected {len(midpoints)} unique midpoints: {start_searches} start searches, {end_searches} end searches")
        return midpoints
