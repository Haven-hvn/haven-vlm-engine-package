"""
VLM Batch Coordinator
Coordinates VLM API calls for the parallel binary search engine.
Handles both OpenAI and Multiplexer VLM clients with intelligent batching.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from PIL import Image
from .vlm_client import OpenAICompatibleVLMClient
from .multiplexer_vlm_client import MultiplexerVLMClient
from .config_models import ModelConfig


class VLMBatchCoordinator:
    """
    Coordinates VLM API calls for binary search engine.
    Provides intelligent batching and optimized VLM communication.
    """
    
    def __init__(self, vlm_client: Union[OpenAICompatibleVLMClient, MultiplexerVLMClient]):
        self.vlm_client = vlm_client
        self.logger = logging.getLogger("logger")
        self.total_calls = 0
        
        # Performance tracking
        self.batch_sizes = []
        self.response_times = []
        
        self.logger.info(f"VLMBatchCoordinator initialized with {type(vlm_client).__name__}")
    
    async def analyze_frame(self, frame: Image.Image) -> Dict[str, float]:
        """
        Analyze a single frame using the VLM client.
        This is the main interface used by the binary search engine.
        """
        import time
        start_time = time.time()
        
        try:
            result = await self.vlm_client.analyze_frame(frame)
            self.total_calls += 1
            
            # Track performance
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            self.logger.debug(f"VLM analysis completed in {response_time:.2f}s, total calls: {self.total_calls}")
            return result
            
        except Exception as e:
            self.logger.error(f"VLM analysis failed: {e}", exc_info=True)
            # Return zero confidence for all tags on failure
            if hasattr(self.vlm_client, 'tag_list'):
                return {tag: 0.0 for tag in self.vlm_client.tag_list}
            else:
                return {}
    
    async def analyze_frames_batch(self, frames: List[Image.Image]) -> List[Dict[str, float]]:
        """
        Analyze multiple frames concurrently.
        Optimizes performance by processing frames in parallel.
        """
        if not frames:
            return []
        
        self.batch_sizes.append(len(frames))
        self.logger.debug(f"Processing batch of {len(frames)} frames")
        
        # Process frames concurrently
        tasks = [self.analyze_frame(frame) for frame in frames]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Frame {i} analysis failed: {result}")
                # Return zero confidence for failed frames
                if hasattr(self.vlm_client, 'tag_list'):
                    processed_results.append({tag: 0.0 for tag in self.vlm_client.tag_list})
                else:
                    processed_results.append({})
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        if not self.response_times:
            return {
                "total_calls": self.total_calls,
                "avg_response_time": 0.0,
                "avg_batch_size": 0.0
            }
        
        return {
            "total_calls": self.total_calls,
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "avg_batch_size": sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0,
            "max_response_time": max(self.response_times),
            "min_response_time": min(self.response_times)
        }


def get_vlm_model(config: Dict[str, Any]) -> Union[OpenAICompatibleVLMClient, MultiplexerVLMClient]:
    """
    Factory function to create VLM client based on configuration.
    Used by the binary search processor to get VLM instances.
    """
    logger = logging.getLogger("logger")
    
    # Check if multiplexer mode is enabled
    use_multiplexer = config.get("use_multiplexer", False)
    
    if use_multiplexer:
        logger.info("Creating MultiplexerVLMClient for load balancing")
        return MultiplexerVLMClient(config)
    else:
        logger.info("Creating OpenAICompatibleVLMClient for single endpoint")
        return OpenAICompatibleVLMClient(config)


class IntegratedVLMCoordinator:
    """
    Integrated VLM coordinator that connects binary search engine 
    with the existing VLM model infrastructure.
    """
    
    def __init__(self, pipeline_models: List[Any]):
        self.pipeline_models = pipeline_models
        self.vlm_client: Optional[Union[OpenAICompatibleVLMClient, MultiplexerVLMClient]] = None
        self.batch_coordinator: Optional[VLMBatchCoordinator] = None
        self.logger = logging.getLogger("logger")
        
        # Find VLM model in pipeline
        self._initialize_vlm_client()
    
    def _initialize_vlm_client(self) -> None:
        """Initialize VLM client from pipeline models"""
        for model_wrapper in self.pipeline_models:
            if hasattr(model_wrapper.model, 'model') and hasattr(model_wrapper.model.model, 'vlm_model'):
                vlm_model = model_wrapper.model.model.vlm_model
                if vlm_model is not None:
                    self.vlm_client = vlm_model
                    self.batch_coordinator = VLMBatchCoordinator(vlm_model)
                    self.logger.info("VLM client found and coordinator initialized")
                    return
        
        self.logger.warning("No VLM client found in pipeline models")
    
    async def ensure_vlm_loaded(self) -> bool:
        """Ensure VLM client is loaded and ready"""
        if self.vlm_client is None:
            self.logger.error("No VLM client available")
            return False
        
        # For MultiplexerVLMClient, ensure it's initialized
        if hasattr(self.vlm_client, '_ensure_initialized'):
            try:
                await self.vlm_client._ensure_initialized()
            except Exception as e:
                self.logger.error(f"Failed to initialize VLM client: {e}")
                return False
        
        return True
    
    async def analyze_frame(self, frame: Image.Image) -> Dict[str, float]:
        """
        Main interface for binary search engine to analyze frames.
        """
        if not await self.ensure_vlm_loaded():
            # Return empty result if VLM not available
            return {}
        
        if self.batch_coordinator is None:
            self.logger.error("Batch coordinator not initialized")
            return {}
        
        return await self.batch_coordinator.analyze_frame(frame)
    
    def get_action_tags(self) -> List[str]:
        """Get the list of action tags from VLM client"""
        if self.vlm_client and hasattr(self.vlm_client, 'tag_list'):
            return self.vlm_client.tag_list
        return []
    
    def get_threshold(self) -> float:
        """Get the detection threshold from VLM client"""
        if self.vlm_client and hasattr(self.vlm_client, 'vlm_detected_tag_confidence'):
            return self.vlm_client.vlm_detected_tag_confidence
        return 0.5  # Default threshold


class MockVLMCoordinator:
    """
    Mock VLM coordinator for testing and development.
    Simulates VLM responses for binary search algorithm validation.
    """
    
    def __init__(self, action_tags: List[str], mock_responses: Optional[Dict[str, float]] = None):
        self.action_tags = action_tags
        self.mock_responses = mock_responses or {}
        self.call_count = 0
        self.logger = logging.getLogger("logger")
        
        self.logger.info(f"MockVLMCoordinator initialized with {len(action_tags)} action tags")
    
    async def analyze_frame(self, frame: Image.Image) -> Dict[str, float]:
        """Mock frame analysis with simulated responses"""
        self.call_count += 1
        
        # Simulate processing delay
        await asyncio.sleep(0.01)
        
        # Return mock responses or random values
        result = {}
        for tag in self.action_tags:
            if tag in self.mock_responses:
                result[tag] = self.mock_responses[tag]
            else:
                # Simulate random detection with bias toward no detection
                import random
                result[tag] = random.random() * 0.3  # Low probability of detection
        
        self.logger.debug(f"Mock VLM analysis call #{self.call_count}: {len([k for k, v in result.items() if v > 0.5])} detections")
        return result
    
    def get_action_tags(self) -> List[str]:
        """Get action tags for mock coordinator"""
        return self.action_tags
    
    def get_threshold(self) -> float:
        """Get mock threshold"""
        return 0.5 