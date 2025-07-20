"""
Binary Search Pipeline Processor

This module provides a new pipeline-based implementation of the binary search video processor
that uses discrete pipeline stages instead of the monolithic ParallelBinarySearchEngine.

The processor orchestrates 5 pipeline stages:
1. MetadataExtractionStage - Video analysis and initialization
2. CandidateProposalStage - Linear scan for action candidates  
3. StartRefinementStage - Binary search for precise start boundaries
4. EndDeterminationStage - Binary search for precise end boundaries
5. ResultCompilationStage - Aggregate results into final format

This maintains 100% API compatibility with the existing BinarySearchProcessor while
providing better modularity and maintainability.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from PIL import Image
import torch

from .models import Model
from .config_models import ModelConfig
from .async_utils import QueueItem, ItemFuture
from .vlm_batch_coordinator import VLMBatchCoordinator, IntegratedVLMCoordinator, MockVLMCoordinator


class BinarySearchPipelineProcessor(Model):
    """
    Pipeline-based binary search processor that orchestrates discrete stages.
    
    Maintains complete external API compatibility with BinarySearchProcessor
    while providing improved modularity through pipeline stages.
    """
    
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.logger = logging.getLogger("logger")
        
        # Configuration
        self.binary_search_enabled = True
        self.process_for_vlm = True
        self.device = model_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half_precision = getattr(model_config, 'use_half_precision', True)
        
        # Pipeline stage instances will be created on demand
        self._metadata_stage = None
        self._candidate_stage = None
        self._start_refinement_stage = None
        self._end_determination_stage = None
        self._result_compilation_stage = None
        
    async def worker_function(self, data: List[QueueItem]) -> None:
        """Process queue items using the pipeline stages"""
        for item in data:
            try:
                await self._process_item_through_pipeline(item)
            except Exception as e:
                self.logger.error(f"Error in BinarySearchPipelineProcessor: {e}", exc_info=True)
                item.item_future.set_exception(e)
    
    async def _process_item_through_pipeline(self, item: QueueItem) -> None:
        """Process a single item through all pipeline stages"""
        item_future = item.item_future
        
        # Extract inputs (same as original BinarySearchProcessor)
        video_path = item_future[item.input_names[0]]
        use_timestamps = item_future[item.input_names[1]]
        frame_interval_override = item_future[item.input_names[2]]
        threshold = item_future[item.input_names[3]] if item.input_names[3] in item_future else 0.5
        vr_video = item_future[item.input_names[5]] if len(item.input_names) > 5 else False
        
        # Get VLM configuration from pipeline
        vlm_config = self._extract_vlm_config(item_future)
        if vlm_config is None:
            self.logger.error("No VLM configuration found")
            await item_future.set_data(item.output_names[0], [])
            return
        
        # Extract action tags from VLM config
        action_tags = vlm_config.get("tag_list", [])
        if not action_tags:
            self.logger.error("No action tags found in VLM config")
            await item_future.set_data(item.output_names[0], [])
            return
        
        if not self.binary_search_enabled or not self.process_for_vlm:
            self.logger.error("Binary search disabled or not in VLM mode - pipeline processor requires VLM mode")
            await item_future.set_data(item.output_names[0], [])
            return
        
        # Get VLM coordinator from pipeline
        vlm_coordinator = self._get_vlm_coordinator(item_future)
        if vlm_coordinator is None:
            self.logger.error("No VLM coordinator available")
            await item_future.set_data(item.output_names[0], [])
            return
        
        # Create VLM analyzer function
        async def vlm_analyze_function(frame_pil: Image.Image) -> Dict[str, float]:
            """Wrapper function for VLM analysis using actual VLM coordinator"""
            return await vlm_coordinator.analyze_frame(frame_pil)
        
        # Progress callback
        callback = item_future.get("callback")
        if callback:
            callback(10)  # Starting pipeline
        
        try:
            # Execute pipeline stages sequentially
            self.logger.info(f"Starting binary search pipeline for video: {video_path}")
            
            # Stage 1: Metadata Extraction
            video_metadata = await self._execute_metadata_extraction(
                video_path, action_tags, threshold, 
                frame_interval_override or 1.0, use_timestamps, 10
            )
            
            if callback:
                callback(20)
            
            # Stage 2: Candidate Proposal
            video_metadata = await self._execute_candidate_proposal(video_metadata, vlm_analyze_function)
            
            if callback:
                callback(50)
            
            # Stage 3: Start Refinement
            video_metadata = await self._execute_start_refinement(video_metadata, vlm_analyze_function)
            
            if callback:
                callback(70)
            
            # Stage 4: End Determination
            video_metadata = await self._execute_end_determination(video_metadata, vlm_analyze_function)
            
            if callback:
                callback(85)
            
            # Stage 5: Result Compilation
            frame_results = await self._execute_result_compilation(video_metadata)
            
            if callback:
                callback(90)
            
            # Convert frame results to children format (same as original processor)
            children = []
            for fr in frame_results:
                frame_index = fr["frame_index"]
                # Convert action_results to actiondetection format
                actiondetection = []
                for action_tag, confidence in fr["action_results"].items():
                    actiondetection.append((action_tag, confidence))
                
                self.logger.debug(f'Creating child for frame_index: {frame_index}, actiondetection: {actiondetection}')
                result_future = await ItemFuture.create(item_future, {}, item_future.handler)
                await result_future.set_data("frame_index", frame_index)
                await result_future.set_data("actiondetection", actiondetection)
                children.append(result_future)
            
            # Set final output
            await item_future.set_data(item.output_names[0], children)
            
            if callback:
                callback(100)
            
            self.logger.info(f"Binary search pipeline completed: {len(frame_results)} frame results")
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    async def _execute_metadata_extraction(
        self, video_path: str, action_tags: List[str], threshold: float,
        frame_interval: float, use_timestamps: bool, max_concurrent_vlm_calls: int
    ) -> Dict[str, Any]:
        """Execute metadata extraction stage"""
        if self._metadata_stage is None:
            from .pipeline_stages import MetadataExtractionStage
            config = ModelConfig(type="metadata_extraction_stage")
            self._metadata_stage = MetadataExtractionStage(config)
        
        # Create mock item for stage processing
        mock_future = await self._create_mock_future({
            "video_path": video_path,
            "action_tags": action_tags,
            "threshold": threshold,
            "frame_interval": frame_interval,
            "use_timestamps": use_timestamps,
            "max_concurrent_vlm_calls": max_concurrent_vlm_calls
        })
        
        mock_item = QueueItem(
            mock_future,
            ["video_path", "action_tags", "threshold", "frame_interval", "use_timestamps", "max_concurrent_vlm_calls"],
            ["video_metadata"]
        )
        
        await self._metadata_stage.process_item(mock_item)
        return mock_future["video_metadata"]
    
    async def _execute_candidate_proposal(self, video_metadata: Dict, vlm_analyze_function) -> Dict[str, Any]:
        """Execute candidate proposal stage"""
        if self._candidate_stage is None:
            from .pipeline_stages import CandidateProposalStage
            config = ModelConfig(type="candidate_proposal_stage")
            self._candidate_stage = CandidateProposalStage(config)
        
        mock_future = await self._create_mock_future({
            "video_metadata": video_metadata,
            "vlm_analyze_function": vlm_analyze_function
        })
        
        mock_item = QueueItem(
            mock_future,
            ["video_metadata", "vlm_analyze_function"],
            ["video_metadata"]
        )
        
        await self._candidate_stage.process_item(mock_item)
        return mock_future["video_metadata"]
    
    async def _execute_start_refinement(self, video_metadata: Dict, vlm_analyze_function) -> Dict[str, Any]:
        """Execute start refinement stage"""
        if self._start_refinement_stage is None:
            from .pipeline_stages import StartRefinementStage
            config = ModelConfig(type="start_refinement_stage")
            self._start_refinement_stage = StartRefinementStage(config)
        
        mock_future = await self._create_mock_future({
            "video_metadata": video_metadata,
            "vlm_analyze_function": vlm_analyze_function
        })
        
        mock_item = QueueItem(
            mock_future,
            ["video_metadata", "vlm_analyze_function"],
            ["video_metadata"]
        )
        
        await self._start_refinement_stage.process_item(mock_item)
        return mock_future["video_metadata"]
    
    async def _execute_end_determination(self, video_metadata: Dict, vlm_analyze_function) -> Dict[str, Any]:
        """Execute end determination stage"""
        if self._end_determination_stage is None:
            from .pipeline_stages import EndDeterminationStage
            config = ModelConfig(type="end_determination_stage")
            self._end_determination_stage = EndDeterminationStage(config)
        
        mock_future = await self._create_mock_future({
            "video_metadata": video_metadata,
            "vlm_analyze_function": vlm_analyze_function
        })
        
        mock_item = QueueItem(
            mock_future,
            ["video_metadata", "vlm_analyze_function"],
            ["video_metadata"]
        )
        
        await self._end_determination_stage.process_item(mock_item)
        return mock_future["video_metadata"]
    
    async def _execute_result_compilation(self, video_metadata: Dict) -> List[Dict[str, Any]]:
        """Execute result compilation stage"""
        if self._result_compilation_stage is None:
            from .pipeline_stages import ResultCompilationStage
            config = ModelConfig(type="result_compilation_stage")
            self._result_compilation_stage = ResultCompilationStage(config)
        
        mock_future = await self._create_mock_future({
            "video_metadata": video_metadata
        })
        
        mock_item = QueueItem(
            mock_future,
            ["video_metadata"],
            ["frame_results"]
        )
        
        await self._result_compilation_stage.process_item(mock_item)
        return mock_future["frame_results"]
    
    async def _create_mock_future(self, data: Dict[str, Any]) -> ItemFuture:
        """Create a mock ItemFuture for stage processing"""
        async def mock_handler(future, key):
            pass
        
        mock_future = await ItemFuture.create(None, data, mock_handler)
        return mock_future
    
    def _extract_vlm_config(self, item_future: ItemFuture) -> Optional[Dict[str, Any]]:
        """Extract VLM configuration from pipeline context"""
        # Same logic as original BinarySearchProcessor
        pipeline = item_future.get("pipeline")
        if pipeline is None:
            return None
        
        # Find VLM model in pipeline
        for model_wrapper in pipeline.models:
            if hasattr(model_wrapper.model, 'model') and hasattr(model_wrapper.model.model, 'client_config'):
                vlm_model = model_wrapper.model.model
                return {
                    "tag_list": getattr(vlm_model.client_config, 'tag_list', []),
                    "threshold": getattr(vlm_model.client_config, 'vlm_detected_tag_confidence', 0.99)
                }
        
        return None
    
    def _get_vlm_coordinator(self, item_future: ItemFuture) -> Optional[VLMBatchCoordinator]:
        """Get VLM coordinator from pipeline context"""
        # Same logic as original BinarySearchProcessor
        pipeline = item_future.get("pipeline")
        if pipeline is None:
            return None
        
        # Find VLM model in pipeline
        for model_wrapper in pipeline.models:
            if hasattr(model_wrapper.model, 'model') and hasattr(model_wrapper.model.model, 'vlm_model'):
                vlm_model = model_wrapper.model.model
                if vlm_model.vlm_model is not None:
                    return IntegratedVLMCoordinator(vlm_model.vlm_model)
        
        return None
    

