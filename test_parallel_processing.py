#!/usr/bin/env python3
"""
Test script to verify parallel video processing is working correctly.
This script creates dummy video files and tests the parallel processing behavior.
"""
import asyncio
import logging
import time
import tempfile
import os
from pathlib import Path

# Configure logging to see timing information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_dummy_video(path: str, duration: int = 5):
    """Create a dummy video file for testing"""
    # Create a minimal MP4 file (just headers, not actual video data)
    # This is enough for the preprocessing to work
    dummy_mp4_data = b'\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00mp42isom\x00\x00\x00\x08free'
    with open(path, 'wb') as f:
        f.write(dummy_mp4_data)
    print(f"Created dummy video: {path}")

async def test_parallel_processing():
    """Test parallel video processing"""
    print("🧪 Testing Parallel Video Processing")
    print("=" * 50)
    
    # Create temporary directory for test videos
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple test videos
        test_videos = []
        for i in range(5):
            video_path = os.path.join(temp_dir, f"test_video_{i}.mp4")
            create_dummy_video(video_path)
            test_videos.append(video_path)
        
        print(f"Created {len(test_videos)} test videos")
        
        # Import the VLM engine
        try:
            from vlm_engine import VLMEngine
            from vlm_engine.config_models import EngineConfig, PipelineConfig, ModelConfig, PipelineModelConfig
        except ImportError as e:
            print(f"❌ Failed to import VLM engine: {e}")
            return
        
        # Create engine configuration
        engine_config = EngineConfig(
            active_ai_models=["vlm_nsfw_model"],
            pipelines={
                "video_pipeline_dynamic": PipelineConfig(
                    inputs=[
                        "video_path", "return_timestamps", "time_interval", 
                        "threshold", "return_confidence", "vr_video",
                        "existing_video_data", "skipped_categories"
                    ],
                    output="results",
                    short_name="dynamic_video",
                    version=2.0,
                    models=[
                        PipelineModelConfig(
                            name="dynamic_video_ai",
                            inputs=["video_path", "return_timestamps", "time_interval", 
                                   "threshold", "return_confidence", "vr_video",
                                   "existing_video_data", "skipped_categories"],
                            outputs="results",
                        ),
                    ],
                )
            },
            models={
                "binary_search_processor_dynamic": ModelConfig(
                    type="binary_search_processor", 
                    model_file_name="binary_search_processor_dynamic",
                    instance_count=10,
                    max_batch_size=1,
                    max_concurrent_requests=20,
                ),
                "vlm_nsfw_model": ModelConfig(
                    type="vlm_model",
                    model_file_name="vlm_nsfw_model",
                    model_category="actiondetection",
                    model_id="Haven-adult",
                    model_identifier=93848,
                    model_version="1.0",
                    use_multiplexer=True,
                    max_concurrent_requests=50,
                    connection_pool_size=100,
                    instance_count=10,
                    max_batch_size=1,
                    multiplexer_endpoints=[
                        {
                            "base_url": "http://localhost:8080/v1/",  # Mock endpoint
                            "api_key": "",
                            "name": "test-endpoint",
                            "weight": 100,
                            "is_fallback": False
                        }
                    ],
                    tag_list=["Test Action"]
                ),
                "result_coalescer": ModelConfig(type="python", model_file_name="result_coalescer"),
                "result_finisher": ModelConfig(type="python", model_file_name="result_finisher"),
                "batch_awaiter": ModelConfig(type="python", model_file_name="batch_awaiter"),
                "video_result_postprocessor": ModelConfig(type="python", model_file_name="video_result_postprocessor"),
            },
            category_config={"actiondetection": {"Test Action": {"TagThreshold": 0.5}}}
        )
        
        # Initialize the engine
        try:
            engine = VLMEngine(config=engine_config)
            await engine.initialize()
            print("✅ Engine initialized successfully")
        except Exception as e:
            print(f"❌ Engine initialization failed: {e}")
            print("This is expected if the VLM endpoint is not running")
            return
        
        # Test parallel processing
        print("\n🚀 Testing Parallel Video Processing")
        print("Watch the timestamps to verify concurrent execution...")
        
        semaphore = asyncio.Semaphore(5)  # Allow 5 concurrent videos
        start_times = {}
        end_times = {}
        
        async def process_video_with_timing(video_path, video_index):
            async with semaphore:
                start_time = time.time()
                start_times[video_index] = start_time
                
                print(f"🎬 [{time.strftime('%H:%M:%S')}] Starting video {video_index + 1}: {Path(video_path).name}")
                
                try:
                    result = await engine.process_video(
                        video_path,
                        frame_interval=1.0,
                        return_timestamps=True,
                        threshold=0.5,
                        return_confidence=True
                    )
                    
                    end_time = time.time()
                    end_times[video_index] = end_time
                    duration = end_time - start_time
                    
                    print(f"✅ [{time.strftime('%H:%M:%S')}] Completed video {video_index + 1} in {duration:.2f}s")
                    return result
                    
                except Exception as e:
                    end_time = time.time()
                    end_times[video_index] = end_time
                    duration = end_time - start_time
                    print(f"❌ [{time.strftime('%H:%M:%S')}] Failed video {video_index + 1} after {duration:.2f}s: {e}")
                    return None
        
        # Create tasks for all videos
        tasks = [
            asyncio.create_task(process_video_with_timing(video_path, i))
            for i, video_path in enumerate(test_videos)
        ]
        
        print(f"\n⚡ Processing {len(test_videos)} videos concurrently...")
        overall_start = time.time()
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        print(f"\n🎉 All videos completed in {overall_duration:.2f}s")
        
        # Analyze timing to verify parallel execution
        print("\n📊 Timing Analysis:")
        successful_videos = [i for i, result in enumerate(results) if result is not None]
        
        if len(successful_videos) > 1:
            # Check for overlapping execution times
            overlaps = 0
            for i in successful_videos:
                for j in successful_videos:
                    if i != j:
                        # Check if video i and j had overlapping execution
                        i_start, i_end = start_times[i], end_times[i]
                        j_start, j_end = start_times[j], end_times[j]
                        
                        if (i_start < j_end and j_start < i_end):
                            overlaps += 1
            
            if overlaps > 0:
                print(f"✅ PARALLEL PROCESSING DETECTED: {overlaps} video pairs had overlapping execution")
                print("   This confirms videos are being processed concurrently!")
            else:
                print("❌ SEQUENTIAL PROCESSING DETECTED: No overlapping execution found")
                print("   Videos appear to be processed one after another")
        
        # Show individual video timing
        for i in successful_videos:
            start_time = start_times[i]
            end_time = end_times[i]
            relative_start = start_time - overall_start
            relative_end = end_time - overall_start
            duration = end_time - start_time
            
            print(f"   Video {i+1}: {relative_start:.2f}s - {relative_end:.2f}s (duration: {duration:.2f}s)")

if __name__ == "__main__":
    asyncio.run(test_parallel_processing())
