import asyncio
import logging
from vlm_engine import VLMEngine
from vlm_engine.config_models import EngineConfig, PipelineConfig, ModelConfig, PipelineModelConfig
from typing import Callable

# Configure logging
logging.basicConfig(level=logging.ERROR)

async def main():
    # Define the configuration for the engine with Binary Search Processor
    engine_config = EngineConfig(
        active_ai_models=["llm_vlm_model"],
        pipelines={
            "video_pipeline_dynamic": PipelineConfig(
                inputs=[
                    "video_path",
                    "return_timestamps",
                    "time_interval",
                    "threshold",
                    "return_confidence",
                    "vr_video",
                    "existing_video_data",
                    "skipped_categories",
                ],
                output="results",
            version=2.0,  # Version 2.0 with binary search optimization
            models=[
                PipelineModelConfig(
                        name="video_analysis_pipeline",  # Updated from dynamic_video_ai
                        inputs=["video_path", "return_timestamps", "time_interval", "threshold", "return_confidence", "vr_video", "existing_video_data", "skipped_categories"],
                        outputs="results",
                    ),
                ],
            )
        },
        models={
            # NEW: Binary Search Processor replaces video_preprocessor_dynamic
            "binary_search_processor_dynamic": ModelConfig(
                type="binary_search_processor",
                instance_count=10,           # Multiple instances for parallel video processing
                max_batch_size=1,            # Process one video at a time for better concurrency
                max_concurrent_requests=20,  # Allow concurrent video processing
            ),
            "llm_vlm_model": ModelConfig(
                type="vlm_model",
                model_category="humanactivityevaluation",
                model_id="Haven-adult",
                use_multiplexer=True,
                # Increase concurrency limits for parallel video processing
                max_concurrent_requests=50,  # Increased from default 20
                connection_pool_size=100,    # Increased from default 50
                instance_count=10,           # Multiple instances for parallel processing
                max_batch_size=1,            # Process one frame at a time for better concurrency
                multiplexer_endpoints=[
                    {
                        "base_url": "https://ethical-jennifer-phys-combining.trycloudflare.com/v1/",
                        "api_key": "",  # Use empty string for endpoints that don't require auth
                        "name": "lm-studio-primary",
                        "weight": 100,  # Higher weight = more requests
                        "is_fallback": False
                    }
                ],
                # IMPORTANT: These action tags drive the binary search
                tag_list = [
                    "Anal Fucking", "Ass Licking", "Ass Penetration", "Ball Licking/Sucking", "Blowjob", "Cum on Person",
                    "Cum Swapping", "Cumshot", "Deepthroat", "Double Penetration", "Fingering", "Fisting", "Footjob",
                    "Gangbang", "Gloryhole", "Grabbing Ass", "Grabbing Boobs", "Grabbing Hair/Head", "Handjob", "Kissing",
                    "Licking Penis", "Masturbation", "Pissing", "Pussy Licking (Clearly Visible)", "Pussy Licking",
                    "Pussy Rubbing", "Sucking Fingers", "Sucking Toy/Dildo", "Wet (Genitals)", "Titjob", "Tribbing/Scissoring",
                    "Undressing", "Vaginal Penetration", "Vaginal Fucking", "Vibrating"
                ]
            ),
            "result_coalescer": ModelConfig(type="python"),
            "result_finisher": ModelConfig(type="python"),
            "batch_awaiter": ModelConfig(type="python"),
            "video_result_postprocessor": ModelConfig(type="python"),
        },
        category_config={
            "humanactivityevaluation": {
                "Anal Fucking": {
                    "RenamedTag": "Anal Fucking",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Ass Licking": {
                    "RenamedTag": "Ass Licking",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Ass Penetration": {
                    "RenamedTag": "Ass Penetration",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Ball Licking/Sucking": {
                    "RenamedTag": "Ball Licking/Sucking",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Blowjob": {
                    "RenamedTag": "Blowjob",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Cum on Person": {
                    "RenamedTag": "Cum on Person",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Cum Swapping": {
                    "RenamedTag": "Cum Swapping",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Cumshot": {
                    "RenamedTag": "Cumshot",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Deepthroat": {
                    "RenamedTag": "Deepthroat",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Double Penetration": {
                    "RenamedTag": "Double Penetration",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Fingering": {
                    "RenamedTag": "Fingering",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Fisting": {
                    "RenamedTag": "Fisting",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Footjob": {
                    "RenamedTag": "Footjob",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Gangbang": {
                    "RenamedTag": "Gangbang",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Gloryhole": {
                    "RenamedTag": "Gloryhole",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Grabbing Ass": {
                    "RenamedTag": "Grabbing Ass",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Grabbing Boobs": {
                    "RenamedTag": "Grabbing Boobs",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Grabbing Hair/Head": {
                    "RenamedTag": "Grabbing Hair/Head",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Handjob": {
                    "RenamedTag": "Handjob",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Kissing": {
                    "RenamedTag": "Kissing",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Licking Penis": {
                    "RenamedTag": "Licking Penis",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Masturbation": {
                    "RenamedTag": "Masturbation",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Pissing": {
                    "RenamedTag": "Pissing",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Pussy Licking (Clearly Visible)": {
                    "RenamedTag": "Pussy Licking (Clearly Visible)",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Pussy Licking": {
                    "RenamedTag": "Pussy Licking",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Pussy Rubbing": {
                    "RenamedTag": "Pussy Rubbing",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Sucking Fingers": {
                    "RenamedTag": "Sucking Fingers",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Sucking Toy/Dildo": {
                    "RenamedTag": "Sucking Toy/Dildo",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Wet (Genitals)": {
                    "RenamedTag": "Wet (Genitals)",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Titjob": {
                    "RenamedTag": "Titjob",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Tribbing/Scissoring": {
                    "RenamedTag": "Tribbing/Scissoring",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Undressing": {
                    "RenamedTag": "Undressing",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Vaginal Penetration": {
                    "RenamedTag": "Vaginal Penetration",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Vaginal Fucking": {
                    "RenamedTag": "Vaginal Fucking",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
                "Vibrating": {
                    "RenamedTag": "Vibrating",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
            }
        }
    )

    # Create an instance of the engine with the configuration object
    engine = VLMEngine(config=engine_config)
    await engine.initialize()

    print("🚀 VLM Engine with Parallel Binary Search initialized!")
    print("📊 Expected Performance Improvement:")
    print("   • 98% reduction in API calls for large videos")
    print("   • O(K × log N) complexity vs O(N) linear sampling")
    print("   • Intelligent action boundary detection")
    print("   • Maintains identical external API compatibility")
    print()

    # Process a video with binary search optimization
    videos = ["sample1.mp4", "sample2.mp4"]
    try:
        # Increase semaphore limit significantly for video processing
        # The issue was that the semaphore was too restrictive (5) for proper video processing
        semaphore = asyncio.Semaphore(20)
        
        video_progress = {v: 0 for v in videos}
        
        def make_callback(vid: str) -> Callable[[int], None]:
            def cb(p: int):
                video_progress[vid] = p
                logging.info(f"Progress: { {k: f'{v}%' for k,v in video_progress.items()} }")
            return cb
        
        async def process_video_with_semaphore(video_path, video_index):
            async with semaphore:
                print(f"🎬 Starting video {video_index + 1}: {video_path}")
                start_time = asyncio.get_event_loop().time()
                
                result = await engine.process_video(
                    video_path,
                    frame_interval=30.0,
                    return_timestamps=True,
                    threshold=0.5,
                    return_confidence=True,
                    progress_callback=make_callback(video_path)
                )
                
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                print(f"✅ Completed video {video_index + 1} in {duration:.2f}s")
                return result

        # Create tasks with proper indexing for debugging
        tasks = [
            asyncio.create_task(process_video_with_semaphore(video_path, i)) 
            for i, video_path in enumerate(videos)
        ]
        
        print(f"⚡ Processing {len(videos)} videos concurrently with semaphore limit of 20...")
        print("📊 Watch for concurrent execution in the output timestamps...")
        
        # Use asyncio.as_completed to see results as they finish (proves concurrency)
        results = []
        completed_tasks = asyncio.as_completed(tasks)
        
        for i, completed_task in enumerate(completed_tasks):
            result = await completed_task
            results.append(result)
            print(f"🏁 Video {i + 1} finished processing")
        
        print(f"🎉 All {len(videos)} videos completed!")
        
        for i, result in enumerate(results, 1):
            print(f"\n📈 Results for video {i}:")
            print(f"✅ Processing completed successfully!")
            print(f"📋 Results structure: {type(result)}")

            if isinstance(result, dict):
                json_result = result.get("json_result", {})
                if "metadata" in json_result:
                    print(f"📊 Video duration: {json_result['metadata'].get('duration', 'N/A')}s")

                if "timespans" in json_result:
                    total_detections = sum(
                        len(tags) for tags in json_result["timespans"].values()
                    )
                    print(f"🎯 Total action detections: {total_detections}")

            print(f"\n📊 Full results for video {i}: {result}")
        
    except Exception as e:
        logging.error(f"❌ Error processing videos: {e}", exc_info=True)
        print(f"\n🔧 Troubleshooting tips:")
        print(f"   • Ensure video paths are correct: {videos}")
        print(f"   • Check VLM API endpoint is running on http://localhost:7045")
        print(f"   • Verify binary search processor is properly initialized")
        print(f"   • Make sure all video files exist in the current directory")

if __name__ == "__main__":
    asyncio.run(main())
