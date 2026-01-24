import asyncio
import logging
from vlm_engine import VLMEngine
from vlm_engine.config_models import EngineConfig, PipelineConfig, ModelConfig, PipelineModelConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)

async def main():
    # Define the configuration for the engine with multiplexer support
    engine_config = EngineConfig(
        active_ai_models=["vlm_multiplexer_model"],
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
                version=1.0,
                models=[
                    PipelineModelConfig(
                        name="dynamic_video_ai",
                        inputs=["video_path", "return_timestamps", "time_interval", "threshold", "return_confidence", "vr_video", "existing_video_data", "skipped_categories"],
                        outputs="results",
                    ),
                ],
            )
        },
        models={
            "video_preprocessor_dynamic": ModelConfig(type="video_preprocessor"),
            "vlm_multiplexer_model": ModelConfig(
                type="vlm_model",
                model_category="humanactivityevaluation",
                model_id="HuggingFaceTB/SmolVLM-Instruct",
                # Enable multiplexer mode with performance optimizations
                use_multiplexer=True,
                # Performance optimization settings
                max_concurrent_requests=30,  # Higher concurrency for better throughput
                connection_pool_size=100,    # Larger connection pool for multiple endpoints
                # Configure multiple endpoints for load balancing
                multiplexer_endpoints=[
                    {
                        "base_url": "https://tricks-wellness-villas-anne.trycloudflare.com/v1",
                        "api_key": "",  # Use empty string for endpoints that don't require auth
                        "name": "haven-adult",
                        "weight": 100,  # Higher weight = more requests
                        "is_fallback": False
                    }
                ],
                tag_list=[
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
                "69": {
                    "RenamedTag": "69",
                    "MinMarkerDuration": "1s",
                    "MaxGap": "30s",
                    "RequiredDuration": "1s",
                    "TagThreshold": 0.5,
                },
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
                "Blowjob": {
                    "RenamedTag": "Blowjob",
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
                "Masturbation": {
                    "RenamedTag": "Masturbation",
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
                "Vaginal Fucking": {
                    "RenamedTag": "Vaginal Fucking",
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

    # Process a video using the multiplexer for load balancing
    video_path = "/home/user/haven-vlm-engine-package/sample.mp4"  # Replace with a valid video path
    try:
        print("Processing video with multiplexer load balancing...")
        results = await engine.process_video(
            video_path,
            frame_interval=2.0  # Process every 2 seconds
        )
        print(f"Video Processing Results: result={results}")
    except Exception as e:
        logging.error(f"Error processing video: {e}", exc_info=True)

if __name__ == "__main__":
    # To run this example, you need to have multiple VLM servers running at the specified endpoints
    # and a video file at the specified path.
    # 
    # Example setup:
    # 1. Start LM Studio or similar on multiple machines/ports
    # 2. Enable OpenAI-compatible API on each
    # 3. Update the endpoint URLs in the multiplexer_endpoints configuration above
    # 4. Ensure models are loaded on each endpoint
    #
    # The multiplexer will automatically:
    # - Load balance requests across primary endpoints based on weights
    # - Use fallback endpoints if primary endpoints fail
    # - Provide high-performance async processing
    
    print("Starting VLM Engine with Multiplexer Support")
    print("=" * 50)
    print("This example demonstrates:")
    print("- Load balancing across multiple VLM endpoints")
    print("- Automatic failover to backup endpoints")
    print("- High-performance async video frame processing")
    print("- Seamless integration with existing pipeline architecture")
    print("=" * 50)
    
    asyncio.run(main())
