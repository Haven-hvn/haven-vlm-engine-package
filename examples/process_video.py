import asyncio
import logging
from vlm_engine import VLMEngine
from vlm_engine.config_models import EngineConfig, PipelineConfig, ModelConfig, PipelineModelConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)

async def main():
    # Define the configuration for the engine
    engine_config = EngineConfig(
        active_ai_models=["vlm_nsfw_model"],
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
                short_name="dynamic_video",
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
            "video_preprocessor_dynamic": ModelConfig(type="video_preprocessor", model_file_name="video_preprocessor_dynamic"),
            "vlm_nsfw_model": ModelConfig(
                type="vlm_model",
                model_file_name="vlm_nsfw_model",
                model_category="actiondetection",
                model_id="HuggingFaceTB/SmolVLM-Instruct",
                model_identifier=93848,
                model_version="1.0",
                api_base_url="http://localhost:7045",
                tag_list = [
    "Anal Fucking", "Ass Licking", "Ass Penetration", "Ball Licking/Sucking", "Blowjob", "Cum on Person",
    "Cum Swapping", "Cumshot", "Deepthroat", "Double Penetration", "Fingering", "Fisting", "Footjob",
    "Gangbang", "Gloryhole", "Grabbing Ass", "Grabbing Boobs", "Grabbing Hair/Head", "Handjob", "Kissing",
    "Licking Penis", "Masturbation", "Pissing", "Pussy Licking (Clearly Visible)", "Pussy Licking",
    "Pussy Rubbing", "Sucking Fingers", "Sucking Toy/Dildo", "Wet (Genitals)", "Titjob", "Tribbing/Scissoring",
    "Undressing", "Vaginal Penetration", "Vaginal Fucking", "Vibrating"
]
            ),
            "result_coalescer": ModelConfig(type="python", model_file_name="result_coalescer"),
            "result_finisher": ModelConfig(type="python", model_file_name="result_finisher"),
            "batch_awaiter": ModelConfig(type="python", model_file_name="batch_awaiter"),
            "video_result_postprocessor": ModelConfig(type="python", model_file_name="video_result_postprocessor"),
        },
        category_config={
    "actiondetection": {
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

    # Process a video
    video_path = "K:\\sample.mp4"  # Replace with a valid video path
    try:
        results = await engine.process_video(
            video_path,
            frame_interval=2.0  # Match the expected output
        )
        print(f"Video Processing Results: result={results}")
    except Exception as e:
        logging.error(f"Error processing video: {e}", exc_info=True)

if __name__ == "__main__":
    # To run this example, you need to have a VLM server running at the specified api_base_url
    # and a video file at the specified path.
    asyncio.run(main())
