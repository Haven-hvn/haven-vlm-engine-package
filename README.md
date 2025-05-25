# Haven VLM Engine Package

A reusable Python package for orchestrating Vision Language Model (VLM) data pipelines and complex AI processing tasks. This package is extracted from the original Haven VLM Engine server to provide its core functionalities in a modular and importable format.

## Features

- Dynamic pipeline configuration and execution.
- Management of local AI models and clients for external VLM services.
- Queue-based asynchronous processing with batching.
- Image and video preprocessing.
- Result aggregation and post-processing.

## Installation

\`\`\`bash
pip install haven-vlm-engine
\`\`\`
(Once published to PyPI)

Alternatively, install directly from source:
\`\`\`bash
git clone https://github.com/example/haven-vlm-engine-package.git # Replace with actual URL
cd haven-vlm-engine-package
pip install .
\`\`\`

## Basic Usage

\`\`\`python
from haven_vlm_engine import VLMEngine

# Define your pipeline and model configurations
pipeline_config = {
    "my_video_pipeline": {
        "inputs": ["video_path", "threshold"],
        "output": "final_results",
        "models": [
            {
                "name": "video_preprocessor_1",
                "inputs": ["video_path"],
                "outputs": ["frames_data"]
            },
            {
                "name": "my_vlm_model_1",
                "inputs": ["frames_data", "threshold"],
                "outputs": ["vlm_analysis"]
            },
            {
                "name": "result_postprocessor_1",
                "inputs": ["vlm_analysis"],
                "outputs": ["final_results"]
            }
        ]
    }
}

model_config = {
    "video_preprocessor_1": {
        "type": "video_preprocessor", # Special type
        "image_size": 224,
        "frame_interval": 1.0,
        # ... other preprocessor params
    },
    "my_vlm_model_1": {
        "type": "vlm_model",
        "api_base_url": "http://localhost:1234/v1",
        "model_id": "your-vlm-model-id",
        "tag_list": ["cat", "dog", "person"],
        "model_category": "object_detection",
        # ... other VLM client params
    },
    "result_postprocessor_1": {
        "type": "python_function", # Example for a custom Python function model
        "function_name": "my_custom_postprocessor_function", 
        # ...
    }
}

# Initialize the engine
engine = VLMEngine(pipelines=pipeline_config, models=model_config)

async def main():
    # Process a video
    video_results = await engine.process_video(
        video_path="path/to/your/video.mp4",
        pipeline_name="my_video_pipeline",
        threshold=0.7
    )
    print(video_results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
\`\`\`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
