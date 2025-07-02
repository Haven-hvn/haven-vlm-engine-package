# VLM Engine

A high-performance Python package for Vision-Language Model (VLM) based content tagging and analysis. This package provides an advanced implementation for automatic content detection and tagging, delivering superior accuracy compared to traditional image classification methods.

## Features

- **Remote VLM Integration**: Connects to any OpenAI-compatible VLM endpoint (no local model loading required)
- **Context-Aware Detection**: Leverages Vision-Language Models' understanding of visual relationships for accurate content tagging
- **Flexible Architecture**: Modular pipeline system with configurable models and processing stages
- **Asynchronous Processing**: Built on asyncio for efficient video and image processing
- **Customizable Tag Sets**: Easy configuration of detection categories
- **Production Ready**: Includes retry logic, error handling, and comprehensive logging

## Installation

### From PyPI (when published)
```bash
pip install vlm-engine
```

### From Source
```bash
git clone https://github.com/Haven-hvn/haven-vlm-engine-package.git
cd vlm-engine-package
pip install -e .
```

### Requirements
- Python 3.8+
- **Sufficient RAM**: Video preprocessing loads entire videos into memory (not GPU memory)
- Compatible VLM server endpoint:
  - Remote OpenAI-compatible API (recommended)
  - Local server using [LM Studio](https://lmstudio.ai/)
  - Haven's custom VLM available at [https://havenmodels.orbiter.website/](https://havenmodels.orbiter.website/)

## Quick Start

```python
import asyncio
from vlm_engine import VLMEngine
from vlm_engine.config_models import EngineConfig, ModelConfig

# Configure the engine
config = EngineConfig(
    active_ai_models=["vlm_nsfw_model"],
    models={
        "vlm_nsfw_model": ModelConfig(
            type="vlm_model",
            model_id="HuggingFaceTB/SmolVLM-Instruct",
            api_base_url="http://localhost:7045",
            tag_list=["tag1", "tag2", "tag3"]  # Your custom tags
        )
    }
)

# Initialize and use
async def main():
    engine = VLMEngine(config)
    await engine.initialize()
    
    results = await engine.process_video(
        "path/to/video.mp4",
        frame_interval=2.0,
        threshold=0.5
    )
    print(f"Detected tags: {results}")

asyncio.run(main())
```

### Multiplexer Configuration (Load Balancing)

For high-performance deployments, you can configure multiple VLM endpoints with automatic load balancing:

```python
from vlm_engine.config_models import EngineConfig, ModelConfig

config = EngineConfig(
    active_ai_models=["vlm_multiplexer_model"],
    models={
        "vlm_multiplexer_model": ModelConfig(
            type="vlm_model",
            model_id="HuggingFaceTB/SmolVLM-Instruct",
            use_multiplexer=True,  # Enable multiplexer mode
            multiplexer_endpoints=[
                {
                    "base_url": "http://server1:7045/v1",
                    "api_key": "",
                    "name": "primary-server",
                    "weight": 5,  # Higher weight = more requests
                    "is_fallback": False
                },
                {
                    "base_url": "http://server2:7045/v1",
                    "api_key": "",
                    "name": "secondary-server",
                    "weight": 3,
                    "is_fallback": False
                },
                {
                    "base_url": "http://backup:7045/v1",
                    "api_key": "",
                    "name": "backup-server",
                    "weight": 1,
                    "is_fallback": True  # Used only when primaries fail
                }
            ],
            tag_list=["tag1", "tag2", "tag3"]
        )
    }
)
```

## Architecture

### Core Components

1. **VLMEngine**: Main entry point for the package
   - Manages model initialization and pipeline execution
   - Handles asynchronous processing of videos and images

2. **VLMClient**: OpenAI-compatible API client with multiplexer support
   - Supports any VLM with chat completions endpoint
   - Load balancing across multiple endpoints using multiplexer-llm
   - Automatic failover for high availability
   - Includes retry logic with exponential backoff and jitter
   - Handles image encoding and prompt formatting

3. **Pipeline System**: Flexible processing pipeline
   - Modular design allows custom processing stages
   - Built-in support for preprocessing, analysis, and postprocessing
   - Configurable through YAML or Python objects

4. **Model Management**: Dynamic model loading
   - Supports multiple model types (VLM, preprocessors, postprocessors)
   - Lazy loading for efficient resource usage
   - Thread-safe model access

## Configuration

### Basic Configuration

```python
from vlm_engine.config_models import EngineConfig, ModelConfig, PipelineConfig

config = EngineConfig(
    active_ai_models=["my_vlm_model"],
    models={
        "my_vlm_model": ModelConfig(
            type="vlm_model",
            model_id="model-name",
            api_base_url="http://localhost:8000",
            tag_list=["action1", "action2", "action3"],
            max_new_tokens=128,
            request_timeout=70,
            vlm_detected_tag_confidence=0.99
        )
    },
    pipelines={
        "video_pipeline": PipelineConfig(
            inputs=["video_path", "frame_interval"],
            output="results",
            models=[{"name": "my_vlm_model", "inputs": ["frame"], "outputs": "tags"}]
        )
    }
)
```

#### Multiplexer Benefits

- **Load Balancing**: Distribute requests across multiple VLM endpoints based on configurable weights
- **High Availability**: Automatic failover to backup endpoints when primary endpoints fail
- **Improved Performance**: Parallel processing across multiple servers for higher throughput
- **Seamless Integration**: Drop-in replacement for single endpoint configurations
- **Flexible Configuration**: Mix of primary and fallback endpoints with custom weights

### Advanced Configuration

The package supports complex configurations including:
- Multiple models in a pipeline
- Custom preprocessing and postprocessing stages
- Category-specific settings (thresholds, durations, etc.)
- Batch processing configurations

See the [examples](examples/) directory for detailed configuration examples.

For comprehensive multiplexer setup and configuration, see [MULTIPLEXER_INTEGRATION.md](MULTIPLEXER_INTEGRATION.md).

## API Reference

### VLMEngine

```python
class VLMEngine:
    def __init__(self, config: EngineConfig)
    async def initialize()
    async def process_video(video_path: str, **kwargs) -> Dict[str, Any]
```

### Processing Parameters

- `video_path`: Path to the video file
- `frame_interval`: Seconds between frame samples (default: 0.5)
- `threshold`: Confidence threshold for tag detection (default: 0.5)
- `return_timestamps`: Include timestamp information (default: True)
- `return_confidence`: Include confidence scores (default: True)

## Performance Optimization

### Memory Requirements
- **Important**: Video preprocessing loads the entire video into system RAM (not GPU memory)
- Ensure sufficient RAM for your video sizes (e.g., a 1GB video may require 4-8GB of available RAM)
- Consider processing videos in segments for very large files

### API Optimization
- Configure retry settings based on your VLM server's capacity
- Adjust `max_new_tokens` to balance speed vs accuracy
- Use appropriate `frame_interval` to reduce processing time and API calls

### Processing Speed
- Increase `frame_interval` to sample fewer frames (faster but less accurate)
- Use batch processing when your VLM endpoint supports it
- Consider running multiple VLM instances for parallel processing

## Extending the Package

### Custom Models

Create custom model classes by inheriting from the base Model class:

```python
from vlm_engine.models import Model

class CustomModel(Model):
    async def process(self, inputs):
        # Your custom processing logic
        return results
```

### Custom Pipelines

Define custom pipelines for specific use cases:

```python
custom_pipeline = PipelineConfig(
    inputs=["image_path"],
    output="analysis",
    models=[
        {"name": "preprocessor", "inputs": ["image_path"], "outputs": "processed_image"},
        {"name": "analyzer", "inputs": ["processed_image"], "outputs": "analysis"}
    ]
)
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Ensure your VLM server is running and accessible
   - Check the `api_base_url` configuration
   - Verify firewall settings

2. **GPU Memory Errors**
   - Reduce batch size or frame interval
   - Ensure proper CUDA installation
   - Check GPU memory availability

3. **Slow Processing**
   - Increase frame interval for faster processing
   - Use GPU acceleration if available
   - Optimize VLM server settings

### Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/yourusername/vlm-engine.git
cd vlm-engine
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- Built on top of modern Python async patterns
- Inspired by production ML serving architectures
- Haven's custom VLM models trained using [SmolVLM-Finetune](https://github.com/Haven-hvn/SmolVLM-Finetune) - Model Download found on [https://havenmodels.orbiter.website/](https://havenmodels.orbiter.website/)

- Designed for integration with OpenAI-compatible VLM endpoints

## Support

For issues and feature requests, please use the [GitHub issue tracker](https://github.com/Haven-hvn/haven-vlm-engine-package/issues).

For questions and discussions, join our community:
- Discord: [Link to Discord](https://discord.gg/57mPMDfQew)

---

**Note**: This package requires an OpenAI-compatible VLM endpoint. Options include:

### Remote Services
- Any OpenAI-compatible API endpoint 
- Akash deployment - https://github.com/Haven-hvn/haven-inference

### Local Setup
- [LM Studio](https://lmstudio.ai/) - Easy local VLM hosting with OpenAI-compatible API

The package **does not** load VLM models directly - it communicates with external VLM services via API.
