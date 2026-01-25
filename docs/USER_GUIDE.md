# VLM Engine Configuration User Guide

## Introduction

This guide provides a comprehensive overview of all configuration parameters for the Haven VLM Engine. It simplifies the detailed technical documentation into practical information that users need to know when configuring the system.

The VLM Engine uses four main configuration structures:
1. **EngineConfig** - Global engine settings and behavior (dictionary containing `pipelines`, `models`, `category_config`)
2. **PipelineConfig** - Defines processing workflows (dictionary with `inputs`, `output`, `version`, `models`)
3. **ModelConfig** - Configures individual AI models and processors (dictionary with model-specific settings)
4. **PipelineModelConfig** - Defines how models integrate into pipelines (dictionary with `name`, `inputs`, `outputs`)

**Important:** All configurations are defined as Python dictionaries. The system automatically converts these to Pydantic models internally, but you should define them as dictionaries in your configuration files.

Configuration can be provided either as dictionaries or as Pydantic model objects. This guide shows dictionary format, which is the most common approach for configuration files.

## EngineConfig Parameters

EngineConfig defines the overall engine configuration including available models, pipelines, and global settings.

### `pipelines` (Required)
Dictionary of pipeline configurations where keys are pipeline names and values are dictionaries containing pipeline settings. Each pipeline represents a complete processing workflow.

**Example:**
```python
pipelines={
    "video_pipeline_dynamic": {
        "inputs": ["video_path", "return_timestamps"],
        "output": "results",
        "short_name": "video",
        "version": 1.0,
        "models": [...]
    },
    "image_analyzer": {
        "inputs": ["image_path"],
        "output": "detections",
        "short_name": "image",
        "version": 1.0,
        "models": [...]
    }
}
```

### `models` (Required)
Dictionary of model configurations where keys are model names and values are dictionaries containing model settings. Models defined here can be referenced by pipelines.

**Example:**
```python
models={
    "vlm_multiplexer_model": {
        "type": "vlm_model",
        "model_id": "zai-org/glm-4.6v-flash",
        "tag_list": ["throw"]
    },
    "binary_search_processor_dynamic": {
        "type": "binary_search_processor",
        "model_file_name": "binary_search_processor_dynamic"
    }
}
```

### `category_config` (Required)
Defines tag filtering and transformation rules for video processing results. Controls how tags are filtered, mapped, normalized, and reported.

**Structure:**
```python
category_config={
    "category_name": {
        "tag_name": {
            "RenamedTag": "new_name",      # Custom name for output
            "MinMarkerDuration": "1s",      # Minimum occurrence duration
            "MaxGap": "30s",                # Max gap between occurrences
            "RequiredDuration": "10s",      # Total required duration
            "TagThreshold": 0.5             # Confidence threshold
        }
    }
}
```

### `active_ai_models` (Optional)
List of AI model names that should be active for dynamic pipeline processing. Defaults to `["vlm_multiplexer_model"]`.

**Example:**
```python
active_ai_models=["vlm_multiplexer_model", "secondary_model"]
```

## PipelineConfig Parameters

PipelineConfig defines individual processing pipelines that execute sequences of models.

### `inputs` (Required)
List of input parameter names required for pipeline execution. Specifies what data the pipeline expects and validates input completeness.

**Example:**
```python
inputs=["video_path", "return_timestamps", "threshold"]
```

### `output` (Required)
Key name where the final pipeline output will be stored. Users retrieve results using this key.

**Example:**
```python
output="detected_objects"
```

### `version` (Required)
Pipeline version number for tracking changes and backward compatibility.

**Example:**
```python
version=1.0
```

### `models` (Required)
List of dictionaries defining the sequence and configuration of models in the pipeline.

**Example:**
```python
models=[
    {
        "name": "dynamic_video_ai",
        "inputs": ["video_path", "return_timestamps", "time_interval", "threshold", "return_confidence", "vr_video", "existing_video_data", "skipped_categories"],
        "outputs": "results"
    },
    {
        "name": "binary_search_processor_dynamic",
        "inputs": ["video_path"],
        "outputs": "results"
    }
]
```

## ModelConfig Parameters

ModelConfig defines settings for individual AI models and processors. Parameters are grouped by functionality.

### Model Type & Identification

#### `type` (Required)
Determines which model class to instantiate for processing.

**Valid Values:**
- `"vlm_model"` - Vision Language Model for tag detection (uses HTTP APIs)
- `"video_preprocessor"` - Video frame extraction and preprocessing
- `"binary_search_processor"` - Optimized binary search processing for video
- `"python"` - Custom Python function-based processing

#### `model_id` (Required for VLM models)
Primary identifier for the model in API calls. For LLM-based models, this is the complete model identifier (e.g., `"zai-org/glm-4.6v-flash"`).

#### `model_identifier` (Optional)
Numeric identifier for the model (e.g., `93848`).

#### `model_version` (Optional)
Version string for the model (e.g., `"1.0"`).

#### `model_file_name` (Optional)
Filename for the model processor.

#### `api_base_url` (Required for single-endpoint VLM models)
Base URL for the VLM API endpoint. Forms the complete endpoint URL combined with `model_id`.

**Note:** In multiplexer mode, use `base_url` instead of `api_base_url` in endpoint configurations.

### Processing Configuration

#### `max_queue_size` (Optional)
Maximum size of the processing queue. Controls memory usage by limiting buffered items.
- `None` (default): Unbounded queue
- `N`: Queue limited to N items

#### `max_batch_size` (Optional, default: 5)
Maximum number of items to process in a single batch. Controls parallelism and throughput.
- Larger batches improve API efficiency
- Smaller batches reduce memory usage

#### `max_concurrent_requests` (Optional, default: 5)
Number of concurrent requests to process. Controls parallel processing capacity.

#### `fill_to_batch_size` (Optional, default: True)
Whether to wait for a full batch before processing.
- `True`: Wait for `max_batch_size` items (better throughput)
- `False`: Process items as they arrive (lower latency)

### Output Configuration

#### `tag_list` (Required for VLM models)
List of detectable tags that the model should recognize and classify.

**Example:**
```python
tag_list=["basketball", "soccer", "tennis", "swimming"]
```

#### `model_return_tags` (Optional, default: True)
Controls whether the model returns only tag names or tag names with confidence scores.
- `True` (default): Returns list of tag names `["tag1", "tag2"]`
- `False`: Returns list of tuples `[("tag1", 0.95), ("tag2", 0.87)]`

#### `model_return_confidence` (Optional, default: True)
Controls whether confidence scores are included in output.
- `True` (default): Include confidence scores
- `False`: Exclude confidence scores

#### `model_category` (Optional)
Category filter for detections. Can be a single category string or list of categories.

### Advanced Configuration

#### `model_image_size` (Optional)
Image size for processing. Can be a single integer (square) or tuple (width, height).
- Example: `1024` (1024x1024 pixels)
- Example: `(1920, 1080)` (HD resolution)

#### `normalization_config` (Optional, default: 1)
Configuration for normalizing and standardizing tag detection results. Can be an integer or dictionary with normalization rules.

#### `category_mappings` (Optional)
Maps category IDs to tag IDs for post-processing transformations.

### Multiplexer Configuration

#### `use_multiplexer` (Optional, default: False)
Enables multiplexer mode for load balancing across multiple API endpoints.
- `False`: Use single `api_base_url`
- `True`: Use `multiplexer_endpoints` for load balancing

#### `multiplexer_endpoints` (Required when `use_multiplexer=True`)
List of API endpoints for load balancing and high availability.

**Example:**
```python
multiplexer_endpoints=[
    {
        "base_url": "http://localhost:1234/v1",
        "api_key": "",
        "name": "lm-studio-primary",
        "weight": 9,
        "is_fallback": False,
        "max_concurrent": 10
    },
    {
        "base_url": "https://cloudagnostic.com:443/v1",
        "api_key": "",
        "name": "cloud-fallback",
        "weight": 1,
        "is_fallback": True,
        "max_concurrent": 2
    }
]
```

## PipelineModelConfig Parameters

PipelineModelConfig defines how models are integrated into pipelines, specifying input/output mappings.

### `name` (Required)
Unique identifier that references a ModelConfig defined in `EngineConfig.models`.

**Example:**
```python
name="dynamic_video_ai"  # Must match a key in EngineConfig.models dictionary
```

### `inputs` (Required)
List of input names that this model expects. Must match pipeline inputs or outputs from previous models.

**Example:**
```python
inputs=["video_frames", "category_config"]
```

### `outputs` (Required)
Output specification for this model. Can be a single string or list of strings.

**Example:**
```python
outputs="detected_tags"
# or
outputs=["detected_tags", "confidence_scores"]
```

## Complete Configuration Example

Here's a complete example showing all configuration structures working together:

```python
# Define model configurations
models = {
    "vlm_multiplexer_model": {
        "type": "vlm_model",
        "model_id": "zai-org/glm-4.6v-flash",
        "model_identifier": 93848,
        "model_version": "1.0",
        "model_file_name": "vlm_multiplexer_model",
        "model_category": "actiondetection",
        "tag_list": ["throw"],
        "use_multiplexer": True,
        "max_concurrent_requests": 13,
        "max_batch_size": 4,
        "multiplexer_endpoints": [
            {
                "base_url": "http://localhost:1234/v1",
                "api_key": "",
                "name": "lm-studio-primary",
                "weight": 9,
                "is_fallback": False,
                "max_concurrent": 10
            },
            {
                "base_url": "https://cloudagnostic.com:443/v1",
                "api_key": "",
                "name": "cloud-fallback",
                "weight": 1,
                "is_fallback": True,
                "max_concurrent": 2
            }
        ]
    },
    "binary_search_processor_dynamic": {
        "type": "binary_search_processor",
        "model_file_name": "binary_search_processor_dynamic"
    },
    "result_coalescer": {
        "type": "python",
        "model_file_name": "result_coalescer"
    },
    "result_finisher": {
        "type": "python",
        "model_file_name": "result_finisher"
    },
    "batch_awaiter": {
        "type": "python",
        "model_file_name": "batch_awaiter"
    },
    "video_result_postprocessor": {
        "type": "python",
        "model_file_name": "video_result_postprocessor"
    }
}

# Define pipeline configuration
pipelines = {
    "video_pipeline_dynamic": {
        "inputs": ["video_path", "return_timestamps", "time_interval", "threshold", "return_confidence", "vr_video", "existing_video_data", "skipped_categories"],
        "output": "results",
        "short_name": "dynamic_video",
        "version": 1.0,
        "models": [
            {
                "name": "dynamic_video_ai",
                "inputs": ["video_path", "return_timestamps", "time_interval", "threshold", "return_confidence", "vr_video", "existing_video_data", "skipped_categories"],
                "outputs": "results"
            },
            {
                "name": "binary_search_processor_dynamic",
                "inputs": ["video_path"],
                "outputs": "results"
            }
        ]
    }
}

# Define category configuration
category_config = {
    "actiondetection": {
        "throw": {
            "RenamedTag": "throw",
            "MinMarkerDuration": "1s",
            "MaxGap": "30s",
            "RequiredDuration": "1s",
            "TagThreshold": 0.5
        }
    }
}

# Create complete engine configuration
engine_config = {
    "pipelines": pipelines,
    "models": models,
    "category_config": category_config,
    "active_ai_models": ["vlm_multiplexer_model"]
}
```

## Quick Start Configuration

For a simple video processing setup:

```python
config = {
    "models": {
        "simple_vlm": {
            "type": "vlm_model",
            "model_id": "gpt-4-vision-preview",
            "tag_list": ["person", "car", "dog", "cat"],
            "api_base_url": "https://api.openai.com/v1",
            "max_batch_size": 5
        }
    },
    "pipelines": {
        "simple_processor": {
            "inputs": ["image_path"],
            "output": "detections",
            "version": 1.0,
            "models": [
                {
                    "name": "simple_vlm",
                    "inputs": ["image_path"],
                    "outputs": ["detections"]
                }
            ]
        }
    },
    "category_config": {
        "objects": {
            "person": {"TagThreshold": 0.5},
            "car": {"TagThreshold": 0.5},
            "dog": {"TagThreshold": 0.5},
            "cat": {"TagThreshold": 0.5}
        }
    }
}
```

## Best Practices

### Model Configuration
1. **Choose the right model type**: Use `vlm_model` for LLM-based detection, `video_preprocessor` for frame extraction, `python` for custom processing.
2. **Set appropriate batch sizes**: Balance memory usage vs throughput (5-10 for most use cases).
3. **Configure tag lists carefully**: Only include tags relevant to your use case for better accuracy.

### Pipeline Design
1. **Logical sequencing**: Preprocess → Analyze → Postprocess
2. **Clear input/output naming**: Use descriptive names that indicate data content
3. **Version your pipelines**: Track changes with version numbers

### Category Configuration
1. **Set realistic thresholds**: Start with 0.5-0.7 and adjust based on results
2. **Configure durations appropriately**: Match to your video content and use case
3. **Use RenamedTag for standardization**: Ensure consistent output formats

### Performance Tuning
1. **Batch processing**: Use `fill_to_batch_size=True` for better API efficiency
2. **Multiple instances**: Increase `max_concurrent_requests` for higher throughput
3. **Queue management**: Set `max_queue_size` to prevent memory issues

## Common Use Cases

### Video Analysis Pipeline
```python
# For analyzing sports videos
pipelines={
    "sports_analyzer": {
        "inputs": ["video_path", "sport_type"],
        "output": "analysis_results",
        "version": 1.0,
        "models": [
            {
                "name": "frame_extractor",
                "inputs": ["video_path"],
                "outputs": ["video_frames"]
            },
            {
                "name": "sports_detector",
                "inputs": ["video_frames", "sport_type"],
                "outputs": ["detected_actions"]
            },
            {
                "name": "summary_generator",
                "inputs": ["detected_actions"],
                "outputs": ["analysis_results"]
            }
        ]
    }
}
```

### Multi-Model Detection
```python
# Using multiplexer for high availability
models={
    "high_availability_vlm": {
        "type": "vlm_model",
        "model_id": "gpt-4-vision-preview",
        "use_multiplexer": True,
        "multiplexer_endpoints": [
            {
                "base_url": "https://primary-api.com/v1",
                "api_key": "",
                "name": "primary-endpoint",
                "weight": 9,
                "is_fallback": False,
                "max_concurrent": 10
            },
            {
                "base_url": "https://backup-api.com/v1",
                "api_key": "",
                "name": "backup-endpoint",
                "weight": 1,
                "is_fallback": True,
                "max_concurrent": 2
            }
        ],
        "tag_list": ["cat", "dog", "person"]
    }
}
```

## Troubleshooting

### Common Issues

1. **Model not found in pipeline**: Ensure `PipelineModelConfig.name` matches a key in `EngineConfig.models`
2. **Missing inputs**: Verify all pipeline inputs are provided when executing
3. **API connection errors**: Check `api_base_url` accessibility and `model_id` validity
4. **Memory issues**: Reduce `max_batch_size` or `max_queue_size`

### Validation Checks
- All pipeline models must reference valid ModelConfig names
- Input/output names must match across pipeline models
- Required parameters must be provided for each model type
- Category names in `category_config` must match model outputs

---

*Last Updated: January 2025*  
*For technical details, refer to the individual parameter documentation in the MODEL_CONFIG/, PIPELINE_CONFIG/, and ENGINE_CONFIG/ directories.*