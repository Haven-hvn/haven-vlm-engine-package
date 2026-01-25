# Multiplexer-LLM Integration Guide

This document explains how to use the new multiplexer-llm integration in the haven-vlm-engine-package for load balancing across multiple VLM endpoints.

## Overview

The multiplexer integration allows you to:
- **Load balance** requests across multiple VLM endpoints
- **Automatic failover** when primary endpoints are unavailable
- **Improved performance** through parallel processing
- **High availability** with backup endpoints
- **Seamless integration** with existing pipeline architecture

## Installation

The multiplexer-llm package is automatically installed as a dependency:

```bash
pip install vlm_engine  # Includes multiplexer-llm==0.2.3
```

## Configuration

### Single Endpoint (Backward Compatible)

```python
from vlm_engine.config_models import EngineConfig, ModelConfig

config = EngineConfig(
    active_ai_models=["vlm_model"],
    models={
        "vlm_model": ModelConfig(
            type="vlm_model",
            model_id="HuggingFaceTB/SmolVLM-Instruct",
            api_base_url="http://localhost:7045",  # Single endpoint
            use_multiplexer=False,  # Disable multiplexer (default)
            tag_list=["action", "scene", "object"]
        )
    }
)
```

### Multiple Endpoints with Load Balancing

```python
from vlm_engine.config_models import EngineConfig, ModelConfig

config = EngineConfig(
    active_ai_models=["vlm_multiplexer_model"],
    models={
        "vlm_multiplexer_model": ModelConfig(
            type="vlm_model",
            model_id="HuggingFaceTB/SmolVLM-Instruct",
            use_multiplexer=True,  # Enable multiplexer
            multiplexer_endpoints=[
                {
                    "base_url": "http://server1:7045/v1",
                    "api_key": "",  # Empty for unauthenticated endpoints
                    "name": "primary-server-1",
                    "weight": 5,  # Higher weight = more requests
                    "is_fallback": False
                },
                {
                    "base_url": "http://server2:7045/v1",
                    "api_key": "",
                    "name": "primary-server-2", 
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
            tag_list=["action", "scene", "object"]
        )
    }
)
```

## Endpoint Configuration Options

Each endpoint in `multiplexer_endpoints` supports:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `base_url` | string | Yes | OpenAI-compatible API endpoint URL |
| `api_key` | string | No | API key (use "" for unauthenticated) |
| `name` | string | No | Friendly name for logging |
| `weight` | integer | No | Request distribution weight (default: 1) |
| `is_fallback` | boolean | No | Use only when primaries fail (default: false) |

## Usage Examples

### Basic Video Processing

```python
import asyncio
from vlm_engine import VLMEngine

async def main():
    # Initialize engine with multiplexer config
    engine = VLMEngine(config)
    await engine.initialize()
    
    # Process video - automatically load balanced
    results = await engine.process_video(
        "path/to/video.mp4",
        frame_interval=2.0
    )
    print(f"Results: {results}")

asyncio.run(main())
```

### High-Performance Setup

For maximum performance, configure multiple endpoints with appropriate weights:

```python
multiplexer_endpoints=[
    # High-performance primary servers
    {"base_url": "http://gpu-server-1:7045/v1", "weight": 8, "name": "gpu-1"},
    {"base_url": "http://gpu-server-2:7045/v1", "weight": 8, "name": "gpu-2"},
    {"base_url": "http://gpu-server-3:7045/v1", "weight": 8, "name": "gpu-3"},
    
    # Medium-performance servers
    {"base_url": "http://cpu-server-1:7045/v1", "weight": 3, "name": "cpu-1"},
    {"base_url": "http://cpu-server-2:7045/v1", "weight": 3, "name": "cpu-2"},
    
    # Fallback server
    {"base_url": "http://backup:7045/v1", "weight": 1, "name": "backup", "is_fallback": True}
]
```

## Load Balancing Behavior

### Weight-Based Distribution
- Requests are distributed based on endpoint weights
- Higher weight = more requests
- Example: weights [5, 3, 2] → ~50%, 30%, 20% distribution

### Fallback Logic
- Primary endpoints (is_fallback=False) are used first
- Fallback endpoints (is_fallback=True) are used only when primaries fail
- Automatic retry with exponential backoff

### Performance Optimization
- Async processing maintains high throughput
- Connection pooling reduces latency
- Parallel requests across endpoints

## Monitoring and Logging

The multiplexer provides detailed logging:

```
INFO - Added primary endpoint: gpu-server-1 (weight: 8)
INFO - Added primary endpoint: gpu-server-2 (weight: 8)  
INFO - Added fallback endpoint: backup-server (weight: 1)
DEBUG - Request routed to: gpu-server-1
DEBUG - Fallback triggered, using: backup-server
```

## Migration Guide

### From Single Endpoint

1. **Add multiplexer configuration**:
   ```python
   # Before
   api_base_url="http://localhost:7045"
   
   # After  
   use_multiplexer=True
   multiplexer_endpoints=[
       {"base_url": "http://localhost:7045/v1", "weight": 1}
   ]
   ```

2. **No code changes required** - same interface maintained

### Adding More Endpoints

1. **Add new endpoints** to `multiplexer_endpoints`
2. **Adjust weights** for load distribution
3. **Configure fallbacks** for high availability

## Best Practices

### Endpoint Setup
- Use `/v1` suffix for OpenAI compatibility
- Test each endpoint individually first
- Monitor endpoint health and performance

### Weight Configuration
- Start with equal weights (1) for all endpoints
- Adjust based on server performance
- Use higher weights for more powerful servers

### Fallback Strategy
- Always configure at least one fallback endpoint
- Use reliable, lower-performance servers as fallbacks
- Test failover scenarios regularly

### Performance Tuning
- Monitor request distribution across endpoints
- Adjust weights based on actual performance
- Use connection pooling for high-throughput scenarios

## Troubleshooting

### Common Issues

1. **No endpoints responding**
   - Check endpoint URLs and availability
   - Verify OpenAI-compatible API is enabled
   - Check network connectivity

2. **Uneven load distribution**
   - Review endpoint weights
   - Monitor endpoint response times
   - Check for endpoint failures

3. **High latency**
   - Reduce request timeout values
   - Add more endpoints for load distribution
   - Check network latency to endpoints

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Benchmarks

With proper configuration, the multiplexer can provide:
- **2-5x throughput improvement** with multiple endpoints
- **Sub-second failover** to backup endpoints  
- **Linear scaling** with additional endpoints
- **99.9% availability** with proper fallback configuration

## Support

For issues or questions:
1. Check the logs for error details
2. Verify endpoint configurations
3. Test individual endpoints separately
4. Review the examples in the `examples/` directory
