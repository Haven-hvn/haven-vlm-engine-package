import asyncio
import logging
from vlm_engine import VLMEngine
from vlm_engine.config_models import EngineConfig, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

async def single_endpoint_example():
    """Example using a single VLM endpoint (backward compatible)."""
    print("\n=== Single Endpoint Example ===")
    
    config = EngineConfig(
        active_ai_models=["vlm_single_model"],
        models={
            "vlm_single_model": ModelConfig(
                type="vlm_model",
                model_id="HuggingFaceTB/SmolVLM-Instruct",
                api_base_url="http://localhost:7045",
                use_multiplexer=False,  # Use single endpoint
                tag_list=["action", "scene", "object"]
            )
        }
    )
    
    engine = VLMEngine(config)
    await engine.initialize()
    print("✓ Single endpoint VLM engine initialized")

async def multiplexer_example():
    """Example using multiple VLM endpoints with load balancing."""
    print("\n=== Multiplexer Example ===")
    
    config = EngineConfig(
        active_ai_models=["vlm_multiplexer_model"],
        models={
            "vlm_multiplexer_model": ModelConfig(
                type="vlm_model",
                model_id="HuggingFaceTB/SmolVLM-Instruct",
                use_multiplexer=True,  # Enable multiplexer
                multiplexer_endpoints=[
                    {
                        "base_url": "http://192.168.68.70:1234/v1",
                        "api_key": "",
                        "name": "primary-endpoint",
                        "weight": 5,
                        "is_fallback": False
                    },
                    {
                        "base_url": "http://192.168.68.67:7045/v1",
                        "api_key": "",
                        "name": "secondary-endpoint", 
                        "weight": 3,
                        "is_fallback": False
                    },
                    {
                        "base_url": "http://localhost:7045/v1",
                        "api_key": "",
                        "name": "fallback-endpoint",
                        "weight": 1,
                        "is_fallback": True
                    }
                ],
                tag_list=["action", "scene", "object"]
            )
        }
    )
    
    engine = VLMEngine(config)
    await engine.initialize()
    print("✓ Multiplexer VLM engine initialized with load balancing")

async def main():
    """Main function demonstrating both configurations."""
    print("VLM Engine Multiplexer Integration Examples")
    print("=" * 50)
    
    try:
        # Example 1: Single endpoint (backward compatible)
        await single_endpoint_example()
        
        # Example 2: Multiple endpoints with load balancing
        await multiplexer_example()
        
        print("\n✅ All examples completed successfully!")
        print("\nKey Benefits of Multiplexer Integration:")
        print("- Load balancing across multiple VLM endpoints")
        print("- Automatic failover for high availability")
        print("- Improved throughput and reduced latency")
        print("- Seamless integration with existing pipeline architecture")
        print("- Backward compatibility with single endpoint configurations")
        
    except Exception as e:
        logging.error(f"Error in examples: {e}", exc_info=True)

if __name__ == "__main__":
    print("Starting VLM Engine Multiplexer Examples...")
    asyncio.run(main())
