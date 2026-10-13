"""System One (Jev /v1/systemone) usage example for haven-vlm-engine-package.

Demonstrates asking atomic noul/choice/score questions through a dedicated
Jev endpoint pool, with the same weighted selection, capacity slots, and
automatic failover as chat completions. The Jev pool is kept separate from
the chat/VLM pool - each pool serves exactly one endpoint type.
"""

import asyncio
import logging
import os

from vlm_engine.multiplexer_vlm_client import MultiplexerVLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def systemone_example():
    """Ask atomic questions via the Jev /v1/systemone contract."""
    print("\n=== System One (Jev /v1/systemone) Example ===")

    config = {
        "model_id": "HuggingFaceTB/SmolVLM-Instruct",
        "tag_list": ["action", "scene", "object"],
        # Chat/VLM pool - unchanged generic VLM calls
        "multiplexer_endpoints": [
            {
                "base_url": os.getenv("VLM_ENDPOINT", "http://localhost:7045/v1"),
                "api_key": "",
                "name": "vlm-primary",
                "weight": 5,
            }
        ],
        # Dedicated Jev pool - only ever serves /v1/systemone
        "systemone_endpoints": [
            {
                "base_url": os.getenv("JEV_ENDPOINT", "http://localhost:8000/v1"),
                "api_key": os.getenv("JEV_API_KEY", ""),
                "name": "jev-primary",
                "weight": 5,
                "max_concurrent": 4,
            },
            {
                "base_url": os.getenv("JEV_BACKUP", "http://localhost:8001/v1"),
                "api_key": os.getenv("JEV_API_KEY", ""),
                "name": "jev-backup",
                "weight": 1,
                "is_fallback": True,  # Only used when the primary fails
            },
        ],
    }

    async with MultiplexerVLMClient(config) as client:
        result = await client.ask_systemone(
            state="refund please",
            questions={
                "greeting": {
                    "type": "noul",
                    "instructions": "Is this a greeting?",
                },
                "topic": {
                    "type": "choice",
                    "instructions": "Pick a category.",
                    "criteria": {"billing": "billing issue", "other": None},
                },
                "sentiment": {
                    "type": "score",
                    "instructions": "Rate the sentiment.",
                    "criteria": [
                        "very negative",
                        "negative",
                        "neutral",
                        "positive",
                        "very positive",
                    ],
                },
            },
        )

        print(f"model: {result['model']}")
        print(f"greeting probability: {result['answers']['greeting']['noul']}")
        print(f"chosen topic: {result['answers']['topic']['choice']}")
        print(f"sentiment score: {result['answers']['sentiment']['score']}")
        print(f"usage: {result['usage']}")

        # Per-endpoint routing stats for the System One pool
        print(f"systemone stats: {client.systemone_multiplexer.get_stats()}")


async def main():
    await systemone_example()


if __name__ == "__main__":
    asyncio.run(main())
