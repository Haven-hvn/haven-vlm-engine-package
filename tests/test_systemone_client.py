#!/usr/bin/env python3
"""
Test script for the System One (Jev /v1/systemone) integration with haven-vlm-engine-package.

This script validates:
1. MultiplexerVLMClient accepts systemone_endpoints configuration
2. Pool separation: chat endpoints and Jev endpoints never share a pool
3. ask_systemone posts the canonical Jev envelope through the multiplexer
4. Automatic failover from a rate-limited Jev endpoint to its fallback
5. Terminal (non-retryable) errors are raised, not silently folded to zeros
6. Lazy initialization: chat-only configurations never build the System One pool

Note: This test script doesn't require actual endpoints to be running -
AsyncOpenAI is replaced with a recording fake, so no network is used.
"""

import asyncio
import sys
from typing import Any, Dict, List, Optional

import vlm_engine.multiplexer_vlm_client as mvlm_module
from vlm_engine.multiplexer_vlm_client import MultiplexerVLMClient

QUESTIONS: Dict[str, Any] = {
    "greeting": {"type": "noul", "instructions": "Is this a greeting?"},
    "topic": {
        "type": "choice",
        "instructions": "Pick a category.",
        "criteria": {"billing": "billing issue", "other": None},
    },
}


class FakeRateLimitError(Exception):
    """Mock rate limit error carrying a 429 status (multiplexer classification input)."""

    def __init__(self, message: str = "Rate limit exceeded", status_code: int = 429):
        super().__init__(message)
        self.status_code = status_code


class FakePost:
    """Records generic post() calls and replays a queued response or error."""

    def __init__(self, owner: "FakeAsyncOpenAI") -> None:
        self._owner = owner
        self.calls: List[Dict[str, Any]] = []

    async def __call__(
        self, path: str, *, body: Any = None, cast_to: Any = dict, **kwargs: Any
    ) -> Any:
        self.calls.append({"path": path, "body": body, "cast_to": cast_to})
        if self._owner.error is not None:
            raise self._owner.error
        return dict(self._owner.response)


class FakeAsyncOpenAI:
    """Stand-in for AsyncOpenAI exposing only the generic post() verb."""

    instances: List["FakeAsyncOpenAI"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.base_url = str(kwargs.get("base_url"))
        self.error: Optional[Exception] = None
        self.response: Dict[str, Any] = {
            "model": "jev-echo",
            "answers": {
                "greeting": {"type": "noul", "noul": 0.97},
                "topic": {"type": "choice", "choice": "billing"},
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }
        self.post = FakePost(self)
        FakeAsyncOpenAI.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    @classmethod
    def by_url(cls, base_url: str) -> "FakeAsyncOpenAI":
        for instance in cls.instances:
            if instance.base_url == base_url:
                return instance
        raise AssertionError(f"No fake client created for {base_url}")


def make_config(systemone_endpoints: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build a client config with both a chat pool and an optional System One pool."""
    config = {
        "model_id": "test-model",
        "tag_list": ["action", "scene", "object"],
        "multiplexer_endpoints": [
            {
                "base_url": "http://vlm-primary:8000/v1",
                "api_key": "",
                "name": "vlm-primary",
                "weight": 5,
            }
        ],
    }
    if systemone_endpoints is not None:
        config["systemone_endpoints"] = systemone_endpoints
    return config


def systemone_pool() -> List[Dict[str, Any]]:
    """A healthy Jev primary plus a fallback endpoint."""
    return [
        {
            "base_url": "http://jev-primary:9000/v1",
            "api_key": "",
            "name": "jev-primary",
            "weight": 5,
        },
        {
            "base_url": "http://jev-backup:9001/v1",
            "api_key": "",
            "name": "jev-backup",
            "weight": 1,
            "is_fallback": True,
        },
    ]


async def run_initialization():
    """Test MultiplexerVLMClient accepts and stores System One configuration."""
    print("🧪 Testing System One configuration handling...")

    original = mvlm_module.AsyncOpenAI
    mvlm_module.AsyncOpenAI = FakeAsyncOpenAI
    try:
        # Test 1: Chat pool + System One pool both accepted
        FakeAsyncOpenAI.reset()
        client = MultiplexerVLMClient(make_config(systemone_pool()))
        assert len(client.systemone_endpoints) == 2, "System One endpoints not stored"
        print("✅ Configuration with systemone_endpoints accepted")

        # Test 2: Chat-only configuration unchanged (System One optional)
        FakeAsyncOpenAI.reset()
        client = MultiplexerVLMClient(make_config())
        assert client.systemone_endpoints == [], "System One endpoints should default to empty"
        print("✅ Chat-only configuration still accepted (System One optional)")

        # Test 3: ask_systemone without a System One pool raises a clear error
        try:
            await client.ask_systemone(state="hello", questions=QUESTIONS)
            print("❌ ask_systemone should require systemone_endpoints")
            return False
        except ValueError as e:
            assert "systemone_endpoints" in str(e)
            print("✅ ask_systemone without systemone_endpoints correctly rejected")

        return True
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        mvlm_module.AsyncOpenAI = original


async def run_pool_separation():
    """Test that chat endpoints and Jev endpoints are routed through separate pools."""
    print("🧪 Testing pool separation...")

    original = mvlm_module.AsyncOpenAI
    mvlm_module.AsyncOpenAI = FakeAsyncOpenAI
    try:
        FakeAsyncOpenAI.reset()
        client = MultiplexerVLMClient(make_config(systemone_pool()))

        async with client:
            # Both pools initialized: one multiplexer each, distinct instances
            assert client.multiplexer is not client.systemone_multiplexer, (
                "Chat and System One pools must not share a multiplexer"
            )
            assert len(client.multiplexer._weighted_models) == 1
            assert len(client.systemone_multiplexer._weighted_models) == 1
            assert len(client.systemone_multiplexer._fallback_models) == 1
            print("✅ Distinct multiplexers per pool (no endpoint type mixing)")

            # A System One query only touches Jev endpoints
            result = await client.ask_systemone(
                state="refund please", questions=QUESTIONS
            )
            assert "answers" in result, "System One query did not return answers"

            vlm_fake = FakeAsyncOpenAI.by_url("http://vlm-primary:8000/v1")
            jev_fake = FakeAsyncOpenAI.by_url("http://jev-primary:9000/v1")
            assert jev_fake.post.calls, "Jev endpoint never received the query"
            assert vlm_fake.post.calls == [], "Chat endpoint received a System One query"
            print("✅ System One queries only reach Jev endpoints")

        return True
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        mvlm_module.AsyncOpenAI = original


async def run_canonical_payload():
    """Test that ask_systemone posts the canonical Jev envelope."""
    print("🧪 Testing canonical Jev payload...")

    original = mvlm_module.AsyncOpenAI
    mvlm_module.AsyncOpenAI = FakeAsyncOpenAI
    try:
        FakeAsyncOpenAI.reset()
        client = MultiplexerVLMClient(make_config(systemone_pool()))

        async with client:
            result = await client.ask_systemone(
                state="refund please",
                questions=QUESTIONS,
                choice_head="support",
            )

        call = FakeAsyncOpenAI.by_url("http://jev-primary:9000/v1").post.calls[0]
        assert call["path"] == "/systemone", f"Wrong path: {call['path']}"
        assert call["cast_to"] is dict, "Response should be parsed as a plain dict"
        assert call["body"]["state"] == "refund please"
        assert call["body"]["questions"] == QUESTIONS
        # model is overridden by the selected endpoint's registered name
        assert call["body"]["model"] == "jev-primary", f"Wrong model: {call['body']['model']}"
        # extra body fields ride along
        assert call["body"]["choice_head"] == "support"
        # response is stamped with the selected endpoint's name and keeps answers/usage
        assert result["model"] == "jev-primary", f"Wrong model stamp: {result['model']}"
        assert result["answers"]["greeting"]["noul"] == 0.97
        assert result["usage"]["input_tokens"] == 12
        print("✅ Canonical {state, model, questions} envelope posted and parsed")
        return True
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        mvlm_module.AsyncOpenAI = original


async def run_failover():
    """Test failover from a rate-limited Jev primary to the fallback endpoint."""
    print("🧪 Testing System One failover...")

    original = mvlm_module.AsyncOpenAI
    mvlm_module.AsyncOpenAI = FakeAsyncOpenAI
    try:
        FakeAsyncOpenAI.reset()
        client = MultiplexerVLMClient(make_config(systemone_pool()))

        async with client:
            primary_fake = FakeAsyncOpenAI.by_url("http://jev-primary:9000/v1")
            fallback_fake = FakeAsyncOpenAI.by_url("http://jev-backup:9001/v1")
            primary_fake.error = FakeRateLimitError()

            result = await client.ask_systemone(
                state="refund please", questions=QUESTIONS
            )

            assert fallback_fake.post.calls, "Fallback endpoint never received the query"
            assert result["model"] == "jev-backup", f"Wrong model stamp: {result['model']}"

            stats = client.systemone_multiplexer.get_stats()
            assert stats["jev-primary"]["rateLimited"] == 1, f"stats: {stats}"
            assert stats["jev-backup"]["success"] == 1, f"stats: {stats}"
            print("✅ 429 on primary fails over to fallback with stats recorded")

        # All endpoints failing surfaces the multiplexer's RateLimitError.
        # Both endpoints are disabled after their first 429, so the multiplexer
        # raises on the next selection pass without any backoff sleep.
        FakeAsyncOpenAI.reset()
        from multiplexer_llm import RateLimitError as MuxRateLimitError

        async with MultiplexerVLMClient(make_config(systemone_pool())) as client2:
            await client2._ensure_systemone_initialized()
            FakeAsyncOpenAI.by_url("http://jev-primary:9000/v1").error = FakeRateLimitError()
            FakeAsyncOpenAI.by_url("http://jev-backup:9001/v1").error = FakeRateLimitError()

            try:
                await client2.ask_systemone(state="hello", questions=QUESTIONS)
                print("❌ All endpoints rate limited should raise")
                return False
            except MuxRateLimitError as e:
                assert "Rate limit exceeded" in str(e)
                print("✅ All endpoints rate limited raises RateLimitError")

        return True
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        mvlm_module.AsyncOpenAI = original


async def run_terminal_errors():
    """Test that non-retryable errors propagate instead of returning zeros."""
    print("🧪 Testing terminal error propagation...")

    original = mvlm_module.AsyncOpenAI
    mvlm_module.AsyncOpenAI = FakeAsyncOpenAI
    try:
        FakeAsyncOpenAI.reset()
        client = MultiplexerVLMClient(make_config(systemone_pool()))

        async with client:
            primary_fake = FakeAsyncOpenAI.by_url("http://jev-primary:9000/v1")
            fallback_fake = FakeAsyncOpenAI.by_url("http://jev-backup:9001/v1")
            # 422 is neither rate limit nor 5xx: the multiplexer propagates it
            # immediately instead of failing over
            primary_fake.error = FakeRateLimitError(
                message="Unprocessable request", status_code=422
            )

            from multiplexer_llm import MultiplexerError as MuxError

            try:
                await client.ask_systemone(state="hello", questions=QUESTIONS)
                print("❌ Invalid request should raise, not return")
                return False
            except MuxError as e:
                assert "jev-primary" in str(e)
                print("✅ 422 error raises MultiplexerError naming the endpoint")

            # The fallback must NOT have been tried for a non-retryable error
            assert fallback_fake.post.calls == [], "Fallback tried a non-retryable error"
            print("✅ Non-retryable errors do not fail over")

        return True
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        mvlm_module.AsyncOpenAI = original


async def run_lazy_initialization():
    """Test that chat-only configurations never build the System One pool."""
    print("🧪 Testing lazy System One initialization...")

    original = mvlm_module.AsyncOpenAI
    mvlm_module.AsyncOpenAI = FakeAsyncOpenAI
    try:
        FakeAsyncOpenAI.reset()
        client = MultiplexerVLMClient(make_config())

        async with client:
            assert client.multiplexer is not None, "Chat pool should be initialized"
            assert client.systemone_multiplexer is None, (
                "System One pool should stay uninitialized when not configured"
            )
            print("✅ Chat-only client never builds the System One pool")

        return True
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        mvlm_module.AsyncOpenAI = original


async def main():
    """Run all tests."""
    print("🚀 Haven VLM Engine - System One (Jev /v1/systemone) Integration Tests")
    print("=" * 60)

    tests = [
        ("Configuration Handling", run_initialization),
        ("Pool Separation", run_pool_separation),
        ("Canonical Jev Payload", run_canonical_payload),
        ("Failover", run_failover),
        ("Terminal Error Propagation", run_terminal_errors),
        ("Lazy Initialization", run_lazy_initialization),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System One integration is working correctly.")
        print("\n🔧 Integration Summary:")
        print("- ✅ systemone_endpoints configuration accepted")
        print("- ✅ Dedicated pool per endpoint type (no mixing)")
        print("- ✅ Canonical {state, model, questions} envelope via multiplexer")
        print("- ✅ Automatic failover on 429/529 with stats")
        print("- ✅ Terminal errors raised, non-retryable errors do not fail over")
        print("- ✅ Chat-only configurations unaffected")

        print("\n🚀 Ready for production use!")
        return True
    else:
        print("❌ Some tests failed. Please review the integration.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
