"""
Pytest tests for NVIDIA LLM model access.
Verifies that NVIDIA API models are accessible.

Run with: pytest test_nvidia_model.py -v
"""

import pytest
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from config import NVIDIA_API_KEY


class TestNVIDIAModelAccess:
    """Test suite for NVIDIA model access verification."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """Verify API key is configured before running tests."""
        if not NVIDIA_API_KEY:
            pytest.skip("NVIDIA_API_KEY not configured")

    def test_llama_3_3_70b_instruct(self):
        """Test access to meta/llama-3.3-70b-instruct model."""
        model_name = "meta/llama-3.3-70b-instruct"
        try:
            llm = ChatNVIDIA(model=model_name, temperature=0.0)
            response = llm.invoke("Say 'hello' and nothing else.")
            assert response is not None
            assert hasattr(response, "content")
            assert len(response.content) > 0
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "Authorization failed" in error_str:
                pytest.skip(f"API key does not have access to {model_name}: {error_str[:100]}")
            else:
                pytest.fail(f"Failed to access {model_name}: {error_str[:100]}")

    def test_llama_4_maverick_17b_instruct(self):
        """Test access to meta/llama-4-maverick-17b-128e-instruct model."""
        model_name = "meta/llama-4-maverick-17b-128e-instruct"
        try:
            llm = ChatNVIDIA(model=model_name, temperature=0.0)
            response = llm.invoke("Say 'hello' and nothing else.")
            assert response is not None
            assert hasattr(response, "content")
            assert len(response.content) > 0
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "Authorization failed" in error_str:
                pytest.skip(f"API key does not have access to {model_name}: {error_str[:100]}")
            else:
                pytest.fail(f"Failed to access {model_name}: {error_str[:100]}")

    @pytest.mark.parametrize("model_name", [
        "meta/llama-3.3-70b-instruct",
        "meta/llama-4-maverick-17b-128e-instruct",
    ])
    def test_model_initialization(self, model_name):
        """Test that models can be initialized without errors."""
        llm = ChatNVIDIA(model=model_name, temperature=0.0)
        assert llm is not None
        assert llm.model == model_name


@pytest.mark.integration
class TestNVIDIAModelResponses:
    """Integration tests for NVIDIA model response quality."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """Verify API key is configured before running tests."""
        if not NVIDIA_API_KEY:
            pytest.skip("NVIDIA_API_KEY not configured")

    def test_simple_response(self):
        """Test that model returns expected response."""
        try:
            llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.0)
            response = llm.invoke("What is 2+2? Answer with just the number.")
            assert response is not None
            assert "4" in response.content
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "Authorization failed" in error_str:
                pytest.skip(f"API key does not have model access: {error_str[:100]}")
            else:
                raise

    def test_model_configuration(self):
        """Test model initialization with different configurations."""
        configs = [
            {"temperature": 0.0},
            {"temperature": 0.5},
            {"temperature": 1.0},
        ]
        for config in configs:
            llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", **config)
            assert llm is not None
