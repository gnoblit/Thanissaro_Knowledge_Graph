# tests/test_llm_helpers.py (Corrected Version)

import pytest
import os
from unittest.mock import patch, MagicMock  # <-- CORRECTED LINE

from src.utils.llm_helpers import get_llm_client, GeminiClient, OpenAIClient

def test_get_llm_client_returns_gemini():
    """Test that the factory returns a GeminiClient for a gemini model_id."""
    config = {'model_id': 'gemini-1.5-flash'}
    # Patch os.environ to simulate the presence of the API key
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
        client = get_llm_client(config, "sys_prompt", MagicMock())
    assert isinstance(client, GeminiClient)

def test_get_llm_client_returns_openai():
    """Test that the factory returns an OpenAIClient for a deepseek model_id."""
    config = {'model_id': 'deepseek-chat'}
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"}):
        client = get_llm_client(config, "sys_prompt", MagicMock())
    assert isinstance(client, OpenAIClient)

def test_get_llm_client_raises_for_missing_gemini_key(monkeypatch):
    """Test ValueError is raised if GEMINI_API_KEY is missing."""
    # monkeypatch is a pytest fixture that safely modifies/deletes environment variables
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = {'model_id': 'gemini-pro'}
    with pytest.raises(ValueError, match="GEMINI_API_KEY not found"):
        get_llm_client(config, "sys_prompt", MagicMock())

def test_get_llm_client_raises_for_unsupported_model():
    """Test ValueError for an unsupported model provider."""
    config = {'model_id': 'unsupported-model-v1'}
    with pytest.raises(ValueError, match="Unsupported model provider"):
        get_llm_client(config, "sys_prompt", MagicMock())