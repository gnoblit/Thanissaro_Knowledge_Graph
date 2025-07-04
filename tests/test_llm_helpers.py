# tests/test_llm_helpers.py

import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

# Import the actual classes and functions to be tested
from utils.llm_helpers import get_llm_client, GeminiClient, OpenAIClient

# --- Fixtures for Configuration ---
@pytest.fixture
def mock_gemini_config():
    return {
        'model_id': 'gemini-1.5-flash',
        'temperature': 0.7
    }

@pytest.fixture
def mock_deepseek_config():
    return {
        'model_id': 'deepseek-coder',
        'temperature': 0.7 # Note: this is ignored by OpenAIClient, but good to have
    }

# A dummy Pydantic model for testing purposes
class DummySchema(BaseModel):
    pass

# --- Tests for the Factory Function: get_llm_client ---

@patch('utils.llm_helpers.os.getenv', return_value='fake-gemini-key')
@patch('utils.llm_helpers.GeminiClient')
def test_get_llm_client_gemini(mock_gemini_client_class, mock_getenv, mock_gemini_config):
    """Test factory function for Gemini client."""
    # Use the dummy schema class
    get_llm_client(mock_gemini_config, "system prompt", DummySchema)
    mock_getenv.assert_called_with("GEMINI_API_KEY")
    mock_gemini_client_class.assert_called_once_with(mock_gemini_config, DummySchema, "system prompt")

@patch('utils.llm_helpers.os.getenv', return_value='fake-deepseek-key')
@patch('utils.llm_helpers.OpenAIClient')
def test_get_llm_client_deepseek(mock_openai_client_class, mock_getenv, mock_deepseek_config):
    """Test factory function for DeepSeek (OpenAI-compatible) client."""
    get_llm_client(mock_deepseek_config, "system prompt", DummySchema)
    mock_getenv.assert_called_with("DEEPSEEK_API_KEY")
    mock_openai_client_class.assert_called_once_with(mock_deepseek_config, "system prompt")

@patch('utils.llm_helpers.os.getenv', return_value=None)
def test_get_llm_client_raises_no_gemini_key(mock_getenv, mock_gemini_config):
    """Test factory raises ValueError if Gemini API key is missing."""
    with pytest.raises(ValueError, match="GEMINI_API_KEY not found"):
        get_llm_client(mock_gemini_config, "prompt", DummySchema)

@patch('utils.llm_helpers.os.getenv', return_value=None)
def test_get_llm_client_raises_no_deepseek_key(mock_getenv, mock_deepseek_config):
    """Test factory raises ValueError if DeepSeek API key is missing."""
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY not found"):
        get_llm_client(mock_deepseek_config, "prompt", DummySchema)

def test_get_llm_client_raises_unsupported_model():
    """Test factory raises ValueError for an unsupported model."""
    config = {'model_id': 'unsupported-model-v1'}
    with pytest.raises(ValueError, match="Unsupported model provider"):
        get_llm_client(config, "prompt", DummySchema)

# --- Tests for Client Implementations ---

@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
@patch('utils.llm_helpers.genai.Client')
def test_gemini_client_initialization_and_generate(mock_genai_client, mock_gemini_config):
    """Test GeminiClient initialization and content generation call."""
    # Setup Mocks
    mock_client_instance = MagicMock()
    mock_genai_client.return_value = mock_client_instance
    mock_response = MagicMock()
    mock_response.text = '{"result": "success"}'
    mock_client_instance.models.generate_content.return_value = mock_response

    # Action
    # FIX: Pass the DummySchema class instead of a string
    gemini_client = GeminiClient(mock_gemini_config, DummySchema, "system_prompt")
    result = gemini_client.generate_content("sutta body")
    
    # Assertions
    mock_genai_client.assert_called_once_with(api_key='test-key')
    mock_client_instance.models.generate_content.assert_called_once()
    assert result == '{"result": "success"}'

@patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
@patch('utils.llm_helpers.OpenAI')
def test_openai_client_initialization_and_generate(mock_openai_class, mock_deepseek_config):
    """Test OpenAIClient initialization and content generation call."""
    # Setup Mocks
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"result": "openai_success"}'
    mock_client_instance.chat.completions.create.return_value = mock_response

    # Action
    openai_client = OpenAIClient(mock_deepseek_config, "system_prompt_for_openai")
    result = openai_client.generate_content("sutta body")

    # Assertions
    mock_openai_class.assert_called_once_with(api_key='test-key', base_url="https://api.deepseek.com")
    mock_client_instance.chat.completions.create.assert_called_once()
    assert result == '{"result": "openai_success"}'