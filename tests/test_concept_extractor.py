import pytest
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
import json

from processing.concept_extractor import ConceptExtractor
from utils.schemas import SuttaConceptsDiscovery, SuttaConceptsFixed

# A mock config that the test can use and modify
@pytest.fixture
def mock_cfg_manager():
    manager = MagicMock()
    manager.config = {
        'concept_extraction': {
            'mode': 'discovery',
            'model_id': 'gemini-1.5-flash',
            'base_prompt_beginning': 'BEGIN\n',
            'discovery_instructions': 'DISCOVERY_INSTRUCTIONS\n',
            'fixed_instructions': 'FIXED_INSTRUCTIONS\n',
            'base_prompt_end': 'END',
            'output_path_template': 'data/test_{mode}_{model_id}.jsonl',
            'log_path_template': 'logs/test_{mode}_{model_id}.jsonl',
            'temperature': 1.0
        },
        'output_paths': {
            'raw_data': 'data/raw_test.jsonl'
        }
    }
    # Mock the get_path to return a predictable, absolute-like path
    manager.get_path.side_effect = lambda key, format_args=None: f"mock/path/to/{key}" if not format_args else f"mock/path/to/{key.format(**format_args)}"
    return manager

# --- Tests for Initialization ---

@patch('processing.concept_extractor.get_llm_client')
def test_initialization_discovery_mode(mock_get_llm_client, mock_cfg_manager):
    """Test correct initialization in 'discovery' mode."""
    extractor = ConceptExtractor(mock_cfg_manager)
    
    assert extractor.strategy == 'discovery'
    expected_prompt = 'BEGIN\nDISCOVERY_INSTRUCTIONS\nEND'
    assert extractor.system_prompt == expected_prompt
    assert extractor.response_schema_class == SuttaConceptsDiscovery
    
    # Check that the LLM client was initialized correctly
    mock_get_llm_client.assert_called_once_with(
        extraction_config=extractor.extraction_config,
        system_prompt=expected_prompt,
        response_schema_class=SuttaConceptsDiscovery
    )

@patch('processing.concept_extractor.get_llm_client')
def test_initialization_fixed_mode(mock_get_llm_client, mock_cfg_manager):
    """Test correct initialization in 'fixed' mode."""
    mock_cfg_manager.config['concept_extraction']['mode'] = 'fixed'
    extractor = ConceptExtractor(mock_cfg_manager)

    assert extractor.strategy == 'fixed'
    expected_prompt = 'BEGIN\nFIXED_INSTRUCTIONS\nEND'
    assert extractor.system_prompt == expected_prompt
    assert extractor.response_schema_class == SuttaConceptsFixed
    
    # Check that the LLM client was initialized correctly
    mock_get_llm_client.assert_called_once_with(
        extraction_config=extractor.extraction_config,
        system_prompt=expected_prompt,
        response_schema_class=SuttaConceptsFixed
    )

def test_initialization_invalid_mode(mock_cfg_manager):
    """Test that initialization fails with an invalid mode."""
    mock_cfg_manager.config['concept_extraction']['mode'] = 'invalid_mode'
    with pytest.raises(ValueError, match="Invalid extraction strategy: invalid_mode"):
        ConceptExtractor(mock_cfg_manager)

# --- NEW TEST TO VERIFY THE FIX ---
@patch('processing.concept_extractor.get_llm_client')
def test_path_sanitization_handles_slashes(mock_get_llm, mock_cfg_manager):
    """Test that model_id with slashes is correctly sanitized for file paths."""
    # Arrange: Use a model_id with a slash
    mock_cfg_manager.config['concept_extraction']['model_id'] = 'google/gemini-1.5-pro'
    extractor = ConceptExtractor(mock_cfg_manager)

    # Act: Trigger the path generation by accessing the property
    _ = extractor.output_path
    
    # Assert: Check that the format_args passed to the mocked get_path are correct
    expected_sanitized_id = 'google_gemini_1_5_pro'
    expected_format_args = {'mode': 'discovery', 'model_id': expected_sanitized_id}
    
    # We check the arguments of the *last* call to the mock
    mock_cfg_manager.get_path.assert_called_with(
        'concept_extraction.output_path_template',
        expected_format_args
    )

# --- Tests for Core Logic (_process_item) ---

@patch('processing.concept_extractor.get_llm_client')
def test_process_item_success(mock_get_llm, mock_cfg_manager):
    """Test the successful processing of a single item."""
    # Mock the LLM client and its response
    mock_llm_client = MagicMock()
    mock_llm_client.generate_content.return_value = '{"concepts": [{"concept_name": "Test", "concept_type": "Person", "evidence_quote": "A test."}]}'
    mock_get_llm.return_value = mock_llm_client

    extractor = ConceptExtractor(mock_cfg_manager)
    
    # Mock datetime to have a predictable timestamp
    mock_dt = datetime(2023, 1, 1, 12, 0)
    with patch('processing.concept_extractor.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        # Re-initialize to get the mocked datetime
        extractor = ConceptExtractor(mock_cfg_manager)

        sutta = {'sutta_id': 1, 'body': 'This is a test sutta.'}
        result = extractor._process_item(sutta)

        mock_llm_client.generate_content.assert_called_once_with('This is a test sutta.')
        
        assert result['sutta_id'] == 1
        assert result['model_id'] == 'gemini-1.5-flash'
        assert result['time_of_run'] == mock_dt.strftime("%Y-%m-%d_%H-%M")
        assert result['mode'] == 'discovery'
        assert len(result['concepts']) == 1
        assert result['concepts'][0]['concept_name'] == 'Test'

@patch('processing.concept_extractor.get_llm_client')
def test_process_item_raises_on_empty_body(mock_get_llm, mock_cfg_manager):
    """Test that _process_item raises a ValueError for an empty sutta body."""
    extractor = ConceptExtractor(mock_cfg_manager)
    sutta = {'sutta_id': 2, 'body': '  '} # Whitespace only
    with pytest.raises(ValueError, match="Sutta body is empty."):
        extractor._process_item(sutta)

@patch('processing.concept_extractor.get_llm_client')
def test_process_item_raises_on_validation_error(mock_get_llm, mock_cfg_manager):
    """Test that _process_item raises a specific ValueError on schema validation failure."""
    mock_llm_client = MagicMock()
    # Malformed JSON (missing closing brace and required fields)
    bad_json_response = '{"concepts": [{"concept_name": "Test"}]' 
    mock_llm_client.generate_content.return_value = bad_json_response
    mock_get_llm.return_value = mock_llm_client

    extractor = ConceptExtractor(mock_cfg_manager)
    sutta = {'sutta_id': 3, 'body': 'Another test sutta.'}
    
    with pytest.raises(ValueError, match="Schema validation failed"):
        extractor._process_item(sutta)

# --- Test for Pipeline Orchestration (run_pipeline) ---

@patch('processing.base_processor.jsonlines.open', new_callable=mock_open)
@patch('processing.base_processor.get_unprocessed_items')
@patch('processing.base_processor.get_processed_ids')
@patch('processing.concept_extractor.get_llm_client')
def test_run_pipeline_orchestration(mock_get_llm, mock_get_ids, mock_get_items, mock_json_open, mock_cfg_manager):
    """
    Test that run_pipeline correctly orchestrates loading, processing, and saving.
    We mock the sub-functions to ensure they are called correctly.
    """
    # Setup: Mock the data loading to return one item
    mock_get_ids.return_value = {1, 3}
    mock_get_items.return_value = [{'sutta_id': 2, 'body': 'Process this one.'}]
    
    # Setup: Mock the LLM and the result from _process_item
    mock_llm_client = MagicMock()
    mock_llm_client.generate_content.return_value = '{"concepts": [{"concept_name": "Item 2", "concept_type": "Place", "evidence_quote": "..."}]}'
    mock_get_llm.return_value = mock_llm_client
    
    # Instantiate the extractor
    extractor = ConceptExtractor(mock_cfg_manager)
    
    # Run the pipeline
    with patch('processing.base_processor.time.sleep'): # Patch sleep to speed up test
      extractor.run_pipeline()

    # --- Assertions ---
    # 1. Assert data loading was called correctly
    mock_get_ids.assert_called_once_with(
        processed_path=extractor.output_path,
        id_key='sutta_id',
        model_id=extractor.model_id,
        mode=extractor.strategy
    )
    mock_get_items.assert_called_once_with(
        source_path=extractor.source_path,
        source_id_key='sutta_id',
        processed_ids_set={1, 3}
    )
    
    # 2. Assert the LLM was called with the item's body
    mock_llm_client.generate_content.assert_called_once_with('Process this one.')

    # 3. Assert the output file was opened in append mode
    mock_json_open.assert_called_with(extractor.output_path, mode='a')
    
    # 4. Assert that the result was written to the file
    mock_writer = mock_json_open()
    mock_writer.write.assert_called_once()
    written_data = mock_writer.write.call_args[0][0]
    assert written_data['sutta_id'] == 2
    assert written_data['concepts'][0]['concept_name'] == 'Item 2'