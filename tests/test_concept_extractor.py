import pytest
from unittest.mock import patch, MagicMock

from processing.concept_extractor import ConceptExtractor
from utils.schemas import SuttaConceptsDiscovery, SuttaConceptsFixed

# A mock config that the test can use
@pytest.fixture
def mock_cfg_manager():
    manager = MagicMock()
    manager.config = {
        'concept_extraction': {
            'mode': 'discovery',
            'model_id': 'gemini-1.5-flash',
            'base_prompt_beginning': 'BEGIN',
            'discovery_instructions': 'DISCOVERY',
            'fixed_instructions': 'FIXED',
            'base_prompt_end': 'END',
            'output_path_template': 'data/test_{mode}.jsonl',
            'log_path_template': 'logs/test_{mode}.jsonl',
        },
        'output_paths': {
            'raw_data': 'data/raw_test.jsonl'
        }
    }
    manager.get_path.side_effect = lambda key, a_format=None: f"mock/path/to/{key.format(**a_format) if a_format else key}"
    return manager

@patch('processing.concept_extractor.get_llm_client')
def test_concept_extractor_initialization(mock_get_llm_client, mock_cfg_manager):
    extractor = ConceptExtractor(mock_cfg_manager)
    assert extractor.strategy == 'discovery'
    expected_prompt = 'BEGINDISCOVERYEND'
    assert extractor.system_prompt == expected_prompt
    assert extractor.response_schema_class == SuttaConceptsDiscovery
    mock_get_llm_client.assert_called_once()

@patch('processing.concept_extractor.get_llm_client')
def test_concept_extractor_initialization_fixed_mode(mock_get_llm_client, mock_cfg_manager):
    mock_cfg_manager.config['concept_extraction']['mode'] = 'fixed'
    extractor = ConceptExtractor(mock_cfg_manager)
    assert extractor.strategy == 'fixed'
    assert extractor.system_prompt == 'BEGINFIXEDEND'
    assert extractor.response_schema_class == SuttaConceptsFixed

def test_concept_extractor_invalid_mode(mock_cfg_manager):
    mock_cfg_manager.config['concept_extraction']['mode'] = 'invalid_mode'
    with pytest.raises(ValueError, match="Invalid extraction strategy: invalid_mode"):
        ConceptExtractor(mock_cfg_manager)

@patch('processing.concept_extractor.get_unprocessed_items')
@patch('processing.concept_extractor.get_processed_ids')
@patch('processing.concept_extractor.get_llm_client')
def test_load_data_logic(mock_get_llm, mock_get_processed, mock_get_unprocessed, mock_cfg_manager):
    extractor = ConceptExtractor(mock_cfg_manager)
    mock_get_processed.return_value = {1, 3}
    mock_get_unprocessed.return_value = [{'sutta_id': 2}]
    result = extractor._load_data()
    mock_get_processed.assert_called_once()
    mock_get_unprocessed.assert_called_once_with(
        source_path=extractor.raw_data_path,
        source_id_key='sutta_id',
        processed_ids_set={1, 3}
    )
    assert result == [{'sutta_id': 2}]

@patch('processing.concept_extractor.jsonlines')
@patch('processing.concept_extractor.time.sleep')
@patch('processing.concept_extractor.get_llm_client')
def test_process_suttas_success(mock_get_llm, mock_sleep, mock_jsonlines, mock_cfg_manager):
    mock_llm_client = MagicMock()
    mock_llm_client.generate_content.return_value = '{"concepts": [{"concept_name": "Test", "concept_type": "Person", "evidence_quote": "A test."}]}'
    mock_get_llm.return_value = mock_llm_client
    extractor = ConceptExtractor(mock_cfg_manager)
    suttas_to_process = [{'sutta_id': 1, 'body': 'This is a test sutta.'}]
    mock_writer = MagicMock()
    mock_jsonlines.open.return_value.__enter__.return_value = mock_writer
    extractor._process_suttas(suttas_to_process)
    mock_llm_client.generate_content.assert_called_once_with('This is a test sutta.')
    mock_jsonlines.open.assert_called_with(extractor.output_path, mode='a')
    assert mock_writer.write.call_count == 1

@patch('processing.concept_extractor.jsonlines')
@patch('processing.concept_extractor.time.sleep')
@patch('processing.concept_extractor.get_llm_client')
def test_process_suttas_validation_error(mock_get_llm, mock_sleep, mock_jsonlines, mock_cfg_manager):
    mock_llm_client = MagicMock()
    bad_json_response = '{"concepts": [{"concept_name": "Test"}]' 
    mock_llm_client.generate_content.return_value = bad_json_response
    mock_get_llm.return_value = mock_llm_client
    extractor = ConceptExtractor(mock_cfg_manager)
    suttas_to_process = [{'sutta_id': 2, 'body': 'Another test sutta.'}]
    mock_log_writer = MagicMock()
    def open_side_effect(path, mode):
        if path == extractor.log_path:
            return MagicMock(__enter__=MagicMock(return_value=mock_log_writer))
        return MagicMock()
    mock_jsonlines.open.side_effect = open_side_effect
    extractor._process_suttas(suttas_to_process)
    mock_jsonlines.open.assert_any_call(extractor.log_path, mode='a')
    assert mock_log_writer.write_all.call_count == 1
    logged_data = mock_log_writer.write_all.call_args[0][0]
    assert logged_data[0]['sutta_id'] == 2
    assert "Schema validation failed" in logged_data[0]['reason']

@patch('processing.concept_extractor.sys.exit')
@patch('processing.concept_extractor.jsonlines')
@patch('processing.concept_extractor.time.sleep')
@patch('processing.concept_extractor.get_llm_client')
def test_process_suttas_resource_exhausted(mock_get_llm, mock_sleep, mock_jsonlines, mock_sys_exit, mock_cfg_manager):
    mock_llm_client = MagicMock()
    mock_llm_client.generate_content.side_effect = Exception("API error: RESOURCE_EXHAUSTED")
    mock_get_llm.return_value = mock_llm_client
    extractor = ConceptExtractor(mock_cfg_manager)
    suttas_to_process = [{'sutta_id': 4, 'body': 'This will fail.'}]
    extractor._process_suttas(suttas_to_process)
    mock_sys_exit.assert_called_once_with(1)