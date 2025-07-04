import pytest
from unittest.mock import patch, MagicMock

from src.processing.concept_extractor import ConceptExtractor
from src.utils.schemas import SuttaConceptsDiscovery, SuttaConceptsFixed

# A mock config that the test can use
@pytest.fixture
def mock_cfg_manager():
    manager = MagicMock()
    manager.config = {
        'concept_extraction': {
            'mode': 'discovery',
            'model_id': 'gemini-1.5-flash',
            'base_prompt_beginning': 'BEGIN',
            'discovery_instruction': 'DISCOVERY',
            'base_prompt_end': 'END',
            'output_path_template': 'data/test_{mode}.jsonl',
            'log_path_template': 'logs/test_{mode}.jsonl',
        },
        'output_paths': {
            'raw_data': 'data/raw_test.jsonl'
        }
    }
    # Mock get_path to return a simplified path
    manager.get_path.side_effect = lambda key, a_format=None: f"mock/path/to/{key}"
    return manager

@patch('src.processing.concept_extractor.get_llm_client')
def test_concept_extractor_initialization(mock_get_llm_client, mock_cfg_manager):
    """
    Test that ConceptExtractor initializes correctly and calls the LLM factory.
    """
    # Act
    extractor = ConceptExtractor(mock_cfg_manager)

    # Assert
    # 1. The strategy is correctly read from the config
    assert extractor.strategy == 'discovery'
    
    # 2. The system prompt is constructed correctly
    expected_prompt = 'BEGINDISCOVERYEND'
    assert extractor.system_prompt == expected_prompt

    # 3. The correct Pydantic schema was chosen
    assert extractor.response_schema_class == SuttaConceptsDiscovery

    # 4. The LLM client factory was called with the correct arguments
    mock_get_llm_client.assert_called_once_with(
        extraction_config=mock_cfg_manager.config['concept_extraction'],
        system_prompt=expected_prompt,
        response_schema_class=SuttaConceptsDiscovery
    )

@patch('src.processing.concept_extractor.get_unprocessed_items')
@patch('src.processing.concept_extractor.get_processed_ids')
def test_load_data_logic(mock_get_processed, mock_get_unprocessed, mock_cfg_manager):
    """
    Test that the _load_data method correctly orchestrates its helpers.
    """
    # Arrange
    extractor = ConceptExtractor(mock_cfg_manager)
    mock_get_processed.return_value = {1, 3} # Simulate that sutta 1 and 3 are done
    mock_get_unprocessed.return_value = [{'sutta_id': 2}] # Expect to get sutta 2 back

    # Act
    result = extractor._load_data()

    # Assert
    # Verify that we first ask for the IDs of already processed items
    mock_get_processed.assert_called_once_with(
        processed_path=extractor.output_path,
        id_key='sutta_id',
        model_id=extractor.model_id,
        mode=extractor.strategy
    )

    # Verify we then ask for unprocessed items, passing the processed IDs
    mock_get_unprocessed.assert_called_once_with(
        source_path=extractor.raw_data_path,
        source_id_key='sutta_id',
        processed_ids_set={1, 3}
    )

    # Verify the final result is what the helper returned
    assert result == [{'sutta_id': 2}]