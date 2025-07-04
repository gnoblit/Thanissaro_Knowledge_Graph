import jsonlines
from utils.data_helpers import get_processed_ids, get_unprocessed_items

def test_get_processed_ids(tmp_path):
    """
    Test that get_processed_ids correctly filters by model and mode.
    """
    processed_file = tmp_path / "processed.jsonl"
    records = [
        {'sutta_id': 1, 'model_id': 'gemini-1.5', 'mode': 'discovery'},
        {'sutta_id': 2, 'model_id': 'gpt-4', 'mode': 'discovery'}, # Wrong model
        {'sutta_id': 3, 'model_id': 'gemini-1.5', 'mode': 'fixed'},     # Wrong mode
        {'sutta_id': 4, 'model_id': 'gemini-1.5', 'mode': 'discovery'},
    ]
    with jsonlines.open(processed_file, 'w') as writer:
        writer.write_all(records)

    # Act
    processed_set = get_processed_ids(
        processed_path=str(processed_file),
        id_key='sutta_id',
        model_id='gemini-1.5',
        mode='discovery'
    )

    # Assert
    assert processed_set == {1, 4}

def test_get_processed_ids_file_not_exist(tmp_path):
    """Test it returns an empty set if the processed file doesn't exist."""
    processed_set = get_processed_ids(
        processed_path=str(tmp_path / "nonexistent.jsonl"),
        id_key='sutta_id',
        model_id='any',
        mode='any'
    )
    assert processed_set == set()

def test_get_unprocessed_items(tmp_path):
    """
    Test that get_unprocessed_items correctly filters items from a source file.
    """
    source_file = tmp_path / "source.jsonl"
    source_items = [
        {'sutta_id': 1, 'text': '...'},
        {'sutta_id': 2, 'text': '...'},
        {'sutta_id': 3, 'text': '...'},
    ]
    with jsonlines.open(source_file, 'w') as writer:
        writer.write_all(source_items)
        
    processed_ids = {1, 3}

    # Act
    items_to_process = get_unprocessed_items(
        source_path=str(source_file),
        source_id_key='sutta_id',
        processed_ids_set=processed_ids
    )

    # Assert
    assert len(items_to_process) == 1
    assert items_to_process[0]['sutta_id'] == 2