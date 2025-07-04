import os
import jsonlines

def get_processed_ids(processed_path: str, id_key: str, model_id: str, mode: str) -> set:
    """
    Loads IDs from a processed file, filtering by the specific model and mode used.
    This ensures that a run with a new model/mode will re-process all items.
    
    Args:
        processed_path (str): Path to the .jsonl file of already processed items.
        id_key (str): The key in the processed file that holds the unique ID.
        model_id (str): The model ID to filter by.
        mode (str): The mode ('discovery' or 'fixed') to filter by.

    Returns:
        set: A set of unique IDs that have been processed with the given settings.
    """
    processed_ids = set()
    if not os.path.exists(processed_path):
        return processed_ids
        
    with jsonlines.open(processed_path) as reader:
        for record in reader:
            # Only count a record as "processed" if it matches the current run's settings
            if record.get('model_id') == model_id and record.get('mode') == mode:
                if record.get(id_key):
                    processed_ids.add(record[id_key])
    return processed_ids

def get_unprocessed_items(source_path: str, source_id_key: str, processed_ids_set: set) -> list:
    """
    Loads items from a source file and filters out those whose IDs are in the provided set.

    Args:
        source_path (str): Path to the source .jsonl file.
        source_id_key (str): The key in the source file that holds the unique ID.
        processed_ids_set (set): A set of IDs to filter out.

    Returns:
        list: A list of item dictionaries that have not yet been processed.
    """
    items_to_process = []
    total_items = 0
    with jsonlines.open(source_path) as reader:
        for item in reader:
            total_items += 1
            if item.get(source_id_key) not in processed_ids_set:
                items_to_process.append(item)

    print(f"Found {total_items} total items in source.")
    print(f"{len(processed_ids_set)} items already processed for this configuration.")
    print(f"Returning {len(items_to_process)} new items for processing.")
    
    return items_to_process