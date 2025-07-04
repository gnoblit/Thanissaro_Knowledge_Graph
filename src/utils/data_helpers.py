import os
import jsonlines
import json 

def get_processed_ids(processed_path: str, id_key: str, **run_config) -> set:
    """
    Loads IDs from a processed file, filtering by the specific run configuration.
    This ensures that a run with new settings (e.g., a different model, mode, or
    any other parameter) will re-process all items.
    
    Args:
        processed_path (str): Path to the .jsonl file of already processed items.
        id_key (str): The key in the processed file that holds the unique ID.
        **run_config: A dictionary of key-value pairs that define a unique run.
                      A record is considered "processed" if it matches all of these pairs.

    Returns:
        set: A set of unique IDs that have been processed with the given settings.
    """
    processed_ids = set()
    if not os.path.exists(processed_path):
        return processed_ids
        
    with jsonlines.open(processed_path) as reader:
        # Wrap in a try-except to handle corrupted lines in the file.
        for line_num, record in enumerate(reader, 1):
            try:
                # A record is a match if all key-value pairs in run_config
                # exist and are equal in the record.
                is_match = all(record.get(key) == value for key, value in run_config.items())
                
                if is_match and record.get(id_key):
                    processed_ids.add(record[id_key])

            except (TypeError, json.JSONDecodeError):
                print(f"Warning: Skipping corrupted or invalid line {line_num} in {processed_path}")
                continue
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