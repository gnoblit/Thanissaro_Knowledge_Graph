import os
import jsonlines

def get_unprocessed_items(source_path, processed_path, source_id_key, processed_id_key):
    """
    Loads items from a source file and filters out those already present in a processed file.

    Args:
        source_path (str): Path to the source .jsonl file.
        processed_path (str): Path to the .jsonl file of already processed items.
        source_id_key (str): The key in the source file that holds the unique ID.
        processed_id_key (str): The key in the processed file that holds the unique ID.

    Returns:
        list: A list of item dictionaries that have not yet been processed.
    """
    all_items = []
    with jsonlines.open(source_path) as reader:
        for item in reader:
            if item.get(source_id_key):
                all_items.append(item)

    processed_ids = set()
    if os.path.exists(processed_path):
        with jsonlines.open(processed_path) as reader:
            for record in reader:
                if record.get(processed_id_key):
                    processed_ids.add(record[processed_id_key])
    
    items_to_process = [
        item for item in all_items if item[source_id_key] not in processed_ids
    ]
    
    print(f"Found {len(all_items)} total items in source.")
    print(f"{len(processed_ids)} items already processed.")
    print(f"Returning {len(items_to_process)} new items for processing.")
    
    return items_to_process