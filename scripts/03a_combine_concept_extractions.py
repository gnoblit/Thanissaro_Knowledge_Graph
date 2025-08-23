import os
from collections import defaultdict
import jsonlines
from utils.config_helpers import ConfigManager, sanitize_for_filename

def main():
    """
    Combines the results from the initial concept extraction and the
    second look extraction into a single, unified file.
    """
    print("--- Combining First and Second Look Concept Extractions ---")
    cfg_manager = ConfigManager()
    config = cfg_manager.config

    # --- Get model and mode from the primary extraction config ---
    ext_config = config['concept_extraction']
    s_look_config = config['second_look_concept_extraction']
    
    mode = ext_config['mode']
    model_id_first_pass = ext_config['model_id']
    model_id_second_pass = s_look_config['model_id']

    # --- Define Input Paths ---
    s_model_id_first = sanitize_for_filename(model_id_first_pass)
    first_pass_path = cfg_manager.get_path(
        'concept_extraction.output_path_template',
        {'mode': mode, 'model_id': s_model_id_first}
    )

    s_model_id_second = sanitize_for_filename(model_id_second_pass)
    second_pass_path = cfg_manager.get_path(
        'second_look_concept_extraction.output_path_template',
        {'model_id': s_model_id_second}
    )

    # --- Define Output Path ---
    # We will add a new template for this in settings.yaml
    output_path = cfg_manager.get_path(
        'output_paths.combined_concepts',
        {'mode': mode, 'model_id': s_model_id_first} # Base the name on the first pass
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Logic for Combining ---
    combined_data = {}

    # 1. Read the first pass results as the base
    print(f"Loading initial concepts from: {first_pass_path}")
    with jsonlines.open(first_pass_path) as reader:
        for record in reader:
            sutta_id = record.get('sutta_id')
            if sutta_id:
                combined_data[sutta_id] = record
    print(f"Loaded {len(combined_data)} records from the first pass.")

    # 2. Read the second pass results and merge them in
    if os.path.exists(second_pass_path):
        print(f"Loading second look concepts from: {second_pass_path}")
        found_second_look = 0
        with jsonlines.open(second_pass_path) as reader:
            for record in reader:
                sutta_id = record.get('sutta_id')
                if sutta_id in combined_data:
                    new_concepts = record.get('newly_found_concepts', [])
                    if new_concepts:
                        combined_data[sutta_id]['concepts'].extend(new_concepts)
                        found_second_look += 1
        print(f"Merged new concepts for {found_second_look} records.")
    else:
        print(f"Warning: Second look file not found at {second_pass_path}. Proceeding with first pass data only.")

    # 3. Write the combined data to the new file
    print(f"Writing combined data to: {output_path}")
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(combined_data.values())
        
    print(f"\nCombination complete. Saved {len(combined_data)} total records.")

if __name__ == "__main__":
    main()