from utils.config_helpers import ConfigManager
from processing.second_look_concept_extractor import SecondLookConceptExtractor

def main():
    """Initializes configuration and runs the second look concept extraction."""
    # 1. Initialize configuration
    cfg_manager = ConfigManager()
    model_id = cfg_manager.config['second_look_concept_extraction']['model_id']
    
    print(f"--- Running Second Look Concept Extraction for '{model_id}' Model ---")

    # 2. Initialize and run the extraction pipeline
    extractor = SecondLookConceptExtractor(cfg_manager)
    extractor.run_pipeline()
    
    print(f"\nSecond look concept extraction process completed.")

if __name__ == "__main__":
    main()