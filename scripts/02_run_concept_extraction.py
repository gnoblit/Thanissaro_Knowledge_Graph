from utils.config_helpers import ConfigManager
from processing.concept_extractor import ConceptExtractor

def main():
    """Initializes configuration and runs the concept extraction pipeline."""
    # 1. Initialize configuration
    cfg_manager = ConfigManager()
    mode = cfg_manager.config['concept_extraction']['mode']
    model_id = cfg_manager.config['concept_extraction']['model_id']
    
    print(f"--- Running Concept Extraction in '{mode.upper()}' Mode for '{model_id}' Model ---")

    # 2. Initialize and run the extraction pipeline
    # The extractor now gets the mode from the config itself.
    extractor = ConceptExtractor(cfg_manager)
    extractor.run_pipeline()
    
    print(f"\nConcept extraction process ('{mode}' mode) completed.")

if __name__ == "__main__":
    main()