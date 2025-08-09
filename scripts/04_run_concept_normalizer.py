from utils.config_helpers import ConfigManager
from processing.concept_normalizer import ConceptNormalizer

def main():
    """Initializes configuration and runs the concept normalization pipeline."""
    # 1. Initialize configuration
    cfg_manager = ConfigManager()
    
    # 2. Initialize and run the normalization pipeline
    normalizer = ConceptNormalizer(cfg_manager)
    normalizer.run_pipeline()
    
    print("\nConcept normalization process completed.")

if __name__ == "__main__":
    main()