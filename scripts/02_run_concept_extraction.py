import argparse
from utils.config_helpers import ConfigManager
from processing.concept_extractor import ConceptExtractor

def main():
    """Parses command-line arguments to select and run an extraction strategy."""
    parser = argparse.ArgumentParser(description="Run concept extraction with a specified strategy.")
    parser.add_argument("strategy", choices=['discovery', 'fixed'], help="The extraction strategy to use: 'discovery' or 'fixed'.")
    args = parser.parse_args()

    print(f"--- Running Concept Extraction in '{args.strategy.upper()}' Mode ---")

    # 1. Initialize configuration
    cfg_manager = ConfigManager()

    # 2. Initialize and run the extraction pipeline
    extractor = ConceptExtractor(cfg_manager, args.strategy)
    extractor.run_pipeline()
    
    print(f"\nConcept extraction process ('{args.strategy}' mode) completed.")

if __name__ == "__main__":
    main()