import yaml
import os
from utils.config_helpers import load_config 

from data_acquisition.scraper import run_scraper

def main():
    """Loads configuration and runs the scraper."""
    # --- Setup ---
    # Use the helper to load config and get project_root
    config_dict = load_config()
    config = config_dict['config']
    project_root = config_dict['project_root']

    # --- Path Management ---
    # Construct the absolute path for the output file
    raw_data_path = os.path.join(project_root, config['output_paths']['raw_data'])
    
    # The scraper function will handle os.makedirs
    
    # --- Run Scraper ---
    # Pass both the config and the absolute output path
    run_scraper(config, raw_data_path)
    
    print("\nScraping process completed.")

if __name__ == "__main__":
    main()