from utils.config_helpers import ConfigManager
from data_acquisition.scraper import run_scraper

def main():
    """Loads configuration and runs the scraper."""
    # --- Setup ---
    # Use the ConfigManager to load config and handle paths
    cfg_manager = ConfigManager()
    config = cfg_manager.config
    
    # --- Path Management ---
    # Get the absolute path for the output file from the manager
    raw_data_path = cfg_manager.get_path('output_paths.raw_data')
    
    # The scraper function will handle os.makedirs
    
    # --- Run Scraper ---
    # Pass both the config and the absolute output path
    run_scraper(config, raw_data_path)
    
    print("\nScraping process completed.")

if __name__ == "__main__":
    main()