import yaml
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data_acquisition.scraper import run_scraper

def main():
    """Loads configuration and runs the scraper."""
    # Load config file
    config_path = os.path.join(project_root, 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure data directory exists
    raw_data_path = os.path.join(project_root, config['output_paths']['raw_data'])
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    
    # Run the main scraping logic
    run_scraper(config)
    
    print("Scraping process completed.")

if __name__ == "__main__":
    main()