import yaml
import os
from dotenv import load_dotenv

class ConfigManager:
    """
    Manages loading configuration and constructing absolute paths for the project.
    """
    def __init__(self, config_filename='config/settings.yaml'):
        # Load API keys
        load_dotenv()
        
        # Find the project root by going up two directories
        # from this file's location (src/utils -> src -> project_root)
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        config_path = os.path.join(self.project_root, config_filename)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_path(self, key_path, a_format=None):
        """
        Retrieves a path from config, constructs the absolute path, 
        and optionally formats it.
        """
        keys = key_path.split('.')
        path_template = self.config
        for key in keys:
            path_template = path_template[key]
        
        if a_format:
            path_template = path_template.format(**a_format)

        return os.path.join(self.project_root, path_template)
    
def sanitize_for_filename(text: str) -> str:
    """
    Replaces characters in a string that are problematic for filenames
    with underscores.
    
    This ensures consistency when generating and reading files based on
    dynamic values like model IDs.
    """
    return text.replace('/', '_').replace('-', '_').replace('.', '')