import yaml
import os

def load_config():
    """
    Finds project root then loads configuration from YAML file.
    Returns:
            return_dict (dict): 'project_root', 'config_path', 'config' keys
    """
    # Find the project root by going up two directories
    # from this file's location (src/utils -> src -> project_root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    config_path = os.path.join(project_root, 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return_dict = {
        'project_root': project_root,
        'config_path': config_path,
        'config': config
    }
    
    return return_dict