# tests/test_config_helpers.py

import pytest
import os
import yaml
from unittest.mock import patch

# Keep the original import
from utils.config_helpers import ConfigManager

# This fixture is correct as is
@pytest.fixture
def temp_project(tmp_path):
    project_root = tmp_path
    src_dir = project_root / "src"
    utils_dir = src_dir / "utils"
    utils_dir.mkdir(parents=True)
    
    config_dir = project_root / "config"
    config_dir.mkdir()
    settings_file = config_dir / "settings.yaml"
    
    test_config = {
        "key1": {
            "nested_key": "value1"
        },
        "path_templates": {
            "templated_path": "output/file_{mode}.txt"
        }
    }
    
    with open(settings_file, "w") as f:
        yaml.dump(test_config, f)
        
    # Return the fake path to the config helper file for the patch
    return project_root, str(utils_dir / "config_helpers.py")

def test_config_manager_initialization_and_get_path(temp_project):
    """
    Test that ConfigManager can initialize, load config, and retrieve paths.
    """
    project_root, fake_helper_path = temp_project
    
    # CORRECTED: Instead of patching the instance attribute, we patch the 'os.path.abspath'
    # call that the __init__ method uses. This makes the test more robust.
    with patch('utils.config_helpers.os.path.abspath') as mock_abspath:
        # We make abspath return the path to our *fake* config_helpers file
        mock_abspath.return_value = fake_helper_path
        
        # Now, when ConfigManager is initialized, it will correctly calculate
        # project_root as our temporary directory.
        cfg_manager = ConfigManager(config_filename='config/settings.yaml')

        # Test that config is loaded correctly
        assert cfg_manager.config["key1"]["nested_key"] == "value1"
        # FIX: Convert the Path object to a string for comparison
        assert cfg_manager.project_root == str(project_root)
        
        # Test get_path without formatting
        simple_path = cfg_manager.get_path("key1.nested_key")
        expected_simple = os.path.join(str(project_root), "value1")
        assert simple_path == expected_simple
        
        # Test get_path with formatting
        formatted_path = cfg_manager.get_path("path_templates.templated_path", a_format={"mode": "discovery"})
        expected_formatted = os.path.join(str(project_root), "output/file_discovery.txt")
        assert formatted_path == expected_formatted