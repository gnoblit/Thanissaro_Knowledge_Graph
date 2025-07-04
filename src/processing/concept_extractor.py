from .base_processor import BaseProcessor
from utils.schemas import SuttaConceptsDiscovery, SuttaConceptsFixed
from utils.llm_helpers import get_llm_client 
import json
from pydantic import ValidationError
from datetime import datetime

class ConceptExtractor(BaseProcessor):
    def __init__(self, cfg_manager):
        # Base class __init__ will handle path setup
        super().__init__(cfg_manager)

        # Specific setup for concept extraction
        self.extraction_config = self.config['concept_extraction']
        self.strategy = self.extraction_config['mode'] 
        self.model_id = self.extraction_config['model_id']
        self.dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # --- FIX START ---
        # The following logic replaces the invalid "..." placeholders.
        # It dynamically builds the prompt and selects the Pydantic schema
        # based on the extraction mode specified in the config.

        if self.strategy == 'discovery':
            instructions = self.extraction_config['discovery_instructions']
            self.response_schema_class = SuttaConceptsDiscovery
        elif self.strategy == 'fixed':
            instructions = self.extraction_config['fixed_instructions']
            self.response_schema_class = SuttaConceptsFixed
        else:
            raise ValueError(f"Invalid extraction strategy: {self.strategy}")

        # Assemble the full system prompt from the config parts
        self.system_prompt = (
            f"{self.extraction_config['base_prompt_beginning']}"
            f"{instructions}"
            f"{self.extraction_config['base_prompt_end']}"
        )

        # Initialize the appropriate LLM client with the constructed prompt and schema
        self.llm_client = get_llm_client(
            extraction_config=self.extraction_config,
            system_prompt=self.system_prompt,
            response_schema_class=self.response_schema_class
        )
        # --- FIX END ---

    # --- Implementation of abstract methods ---
    def _get_source_path(self) -> str:
        return self.cfg_manager.get_path('output_paths.raw_data')

    def _get_output_path(self) -> str:
        # Use a sanitized model_id for file paths
        s_model_id = self.model_id.replace('-', '_').replace('.', '')
        format_args = {'mode': self.strategy, 'model_id': s_model_id}
        return self.cfg_manager.get_path('concept_extraction.output_path_template', format_args)

    def _get_log_path(self) -> str:
        # Use a sanitized model_id for file paths
        s_model_id = self.model_id.replace('-', '_').replace('.', '')
        format_args = {'mode': self.strategy, 'model_id': s_model_id}
        return self.cfg_manager.get_path('concept_extraction.log_path_template', format_args)

    def _get_run_config(self) -> dict:
        return {'model_id': self.model_id, 'mode': self.strategy}

    def _process_item(self, sutta: dict) -> dict:
        """Core logic for one sutta, moved from the old loop."""
        sutta_body = sutta.get("body")
        if not sutta_body or not sutta_body.strip():
            raise ValueError("Sutta body is empty.")
        
        response_text = self.llm_client.generate_content(sutta_body)
        
        try:
            parsed_data = self.response_schema_class.model_validate_json(response_text)
        except (ValidationError, json.JSONDecodeError) as e:
            # Raise a specific exception that the base class can catch
            raise ValueError(f"Schema validation failed: {e}. Raw response: {response_text}") from e

        return {
            'sutta_id': sutta.get("sutta_id"),
            'model_id': self.model_id,
            'time_of_run': self.dt_string,
            'mode': self.strategy,
            'concepts': parsed_data.model_dump()['concepts'],
        }