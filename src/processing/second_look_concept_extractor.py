import json
import re
from datetime import datetime

import jsonlines
from pydantic import ValidationError

from .base_processor import BaseProcessor
from utils.config_helpers import sanitize_for_filename
from utils.data_helpers import get_processed_ids
from utils.llm_helpers import get_llm_client
from utils.schemas import SuttaConceptsDiscovery  # Re-use the existing schema

class SecondLookConceptExtractor(BaseProcessor):
    def __init__(self, cfg_manager):
        # Specific setup for this extractor
        self.second_look_config = cfg_manager.config['second_look_concept_extraction']
        self.concept_extraction_config = cfg_manager.config['concept_extraction']
        
        self.model_id = self.second_look_config['model_id']
        self.system_prompt_template = self.second_look_config['prompt']

        # Call super() AFTER defining the attributes it needs (_get_output_path etc.)
        super().__init__(cfg_manager)

        self.dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # The schema for the response is the same as the original extraction
        self.response_schema_class = SuttaConceptsDiscovery

        # The system prompt is dynamic per item, so we pass a dummy prompt here.
        # The full, formatted prompt will be passed to `generate_content` later.
        self.llm_client = get_llm_client(
            extraction_config=self.second_look_config,
            system_prompt="You are a helpful assistant.", 
            response_schema_class=self.response_schema_class
        )

    # --- Implementation of abstract methods ---
    def _get_source_path(self) -> str:
        """The source is the output of the first concept extraction pass."""
        s_model_id = sanitize_for_filename(self.concept_extraction_config['model_id'])
        mode = self.concept_extraction_config['mode']
        format_args = {'mode': mode, 'model_id': s_model_id}
        return self.cfg_manager.get_path('concept_extraction.output_path_template', format_args)

    def _get_output_path(self) -> str:
        s_model_id = sanitize_for_filename(self.model_id)
        format_args = {'model_id': s_model_id}
        return self.cfg_manager.get_path('second_look_concept_extraction.output_path_template', format_args)

    def _get_log_path(self) -> str:
        s_model_id = sanitize_for_filename(self.model_id)
        format_args = {'model_id': s_model_id}
        return self.cfg_manager.get_path('second_look_concept_extraction.log_path_template', format_args)

    def _get_run_config(self) -> dict:
        """Defines a unique run for this processor."""
        return {'model_id': self.model_id}

    def _load_unprocessed_items(self) -> list:
        """
        Overrides BaseProcessor._load_unprocessed_items.
        Loads both the raw sutta text and the initial concepts, then combines them.
        Filters out items that have already had a "second look".
        """
        # 1. Find which suttas have already been processed for this "second look" run
        processed_ids = get_processed_ids(
            processed_path=self.output_path,
            id_key='sutta_id',
            **self._get_run_config()
        )

        # 2. Load all raw suttas into a dictionary for quick lookup
        raw_suttas_path = self.cfg_manager.get_path('output_paths.raw_data')
        suttas_by_id = {}
        with jsonlines.open(raw_suttas_path) as reader:
            for sutta in reader:
                if 'sutta_id' in sutta and 'body' in sutta:
                    suttas_by_id[sutta['sutta_id']] = sutta['body']
        
        # 3. Load initial concepts and combine with raw text
        initial_concepts_path = self._get_source_path()
        items_to_process = []
        with jsonlines.open(initial_concepts_path) as reader:
            for record in reader:
                sutta_id = record.get('sutta_id')
                if sutta_id and sutta_id not in processed_ids:
                    if sutta_id in suttas_by_id:
                        items_to_process.append({
                            'sutta_id': sutta_id,
                            'body': suttas_by_id[sutta_id],
                            'existing_concepts': record.get('concepts', [])
                        })

        print(f"Found {len(items_to_process)} new items for second look processing.")
        return items_to_process


    def _process_item(self, item: dict) -> dict:
        """Core logic for reviewing one sutta."""
        sutta_body_raw = item.get("body")
        if not sutta_body_raw or not sutta_body_raw.strip():
            raise ValueError("Sutta body is empty.")

        # Clean the text by replacing all whitespace sequences (tabs, newlines) with a single space
        sutta_body = re.sub(r'\s+', ' ', sutta_body_raw).strip()
        existing_concepts = item.get("existing_concepts")

        if not sutta_body or not sutta_body.strip():
            raise ValueError("Sutta body is empty.")
        
        # Format the existing concepts into a clean JSON string for the prompt
        existing_concepts_str = json.dumps(existing_concepts, indent=2)

        # Format the full prompt, which will be treated as the user message
        prompt = self.system_prompt_template.format(
            sutta_body=sutta_body,
            existing_concepts=existing_concepts_str
        )
        
        response_text = self.llm_client.generate_content(prompt)
        
        try:
            # Pydantic validates the JSON response.
            parsed_data = self.response_schema_class.model_validate_json(response_text)
            new_concepts = parsed_data.model_dump().get('concepts', [])
        except (ValidationError, json.JSONDecodeError) as e:
            raise ValueError(f"Schema validation failed: {e}. Raw response: {response_text}") from e

        return {
            'sutta_id': item.get("sutta_id"),
            'model_id': self.model_id,
            'time_of_run': self.dt_string,
            'newly_found_concepts': new_concepts,
        }