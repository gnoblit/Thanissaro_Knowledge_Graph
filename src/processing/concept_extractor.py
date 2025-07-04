import sys
import os
import time
import json
from datetime import datetime
import jsonlines
from tqdm import tqdm
from pydantic import ValidationError

from utils.data_helpers import get_processed_ids, get_unprocessed_items 
from utils.schemas import SuttaConceptsDiscovery, SuttaConceptsFixed
from utils.llm_helpers import get_llm_client 

class ConceptExtractor:
    """
    Orchestrates the concept extraction process from suttas.
    """
    def __init__(self, cfg_manager):
        self.cfg_manager = cfg_manager
        self.config = cfg_manager.config

        self.extraction_config = self.config['concept_extraction']
        self.strategy = self.extraction_config['mode'] 
        self.model_id = self.extraction_config['model_id']


        # Dynamically select prompt and schema
        if self.strategy == 'discovery':
            prompt_key = 'discovery_instructions'
        elif self.strategy == 'fixed':
            prompt_key = 'fixed_instructions'
        else:
            raise ValueError(f"Invalid extraction strategy: {self.strategy}")
        
        self.system_prompt = self.extraction_config['base_prompt_beginning']  + self.extraction_config[prompt_key] + self.extraction_config['base_prompt_end']
        self.response_schema_class = SuttaConceptsDiscovery if self.strategy == 'discovery' else SuttaConceptsFixed
        
        # Get paths
        format_args = {'mode': self.strategy}
        self.output_path = self.cfg_manager.get_path('concept_extraction.output_path_template', format_args)
        self.log_path = self.cfg_manager.get_path('concept_extraction.log_path_template', format_args)
        self.raw_data_path = self.cfg_manager.get_path('output_paths.raw_data')

        # Initialize the LLM client
        self.llm_client = get_llm_client(
            extraction_config=self.extraction_config,
            system_prompt=self.system_prompt,
            response_schema_class=self.response_schema_class
        )

    def run_pipeline(self):
        """Executes the full extraction pipeline."""
        suttas_to_process = self._load_data()
        self._process_suttas(suttas_to_process)

    def _load_data(self):
        """Loads and filters data to find unprocessed suttas."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # 1. Get the set of IDs that have already been processed WITH THE CURRENT CONFIG
        processed_ids = get_processed_ids(
            processed_path=self.output_path,
            id_key='sutta_id',
            model_id=self.model_id,
            mode=self.strategy
        )

        # 2. Get all source items that are not in the processed set
        return get_unprocessed_items(
            source_path=self.raw_data_path,
            source_id_key='sutta_id',
            processed_ids_set=processed_ids
        )
    
    def _process_suttas(self, suttas_to_process):
        """Main extraction loop for processing suttas."""
        skipped_suttas_log = []
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        for sutta in tqdm(suttas_to_process, desc=f"Extracting ({self.strategy} mode)"):
            sutta_identifier = sutta.get("sutta_id", "Unknown")
            response_text = None
            try:
                sutta_body = sutta.get("body")
                if not sutta_body or not sutta_body.strip():
                    raise ValueError("Sutta body is empty.")
                
                response_text = self.llm_client.generate_content(sutta_body)
                parsed_data = self.response_schema_class.model_validate_json(response_text)
                
                result_record = {
                    'sutta_id': sutta_identifier, 'model_id': self.model_id,
                    'time_of_run': dt,
                    'mode': self.strategy,
                    'concepts': parsed_data.model_dump()['concepts'],
                }
                with jsonlines.open(self.output_path, mode='a') as writer:
                    writer.write(result_record)
                time.sleep(0.5)

            except (ValidationError, json.JSONDecodeError) as e:
                reason = f"Schema validation failed: {e}"
                print(f"\nSKIPPING {sutta_identifier}: {reason}")
                skipped_suttas_log.append({"sutta_id": sutta_identifier, "reason": reason, "raw_response": response_text})
            
            except Exception as e:
                error_string = str(e)
                print(f"\nAn error occurred while processing {sutta_identifier}: {error_string}")
                if "RESOURCE_EXHAUSTED" in error_string:
                    print("FATAL: API resource quota exhausted. Terminating script.", file=sys.stderr)
                    if skipped_suttas_log:
                        with jsonlines.open(self.log_path, mode='a') as writer:
                            writer.write_all(skipped_suttas_log)
                    sys.exit(1)
                else:
                    skipped_suttas_log.append({"sutta_id": sutta_identifier, "reason": f"Processing error: {error_string}", "raw_response": response_text})
                    time.sleep(1)

        if skipped_suttas_log:
            print(f"\nINFO: {len(skipped_suttas_log)} suttas were skipped. Logging to {self.log_path}")
            with jsonlines.open(self.log_path, mode='a') as writer:
                writer.write_all(skipped_suttas_log)