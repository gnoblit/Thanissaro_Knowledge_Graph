import os

import jsonlines
import json
from pydantic import ValidationError  

from tqdm import tqdm
from datetime import datetime
import time
import sys
from google import genai
from google.genai.types import GenerateContentConfig

from utils.data_helpers import get_unprocessed_items
from utils.llm_helpers import initialize_gemini_client, SAFETY_SETTINGS

from utils.config_helpers import load_config
from utils.schemas import SuttaConcepts  


def run_extraction_pipeline(config):
    """Main pipeline function to orchestrate extraction process."""
    # --- Setup ---
    # Settings in config
    cfg = config['concept_extraction']
    out_cfg = config['output_paths']
    system_prompt = cfg['system_prompt']
    log_cfg = config['log_paths'] 

    # Client
    model_id = cfg['model_id']
    model_config = GenerateContentConfig(
        temperature=cfg['temperature'],
        safety_settings=SAFETY_SETTINGS,
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=SuttaConcepts,
    )
    client = initialize_gemini_client(cfg)

    # Load remaining suttas
    suttas_to_process = get_unprocessed_items(
        source_path=out_cfg['raw_data'],
        processed_path=out_cfg['raw_concepts'],
        source_id_key='sutta_id',
        processed_id_key='sutta_id'
    )

    # --- Main Loop ---
    # Initialize list for logging skipped items
    skipped_suttas_log = []

    # Initialize datetime string
    dt = datetime.now()
    dt = dt.strftime("%Y-%m-%d_%H-%M")

    for sutta in tqdm(suttas_to_process, desc="Extracting Concepts..."):
        sutta_identifier = sutta.get("sutta_id", "Unknown")

        try:
            sutta_body = sutta.get("body")
            if not sutta_body:
                reason = "No body text found"
                skipped_suttas_log.append({
                    "sutta_id": sutta_identifier,
                    'model_id': model_id,
                    'datetime': dt,
                    "reason": reason,
                })
                continue
                
            # Generate response
            response_text = client.models.generate_content(
                model=model_id,
                config=model_config,
                contents=sutta_body
                ).text
            
            # Simple cleaning
            parsed_data = SuttaConcepts.model_validate_json(response_text)

            result_record = {
                'sutta_id': sutta_identifier,
                'model_id': model_id,
                'datetime': dt,
                'extracted_data': parsed_data.model_dump()['concepts'],
            }

            with jsonlines.open(out_cfg['raw_concepts'], mode='a') as writer:
                writer.write(result_record)

            time.sleep(.01) 

        except (ValidationError, json.JSONDecodeError) as e:
            # Catches both malformed JSON and data that doesn't fit the schema
            reason = f"Schema validation failed: {e}"
            print(f"\nSKIPPING {sutta_identifier}: {reason}")
            skipped_suttas_log.append({
                "sutta_id": sutta_identifier,
                "reason": reason,
                "raw_response": response_text if 'response_text' in locals() else "N/A"
            })

        except Exception as e:
            skipped_suttas_log.append({
                "sutta_id": sutta_identifier,
                'model_id': model_id,
                'datetime': dt,
                "reason":  f"Processing error: {str(e)}",
            })
            print(f"An error occurred while processing {sutta_identifier}: {e}")
            if "RESOURCE_EXHAUSTED" in str(e):
                # If Resource Exhausted, interrupt function.
                print("Resource exhausted, wait to rerun.")
                sys.exit(1)

            time.sleep(1)

    # -- Logs --
    if skipped_suttas_log:
        skipped_log_path = log_cfg['skipped_concepts_log']
        print(f"\nINFO: {len(skipped_suttas_log)} suttas were skipped. Logging to {skipped_log_path}")
        with jsonlines.open(skipped_log_path, mode='a') as writer:
            writer.write_all(skipped_suttas_log)

if __name__ == "__main__":
    config_Dict = load_config()
    project_root = config_Dict['project_root']
    config = config_Dict['config']

    # Ensure data output directory exists
    raw_concepts_path = os.path.join(project_root, config['output_paths']['raw_concepts'])
    os.makedirs(os.path.dirname(raw_concepts_path), exist_ok=True)
    
    # Ensure log_path exists
    skipped_log_path = os.path.join(project_root, config['log_paths']['skipped_concepts_log'])
    os.makedirs(os.path.dirname(skipped_log_path), exist_ok=True)
    
    run_extraction_pipeline(config)
    
    print("\nConcept extraction process completed.")