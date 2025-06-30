import os
import sys
import time
import json
import argparse
from datetime import datetime

import jsonlines
from tqdm import tqdm
from pydantic import ValidationError
from google import genai
from google.genai.types import GenerateContentConfig
from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting

from utils.config_helpers import load_config
from utils.data_helpers import get_unprocessed_items
from utils.schemas import SuttaConceptsDiscovery, SuttaConceptsFixed

def run_extraction_pipeline(strategy, config, project_root):
    """
    Main pipeline function to orchestrate the concept extraction process.
    """
    # --- 1. Get Configuration ---
    extraction_config = config['concept_extraction']
    model_id = extraction_config['model_id']
    
    print(f"--- Running Concept Extraction in '{strategy.upper()}' Mode ---")

    # --- 2. Dynamically Select Prompt and Generate Paths ---
    
    # Select the system prompt dynamically based on the strategy
    prompt_key = f"{strategy}_prompt"
    if prompt_key not in extraction_config:
        raise KeyError(f"Prompt key '{prompt_key}' not found in settings.yaml under concept_extraction.")
    system_prompt = extraction_config[prompt_key]

    # Generate paths dynamically from templates
    output_template = extraction_config['output_path_template']
    log_template = extraction_config['log_path_template']
    
    output_path = os.path.join(project_root, output_template.format(mode=strategy))
    log_path = os.path.join(project_root, log_template.format(mode=strategy))

    # Select the correct Pydantic schema class
    response_schema_class = SuttaConceptsDiscovery if strategy == 'discovery' else SuttaConceptsFixed

    # --- 3. Configure Gemini Client and Model ---
    # Define Satefy Settings
    SAFETY_SETTINGS = [
    SafetySetting(category=c, threshold=HarmBlockThreshold.BLOCK_NONE)
    for c in [
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT]
    ]
    
    model_config = GenerateContentConfig(
        temperature=extraction_config['temperature'],
        safety_settings=SAFETY_SETTINGS,
        response_schema=response_schema_class,
        response_mime_type="application/json",
        system_instruction=system_prompt
    )
    # Initialize Client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # --- 4. Load Data ---
    raw_data_path = os.path.join(project_root, config['output_paths']['raw_data'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    suttas_to_process = get_unprocessed_items(
        source_path=raw_data_path,
        processed_path=output_path,
        source_id_key='sutta_id',
        processed_id_key='sutta_id'
    )

    skipped_suttas_log = []
    dt = datetime.now()
    dt = dt.strftime("%Y-%m-%d_%H-%M")

    # --- 5. Main Extraction Loop ---
    for sutta in tqdm(suttas_to_process, desc=f"Extracting ({strategy} mode)"):
        sutta_identifier = sutta.get("sutta_id", "Unknown")
        response_text = None
        try:
            sutta_body = sutta.get("body")
            if not sutta_body or not sutta_body.strip():
                raise ValueError("Sutta body is empty.")
            
            response_text = client.models.generate_content(
                model=model_id,
                config=model_config,
                contents=sutta_body
                ).text
            parsed_data = response_schema_class.model_validate_json(response_text)
            
            result_record = {
                'sutta_id': sutta_identifier, 'model_id': model_id,
                'time_of_run': dt,
                'concepts': parsed_data.model_dump()['concepts'],
                'mode': strategy
            }
            with jsonlines.open(output_path, mode='a') as writer:
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
                    with jsonlines.open(log_path, mode='a') as writer:
                        writer.write_all(skipped_suttas_log)
                sys.exit(1)
            else:
                skipped_suttas_log.append({"sutta_id": sutta_identifier, "reason": f"Processing error: {error_string}", "raw_response": response_text})
                time.sleep(1)

    # --- 6. Final Logging ---
    if skipped_suttas_log:
        print(f"\nINFO: {len(skipped_suttas_log)} suttas were skipped. Logging to {log_path}")
        with jsonlines.open(log_path, mode='a') as writer:
            writer.write_all(skipped_suttas_log)

def main():
    """Parses command-line arguments to select and run an extraction strategy."""
    parser = argparse.ArgumentParser(description="Run concept extraction with a specified strategy.")
    parser.add_argument("strategy", choices=['discovery', 'fixed'], help="The extraction strategy to use: 'discovery' or 'fixed'.")
    args = parser.parse_args()
    config_dict = load_config()
    run_extraction_pipeline(args.strategy, config_dict['config'], config_dict['project_root'])
    print(f"\nConcept extraction process ('{args.strategy}' mode) completed.")

if __name__ == "__main__":
    main()