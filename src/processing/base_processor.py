import os
import time
import jsonlines
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime

from utils.data_helpers import get_processed_ids, get_unprocessed_items

class BaseProcessor(ABC):
    """
    Abstract base class for a standard data processing pipeline step.
    Handles loading data, checking for previously processed items,
    iterating, saving, and logging.
    """
    def __init__(self, cfg_manager):
        self.cfg_manager = cfg_manager
        self.config = cfg_manager.config
        
        # Paths that are common to most processors
        self.output_path = self._get_output_path()
        self.log_path = self._get_log_path()
        self.source_path = self._get_source_path()

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    # --- Abstract methods for subclasses to implement ---
    @abstractmethod
    def _get_source_path(self) -> str:
        """Return the absolute path to the input data file."""
        pass

    @abstractmethod
    def _get_output_path(self) -> str:
        """Return the absolute path for the processed output."""
        pass

    @abstractmethod
    def _get_log_path(self) -> str:
        """Return the absolute path for the log file."""
        pass
        
    @abstractmethod
    def _get_run_config(self) -> dict:
        """Return a dictionary of config values that define a unique run (e.g., model, mode)."""
        pass

    @abstractmethod
    def _process_item(self, item: dict) -> dict:
        """
        Perform the core processing logic on a single item.
        Should raise an exception on failure.
        Returns the result record to be saved.
        """
        pass

    # --- Concrete methods provided by the base class ---
    def _load_unprocessed_items(self) -> list:
        """Loads and filters data to find unprocessed items."""
        run_config = self._get_run_config()
        processed_ids = get_processed_ids(
            processed_path=self.output_path,
            id_key='sutta_id', # Or make this configurable
            **run_config
        )
        return get_unprocessed_items(
            source_path=self.source_path,
            source_id_key='sutta_id', # Or make this configurable
            processed_ids_set=processed_ids
        )

    def run_pipeline(self):
        """Executes the full, generic processing pipeline."""
        items_to_process = self._load_unprocessed_items()
        
        if not items_to_process:
            print("No new items to process. Exiting.")
            return

        skipped_items_log = []
        
        for item in tqdm(items_to_process, desc=f"Processing ({self.__class__.__name__})"):
            item_id = item.get("sutta_id", "Unknown")
            try:
                result_record = self._process_item(item)
                
                with jsonlines.open(self.output_path, mode='a') as writer:
                    writer.write(result_record)
                
                time.sleep(0.5) # Optional: rate limiting

            except Exception as e:
                # A more generic error handler
                print(f"\nSKIPPING {item_id}: {e}")
                skipped_items_log.append({"item_id": item_id, "reason": str(e)})

        if skipped_items_log:
            print(f"\nINFO: {len(skipped_items_log)} items were skipped. Logging to {self.log_path}")
            with jsonlines.open(self.log_path, mode='a') as writer:
                writer.write_all(skipped_items_log)