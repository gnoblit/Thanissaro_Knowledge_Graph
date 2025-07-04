import jsonlines
from .base_normalizer import BaseNormalizer

class ConceptNormalizer(BaseNormalizer):
    """
    Handles the normalization of extracted concepts by implementing the
    data loading and path generation logic specific to concepts.
    """
    def __init__(self, cfg_manager):
        # The base class __init__ will handle all the setup.
        super().__init__(cfg_manager)
        
        # Get the extraction config, which is specific to concepts
        self.extract_config = self.config['concept_extraction']
        self.normalization_mode = self.norm_config['mode']

    def _get_config_key(self) -> str:
        """Specify the config section for concept normalization."""
        return "concept_normalization"

    def _get_output_path(self) -> str:
        """Construct the output path for concept clusters."""
        extraction_model_id = self.extract_config['model_id']
        s_extraction_model = extraction_model_id.replace('-', '_').replace('.', '')
        # FIX: Sanitize the embedding model ID for the path as well for consistency
        s_embedding_model = self.embedding_model_id.replace('/', '_').replace('-', '_')

        format_args = {
            'extraction_model_id': s_extraction_model,
            'normalization_mode': self.normalization_mode,
            'embedding_model_id': s_embedding_model
        }
        return self.cfg_manager.get_path('concept_normalization.output_path_template', format_args)

    def _prepare_corpus(self) -> tuple[list, dict]:
        """
        Load concepts and prepare the corpus for embedding based on the normalization mode.
        
        - In 'name' mode, it deduplicates concepts by name to embed each unique name once.
        - In 'hybrid' mode, it uses ALL concepts, as the combination of name and
          evidence quote is considered for clustering.
        """
        # 1. Get input path
        format_args = {
            'mode': self.extract_config['mode'], 
            'model_id': self.extract_config['model_id'].replace('-', '_').replace('.', '')
        }
        input_path = self.cfg_manager.get_path('concept_extraction.output_path_template', format_args)
        
        print(f"Loading concepts from {input_path}...")
        print(f"Preparing corpus in '{self.normalization_mode}' mode...")

        corpus = []
        concepts_to_process = []

        # --- FIX START ---
        if self.normalization_mode == 'hybrid':
            # In hybrid mode, we do NOT deduplicate. We process every concept instance
            # because the evidence quote provides crucial context for clustering.
            with jsonlines.open(input_path) as reader:
                for sutta_record in reader:
                    concepts_to_process.extend(sutta_record.get('concepts', []))
            
            print(f"Found {len(concepts_to_process)} total concept instances to process for hybrid mode.")
            corpus = [f"{c['concept_name']} [SEP] {c['evidence_quote']}" for c in concepts_to_process]

        elif self.normalization_mode == 'name':
            # In name mode, we deduplicate by concept_name to find canonical concepts.
            # The original logic is correct for this mode.
            unique_concepts = {}
            with jsonlines.open(input_path) as reader:
                for sutta_record in reader:
                    for concept in sutta_record.get('concepts', []):
                        if concept['concept_name'] not in unique_concepts:
                            unique_concepts[concept['concept_name']] = concept
            
            concepts_to_process = list(unique_concepts.values())
            print(f"Found {len(concepts_to_process)} unique concept names to process for name mode.")
            corpus = [c['concept_name'] for c in concepts_to_process]
        
        else:
            raise ValueError(f"Invalid normalization mode: {self.normalization_mode}")
        # --- FIX END ---
            
        # 4. Map corpus index back to the original concept object
        concept_map = {i: concept for i, concept in enumerate(concepts_to_process)}
        return corpus, concept_map