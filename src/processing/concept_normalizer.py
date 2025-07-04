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
        s_extraction_model = extraction_model_id.replace('/', '_').replace('-', '_').replace('.', '')
        
        # FIX: Sanitize the embedding model ID to ensure consistent and safe filenames.
        # This replaces both '/' (common in Hugging Face model IDs) and '-' with '_'.
        s_embedding_model = self.embedding_model_id.replace('/', '_').replace('-', '_').replace('.', '')

        format_args = {
            'extraction_model_id': s_extraction_model,
            'normalization_mode': self.normalization_mode,
            'embedding_model_id': s_embedding_model
        }
        return self.cfg_manager.get_path('concept_normalization.output_path_template', format_args)

    def _prepare_corpus(self) -> tuple[list, dict]:
        """Load concepts, deduplicate, and prepare the corpus for embedding."""
        # 1. Get input path
        format_args = {
            'mode': self.extract_config['mode'], 
            'model_id': self.extract_config['model_id'].replace('/', '_').replace('-', '_').replace('.', '')
        }
        input_path = self.cfg_manager.get_path('concept_extraction.output_path_template', format_args)
        
        # 2. Load and deduplicate concepts
        print(f"Loading concepts from {input_path}...")
        all_concepts = []
        with jsonlines.open(input_path) as reader:
            for sutta_record in reader:
                all_concepts.extend(sutta_record.get('concepts', []))
        
        print(f"Found {len(all_concepts)} concept instances to process.")
                
        # 3. Prepare corpus based on mode
        print(f"Preparing corpus in '{self.normalization_mode}' mode...")
        corpus = []
        if self.normalization_mode == 'name':
            corpus = [c['concept_name'] for c in all_concepts]
        elif self.normalization_mode == 'hybrid':
            corpus = [f"{c['concept_name']} [SEP] {c['evidence_quote']}" for c in all_concepts]
        else:
            raise ValueError(f"Invalid normalization mode: {self.normalization_mode}")
            
        # 4. Map corpus index back to the original concept object
        concept_map = {i: concept for i, concept in enumerate(all_concepts)}
        return corpus, concept_map