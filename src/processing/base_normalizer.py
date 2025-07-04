import os
import json
import jsonlines
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer, util

class BaseNormalizer(ABC):
    """
    Abstract base class for normalization processes that use embedding and clustering.
    
    This class handles the shared logic of loading a model, generating embeddings,
    clustering, and saving results. Subclasses must implement the logic specific
    to the data they are normalizing (e.g., concepts or relationships).
    """
    def __init__(self, cfg_manager):
        self.cfg_manager = cfg_manager
        self.config = cfg_manager.config
        
        # Subclasses will define which config key to use (e.g., 'concept_normalization')
        self.norm_config = self.config[self._get_config_key()]
        
        # Common parameters
        self.embedding_model_id = self.norm_config['embedding_model_id']
        self.min_community_size = self.norm_config['min_community_size']
        self.threshold = self.norm_config['threshold']
        
        # Initialize the embedding model
        print(f"Loading embedding model: {self.embedding_model_id}...")
        self.model = SentenceTransformer(self.embedding_model_id)
        print("Model loaded.")

    @abstractmethod
    def _get_config_key(self) -> str:
        """Return the key for the relevant section in settings.yaml."""
        pass

    @abstractmethod
    def _get_output_path(self) -> str:
        """Construct the specific output path for the normalization results."""
        pass

    @abstractmethod
    def _prepare_corpus(self) -> tuple[list, dict]:
        """
        Load the source data, deduplicate, and prepare the corpus for embedding.
        
        Returns:
            A tuple containing (corpus_texts, item_map), where item_map maps
            the corpus index back to the original data item.
        """
        pass

    def run_pipeline(self):
        """Executes the full, generic normalization pipeline."""
        print(f"--- Running Normalization for '{self._get_config_key()}' ---")
        
        # 1. Prepare data using subclass-specific logic
        corpus, item_map = self._prepare_corpus()
        
        # 2. Generate embeddings (shared logic)
        embeddings = self._generate_embeddings(corpus)
        
        # 3. Cluster items (shared logic)
        clusters = self._cluster_items(embeddings, item_map)
        
        # 4. Save results (shared logic)
        output_path = self._get_output_path()
        self._save_clusters(clusters, output_path)
        
        print(f"\nNormalization complete. Found {len(clusters)} clusters.")
        print(f"Results saved to: {output_path}")

    def _generate_embeddings(self, corpus: list):
        """Generates embeddings for the given text corpus."""
        print(f"Generating embeddings for {len(corpus)} items...")
        return self.model.encode(
            corpus, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )

    def _cluster_items(self, embeddings, item_map: dict):
        """Performs community detection to cluster items."""
        print("Clustering items using community detection...")
        clusters_indices = util.community_detection(
            embeddings, 
            min_community_size=self.min_community_size, 
            threshold=self.threshold
        )
        
        # Map indices back to full item objects
        final_clusters = []
        for cluster in clusters_indices:
            cluster_items = [item_map[idx] for idx in cluster]
            final_clusters.append(cluster_items)
            
        return final_clusters

    def _save_clusters(self, clusters: list, output_path: str):
        """Saves the final clusters to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clusters, f, indent=2, ensure_ascii=False)