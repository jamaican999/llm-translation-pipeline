"""
Glossary Manager - Handles loading, embedding, and retrieval of glossary terms
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple


class GlossaryManager:
    """Manages glossary terms with vector-based retrieval"""
    
    def __init__(self, glossary_path: str, embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the glossary manager
        
        Args:
            glossary_path: Path to the JSONL glossary file
            embedding_model: Name of the sentence transformer model
        """
        self.glossary_path = glossary_path
        self.glossary_entries = []
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.embeddings = None
        
        # Load and process glossary
        self._load_glossary()
        self._create_embeddings()
        self._build_index()
    
    def _load_glossary(self):
        """Load glossary from JSONL file"""
        with open(self.glossary_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    # Clean up NaN values
                    for key, value in entry.items():
                        if isinstance(value, float) and np.isnan(value):
                            entry[key] = None
                    self.glossary_entries.append(entry)
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(self.glossary_entries)} glossary entries")
    
    def _create_embeddings(self):
        """Create embeddings for all glossary terms"""
        # Create text representations for embedding
        texts = []
        for entry in self.glossary_entries:
            # Combine term with context for better retrieval
            term = entry['term_en']
            domain = entry.get('domain', '')
            pos = entry.get('part_of_speech', '')
            
            # Create rich text representation
            text_parts = [term]
            if domain:
                text_parts.append(f"domain: {domain}")
            if pos:
                text_parts.append(f"type: {pos}")
            
            text = " | ".join(text_parts)
            texts.append(text)
        
        print("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print(f"Created embeddings with shape: {self.embeddings.shape}")
    
    def _build_index(self):
        """Build FAISS index for fast similarity search"""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, target_lang: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve relevant glossary terms for a query
        
        Args:
            query: Source text segment
            target_lang: Target language code (fr, it, jp)
            top_k: Number of top results to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of relevant glossary entries with similarity scores
        """
        # Embed the query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Filter and format results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold:
                entry = self.glossary_entries[idx].copy()
                entry['similarity'] = float(sim)
                entry['target_translation'] = entry.get(target_lang, '')
                results.append(entry)
        
        return results
    
    def get_all_terms_for_language(self, target_lang: str) -> List[Dict]:
        """Get all glossary terms for a specific language"""
        terms = []
        for entry in self.glossary_entries:
            if entry.get(target_lang):
                terms.append({
                    'term_en': entry['term_en'],
                    'translation': entry[target_lang],
                    'dnt': entry.get('dnt', False),
                    'case_sensitive': entry.get('case_sensitive', False),
                    'criticality': entry.get('criticality_level', 'Medium')
                })
        return terms
    
    def format_glossary_for_prompt(self, retrieved_terms: List[Dict], target_lang: str) -> str:
        """
        Format retrieved glossary terms for LLM prompt
        
        Args:
            retrieved_terms: List of retrieved glossary entries
            target_lang: Target language code
            
        Returns:
            Formatted string for prompt injection
        """
        if not retrieved_terms:
            return ""
        
        lang_names = {'fr': 'French', 'it': 'Italian', 'jp': 'Japanese'}
        lang_name = lang_names.get(target_lang, target_lang)
        
        lines = [f"\nGLOSSARY TERMS (use these exact {lang_name} translations):"]
        
        for term in retrieved_terms:
            term_en = term['term_en']
            translation = term.get('target_translation', '')
            dnt = term.get('dnt', False)
            
            if translation:
                if dnt:
                    lines.append(f"- {term_en}: {translation} (DO NOT TRANSLATE - use as-is)")
                else:
                    lines.append(f"- {term_en}: {translation}")
        
        lines.append("\nIMPORTANT: Use the glossary translations exactly as specified above.")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the glossary manager
    manager = GlossaryManager("data/RideHailingGlossar_OneLine.jsonl")
    
    # Test retrieval
    test_query = "Please request a ride and check your driver rating"
    print(f"\nTest query: {test_query}")
    
    for lang in ['fr', 'it', 'jp']:
        print(f"\n--- {lang.upper()} ---")
        results = manager.retrieve(test_query, lang, top_k=5)
        for r in results:
            print(f"  {r['term_en']} -> {r['target_translation']} (sim: {r['similarity']:.3f})")
