"""
Translation Pipeline - LLM-based translation with optional glossary retrieval
"""

from openai import OpenAI
from typing import Dict, List, Optional
import time
from glossary_manager import GlossaryManager


class TranslationPipeline:
    """Translation pipeline with glossary-aware prompting"""
    
    def __init__(self, glossary_manager: Optional[GlossaryManager] = None, model: str = "gpt-4.1-mini"):
        """
        Initialize translation pipeline
        
        Args:
            glossary_manager: Optional glossary manager for retrieval
            model: OpenAI model to use for translation
        """
        self.client = OpenAI()
        self.model = model
        self.glossary_manager = glossary_manager
        
    def translate_baseline(self, source_text: str, target_lang: str) -> Dict:
        """
        Baseline translation without glossary retrieval
        
        Args:
            source_text: Source text in English
            target_lang: Target language code (fr, it, jp)
            
        Returns:
            Dictionary with translation and metadata
        """
        lang_names = {
            'fr': 'French',
            'it': 'Italian', 
            'jp': 'Japanese'
        }
        target_lang_name = lang_names.get(target_lang, target_lang)
        
        prompt = f"""Translate the following English text to {target_lang_name}.

Provide a natural, fluent translation that preserves the meaning and tone of the original text.

Source text: {source_text}

Translation:"""
        
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a professional translator specializing in English to {target_lang_name} translation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        latency = time.time() - start_time
        translation = response.choices[0].message.content.strip()
        
        return {
            'translation': translation,
            'source': source_text,
            'target_lang': target_lang,
            'mode': 'baseline',
            'latency': latency,
            'tokens': {
                'prompt': response.usage.prompt_tokens,
                'completion': response.usage.completion_tokens,
                'total': response.usage.total_tokens
            }
        }
    
    def translate_with_retrieval(self, source_text: str, target_lang: str, 
                                 top_k: int = 5, threshold: float = 0.3) -> Dict:
        """
        Enhanced translation with glossary retrieval
        
        Args:
            source_text: Source text in English
            target_lang: Target language code (fr, it, jp)
            top_k: Number of glossary terms to retrieve
            threshold: Similarity threshold for retrieval
            
        Returns:
            Dictionary with translation and metadata
        """
        if not self.glossary_manager:
            raise ValueError("Glossary manager not initialized")
        
        lang_names = {
            'fr': 'French',
            'it': 'Italian',
            'jp': 'Japanese'
        }
        target_lang_name = lang_names.get(target_lang, target_lang)
        
        # Retrieve relevant glossary terms
        start_retrieval = time.time()
        retrieved_terms = self.glossary_manager.retrieve(source_text, target_lang, top_k, threshold)
        retrieval_time = time.time() - start_retrieval
        
        # Format glossary for prompt
        glossary_text = self.glossary_manager.format_glossary_for_prompt(retrieved_terms, target_lang)
        
        prompt = f"""Translate the following English text to {target_lang_name}.
{glossary_text}

INSTRUCTIONS:
- Use the glossary translations exactly as specified
- Maintain natural fluency and grammar in {target_lang_name}
- Preserve any HTML tags or special formatting unchanged
- For terms marked "DO NOT TRANSLATE", keep them as specified in the glossary

Source text: {source_text}

Translation:"""
        
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a professional translator specializing in English to {target_lang_name} translation with expertise in ride-hailing terminology."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        llm_latency = time.time() - start_time
        total_latency = retrieval_time + llm_latency
        translation = response.choices[0].message.content.strip()
        
        return {
            'translation': translation,
            'source': source_text,
            'target_lang': target_lang,
            'mode': 'with_retrieval',
            'retrieved_terms': retrieved_terms,
            'latency': total_latency,
            'retrieval_latency': retrieval_time,
            'llm_latency': llm_latency,
            'tokens': {
                'prompt': response.usage.prompt_tokens,
                'completion': response.usage.completion_tokens,
                'total': response.usage.total_tokens
            }
        }
    
    def translate_batch(self, segments: List[Dict], mode: str = 'both') -> List[Dict]:
        """
        Translate a batch of segments
        
        Args:
            segments: List of dicts with 'text' and 'target_lang' keys
            mode: 'baseline', 'with_retrieval', or 'both'
            
        Returns:
            List of translation results
        """
        results = []
        
        for i, segment in enumerate(segments):
            source_text = segment['text']
            target_lang = segment['target_lang']
            
            print(f"Translating segment {i+1}/{len(segments)} to {target_lang}...")
            
            result = {
                'segment_id': i,
                'source': source_text,
                'target_lang': target_lang
            }
            
            if mode in ['baseline', 'both']:
                baseline_result = self.translate_baseline(source_text, target_lang)
                result['baseline'] = baseline_result
            
            if mode in ['with_retrieval', 'both'] and self.glossary_manager:
                retrieval_result = self.translate_with_retrieval(source_text, target_lang)
                result['with_retrieval'] = retrieval_result
            
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test the translation pipeline
    print("Initializing glossary manager...")
    glossary = GlossaryManager("data/RideHailingGlossar_OneLine.jsonl")
    
    print("\nInitializing translation pipeline...")
    pipeline = TranslationPipeline(glossary_manager=glossary)
    
    # Test translation
    test_text = "Request a ride and check your driver rating"
    
    print(f"\nTest translation: {test_text}")
    print("\n--- Baseline ---")
    baseline = pipeline.translate_baseline(test_text, 'fr')
    print(f"FR: {baseline['translation']}")
    
    print("\n--- With Retrieval ---")
    enhanced = pipeline.translate_with_retrieval(test_text, 'fr')
    print(f"FR: {enhanced['translation']}")
    print(f"Retrieved terms: {[t['term_en'] for t in enhanced['retrieved_terms']]}")
