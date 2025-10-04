"""
Improved Evaluator - Better term matching with fuzzy logic
"""

import re
from typing import Dict, List
import json
from difflib import SequenceMatcher


class ImprovedEvaluator:
    """Evaluates translation quality with improved term matching"""
    
    def __init__(self, glossary_manager):
        """
        Initialize evaluator
        
        Args:
            glossary_manager: GlossaryManager instance for term lookup
        """
        self.glossary_manager = glossary_manager
        
        # Build term lookup by language
        self.term_lookup = {}
        for lang in ['fr', 'it', 'jp']:
            self.term_lookup[lang] = {}
            for entry in glossary_manager.glossary_entries:
                term_en = entry['term_en']
                term_target = entry.get(lang, '')
                if term_target:
                    self.term_lookup[lang][term_en.lower()] = {
                        'translation': term_target,
                        'dnt': entry.get('dnt', False),
                        'case_sensitive': entry.get('case_sensitive', False)
                    }
    
    def find_relevant_terms_in_source(self, source_text: str) -> List[str]:
        """
        Find which glossary terms are actually present in the source text
        
        Args:
            source_text: Source English text
            
        Returns:
            List of glossary term keys that appear in the source
        """
        source_lower = source_text.lower()
        found_terms = []
        
        # Check all glossary entries
        for entry in self.glossary_manager.glossary_entries:
            term_en = entry['term_en']
            term_lower = term_en.lower()
            
            # Check for exact match or word boundary match
            if term_lower in source_lower:
                # Verify it's a word boundary match, not substring
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                if re.search(pattern, source_lower):
                    found_terms.append(term_en)
        
        return found_terms
    
    def fuzzy_match(self, target_term: str, translation: str, threshold: float = 0.85) -> bool:
        """
        Check if target term appears in translation with fuzzy matching
        
        Args:
            target_term: Expected translation term
            translation: Full translation text
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if term is found
        """
        translation_lower = translation.lower()
        target_lower = target_term.lower()
        
        # Exact substring match
        if target_lower in translation_lower:
            return True
        
        # Check for word-level matches
        target_words = target_lower.split()
        translation_words = translation_lower.split()
        
        # For single-word terms
        if len(target_words) == 1:
            for trans_word in translation_words:
                similarity = SequenceMatcher(None, target_lower, trans_word).ratio()
                if similarity >= threshold:
                    return True
        else:
            # For multi-word terms, check if all key words are present
            key_words = [w for w in target_words if len(w) > 2]
            if key_words:
                matches = sum(1 for kw in key_words if any(
                    SequenceMatcher(None, kw, tw).ratio() >= threshold 
                    for tw in translation_words
                ))
                if matches >= len(key_words) * 0.7:  # At least 70% of key words
                    return True
        
        return False
    
    def calculate_term_accuracy(self, source_text: str, translation: str, target_lang: str) -> Dict:
        """
        Calculate term accuracy based on actual terms in source
        
        Args:
            source_text: Source English text
            translation: Translated text
            target_lang: Target language code
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Find which glossary terms are actually in the source
        relevant_terms = self.find_relevant_terms_in_source(source_text)
        
        if not relevant_terms:
            return {
                'total_terms': 0,
                'found_terms': 0,
                'accuracy': 1.0,  # No terms to check = perfect
                'missing_terms': [],
                'found_details': [],
                'relevant_terms': []
            }
        
        # Check each relevant term
        found_terms = []
        missing_terms = []
        
        for term_en in relevant_terms:
            term_info = self.term_lookup[target_lang].get(term_en.lower())
            if not term_info:
                continue
            
            target_trans = term_info['translation']
            
            # Check if translation contains the term
            if self.fuzzy_match(target_trans, translation):
                found_terms.append({
                    'term_en': term_en,
                    'term_target': target_trans,
                    'found': True
                })
            else:
                missing_terms.append({
                    'term_en': term_en,
                    'term_target': target_trans,
                    'found': False
                })
        
        total = len(relevant_terms)
        found = len(found_terms)
        accuracy = found / total if total > 0 else 1.0
        
        return {
            'total_terms': total,
            'found_terms': found,
            'accuracy': accuracy,
            'missing_terms': missing_terms,
            'found_details': found_terms,
            'relevant_terms': relevant_terms
        }
    
    def evaluate_result(self, result: Dict) -> Dict:
        """
        Evaluate a translation result
        
        Args:
            result: Translation result dictionary
            
        Returns:
            Evaluation metrics
        """
        source = result['source']
        target_lang = result['target_lang']
        
        evaluation = {
            'segment_id': result['segment_id'],
            'source': source,
            'target_lang': target_lang
        }
        
        # Evaluate baseline
        if 'baseline' in result:
            baseline_trans = result['baseline']['translation']
            baseline_metrics = self.calculate_term_accuracy(source, baseline_trans, target_lang)
            evaluation['baseline'] = {
                'translation': baseline_trans,
                'metrics': baseline_metrics,
                'latency': result['baseline']['latency'],
                'tokens': result['baseline']['tokens']
            }
        
        # Evaluate with retrieval
        if 'with_retrieval' in result:
            retrieval_trans = result['with_retrieval']['translation']
            retrieval_metrics = self.calculate_term_accuracy(source, retrieval_trans, target_lang)
            evaluation['with_retrieval'] = {
                'translation': retrieval_trans,
                'metrics': retrieval_metrics,
                'retrieved_terms': result['with_retrieval']['retrieved_terms'],
                'latency': result['with_retrieval']['latency'],
                'retrieval_latency': result['with_retrieval']['retrieval_latency'],
                'llm_latency': result['with_retrieval']['llm_latency'],
                'tokens': result['with_retrieval']['tokens']
            }
        
        return evaluation
    
    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """
        Evaluate a batch of translation results
        
        Args:
            results: List of translation results
            
        Returns:
            Aggregated evaluation metrics
        """
        evaluations = []
        
        for result in results:
            eval_result = self.evaluate_result(result)
            evaluations.append(eval_result)
        
        # Aggregate metrics
        baseline_accuracies = []
        retrieval_accuracies = []
        baseline_latencies = []
        retrieval_latencies = []
        baseline_tokens = []
        retrieval_tokens = []
        retrieval_latencies_only = []
        
        for eval_result in evaluations:
            if 'baseline' in eval_result:
                baseline_accuracies.append(eval_result['baseline']['metrics']['accuracy'])
                baseline_latencies.append(eval_result['baseline']['latency'])
                baseline_tokens.append(eval_result['baseline']['tokens']['total'])
            
            if 'with_retrieval' in eval_result:
                retrieval_accuracies.append(eval_result['with_retrieval']['metrics']['accuracy'])
                retrieval_latencies.append(eval_result['with_retrieval']['latency'])
                retrieval_tokens.append(eval_result['with_retrieval']['tokens']['total'])
                retrieval_latencies_only.append(eval_result['with_retrieval']['retrieval_latency'])
        
        summary = {
            'total_segments': len(evaluations),
            'baseline': {
                'mean_accuracy': sum(baseline_accuracies) / len(baseline_accuracies) if baseline_accuracies else 0,
                'mean_latency': sum(baseline_latencies) / len(baseline_latencies) if baseline_latencies else 0,
                'mean_tokens': sum(baseline_tokens) / len(baseline_tokens) if baseline_tokens else 0,
                'perfect_segments': sum(1 for acc in baseline_accuracies if acc == 1.0)
            },
            'with_retrieval': {
                'mean_accuracy': sum(retrieval_accuracies) / len(retrieval_accuracies) if retrieval_accuracies else 0,
                'mean_latency': sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0,
                'mean_tokens': sum(retrieval_tokens) / len(retrieval_tokens) if retrieval_tokens else 0,
                'mean_retrieval_latency': sum(retrieval_latencies_only) / len(retrieval_latencies_only) if retrieval_latencies_only else 0,
                'perfect_segments': sum(1 for acc in retrieval_accuracies if acc == 1.0)
            },
            'detailed_evaluations': evaluations
        }
        
        # Calculate improvement
        if baseline_accuracies and retrieval_accuracies:
            summary['improvement'] = {
                'accuracy_gain': summary['with_retrieval']['mean_accuracy'] - summary['baseline']['mean_accuracy'],
                'accuracy_gain_pct': ((summary['with_retrieval']['mean_accuracy'] - summary['baseline']['mean_accuracy']) 
                                     / summary['baseline']['mean_accuracy'] * 100) if summary['baseline']['mean_accuracy'] > 0 else 0,
                'latency_overhead': summary['with_retrieval']['mean_latency'] - summary['baseline']['mean_latency'],
                'token_overhead': summary['with_retrieval']['mean_tokens'] - summary['baseline']['mean_tokens'],
                'perfect_segment_gain': summary['with_retrieval']['perfect_segments'] - summary['baseline']['perfect_segments']
            }
        
        return summary
    
    def generate_comparison_report(self, evaluation_summary: Dict, output_path: str):
        """Generate a detailed comparison report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Translation Quality Comparison Report\n\n")
            f.write("## Executive Summary\n\n")
            
            baseline = evaluation_summary['baseline']
            retrieval = evaluation_summary['with_retrieval']
            improvement = evaluation_summary.get('improvement', {})
            
            f.write(f"This report evaluates the impact of glossary retrieval on LLM translation quality across {evaluation_summary['total_segments']} segments in three language pairs (EN→FR, EN→IT, EN→JP).\n\n")
            
            f.write("### Key Findings\n\n")
            f.write(f"**Baseline Translation (No Glossary)**\n")
            f.write(f"- Term accuracy: {baseline['mean_accuracy']:.1%}\n")
            f.write(f"- Perfect segments: {baseline['perfect_segments']}/{evaluation_summary['total_segments']}\n")
            f.write(f"- Average latency: {baseline['mean_latency']:.3f}s\n")
            f.write(f"- Average tokens: {baseline['mean_tokens']:.0f}\n\n")
            
            f.write(f"**With Glossary Retrieval**\n")
            f.write(f"- Term accuracy: {retrieval['mean_accuracy']:.1%}\n")
            f.write(f"- Perfect segments: {retrieval['perfect_segments']}/{evaluation_summary['total_segments']}\n")
            f.write(f"- Average latency: {retrieval['mean_latency']:.3f}s\n")
            f.write(f"- Average tokens: {retrieval['mean_tokens']:.0f}\n")
            f.write(f"- Average retrieval time: {retrieval['mean_retrieval_latency']:.3f}s\n\n")
            
            if improvement:
                f.write(f"**Impact of Glossary Retrieval**\n")
                f.write(f"- Accuracy improvement: +{improvement['accuracy_gain']:.1%} ({improvement['accuracy_gain_pct']:.1f}% relative)\n")
                f.write(f"- Additional perfect segments: +{improvement['perfect_segment_gain']}\n")
                f.write(f"- Latency overhead: +{improvement['latency_overhead']:.3f}s ({improvement['latency_overhead']/baseline['mean_latency']*100:.1f}%)\n")
                f.write(f"- Token overhead: +{improvement['token_overhead']:.0f} tokens ({improvement['token_overhead']/baseline['mean_tokens']*100:.1f}%)\n\n")
            
            # Detailed examples
            f.write("## Sample Translations\n\n")
            
            for i, eval_result in enumerate(evaluation_summary['detailed_evaluations'][:10]):
                f.write(f"### Example {i+1}: {eval_result['target_lang'].upper()}\n\n")
                f.write(f"**Source:** {eval_result['source']}\n\n")
                
                if 'baseline' in eval_result:
                    metrics = eval_result['baseline']['metrics']
                    f.write(f"**Baseline:** {eval_result['baseline']['translation']}\n")
                    f.write(f"- Term accuracy: {metrics['accuracy']:.0%} ({metrics['found_terms']}/{metrics['total_terms']} terms)\n")
                    if metrics['relevant_terms']:
                        f.write(f"- Relevant terms: {', '.join(metrics['relevant_terms'])}\n")
                    if metrics['missing_terms']:
                        f.write(f"- Missing: {', '.join([t['term_en'] for t in metrics['missing_terms']])}\n")
                    f.write("\n")
                
                if 'with_retrieval' in eval_result:
                    metrics = eval_result['with_retrieval']['metrics']
                    f.write(f"**With Retrieval:** {eval_result['with_retrieval']['translation']}\n")
                    f.write(f"- Term accuracy: {metrics['accuracy']:.0%} ({metrics['found_terms']}/{metrics['total_terms']} terms)\n")
                    if metrics['relevant_terms']:
                        f.write(f"- Relevant terms: {', '.join(metrics['relevant_terms'])}\n")
                    if metrics['missing_terms']:
                        f.write(f"- Missing: {', '.join([t['term_en'] for t in metrics['missing_terms']])}\n")
                    f.write(f"- Retrieved {len(eval_result['with_retrieval']['retrieved_terms'])} glossary entries\n")
                    f.write("\n")
                
                f.write("---\n\n")


if __name__ == "__main__":
    from glossary_manager import GlossaryManager
    
    # Test improved evaluator
    glossary = GlossaryManager("data/RideHailingGlossar_OneLine.jsonl")
    evaluator = ImprovedEvaluator(glossary)
    
    # Test term finding
    test_source = "Request a ride and check your driver rating"
    relevant = evaluator.find_relevant_terms_in_source(test_source)
    print(f"Relevant terms in '{test_source}': {relevant}")
    
    # Test accuracy calculation
    test_translation_fr = "Demandez une course et vérifiez la note de votre conducteur"
    metrics = evaluator.calculate_term_accuracy(test_source, test_translation_fr, 'fr')
    print(f"\nTerm accuracy: {metrics['accuracy']:.0%}")
    print(f"Found: {metrics['found_terms']}/{metrics['total_terms']}")
    print(f"Details: {metrics}")
