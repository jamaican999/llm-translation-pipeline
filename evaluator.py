"""
Evaluator - Metrics and analysis for translation quality
"""

import re
from typing import Dict, List
import json


class TranslationEvaluator:
    """Evaluates translation quality with focus on term accuracy"""
    
    def __init__(self, glossary_manager):
        """
        Initialize evaluator
        
        Args:
            glossary_manager: GlossaryManager instance for term lookup
        """
        self.glossary_manager = glossary_manager
    
    def calculate_term_accuracy(self, translation: str, target_lang: str, 
                                expected_terms: List[str]) -> Dict:
        """
        Calculate term accuracy for a translation
        
        Args:
            translation: Translated text
            target_lang: Target language code
            expected_terms: List of expected English terms that should appear
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Get glossary translations for expected terms
        glossary_terms = {}
        for entry in self.glossary_manager.glossary_entries:
            term_en = entry['term_en'].lower()
            if term_en in [t.lower() for t in expected_terms]:
                target_trans = entry.get(target_lang, '')
                if target_trans:
                    glossary_terms[term_en] = target_trans
        
        if not glossary_terms:
            return {
                'total_terms': 0,
                'found_terms': 0,
                'accuracy': 0.0,
                'missing_terms': [],
                'found_details': []
            }
        
        # Check which terms appear in translation
        translation_lower = translation.lower()
        found_terms = []
        missing_terms = []
        
        for term_en, term_target in glossary_terms.items():
            # Check if the target translation appears in the translated text
            # Handle case-insensitive matching for most terms
            term_target_lower = term_target.lower()
            
            if term_target_lower in translation_lower:
                found_terms.append({
                    'term_en': term_en,
                    'term_target': term_target,
                    'found': True
                })
            else:
                # Check for partial matches or variations
                # Split multi-word terms and check if key words are present
                key_words = [w for w in term_target_lower.split() if len(w) > 3]
                if key_words and any(kw in translation_lower for kw in key_words):
                    found_terms.append({
                        'term_en': term_en,
                        'term_target': term_target,
                        'found': True,
                        'partial': True
                    })
                else:
                    missing_terms.append({
                        'term_en': term_en,
                        'term_target': term_target,
                        'found': False
                    })
        
        total = len(glossary_terms)
        found = len(found_terms)
        accuracy = found / total if total > 0 else 0.0
        
        return {
            'total_terms': total,
            'found_terms': found,
            'accuracy': accuracy,
            'missing_terms': missing_terms,
            'found_details': found_terms
        }
    
    def evaluate_result(self, result: Dict, expected_terms: List[str]) -> Dict:
        """
        Evaluate a translation result (with both baseline and retrieval modes)
        
        Args:
            result: Translation result dictionary
            expected_terms: List of expected English terms
            
        Returns:
            Evaluation metrics
        """
        target_lang = result['target_lang']
        evaluation = {
            'segment_id': result['segment_id'],
            'source': result['source'],
            'target_lang': target_lang,
            'expected_terms': expected_terms
        }
        
        # Evaluate baseline
        if 'baseline' in result:
            baseline_trans = result['baseline']['translation']
            baseline_metrics = self.calculate_term_accuracy(
                baseline_trans, target_lang, expected_terms
            )
            evaluation['baseline'] = {
                'translation': baseline_trans,
                'metrics': baseline_metrics,
                'latency': result['baseline']['latency'],
                'tokens': result['baseline']['tokens']
            }
        
        # Evaluate with retrieval
        if 'with_retrieval' in result:
            retrieval_trans = result['with_retrieval']['translation']
            retrieval_metrics = self.calculate_term_accuracy(
                retrieval_trans, target_lang, expected_terms
            )
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
    
    def evaluate_batch(self, results: List[Dict], test_segments: List[Dict]) -> Dict:
        """
        Evaluate a batch of translation results
        
        Args:
            results: List of translation results
            test_segments: List of test segment definitions with expected_terms
            
        Returns:
            Aggregated evaluation metrics
        """
        evaluations = []
        
        # Create lookup for expected terms
        expected_terms_map = {seg['id']: seg['expected_terms'] for seg in test_segments}
        
        for result in results:
            seg_id = result['segment_id']
            expected_terms = expected_terms_map.get(seg_id + 1, [])  # +1 because IDs start at 1
            eval_result = self.evaluate_result(result, expected_terms)
            evaluations.append(eval_result)
        
        # Aggregate metrics
        baseline_accuracies = []
        retrieval_accuracies = []
        baseline_latencies = []
        retrieval_latencies = []
        baseline_tokens = []
        retrieval_tokens = []
        
        for eval_result in evaluations:
            if 'baseline' in eval_result:
                baseline_accuracies.append(eval_result['baseline']['metrics']['accuracy'])
                baseline_latencies.append(eval_result['baseline']['latency'])
                baseline_tokens.append(eval_result['baseline']['tokens']['total'])
            
            if 'with_retrieval' in eval_result:
                retrieval_accuracies.append(eval_result['with_retrieval']['metrics']['accuracy'])
                retrieval_latencies.append(eval_result['with_retrieval']['latency'])
                retrieval_tokens.append(eval_result['with_retrieval']['tokens']['total'])
        
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
                'token_overhead': summary['with_retrieval']['mean_tokens'] - summary['baseline']['mean_tokens']
            }
        
        return summary
    
    def generate_comparison_report(self, evaluation_summary: Dict, output_path: str):
        """
        Generate a detailed comparison report
        
        Args:
            evaluation_summary: Summary from evaluate_batch
            output_path: Path to save the report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Translation Quality Comparison Report\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"Total segments evaluated: {evaluation_summary['total_segments']}\n\n")
            
            f.write("### Baseline (No Retrieval)\n")
            baseline = evaluation_summary['baseline']
            f.write(f"- Mean term accuracy: {baseline['mean_accuracy']:.2%}\n")
            f.write(f"- Perfect segments: {baseline['perfect_segments']}/{evaluation_summary['total_segments']}\n")
            f.write(f"- Mean latency: {baseline['mean_latency']:.3f}s\n")
            f.write(f"- Mean tokens: {baseline['mean_tokens']:.0f}\n\n")
            
            f.write("### With Glossary Retrieval\n")
            retrieval = evaluation_summary['with_retrieval']
            f.write(f"- Mean term accuracy: {retrieval['mean_accuracy']:.2%}\n")
            f.write(f"- Perfect segments: {retrieval['perfect_segments']}/{evaluation_summary['total_segments']}\n")
            f.write(f"- Mean latency: {retrieval['mean_latency']:.3f}s\n")
            f.write(f"- Mean tokens: {retrieval['mean_tokens']:.0f}\n\n")
            
            if 'improvement' in evaluation_summary:
                f.write("### Improvement\n")
                imp = evaluation_summary['improvement']
                f.write(f"- Accuracy gain: +{imp['accuracy_gain']:.2%} ({imp['accuracy_gain_pct']:.1f}% relative improvement)\n")
                f.write(f"- Latency overhead: +{imp['latency_overhead']:.3f}s\n")
                f.write(f"- Token overhead: +{imp['token_overhead']:.0f} tokens\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for eval_result in evaluation_summary['detailed_evaluations']:
                f.write(f"### Segment {eval_result['segment_id'] + 1} ({eval_result['target_lang'].upper()})\n\n")
                f.write(f"**Source:** {eval_result['source']}\n\n")
                
                if 'baseline' in eval_result:
                    f.write(f"**Baseline translation:** {eval_result['baseline']['translation']}\n")
                    f.write(f"- Term accuracy: {eval_result['baseline']['metrics']['accuracy']:.2%}\n")
                    if eval_result['baseline']['metrics']['missing_terms']:
                        f.write(f"- Missing terms: {', '.join([t['term_en'] for t in eval_result['baseline']['metrics']['missing_terms']])}\n")
                    f.write("\n")
                
                if 'with_retrieval' in eval_result:
                    f.write(f"**With retrieval:** {eval_result['with_retrieval']['translation']}\n")
                    f.write(f"- Term accuracy: {eval_result['with_retrieval']['metrics']['accuracy']:.2%}\n")
                    if eval_result['with_retrieval']['metrics']['missing_terms']:
                        f.write(f"- Missing terms: {', '.join([t['term_en'] for t in eval_result['with_retrieval']['metrics']['missing_terms']])}\n")
                    f.write(f"- Retrieved {len(eval_result['with_retrieval']['retrieved_terms'])} glossary terms\n")
                    f.write("\n")
                
                f.write("---\n\n")


if __name__ == "__main__":
    from glossary_manager import GlossaryManager
    
    # Test evaluator
    glossary = GlossaryManager("data/RideHailingGlossar_OneLine.jsonl")
    evaluator = TranslationEvaluator(glossary)
    
    # Test term accuracy calculation
    test_translation = "Demander une course et vérifier l'évaluation de votre conducteur"
    expected_terms = ["Request a ride", "driver", "rating"]
    
    metrics = evaluator.calculate_term_accuracy(test_translation, 'fr', expected_terms)
    print(f"Term accuracy: {metrics['accuracy']:.2%}")
    print(f"Found: {metrics['found_terms']}/{metrics['total_terms']}")
    print(f"Details: {metrics['found_details']}")
