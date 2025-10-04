"""
Re-evaluate existing translations with improved evaluator
"""

import json
from glossary_manager import GlossaryManager
from evaluator_v2 import ImprovedEvaluator


def main():
    print("=" * 80)
    print("Re-evaluating Translations with Improved Metrics")
    print("=" * 80)
    
    # Load existing results
    print("\n[1/3] Loading translation results...")
    with open("results/translation_results.json", 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} translation results")
    
    # Initialize improved evaluator
    print("\n[2/3] Initializing improved evaluator...")
    glossary = GlossaryManager("data/RideHailingGlossar_OneLine.jsonl")
    evaluator = ImprovedEvaluator(glossary)
    
    # Re-evaluate
    print("\n[3/3] Re-evaluating with improved term matching...")
    evaluation_summary = evaluator.evaluate_batch(results)
    
    # Save results
    with open("results/evaluation_summary_v2.json", 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
    
    evaluator.generate_comparison_report(
        evaluation_summary,
        "results/comparison_report_v2.md"
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("IMPROVED EVALUATION RESULTS")
    print("=" * 80)
    
    baseline = evaluation_summary['baseline']
    retrieval = evaluation_summary['with_retrieval']
    improvement = evaluation_summary.get('improvement', {})
    
    print(f"\nTotal segments: {evaluation_summary['total_segments']}")
    
    print("\n📊 BASELINE (No Retrieval)")
    print(f"  Term Accuracy:    {baseline['mean_accuracy']:.1%}")
    print(f"  Perfect Segments: {baseline['perfect_segments']}/{evaluation_summary['total_segments']}")
    print(f"  Avg Latency:      {baseline['mean_latency']:.3f}s")
    print(f"  Avg Tokens:       {baseline['mean_tokens']:.0f}")
    
    print("\n✨ WITH RETRIEVAL")
    print(f"  Term Accuracy:    {retrieval['mean_accuracy']:.1%}")
    print(f"  Perfect Segments: {retrieval['perfect_segments']}/{evaluation_summary['total_segments']}")
    print(f"  Avg Latency:      {retrieval['mean_latency']:.3f}s")
    print(f"  Avg Tokens:       {retrieval['mean_tokens']:.0f}")
    print(f"  Retrieval Time:   {retrieval['mean_retrieval_latency']:.3f}s")
    
    if improvement:
        print("\n📈 IMPROVEMENT")
        print(f"  Accuracy Gain:    +{improvement['accuracy_gain']:.1%} ({improvement['accuracy_gain_pct']:.1f}% relative)")
        print(f"  Perfect Gain:     +{improvement['perfect_segment_gain']} segments")
        print(f"  Latency Overhead: +{improvement['latency_overhead']:.3f}s")
        print(f"  Token Overhead:   +{improvement['token_overhead']:.0f} tokens")
    
    print("\n" + "=" * 80)
    print("✓ Results saved to:")
    print("  - results/evaluation_summary_v2.json")
    print("  - results/comparison_report_v2.md")
    print("=" * 80)
    
    # Show some examples
    print("\n\n" + "=" * 80)
    print("SAMPLE COMPARISONS")
    print("=" * 80)
    
    for i in range(min(5, len(evaluation_summary['detailed_evaluations']))):
        eval_result = evaluation_summary['detailed_evaluations'][i]
        print(f"\n--- Example {i+1}: {eval_result['target_lang'].upper()} ---")
        print(f"Source: {eval_result['source']}")
        
        if 'baseline' in eval_result:
            metrics = eval_result['baseline']['metrics']
            print(f"\nBaseline: {eval_result['baseline']['translation']}")
            print(f"  Accuracy: {metrics['accuracy']:.0%} ({metrics['found_terms']}/{metrics['total_terms']} terms)")
            if metrics['relevant_terms']:
                print(f"  Relevant: {', '.join(metrics['relevant_terms'])}")
        
        if 'with_retrieval' in eval_result:
            metrics = eval_result['with_retrieval']['metrics']
            print(f"\nWith Retrieval: {eval_result['with_retrieval']['translation']}")
            print(f"  Accuracy: {metrics['accuracy']:.0%} ({metrics['found_terms']}/{metrics['total_terms']} terms)")
            if metrics['relevant_terms']:
                print(f"  Relevant: {', '.join(metrics['relevant_terms'])}")


if __name__ == "__main__":
    main()
