"""
Interactive Demo - LLM Translation Pipeline with Glossary Retrieval

This script provides an interactive demonstration of the translation system,
showing side-by-side comparisons and allowing custom input.
"""

import sys
from glossary_manager import GlossaryManager
from translation_pipeline import TranslationPipeline
from evaluator_v2 import ImprovedEvaluator


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_section(text):
    """Print formatted section"""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def demo_single_translation(pipeline, evaluator, source_text, target_lang):
    """Demonstrate translation with side-by-side comparison"""
    
    lang_names = {'fr': 'French', 'it': 'Italian', 'jp': 'Japanese'}
    lang_name = lang_names.get(target_lang, target_lang)
    
    print_section(f"Translation to {lang_name}")
    print(f"\n📝 Source: {source_text}\n")
    
    # Baseline translation
    print("⚙️  Translating (baseline)...")
    baseline = pipeline.translate_baseline(source_text, target_lang)
    
    # Enhanced translation
    print("⚙️  Translating (with retrieval)...")
    enhanced = pipeline.translate_with_retrieval(source_text, target_lang)
    
    # Evaluate
    result = {
        'segment_id': 0,
        'source': source_text,
        'target_lang': target_lang,
        'baseline': baseline,
        'with_retrieval': enhanced
    }
    evaluation = evaluator.evaluate_result(result)
    
    # Display results
    print("\n" + "─" * 80)
    print("📊 BASELINE (No Glossary)")
    print("─" * 40)
    print(f"Translation: {baseline['translation']}")
    print(f"Term Accuracy: {evaluation['baseline']['metrics']['accuracy']:.0%}")
    print(f"Latency: {baseline['latency']:.3f}s")
    print(f"Tokens: {baseline['tokens']['total']}")
    
    if evaluation['baseline']['metrics']['relevant_terms']:
        print(f"Relevant terms: {', '.join(evaluation['baseline']['metrics']['relevant_terms'])}")
    if evaluation['baseline']['metrics']['missing_terms']:
        missing = [t['term_en'] for t in evaluation['baseline']['metrics']['missing_terms']]
        print(f"❌ Missing: {', '.join(missing)}")
    
    print("\n" + "─" * 80)
    print("✨ WITH GLOSSARY RETRIEVAL")
    print("─" * 40)
    print(f"Translation: {enhanced['translation']}")
    print(f"Term Accuracy: {evaluation['with_retrieval']['metrics']['accuracy']:.0%}")
    print(f"Latency: {enhanced['latency']:.3f}s (retrieval: {enhanced['retrieval_latency']:.3f}s)")
    print(f"Tokens: {enhanced['tokens']['total']}")
    print(f"Retrieved: {len(enhanced['retrieved_terms'])} glossary terms")
    
    if evaluation['with_retrieval']['metrics']['relevant_terms']:
        print(f"Relevant terms: {', '.join(evaluation['with_retrieval']['metrics']['relevant_terms'])}")
    if evaluation['with_retrieval']['metrics']['missing_terms']:
        missing = [t['term_en'] for t in evaluation['with_retrieval']['metrics']['missing_terms']]
        print(f"❌ Missing: {', '.join(missing)}")
    
    # Show retrieved terms
    if enhanced['retrieved_terms']:
        print(f"\n📚 Retrieved Glossary Terms:")
        for i, term in enumerate(enhanced['retrieved_terms'][:5], 1):
            print(f"  {i}. {term['term_en']} → {term['target_translation']} (similarity: {term['similarity']:.3f})")
    
    # Comparison
    print("\n" + "─" * 80)
    print("📈 COMPARISON")
    print("─" * 40)
    
    baseline_acc = evaluation['baseline']['metrics']['accuracy']
    enhanced_acc = evaluation['with_retrieval']['metrics']['accuracy']
    improvement = enhanced_acc - baseline_acc
    
    if improvement > 0:
        print(f"✅ Accuracy improved by {improvement:.1%}")
    elif improvement < 0:
        print(f"⚠️  Accuracy decreased by {abs(improvement):.1%}")
    else:
        print(f"➡️  Accuracy unchanged")
    
    latency_overhead = enhanced['latency'] - baseline['latency']
    print(f"⏱️  Latency overhead: +{latency_overhead:.3f}s")
    
    token_overhead = enhanced['tokens']['total'] - baseline['tokens']['total']
    print(f"💰 Token overhead: +{token_overhead} tokens")


def run_preset_examples(pipeline, evaluator):
    """Run preset example translations"""
    
    examples = [
        {
            'text': 'Request a ride and check your driver rating',
            'lang': 'fr',
            'description': 'Basic ride operations'
        },
        {
            'text': 'The base fare includes a per-minute rate and per-mile rate',
            'lang': 'it',
            'description': 'Pricing terminology'
        },
        {
            'text': 'Use the panic button for safety check or contact support',
            'lang': 'jp',
            'description': 'Safety features'
        }
    ]
    
    print_header("PRESET EXAMPLES")
    
    for i, example in enumerate(examples, 1):
        print(f"\n\n{'='*80}")
        print(f"Example {i}/3: {example['description']}")
        print(f"{'='*80}")
        
        demo_single_translation(
            pipeline,
            evaluator,
            example['text'],
            example['lang']
        )
        
        if i < len(examples):
            input("\n\n⏎ Press Enter to continue to next example...")


def run_interactive_mode(pipeline, evaluator):
    """Run interactive translation mode"""
    
    print_header("INTERACTIVE MODE")
    
    print("Enter your own text to translate!")
    print("Type 'quit' or 'exit' to return to main menu.\n")
    
    while True:
        # Get source text
        source_text = input("\n📝 Enter English text to translate: ").strip()
        
        if source_text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not source_text:
            print("⚠️  Please enter some text.")
            continue
        
        # Get target language
        print("\nTarget language:")
        print("  1. French (fr)")
        print("  2. Italian (it)")
        print("  3. Japanese (jp)")
        
        lang_choice = input("\nSelect language (1-3): ").strip()
        
        lang_map = {'1': 'fr', '2': 'it', '3': 'jp', 'fr': 'fr', 'it': 'it', 'jp': 'jp'}
        target_lang = lang_map.get(lang_choice)
        
        if not target_lang:
            print("⚠️  Invalid language choice.")
            continue
        
        # Translate
        demo_single_translation(pipeline, evaluator, source_text, target_lang)


def show_system_info(glossary):
    """Display system information"""
    
    print_header("SYSTEM INFORMATION")
    
    print("📚 Glossary Statistics:")
    print(f"  Total terms: {len(glossary.glossary_entries)}")
    
    # Count by language
    for lang in ['fr', 'it', 'jp']:
        count = sum(1 for entry in glossary.glossary_entries if entry.get(lang))
        print(f"  {lang.upper()} translations: {count}")
    
    # Count by domain
    domains = {}
    for entry in glossary.glossary_entries:
        domain = entry.get('domain', 'Unknown')
        domains[domain] = domains.get(domain, 0) + 1
    
    print(f"\n📁 Domains:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1])[:5]:
        print(f"  {domain}: {count} terms")
    
    print(f"\n🔍 Vector Store:")
    print(f"  Embedding dimension: {glossary.embeddings.shape[1]}")
    print(f"  Index size: {glossary.index.ntotal} vectors")
    print(f"  Model: paraphrase-multilingual-MiniLM-L12-v2")
    
    print(f"\n🤖 Translation Model:")
    print(f"  Model: GPT-4.1-mini")
    print(f"  Temperature: 0.3")
    print(f"  Max tokens: 500")


def main_menu():
    """Display main menu and handle user choice"""
    
    print_header("LLM TRANSLATION PIPELINE WITH GLOSSARY RETRIEVAL")
    
    print("Welcome to the interactive demo!\n")
    print("This system demonstrates how glossary retrieval improves")
    print("LLM translation quality for domain-specific terminology.\n")
    
    print("Menu:")
    print("  1. Run preset examples")
    print("  2. Interactive translation mode")
    print("  3. Show system information")
    print("  4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    return choice


def main():
    """Main demo function"""
    
    # Initialize system
    print("\n🔧 Initializing system...")
    print("Loading glossary and building vector index...")
    
    glossary = GlossaryManager("data/RideHailingGlossar_OneLine.jsonl")
    pipeline = TranslationPipeline(glossary_manager=glossary)
    evaluator = ImprovedEvaluator(glossary)
    
    print("✅ System ready!\n")
    
    # Main loop
    while True:
        choice = main_menu()
        
        if choice == '1':
            run_preset_examples(pipeline, evaluator)
        elif choice == '2':
            run_interactive_mode(pipeline, evaluator)
        elif choice == '3':
            show_system_info(glossary)
        elif choice == '4':
            print("\n👋 Thank you for using the translation pipeline!")
            print("For more information, see README.md and TECHNICAL_WRITEUP.md\n")
            break
        else:
            print("\n⚠️  Invalid choice. Please select 1-4.")
        
        if choice in ['1', '2', '3']:
            input("\n\n⏎ Press Enter to return to main menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Please check your setup and try again.")
        sys.exit(1)
