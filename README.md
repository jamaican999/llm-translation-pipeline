# LLM Translation Pipeline with Glossary Retrieval

A production-ready translation system that enhances LLM translations with semantic glossary retrieval for improved terminology consistency.

## Overview

This project implements an LLM-based translation pipeline that uses vector-based glossary retrieval to improve translation quality for domain-specific terminology. The system demonstrates a **29% relative improvement** in term accuracy compared to baseline LLM translation.

### Key Features

- Semantic Glossary Retrieval**: Uses sentence transformers and FAISS for fast, relevant term retrieval
- Multi-language Support**: Translates EN→FR, EN→IT, EN→JP with domain-specific terminology
- Comprehensive Evaluation**: Automated metrics for term accuracy, latency, and cost analysis
- Production-Ready**: Modular architecture with clear separation of concerns
- Proven Results**: 82.6% term accuracy vs 63.9% baseline

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (set in environment variable `OPENAI_API_KEY`)

### Installation

```bash
# Clone or extract the repository
cd llm-translation-pipeline

# Install dependencies
pip install openai sentence-transformers faiss-cpu scikit-learn

# Verify installation
python3.11 glossary_manager.py
```

### Running the Demo

```bash
# Run the full experiment (54 translations)
cd notebooks
python3.11 translation_experiment.py

# Or re-evaluate existing results
cd ..
python3.11 re_evaluate.py
```

### Quick Test

```python
from glossary_manager import GlossaryManager
from translation_pipeline import TranslationPipeline

# Initialize
glossary = GlossaryManager("data/RideHailingGlossar_OneLine.jsonl")
pipeline = TranslationPipeline(glossary_manager=glossary)

# Translate with retrieval
result = pipeline.translate_with_retrieval(
    "Request a ride and check your driver rating",
    target_lang='fr'
)

print(result['translation'])
# Output: Demandez une course et vérifiez la note de votre conducteur
```

## Project Structure

```
llm-translation-pipeline/
├── data/
│   ├── RideHailingGlossar_OneLine.jsonl  # 81-term glossary
│   └── test_segments.json                 # 18 test segments
├── results/
│   ├── translation_results.json           # Raw translations
│   ├── evaluation_summary_v2.json         # Metrics
│   └── comparison_report_v2.md            # Detailed report
├── notebooks/
│   └── translation_experiment.py          # Main experiment
├── glossary_manager.py                    # Glossary loading & retrieval
├── translation_pipeline.py                # LLM translation
├── evaluator_v2.py                        # Improved evaluation
├── re_evaluate.py                         # Re-evaluation script
├── TECHNICAL_WRITEUP.md                   # Detailed write-up
└── README.md                              # This file
```

## System Architecture

### Components

1. **Glossary Manager** (`glossary_manager.py`)
   2. Loads JSONL glossary with metadata
   3. Embeds terms using sentence transformers
   4. Builds FAISS index for fast retrieval
   5. Retrieves relevant terms by semantic similarity

2. **Translation Pipeline** (`translation_pipeline.py`)
   2. Baseline mode: Direct LLM translation
   3. Enhanced mode: LLM + glossary constraints
   4. Batch processing support
   5. Latency and token tracking

3. **Evaluator** (`evaluator_v2.py`)
   2. Automatic term detection in source text
   3. Fuzzy matching for morphological variations
   4. Comprehensive metrics calculation
   5. Report generation

### Data Flow

```
Source Text
    ↓
[Glossary Retrieval]
    ↓ (top-k relevant terms)
[Prompt Construction]
    ↓ (source + glossary constraints)
[LLM Translation]
    ↓
Target Translation
    ↓
[Evaluation]
    ↓
Metrics & Reports
```

## Results Summary

| Metric               | Baseline | With Retrieval | Improvement  |
| -------------------- | -------- | -------------- | ------------ |
| **Term Accuracy**    | 63.9%    | 82.6%          | **+18.7 pp** |
| **Perfect Segments** | 17/54    | 29/54          | **+12**      |
| **Avg Latency**      | 0.814s   | 0.879s         | +0.066s      |
| **Avg Tokens**       | 81       | 187            | +106         |

### Sample Translation

**Source:** "The base fare includes a per-minute rate and per-mile rate"

**Baseline (IT):** La tariffa base include una tariffa al minuto e una tariffa al miglio.  
*Accuracy: 75%*

**With Retrieval (IT):** La Tariffa base include una Tariffa al minuto e una Tariffa per miglio/km  
*Accuracy: 100%*

## Configuration

### Glossary Format

The glossary is a JSONL file with one term per line:

```json
{
  "term_en": "Driver",
  "fr": "Conducteur",
  "it": "Autista",
  "jp": "ドライバー",
  "case_sensitive": false,
  "dnt": false,
  "criticality_level": "Medium",
  "domain": "Core App"
}
```

### Retrieval Parameters

Adjust in `glossary_manager.py`:

```python
# Number of terms to retrieve
top_k = 5

# Minimum similarity threshold
threshold = 0.3

# Embedding model
embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
```

### LLM Configuration

Adjust in `translation_pipeline.py`:

```python
# Model selection
model = "gpt-4.1-mini"

# Temperature (lower = more deterministic)
temperature = 0.3

# Max output tokens
max_tokens = 500
```

## API Usage

### Translate Single Segment

```python
from glossary_manager import GlossaryManager
from translation_pipeline import TranslationPipeline

glossary = GlossaryManager("data/RideHailingGlossar_OneLine.jsonl")
pipeline = TranslationPipeline(glossary_manager=glossary)

# Baseline translation
baseline = pipeline.translate_baseline(
    "Request a ride",
    target_lang='fr'
)

# With retrieval
enhanced = pipeline.translate_with_retrieval(
    "Request a ride",
    target_lang='fr',
    top_k=5,
    threshold=0.3
)

print(f"Baseline: {baseline['translation']}")
print(f"Enhanced: {enhanced['translation']}")
print(f"Retrieved terms: {len(enhanced['retrieved_terms'])}")
```

### Translate Batch

```python
segments = [
    {'text': 'Request a ride', 'target_lang': 'fr'},
    {'text': 'Check your driver rating', 'target_lang': 'it'},
    {'text': 'Add a payment method', 'target_lang': 'jp'}
]

results = pipeline.translate_batch(segments, mode='both')

for result in results:
    print(f"Source: {result['source']}")
    print(f"Baseline: {result['baseline']['translation']}")
    print(f"Enhanced: {result['with_retrieval']['translation']}")
    print()
```

### Evaluate Translations

```python
from evaluator_v2 import ImprovedEvaluator

evaluator = ImprovedEvaluator(glossary)
evaluation = evaluator.evaluate_batch(results)

print(f"Baseline accuracy: {evaluation['baseline']['mean_accuracy']:.1%}")
print(f"Enhanced accuracy: {evaluation['with_retrieval']['mean_accuracy']:.1%}")
print(f"Improvement: +{evaluation['improvement']['accuracy_gain']:.1%}")
```

## Evaluation Methodology

### Term Accuracy Calculation

1. **Identify Relevant Terms**: Find which glossary terms appear in the source text using word-boundary matching
2. **Check Target Presence**: Verify expected translations appear in output using fuzzy matching (85% similarity)
3. **Calculate Accuracy**: `accuracy = found_terms / total_relevant_terms`

### Metrics Reported

- **Term Accuracy**: Percentage of glossary terms correctly translated
- **Perfect Segments**: Count of segments with 100% term accuracy
- **Latency**: End-to-end translation time (including retrieval)
- **Token Usage**: Total tokens consumed (input + output)
- **Retrieval Time**: Time spent on vector search

## Performance Optimization

### For Large Glossaries (\>1000 terms)

```python
# Use quantized index
import faiss
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(embeddings)
index.add(embeddings)
```

### For High-Volume Translation

```python
# Cache embeddings
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text):
    return model.encode([text])[0]

# Batch API calls
results = pipeline.translate_batch(segments, mode='with_retrieval')
```

## Troubleshooting

### "OpenAI API key not found"

Set the environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### "FAISS index not built"

Ensure glossary is loaded before retrieval:
```python
glossary = GlossaryManager("data/RideHailingGlossar_OneLine.jsonl")
# Wait for "Built FAISS index" message
```

### Low accuracy scores

Check that:
1. Glossary contains relevant terms for your domain
2. Retrieval threshold is not too high (try 0.2-0.3)
3. Source text actually contains glossary terms

## Cost Analysis

**Per Translation (GPT-4.1-mini):**
- Baseline: \~$0.000023 (81 tokens)
- With retrieval: \~$0.000037 (187 tokens)
- **Cost increase: +$0.000014 (+61%)**

**For 1M translations:**
- Baseline: $23,000
- With retrieval: $37,000
- **Additional cost: $14,000 for 29% accuracy improvement**

## Limitations

- **Retrieval noise**: May retrieve semantically similar but irrelevant terms
- **Morphological variations**: Evaluation may miss valid alternative translations
- **Token overhead**: 131% increase in tokens may impact high-volume costs
- **Language coverage**: Currently supports FR, IT, JP only

## Future Enhancements

- [ ] Few-shot prompting for edge cases (morphology, gender, casing)
- [ ] Context-aware retrieval using discourse structure
- [ ] HTML tag preservation and protected token handling
- [ ] Multi-modal glossaries with images for UI elements
- [ ] Fine-tuning for domain-specific models
- [ ] Automated glossary expansion from production data

## Citation

If you use this code in your research or project, please cite:

```
LLM Translation Pipeline with Glossary Retrieval
Translation Pipeline Project, October 2025
https://github.com/your-repo/llm-translation-pipeline
```

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please open an issue on GitHub or contact the project maintainers.

---

**Built with:** OpenAI GPT-4.1-mini, Sentence Transformers, FAISS  
**Languages:** Python 3.11  
**Domain:** Ride-hailing localization
