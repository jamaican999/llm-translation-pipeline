# Contributing to LLM Translation Pipeline

Thank you for your interest in contributing to this project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/llm-translation-pipeline.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests to ensure everything works
6. Commit your changes: `git commit -am 'Add some feature'`
7. Push to the branch: `git push origin feature/your-feature-name`
8. Submit a pull request

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Run tests
python3 glossary_manager.py
python3 translation_pipeline.py
python3 evaluator_v2.py
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing

Before submitting a PR, please ensure:

- All existing tests pass
- New features include appropriate tests
- Code is well-documented

## Areas for Contribution

- **New language pairs**: Add support for additional target languages
- **Improved evaluation**: Enhance metrics and evaluation methods
- **Performance optimization**: Improve retrieval speed or reduce costs
- **Documentation**: Improve README, add tutorials, or create examples
- **Bug fixes**: Report and fix any issues you encounter

## Questions?

Feel free to open an issue for any questions or discussions about the project.
