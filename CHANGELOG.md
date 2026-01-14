# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-14

### Added
- Initial public release of Claude AI Skills Collection
- `/ask-model` skill for accessing 100+ AI models via OpenRouter
- `/ask-codex` skill for OpenAI Codex integration
- `/ask-gemini` skill for Google Gemini integration
- `/compare-answers` skill for multi-model comparison (self-contained, no dependencies)
- `/compare-models` skill for advanced benchmarking workflows
- `/benchmark-http` skill for systematic API performance testing
- Comprehensive documentation and setup guide
- MIT License for open source usage
- Installation script with dependency validation
- Environment configuration template (.env.template)
- Python benchmarking tools in `tools/benchmarks/`
- Example scenario files for testing

### Features
- **Free model support**: Gemini 2.0 Flash, Llama 3.3 70B, Mistral Devstral
- **Paid model support**: GPT-4 Turbo, Claude 3.5 Sonnet, Claude Opus 4
- **Model shortcuts**: Simple names like `gemini`, `gpt4`, `claude`
- **Robust error handling**: Clear error messages and user feedback
- **Environment configuration**: Support for .env files and environment variables
- **Security**: Proper input sanitization and timeout handling

### Security
- Fixed JSON injection vulnerability in ask-model skill
- Added proper JSON escaping for all user inputs
- Added 30-second timeout for API requests
- Created .gitignore to prevent API key commits
- Added security policy (SECURITY.md)

### Documentation
- Complete README with usage examples
- Setup instructions for all platforms
- Troubleshooting guide
- API reference links
- Contributing guidelines (CONTRIBUTING.md)
- Security policy (SECURITY.md)
- Comprehensive examples directory

## [Unreleased]

### Planned
- Additional model integrations
- Caching layer for repeated queries
- Request batching support
- Enhanced error recovery
- Performance optimizations