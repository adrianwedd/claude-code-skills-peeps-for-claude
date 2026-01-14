# Claude AI Skills Collection

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Skills](https://img.shields.io/badge/skills-6-brightgreen.svg)

A collection of reusable skills for Claude Code that enable delegation to other AI models and services. These skills extend Claude's capabilities by allowing seamless integration with OpenAI Codex, Google Gemini, and 100+ models via OpenRouter.

## 🚀 Quick Start

1. **Clone or download** this repository
2. **Copy skills** to your project's `.claude/skills/` directory
3. **Install dependencies** (see [Setup](#setup))
4. **Use skills** with slash commands: `/ask-model`, `/ask-codex`, `/ask-gemini`

## 📦 Included Skills

### `/ask-model` - Universal AI Model Access
Access 100+ AI models through OpenRouter, including free and paid options.

```bash
# Use free models
/ask-model gemini "Explain microservices architecture"
/ask-model llama-70b "Design a database schema"
/ask-model mistral "Optimize this Python function"

# Use paid models for critical tasks
/ask-model gpt4 "Review this security implementation"
/ask-model claude "Analyze this complex algorithm"
```

**Features:**
- 🆓 **Free models**: Gemini 2.0 Flash, Llama 3.3 70B, Mistral Devstral
- 💰 **Paid models**: GPT-4 Turbo, Claude 3.5 Sonnet, Claude Opus 4
- 🔧 **Model shortcuts**: Simple names like `gemini`, `gpt4`, `claude`
- 🌐 **Full OpenRouter support**: Use any model with full ID

### `/ask-codex` - OpenAI Codex Integration
Specialized coding assistant using OpenAI's Codex model.

```bash
/ask-codex "Write a Python function to parse JSON with error handling"
/ask-codex "Why is this regex not matching email addresses correctly?"
/ask-codex "What's the time complexity of this sorting approach?"
```

**Best for:**
- Code generation and debugging
- Algorithm analysis
- Technical explanations

### `/ask-gemini` - Google Gemini Integration
Access Google's Gemini model for research and analysis.

```bash
/ask-gemini "What are the security implications of this API design?"
/ask-gemini "Review this function for potential bugs"
/ask-gemini "What are REST API best practices?"
```

**Best for:**
- Code review and security analysis
- Research and fact-checking
- Alternative perspectives

### `/compare-answers` - Multi-Model Comparison
Ask the same question to multiple AI models and compare their responses side-by-side.

```bash
# Compare free models (default)
/compare-answers "What's the best way to handle API rate limiting?"

# Use fast preset (2 fastest models)
/compare-answers fast "Explain quicksort algorithm"

# Compare paid models for critical questions
/compare-answers paid "What are the security risks in this auth flow?"

# Get diverse perspectives (Gemini + GPT-4 + Claude)
/compare-answers best "Design a caching strategy for high-traffic API"
```

**Presets:**
- `free` - Top 3 free models (Gemini, Llama 70B, Mistral)
- `fast` - 2 fastest free models (Gemini, Llama 3B)
- `all-free` - All 5 tested free models
- `paid` - GPT-4 Turbo + Claude 3.5 Sonnet
- `best` - Mixed (Gemini + GPT-4 + Claude)

**Best for:**
- Important decisions needing multiple perspectives
- Comparing model capabilities
- Getting consensus on best practices
- Validating critical answers

## 🛠️ Setup

### 1. Copy Skills to Your Project

```bash
# Create skills directory in your project
mkdir -p .claude/skills

# Copy all skills
cp claude-ai-skills/skills/*.md .claude/skills/

# Make executable scripts
chmod +x .claude/skills/*.md
```

### 2. Install Dependencies

**For `/ask-model` (OpenRouter):**
```bash
# Get API key from https://openrouter.ai/keys
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# Add to your shell profile
echo 'export OPENROUTER_API_KEY="sk-or-v1-your-key-here"' >> ~/.bashrc
```

**For `/ask-codex` (OpenAI Codex):**
```bash
# Install codex-cli (platform-specific)
npm install -g codex-cli
# or
pip install codex-cli
# Configure with your OpenAI API key
```

**For `/ask-gemini` (Google Gemini):**
```bash
# Install gemini-cli
npm install -g @google-cloud/generative-ai-cli
# Configure with your Gemini API key
```

**For advanced skills (`/compare-models`, `/benchmark-http`):**
```bash
# Install Python dependencies
pip install -r requirements.txt

# This installs:
# - requests (for HTTP API calls)
```

**Note:** The `/compare-answers` skill works without any additional dependencies!

### 3. Environment Setup

**Option A: Using .env file**
```bash
# Create .env in your project root
cat > .env << EOF
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
EOF
```

**Option B: Export variables**
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
export OPENAI_API_KEY="sk-your-openai-key-here"
export GEMINI_API_KEY="your-gemini-key-here"
```

## 📖 Usage Guide

### Model Selection Strategy

```bash
# Free models (no cost)
/ask-model gemini     # Fast, high quality, good for most tasks
/ask-model llama      # Small, efficient, good for simple tasks
/ask-model llama-70b  # Large, powerful, good for complex tasks
/ask-model mistral    # Coding-focused, good for technical tasks

# Paid models (premium quality)
/ask-model gpt4       # Best general performance
/ask-model claude     # Excellent reasoning, analysis
/ask-model opus       # Most capable, for critical tasks
```

### Common Workflows

**Code Review:**
```bash
/ask-gemini "Review this function for security issues"
/ask-codex "Explain this algorithm's time complexity"
/ask-model claude "Suggest improvements for this design"
```

**Research & Learning:**
```bash
/ask-model gemini "Compare GraphQL vs REST APIs"
/ask-gemini "What are current ML best practices?"
/ask-model llama-70b "Explain distributed systems concepts"
```

**Code Generation:**
```bash
/ask-codex "Write a rate-limited API client in Python"
/ask-model mistral "Generate TypeScript interfaces for this API"
/ask-model gpt4 "Create comprehensive unit tests for this module"
```

### Advanced Usage

**Compare Multiple Models:**
```bash
/ask-model gemini "What's the best database for real-time applications?"
/ask-model claude "What's the best database for real-time applications?"
/ask-codex "What's the best database for real-time applications?"
# Compare responses to get well-rounded perspective
```

**Full Model IDs:**
```bash
# Use specific model versions
/ask-model google/gemini-2.0-flash-exp:free "Your question"
/ask-model anthropic/claude-3.5-sonnet "Your question"
/ask-model meta-llama/llama-3.3-70b-instruct:free "Your question"
```

## 🎯 Use Cases

### Software Development
- **Code Generation**: Generate functions, classes, and modules
- **Debugging**: Identify and fix bugs in existing code
- **Code Review**: Get multiple perspectives on code quality
- **Algorithm Design**: Design and analyze algorithms
- **Testing**: Generate comprehensive test suites

### Architecture & Design
- **System Design**: Design scalable, maintainable systems
- **API Design**: Create well-structured APIs
- **Database Schema**: Design efficient data models
- **Security Review**: Identify security vulnerabilities
- **Performance Analysis**: Optimize system performance

### Research & Learning
- **Technology Comparison**: Compare frameworks, languages, tools
- **Best Practices**: Learn industry standards and conventions
- **Concept Explanation**: Understand complex technical concepts
- **Trend Analysis**: Stay current with technology trends

## 🔧 Customization

### Adding New Skills

Create new skills by following the pattern in `skills/`:

```markdown
# Your Skill Name

Brief description of what this skill does.

## Usage

```
/your-skill <arguments>
```

## Implementation

```bash
#!/bin/bash
# Your bash implementation here
```

## Notes

- Dependencies and requirements
- Usage notes and limitations
```

### Modifying Existing Skills

Skills are implemented as Markdown files with embedded bash scripts. You can:

- **Modify model mappings** in `/ask-model`
- **Add new model shortcuts**
- **Customize API endpoints**
- **Add preprocessing/postprocessing**

### Environment Configuration

Customize behavior with environment variables:

```bash
# Model preferences
export DEFAULT_MODEL="claude"
export FALLBACK_MODEL="gemini"

# API timeouts
export API_TIMEOUT=30

# Debug mode
export SKILLS_DEBUG=true
```

## 📚 API References

### OpenRouter API
- **Documentation**: https://openrouter.ai/docs
- **Model List**: https://openrouter.ai/models
- **Free Tier**: 200-1000 requests/day depending on credits
- **Rate Limits**: 20 requests/minute for free models

### OpenAI Codex API
- **Documentation**: https://platform.openai.com/docs
- **Models**: GPT-4, GPT-3.5 Turbo, Codex
- **Pricing**: Per token, varies by model

### Google Gemini API
- **Documentation**: https://ai.google.dev/docs
- **Models**: Gemini Pro, Gemini Ultra
- **Free Tier**: Available with limits

## 🔍 Troubleshooting

### Common Issues

**"Command not found" errors:**
```bash
# Check if CLI tools are installed
which codex     # Should return path if installed
which gemini    # Should return path if installed
```

**API authentication errors:**
```bash
# Verify API keys are set
echo $OPENROUTER_API_KEY  # Should show your key
echo $OPENAI_API_KEY      # Should show your key
echo $GEMINI_API_KEY      # Should show your key
```

**Rate limiting issues:**
```bash
# Use free models for development
/ask-model gemini "your question"  # Free, 20 RPM limit

# Use paid models for production
/ask-model gpt4 "your question"    # Paid, higher limits
```

### Debug Mode

Enable debug output:
```bash
export SKILLS_DEBUG=true
/ask-model gemini "test question"  # Will show detailed output
```

### Logs

Skills log to the console by default. For persistent logging:

```bash
# Redirect output to file
/ask-model gemini "your question" >> skills.log 2>&1

# Monitor logs
tail -f skills.log
```

## 🤝 Contributing

### Adding Support for New Models

1. **Update model mappings** in `ask-model.md`
2. **Add shortcuts** for common models
3. **Test with various prompts**
4. **Update documentation**

### Improving Error Handling

1. **Add validation** for API responses
2. **Improve error messages**
3. **Add retry logic** for transient failures
4. **Handle rate limiting gracefully**

### Performance Optimizations

1. **Cache API responses** for repeated queries
2. **Add request batching** where supported
3. **Optimize JSON parsing**
4. **Add request timeouts**

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/adrianwedd/claude-code-skills-peeps-for-claude/issues)
- **Discussions**: [GitHub Discussions](https://github.com/adrianwedd/claude-code-skills-peeps-for-claude/discussions)
- **Documentation**: [Wiki](https://github.com/adrianwedd/claude-code-skills-peeps-for-claude/wiki)

## 🚀 Related Projects

- **Claude Code**: https://claude.ai/claude-code
- **OpenRouter**: https://openrouter.ai
- **OpenAI API**: https://platform.openai.com
- **Google AI**: https://ai.google.dev

---

**Made with ❤️ for the Claude Code community**