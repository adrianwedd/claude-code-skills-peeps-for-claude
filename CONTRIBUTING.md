# Contributing to Claude AI Skills

Thank you for considering contributing to Claude AI Skills! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/adrianwedd/claude-code-skills-peeps-for-claude.git
   cd claude-code-skills-peeps-for-claude
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and test thoroughly
5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Submit a pull request** via GitHub

## 📝 Types of Contributions

### Bug Reports
- Use GitHub Issues
- Include clear description of the bug
- Provide steps to reproduce
- Include your environment (OS, shell, versions)
- Add relevant error messages

### Feature Requests
- Use GitHub Issues with "enhancement" label
- Clearly describe the feature and use case
- Explain why it would be valuable
- Consider backwards compatibility

### Code Contributions
- Follow existing code style
- Add tests for new features
- Update documentation
- Ensure all tests pass

## 🛠️ Development Guidelines

### Adding New Skills

Skills should follow this structure:

```markdown
# Skill Name

Brief description of what this skill does.

## Usage

```
/skill-name <arguments>
```

## Examples

```bash
# Example 1: Description
/skill-name arg1 arg2

# Example 2: Description
/skill-name arg1
```

## When to Use

- Use case 1
- Use case 2
- Use case 3

## Implementation

```bash
#!/bin/bash
# Your bash implementation here
# Include proper error handling
# Validate inputs
# Provide helpful error messages
```

## Notes

- Dependencies and requirements
- API keys needed
- Rate limits or costs
- Any limitations
```

### Skill Requirements

**Must have:**
- ✅ Clear, concise documentation
- ✅ Input validation
- ✅ Error handling with helpful messages
- ✅ Examples that work
- ✅ Bash best practices

**Should have:**
- Self-contained (minimal external dependencies)
- Timeout handling for network calls
- Secure handling of API keys
- Cross-platform compatibility (Linux, macOS)

**Test your skill with:**
- Valid inputs
- Invalid inputs
- Missing dependencies
- Special characters (quotes, newlines, backslashes)
- Network failures
- Rate limiting scenarios

### Code Style

**Bash:**
- Use `#!/bin/bash` shebang
- Validate all inputs
- Use meaningful variable names
- Add comments for complex logic
- Use `set -e` for error handling where appropriate
- Quote variables: `"$variable"`
- Use `$()` instead of backticks

**Example:**
```bash
#!/bin/bash

# Validate input
if [ -z "$1" ]; then
    echo "Error: Missing required argument"
    exit 1
fi

# Process with error handling
result=$(some_command "$1" 2>&1)
if [ $? -ne 0 ]; then
    echo "Error: Command failed: $result"
    exit 1
fi
```

### Security Guidelines

**Always:**
- ✅ Validate and sanitize user inputs
- ✅ Escape data before using in JSON/shell
- ✅ Use environment variables for API keys
- ✅ Never commit API keys or secrets
- ✅ Use HTTPS for API calls
- ✅ Add timeouts to network requests

**Never:**
- ❌ Trust user input without validation
- ❌ Execute arbitrary user code
- ❌ Hardcode API keys or credentials
- ❌ Use `eval` with user input
- ❌ Ignore error conditions

## 🧪 Testing

### Manual Testing Checklist

Before submitting a PR, test:

1. **Installation**
   ```bash
   ./install.sh
   # Verify skills are copied correctly
   ```

2. **Valid inputs**
   ```bash
   # Test with normal inputs
   /your-skill "normal input"
   ```

3. **Edge cases**
   ```bash
   # Test with special characters
   /your-skill 'Input with "quotes"'
   /your-skill $'Input with\nnewlines'
   ```

4. **Error handling**
   ```bash
   # Test with missing dependencies
   # Test with invalid API keys
   # Test with no input
   ```

5. **Cross-platform** (if possible)
   - Test on Linux
   - Test on macOS
   - Document Windows compatibility

## 📚 Documentation

### Updating README

When adding features:
- Update main README.md
- Add to "Included Skills" section
- Update usage examples
- Add to appropriate use case sections

### Updating CHANGELOG

Follow [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [Unreleased]

### Added
- New skill: /your-skill for [purpose]
- New feature in /existing-skill

### Changed
- Updated /existing-skill to [improvement]

### Fixed
- Fixed bug in /existing-skill
```

## 🔍 Pull Request Process

1. **Before submitting:**
   - Run all tests
   - Update documentation
   - Update CHANGELOG.md
   - Ensure no sensitive data is committed
   - Rebase on latest main if needed

2. **PR Description should include:**
   - What: Clear description of changes
   - Why: Reason for the changes
   - How: Approach taken
   - Testing: How you tested it
   - Screenshots: If applicable

3. **Review process:**
   - Maintainers will review your PR
   - Address feedback and make requested changes
   - Once approved, maintainers will merge

4. **After merge:**
   - Your contribution will be in the next release
   - You'll be credited in release notes

## 🎨 Coding Standards

### Bash Best Practices

```bash
# Good ✅
if [ -z "$variable" ]; then
    echo "Error: variable is empty"
    exit 1
fi

# Bad ❌
if [ -z $variable ]; then  # Unquoted variable
    echo Error message     # Unquoted string
fi
```

### Error Messages

```bash
# Good ✅
echo "Error: OPENROUTER_API_KEY not set"
echo "Get your API key from: https://openrouter.ai/keys"
echo "Then: export OPENROUTER_API_KEY='your-key-here'"

# Bad ❌
echo "error"  # Not helpful
```

### JSON Handling

```bash
# Good ✅ - Proper escaping
prompt_json=$(python3 -c "import json, sys; print(json.dumps(sys.argv[1]))" "$prompt")
curl -d "{\"content\": $prompt_json}"

# Bad ❌ - JSON injection vulnerability
curl -d "{\"content\": \"$prompt\"}"
```

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on what's best for the project
- Show empathy towards others

### Communication

- **Issues**: For bugs and feature requests
- **Pull Requests**: For code contributions
- **Discussions**: For questions and ideas

### Response Times

- We aim to respond to issues within 48 hours
- PRs will be reviewed within 1 week
- Please be patient with maintainers

## 📊 Project Structure

```
claude-code-skills-peeps-for-claude/
├── skills/              # Skill definitions
│   ├── ask-model.md
│   ├── ask-codex.md
│   └── ask-gemini.md
├── examples/            # Usage examples
├── .claude/            # Installed skills (gitignored)
├── install.sh          # Installation script
├── .env.template       # Environment template
├── README.md           # Main documentation
├── CHANGELOG.md        # Version history
├── CONTRIBUTING.md     # This file
├── SECURITY.md         # Security policy
└── LICENSE             # MIT License
```

## 🏆 Recognition

Contributors will be:
- Listed in release notes
- Credited in README if significant contribution
- Thanked in the community

## ❓ Questions?

- Open an issue for questions
- Start a discussion for ideas
- Reach out to maintainers

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Claude AI Skills! 🎉**
