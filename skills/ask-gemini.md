# Ask Gemini

Delegate a task or question to Google Gemini via gemini-cli.

## Usage

```
/ask-gemini <your question or task>
```

## Examples

```bash
# Ask a question
/ask-gemini "What are the security implications of this API design?"

# Code review
/ask-gemini "Review this function for potential bugs"

# Research
/ask-gemini "What are the best practices for rate limiting in REST APIs?"

# Quick fact check
/ask-gemini "What's the difference between OAuth 2.0 and OAuth 1.0?"
```

## When to Use

- Second opinion on design decisions
- Quick research or fact-checking
- Alternative perspective on code review
- When you want a different model's approach

## Implementation

```bash
#!/bin/bash
# Get the prompt from all arguments
prompt="$*"

if [ -z "$prompt" ]; then
    echo "Usage: /ask-gemini <your question or task>"
    exit 1
fi

# Check if gemini-cli is available
if ! command -v gemini &> /dev/null; then
    echo "Error: gemini-cli not found. Install from: https://github.com/google/generative-ai-cli"
    exit 1
fi

echo "Asking Gemini..."
echo ""

# Call gemini-cli with the prompt
gemini "$prompt"
```

## Notes

- Requires `gemini-cli` to be installed
- Uses your Gemini API credentials
- Free tier available with rate limits
