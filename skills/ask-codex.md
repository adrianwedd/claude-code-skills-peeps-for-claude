# Ask Codex

Delegate a task or question to OpenAI Codex via codex-cli.

## Usage

```
/ask-codex <your question or task>
```

## Examples

```bash
# Code generation
/ask-codex "Write a Python function to parse JSON with error handling"

# Debugging help
/ask-codex "Why is this regex not matching email addresses correctly?"

# Code explanation
/ask-codex "Explain what this SQL query does"

# Algorithm help
/ask-codex "What's the time complexity of this sorting approach?"
```

## When to Use

- Code generation tasks
- Algorithm and data structure questions
- Debugging assistance
- When you want OpenAI's coding-focused model

## Implementation

```bash
#!/bin/bash
# Get the prompt from all arguments
prompt="$*"

if [ -z "$prompt" ]; then
    echo "Usage: /ask-codex <your question or task>"
    exit 1
fi

# Check if codex-cli is available
if ! command -v codex &> /dev/null; then
    echo "Error: codex-cli not found. Install from your package manager or relevant source"
    exit 1
fi

echo "Asking Codex..."
echo ""

# Call codex-cli with the prompt
codex "$prompt"
```

## Notes

- Requires `codex-cli` to be installed
- Uses your OpenAI API credentials
- May incur API costs depending on your plan
