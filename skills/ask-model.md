# Ask Model

Delegate a task to any OpenRouter model (100+ models including free options).

## Usage

```
/ask-model [model] <your question or task>
```

## Model Shortcuts

**Free models:**
- `gemini` - Google Gemini 2.0 Flash (fast, high quality)
- `llama` - Meta Llama 3.2 3B (small, efficient)
- `llama-70b` - Meta Llama 3.3 70B (large, powerful)
- `mistral` - Mistral Devstral 2512 (coding focused)

**Paid models:**
- `gpt4` - OpenAI GPT-4 Turbo
- `claude` - Anthropic Claude 3.5 Sonnet
- `opus` - Anthropic Claude Opus 4

Or use full model ID from OpenRouter (e.g., `google/gemini-2.0-flash-exp:free`)

## Examples

```bash
# Quick free model query
/ask-model gemini "What are the trade-offs of microservices vs monolithic architecture?"

# Use powerful free model for complex task
/ask-model llama-70b "Design a database schema for a multi-tenant SaaS application"

# Use paid model for critical task
/ask-model gpt4 "Review this security implementation for vulnerabilities"

# Coding-specific question
/ask-model mistral "Optimize this Python function for performance"

# Default to gemini if no model specified
/ask-model "Explain how JWT tokens work"
```

## When to Use

- Access to 100+ models through one interface
- Try different models for comparison
- Use free models for most tasks
- Use paid models for critical/complex tasks

## Implementation

```bash
#!/bin/bash
# Parse model and prompt
if [ $# -eq 0 ]; then
    echo "Usage: /ask-model [model] <your question or task>"
    echo ""
    echo "Models: gemini (default), llama, llama-70b, mistral, gpt4, claude, opus"
    echo "Or use full OpenRouter model ID"
    exit 1
fi

# Check if first arg is a model shortcut
case "$1" in
    gemini|llama|llama-70b|mistral|gpt4|claude|opus)
        model="$1"
        shift
        prompt="$*"
        ;;
    google/*|meta-llama/*|mistralai/*|openai/*|anthropic/*|nvidia/*)
        # Full model ID
        model="$1"
        shift
        prompt="$*"
        ;;
    *)
        # No model specified, default to gemini
        model="gemini"
        prompt="$*"
        ;;
esac

if [ -z "$prompt" ]; then
    echo "Error: No prompt provided"
    exit 1
fi

# Map shortcuts to full model IDs
case "$model" in
    gemini)
        model_id="google/gemini-2.0-flash-exp:free"
        ;;
    llama)
        model_id="meta-llama/llama-3.2-3b-instruct:free"
        ;;
    llama-70b)
        model_id="meta-llama/llama-3.3-70b-instruct:free"
        ;;
    mistral)
        model_id="mistralai/devstral-2512:free"
        ;;
    gpt4)
        model_id="openai/gpt-4-turbo"
        ;;
    claude)
        model_id="anthropic/claude-3.5-sonnet"
        ;;
    opus)
        model_id="anthropic/claude-opus-4"
        ;;
    *)
        model_id="$model"
        ;;
esac

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY not set"
    echo "Get your API key from: https://openrouter.ai/keys"
    echo "Then: export OPENROUTER_API_KEY='your-key-here'"
    exit 1
fi

echo "Asking $model_id..."
echo ""

# Properly escape the prompt for JSON to prevent injection
prompt_json=$(python3 -c "import json, sys; print(json.dumps(sys.argv[1]))" "$prompt")

# Call OpenRouter API with timeout
response=$(curl -s --max-time 30 https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$model_id\",
    \"messages\": [
      {\"role\": \"user\", \"content\": $prompt_json}
    ]
  }")

# Check for timeout
if [ $? -eq 28 ]; then
    echo "Error: Request timed out after 30 seconds"
    exit 1
fi

# Extract and display response
echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'choices' in data and len(data['choices']) > 0:
        print(data['choices'][0]['message']['content'])
    elif 'error' in data:
        print(f\"Error: {data['error']['message']}\", file=sys.stderr)
        sys.exit(1)
    else:
        print('Unexpected response format', file=sys.stderr)
        sys.exit(1)
except json.JSONDecodeError as e:
    print(f'Failed to parse response: {e}', file=sys.stderr)
    sys.exit(1)
"
```

## Notes

- Requires `OPENROUTER_API_KEY` environment variable
- Free models have rate limits (20 requests/min)
- Paid models incur costs per request
- Get API key from: https://openrouter.ai/keys
