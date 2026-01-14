# Compare Answers

Ask the same question to multiple AI models and compare their responses.

## Usage

```
/compare-answers [preset] <your question>
```

## Presets

- `free` - Top 3 free models (Gemini, Llama 70B, Mistral)
- `fast` - Fastest free models (Gemini, Llama 3B)
- `all-free` - All 5 tested free models
- `paid` - GPT-4 + Claude 3.5
- `best` - Best models across free and paid (Gemini, GPT-4, Claude)

Default: `free`

## Examples

```bash
# Compare free models
/compare-answers "What's the best way to handle API rate limiting?"

# Specify preset
/compare-answers fast "Explain quicksort algorithm"

# Compare paid models for critical question
/compare-answers paid "What are the security risks in this auth flow?"

# Get diverse perspectives
/compare-answers best "Design a caching strategy for high-traffic API"
```

## When to Use

- Important decisions needing multiple perspectives
- Comparing model capabilities
- Getting consensus on best practices
- Validating critical answers

## Implementation

```bash
#!/bin/bash

# Parse preset and prompt
if [ $# -eq 0 ]; then
    echo "Usage: /compare-answers [preset] <your question>"
    echo ""
    echo "Presets: free (default), fast, all-free, paid, best"
    exit 1
fi

# Check if first arg is a preset
case "$1" in
    free|fast|all-free|paid|best)
        preset="$1"
        shift
        prompt="$*"
        ;;
    *)
        preset="free"
        prompt="$*"
        ;;
esac

if [ -z "$prompt" ]; then
    echo "Error: No question provided"
    exit 1
fi

# Define model sets
case "$preset" in
    free)
        models=("google/gemini-2.0-flash-exp:free" "meta-llama/llama-3.3-70b-instruct:free" "mistralai/devstral-2512:free")
        names=("Gemini 2.0" "Llama 70B" "Mistral")
        ;;
    fast)
        models=("google/gemini-2.0-flash-exp:free" "meta-llama/llama-3.2-3b-instruct:free")
        names=("Gemini 2.0" "Llama 3B")
        ;;
    all-free)
        models=("google/gemini-2.0-flash-exp:free" "meta-llama/llama-3.3-70b-instruct:free" "mistralai/devstral-2512:free" "meta-llama/llama-3.2-3b-instruct:free" "nvidia/nemotron-3-nano-30b-a3b:free")
        names=("Gemini 2.0" "Llama 70B" "Mistral" "Llama 3B" "Nemotron 30B")
        ;;
    paid)
        models=("openai/gpt-4-turbo" "anthropic/claude-3.5-sonnet")
        names=("GPT-4 Turbo" "Claude 3.5")
        ;;
    best)
        models=("google/gemini-2.0-flash-exp:free" "openai/gpt-4-turbo" "anthropic/claude-3.5-sonnet")
        names=("Gemini 2.0" "GPT-4 Turbo" "Claude 3.5")
        ;;
esac

# Check for API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY not set"
    exit 1
fi

echo "Question: $prompt"
echo ""
echo "Comparing ${#models[@]} models..."
echo "========================================"
echo ""

# Query each model
for i in "${!models[@]}"; do
    model="${models[$i]}"
    name="${names[$i]}"

    echo "[$((i+1))/${#models[@]}] $name"
    echo "----------------------------------------"

    # Properly escape the prompt for JSON to prevent injection
    prompt_json=$(python3 -c "import json, sys; print(json.dumps(sys.argv[1]))" "$prompt")

    response=$(curl -s --max-time 30 https://openrouter.ai/api/v1/chat/completions \
      -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$model\",
        \"messages\": [{\"role\": \"user\", \"content\": $prompt_json}]
      }")

    # Check for timeout
    if [ $? -eq 28 ]; then
        echo "Error: Request timed out after 30 seconds"
        continue
    fi

    echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'choices' in data and len(data['choices']) > 0:
        print(data['choices'][0]['message']['content'])
    elif 'error' in data:
        print(f\"Error: {data['error']['message']}\")
    else:
        print('No response')
except:
    print('Failed to parse response')
"

    echo ""
    echo ""

    # Rate limiting for free models
    if [[ "$model" == *":free" ]]; then
        sleep 3
    fi
done

echo "========================================"
echo "✓ Comparison complete"
```

## Notes

- Free models: Rate limited to 20 req/min (3s delays)
- Paid models: Costs apply per request
- Useful for critical decisions
- Shows different model perspectives
