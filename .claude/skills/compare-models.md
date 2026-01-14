# Compare Models

Quick comparison of multiple models on same scenarios.

## Usage

```
/compare-models [scenarios] [preset]
```

## Presets

- `free` - Top 3 free models (Gemini, Llama 3B, Mistral)
- `free-all` - All 5 tested free models
- `paid` - GPT-4 Turbo + Claude 3.5 Sonnet
- `mixed` - Gemini (free) + GPT-4 + Claude

## Examples

```bash
# Compare top free models
/compare-models examples/test_scenarios.jsonl free

# Compare paid models
/compare-models data/splits/sanity_pack.jsonl paid

# Mixed comparison
/compare-models examples/test_scenarios.jsonl mixed

# Compare free models on episodes (multi-turn)
/compare-models data/episodes/stateful_episodes_v0.1.jsonl free

# Compare all models on episodes
/compare-models data/episodes/stateful_episodes_v0.1.jsonl free-all
```

## Implementation

```bash
#!/bin/bash
scenarios="${1:-examples/test_scenarios.jsonl}"
preset="${2:-free}"

case "$preset" in
    free)
        models=(
            "google/gemini-2.0-flash-exp:free"
            "meta-llama/llama-3.2-3b-instruct:free"
            "mistralai/devstral-2512:free"
        )
        ;;
    free-all)
        models=(
            "google/gemini-2.0-flash-exp:free"
            "meta-llama/llama-3.2-3b-instruct:free"
            "mistralai/devstral-2512:free"
            "meta-llama/llama-3.3-70b-instruct:free"
            "nvidia/nemotron-3-nano-30b-a3b:free"
        )
        ;;
    paid)
        models=(
            "openai/gpt-4-turbo"
            "anthropic/claude-3.5-sonnet"
        )
        ;;
    mixed)
        models=(
            "google/gemini-2.0-flash-exp:free"
            "openai/gpt-4-turbo"
            "anthropic/claude-3.5-sonnet"
        )
        ;;
    *)
        echo "Unknown preset: $preset"
        echo "Available: free, free-all, paid, mixed"
        exit 1
        ;;
esac

# Create comparison directory with timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
output="runs/comparison_${preset}_${timestamp}"

echo "Comparing ${#models[@]} models on $scenarios"
echo "Output: $output"
echo ""

# Build command
cmd="python tools/benchmarks/run_benchmark_http.py --scenarios \"$scenarios\" --models"
for model in "${models[@]}"; do
    cmd="$cmd \"$model\""
done
cmd="$cmd --output \"$output\""

eval "$cmd"

echo ""
echo "✓ Comparison complete!"
echo ""
echo "Next steps:"
echo "  1. View traces: cat $output/*.jsonl | jq"
echo "  2. Generate report: python tools/benchmarks/score_report_v1.0.py --traces $output/*.jsonl"
```
