# Benchmark HTTP (OpenRouter)

Run HTTP API benchmarks through OpenRouter with 32+ free models.

## Usage

```
/benchmark-http [scenarios] [models...] [options]
```

## Examples

```bash
# Quick test with free Gemini
/benchmark-http examples/test_scenarios.jsonl gemini

# Multi-model comparison
/benchmark-http examples/test_scenarios.jsonl gemini llama mistral

# Full sanity pack
/benchmark-http data/splits/sanity_pack.jsonl gemini --limit 20

# Episode benchmarking
/benchmark-http data/episodes/stateful_episodes_v0.1.jsonl gemini --limit 1
```

## Model Shortcuts

- `gemini` → `google/gemini-2.0-flash-exp:free`
- `llama` → `meta-llama/llama-3.2-3b-instruct:free`
- `mistral` → `mistralai/devstral-2512:free`
- `llama-70b` → `meta-llama/llama-3.3-70b-instruct:free`
- `gpt4` → `openai/gpt-4-turbo`
- `claude` → `anthropic/claude-3.5-sonnet`

## Options

- `--limit N` - Run only N scenarios
- `--output DIR` - Output directory (default: traces/)
- `--daily-limit N` - Override daily limit for free models

## Implementation

```bash
#!/bin/bash
scenarios="${1:-examples/test_scenarios.jsonl}"
shift

# Parse model shortcuts
models=()
limit=""
output="traces/"
daily_limit=""

for arg in "$@"; do
    case "$arg" in
        --limit)
            shift
            limit="--limit $1"
            shift
            ;;
        --output)
            shift
            output="$1"
            shift
            ;;
        --daily-limit)
            shift
            daily_limit="--daily-limit $1"
            shift
            ;;
        gemini)
            models+=("google/gemini-2.0-flash-exp:free")
            ;;
        llama)
            models+=("meta-llama/llama-3.2-3b-instruct:free")
            ;;
        mistral)
            models+=("mistralai/devstral-2512:free")
            ;;
        llama-70b)
            models+=("meta-llama/llama-3.3-70b-instruct:free")
            ;;
        gpt4)
            models+=("openai/gpt-4-turbo")
            ;;
        claude)
            models+=("anthropic/claude-3.5-sonnet")
            ;;
        *)
            # Assume it's a full model ID
            models+=("$arg")
            ;;
    esac
done

# Default to gemini if no models specified
if [ ${#models[@]} -eq 0 ]; then
    models=("google/gemini-2.0-flash-exp:free")
fi

# Build command
cmd="python tools/benchmarks/run_benchmark_http.py --scenarios \"$scenarios\" --models"
for model in "${models[@]}"; do
    cmd="$cmd \"$model\""
done
cmd="$cmd --output \"$output\" $limit $daily_limit"

echo "Running: $cmd"
eval "$cmd"
```
