#!/bin/bash
# Claude AI Skills Installation Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Claude AI Skills Installation${NC}"
echo "================================="

# Check if we're in a project directory
if [ ! -d ".claude" ] && [ ! -d ".git" ]; then
    echo -e "${YELLOW}Warning: No .claude or .git directory found.${NC}"
    echo "Are you in a project directory? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Please run this script from your project directory."
        exit 1
    fi
fi

# Create .claude/skills directory
echo -e "${BLUE}Creating .claude/skills directory...${NC}"
mkdir -p .claude/skills

# Copy skills
echo -e "${BLUE}Installing skills...${NC}"
cp skills/*.md .claude/skills/

# Skills are markdown files containing bash scripts, no need to make executable
echo -e "${GREEN}✓ Skills copied successfully${NC}"

echo -e "${GREEN}✓ Skills installed successfully!${NC}"
echo ""

# Check for API dependencies
echo -e "${BLUE}Checking dependencies...${NC}"

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${YELLOW}⚠ OpenRouter API key not found${NC}"
    echo "To use /ask-model, set your OpenRouter API key:"
    echo "  export OPENROUTER_API_KEY='sk-or-v1-your-key-here'"
    echo "  Get a key from: https://openrouter.ai/keys"
    echo ""
fi

# Check for codex CLI
if ! command -v codex &> /dev/null; then
    echo -e "${YELLOW}⚠ codex-cli not found${NC}"
    echo "To use /ask-codex, install codex-cli:"
    echo "  npm install -g codex-cli"
    echo "  # or pip install codex-cli"
    echo ""
fi

# Check for gemini CLI
if ! command -v gemini &> /dev/null; then
    echo -e "${YELLOW}⚠ gemini-cli not found${NC}"
    echo "To use /ask-gemini, install gemini-cli:"
    echo "  npm install -g @google-cloud/generative-ai-cli"
    echo ""
fi

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Set up API keys (see warnings above)"
echo "2. Test with: /ask-model gemini 'Hello world'"
echo "3. Read README.md for full usage guide"
echo ""
echo -e "${GREEN}Happy prompting! 🚀${NC}"