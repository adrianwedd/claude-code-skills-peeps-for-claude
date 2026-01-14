# Basic Usage Examples

This file contains practical examples of using Claude AI Skills in real development scenarios.

## Quick Start

### Ask a Simple Question
```bash
/ask-model gemini "What's the difference between REST and GraphQL?"
```

### Get Multiple Perspectives
```bash
/ask-model gemini "Best practices for error handling in APIs?"
/ask-codex "Best practices for error handling in APIs?"
/ask-model claude "Best practices for error handling in APIs?"
```

## Code Generation

### Python Function
```bash
/ask-codex "Write a Python function to retry HTTP requests with exponential backoff"
```

### TypeScript Interface
```bash
/ask-model mistral "Generate TypeScript interfaces for a user management API"
```

### Database Schema
```bash
/ask-model llama-70b "Design a PostgreSQL schema for a blog platform with users, posts, and comments"
```

## Code Review

### Security Review
```bash
/ask-gemini "Review this authentication function for security vulnerabilities:
def authenticate_user(username, password):
    user = db.get_user(username)
    if user and user.password == password:
        return generate_token(user)
    return None"
```

### Performance Analysis
```bash
/ask-codex "Analyze the time complexity of this sorting function and suggest optimizations:
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr"
```

## Architecture & Design

### System Design
```bash
/ask-model gpt4 "Design a microservices architecture for an e-commerce platform. Include services for user management, product catalog, order processing, and payment."
```

### Database Choice
```bash
/ask-model claude "Compare PostgreSQL, MongoDB, and Redis for a real-time chat application. Consider scalability, consistency, and performance."
```

### API Design
```bash
/ask-gemini "Design a RESTful API for a task management system. Include endpoints for users, projects, tasks, and comments."
```

## Debugging & Problem Solving

### Error Analysis
```bash
/ask-codex "This JavaScript code throws 'Cannot read property of undefined'. Help me debug:
const data = response.data.users;
const userName = data.map(user => user.profile.name);"
```

### Regex Help
```bash
/ask-model mistral "Write a regex to validate email addresses that supports international domains"
```

### Performance Issue
```bash
/ask-model gpt4 "My React component re-renders too often. Identify the issue and suggest fixes:
function UserList({ users, searchTerm }) {
    const filteredUsers = users.filter(user =>
        user.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
    return (
        <div>
            {filteredUsers.map(user => (
                <UserCard key={user.id} user={user} />
            ))}
        </div>
    );
}"
```

## Learning & Research

### Technology Comparison
```bash
/ask-model gemini "Compare React, Vue, and Svelte for building single-page applications. Include pros, cons, and use cases."
```

### Best Practices
```bash
/ask-gemini "What are the current best practices for securing Node.js applications?"
```

### Algorithm Explanation
```bash
/ask-codex "Explain how the A* pathfinding algorithm works and provide a Python implementation"
```

## Advanced Workflows

### Multi-Step Code Generation
```bash
# Step 1: Design the API
/ask-model claude "Design a REST API for a library management system"

# Step 2: Implement endpoints
/ask-codex "Implement the book management endpoints in Express.js"

# Step 3: Add tests
/ask-model mistral "Generate comprehensive unit tests for these endpoints"

# Step 4: Add documentation
/ask-gemini "Create API documentation for these endpoints"
```

### Code Refactoring Workflow
```bash
# Step 1: Analyze current code
/ask-codex "Analyze this legacy function for maintainability issues"

# Step 2: Suggest improvements
/ask-model gpt4 "Suggest refactoring strategies for this code"

# Step 3: Implement refactor
/ask-codex "Refactor this function following SOLID principles"

# Step 4: Verify changes
/ask-gemini "Review the refactored code for potential issues"
```

## Model-Specific Use Cases

### Free Models

**Gemini (Fast, versatile)**
```bash
/ask-model gemini "Quick explanation of Docker containers"
/ask-model gemini "Code review for this React component"
```

**Llama 3.3 70B (Powerful reasoning)**
```bash
/ask-model llama-70b "Design a complex distributed system architecture"
/ask-model llama-70b "Analyze this algorithm's computational complexity"
```

**Mistral (Coding-focused)**
```bash
/ask-model mistral "Generate optimized Python code for data processing"
/ask-model mistral "Debug this performance issue in my application"
```

### Paid Models

**GPT-4 Turbo (Best general performance)**
```bash
/ask-model gpt4 "Comprehensive security audit of this application"
/ask-model gpt4 "Design enterprise-grade system architecture"
```

**Claude 3.5 Sonnet (Excellent reasoning)**
```bash
/ask-model claude "Analyze complex business logic requirements"
/ask-model claude "Review system design for edge cases"
```

**Claude Opus 4 (Most capable)**
```bash
/ask-model opus "Critical security review for production deployment"
/ask-model opus "Complex algorithm optimization with formal analysis"
```

## Environment-Specific Examples

### Development Environment
```bash
# Quick debugging
/ask-model gemini "Why is my development server not starting?"

# Code generation
/ask-codex "Generate a development configuration for webpack"
```

### Production Environment
```bash
# Security review (use paid models for critical systems)
/ask-model gpt4 "Security review of this production deployment script"

# Performance optimization
/ask-model claude "Optimize this database query for production load"
```

### Learning Environment
```bash
# Concept explanation
/ask-gemini "Explain microservices patterns with examples"

# Practice problems
/ask-codex "Generate practice coding problems for learning algorithms"
```

## Tips for Better Results

### Be Specific
```bash
# Instead of: "Help with Python"
/ask-codex "Write a Python function to parse CSV files with error handling for missing columns"
```

### Provide Context
```bash
# Good: Include relevant code context
/ask-gemini "Review this authentication middleware for security issues:
[paste your code here]
This is for a Node.js Express application handling user sessions."
```

### Use Appropriate Models
```bash
# Simple questions: use free models
/ask-model gemini "What is REST API?"

# Complex analysis: use paid models
/ask-model gpt4 "Comprehensive security architecture review for banking application"
```

### Iterate and Refine
```bash
# Start broad
/ask-model gemini "How to implement caching in a web application?"

# Get specific
/ask-codex "Implement Redis caching for this Express.js route handler"

# Refine further
/ask-model claude "Optimize this Redis caching implementation for high concurrency"
```