# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### Reporting Process

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. **Report via GitHub Security Advisories**: Go to the Security tab and click "Report a vulnerability"
3. **Or email**: Include "SECURITY" in the subject line (update with maintainer email)
4. **Provide details**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium/Low: Next release cycle
- **Disclosure**: Coordinated disclosure after fix is released

## Security Best Practices for Users

### API Key Protection

**DO:**
- ✅ Store API keys in environment variables
- ✅ Use `.env` files (included in `.gitignore`)
- ✅ Rotate API keys regularly
- ✅ Use separate keys for dev/prod
- ✅ Review API key permissions

**DON'T:**
- ❌ Commit API keys to repositories
- ❌ Share API keys in chat/email
- ❌ Use production keys in development
- ❌ Store keys in code files
- ❌ Share your `.env` file

### Environment Setup

```bash
# Good ✅ - Environment variable
export OPENROUTER_API_KEY="sk-or-v1-..."

# Good ✅ - .env file (gitignored)
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env

# Bad ❌ - Hardcoded in script
OPENROUTER_API_KEY="sk-or-v1-..."  # Never do this!
```

### Input Validation

This project validates all user inputs to prevent:
- JSON injection attacks
- Shell injection
- Command injection
- Path traversal

**As of v1.0.0:**
- ✅ All user prompts are properly escaped before JSON serialization
- ✅ API requests have 30-second timeouts
- ✅ Error messages don't expose sensitive information

## Security Features

### Current Protections

1. **Input Sanitization**
   - All user inputs are escaped before use in JSON/API calls
   - Special characters (quotes, newlines) are handled safely
   - Uses Python's `json.dumps()` for safe JSON serialization

2. **Network Security**
   - All API calls use HTTPS
   - 30-second timeout on all requests
   - No sensitive data in URL parameters

3. **API Key Protection**
   - Keys stored in environment variables only
   - `.gitignore` prevents accidental commits
   - Keys never logged or displayed in output
   - No keys in error messages

4. **Error Handling**
   - Graceful failure without exposing system details
   - Clear error messages without sensitive information
   - Proper exit codes for error conditions

### Known Limitations

1. **Shell Execution**
   - Skills execute bash scripts with user input
   - Input is validated but skills run with user's permissions
   - Use caution when running untrusted skill files

2. **Third-Party APIs**
   - Skills make requests to third-party services (OpenRouter, OpenAI, Google)
   - Data is transmitted to these services
   - Review each service's privacy policy

3. **API Keys in Memory**
   - API keys are in environment variables (accessible to all processes)
   - Consider using more secure credential stores for sensitive environments

## Security Considerations for Contributors

### When Adding Features

**Always:**
- ✅ Validate all user inputs
- ✅ Escape data before using in JSON/shell/URLs
- ✅ Use parameterized queries/commands
- ✅ Handle errors gracefully
- ✅ Add timeouts to network calls
- ✅ Test with malicious inputs

**Never:**
- ❌ Use `eval` with user input
- ❌ Execute arbitrary code
- ❌ Trust user input without validation
- ❌ Expose sensitive data in errors
- ❌ Use `curl` without timeouts

### Code Review Checklist

Before merging code, verify:

- [ ] No hardcoded credentials
- [ ] User input is validated
- [ ] Proper escaping for JSON/shell
- [ ] Error messages don't leak sensitive info
- [ ] Network calls have timeouts
- [ ] No new security warnings

### Example: Safe JSON Construction

```bash
# Good ✅ - Proper escaping
prompt_json=$(python3 -c "import json, sys; print(json.dumps(sys.argv[1]))" "$prompt")
curl -d "{\"content\": $prompt_json}"

# Bad ❌ - Vulnerable to injection
curl -d "{\"content\": \"$prompt\"}"  # DON'T DO THIS
```

### Example: Safe Shell Execution

```bash
# Good ✅ - Quoted variables
if [ -f "$user_file" ]; then
    cat "$user_file"
fi

# Bad ❌ - Unquoted (vulnerable to injection)
if [ -f $user_file ]; then
    cat $user_file
fi
```

## Vulnerability Disclosure Policy

### Scope

**In scope:**
- Code in this repository
- Installation scripts
- Skill implementations
- Documentation that could lead to security issues

**Out of scope:**
- Third-party dependencies (report to their maintainers)
- Issues in Claude Code itself
- Social engineering attacks
- Physical attacks

### Severity Levels

**Critical:**
- Remote code execution
- API key exposure
- Authentication bypass
- Data exfiltration

**High:**
- Injection vulnerabilities
- Privilege escalation
- Denial of service
- Information disclosure

**Medium:**
- Error message information leakage
- Missing input validation (non-exploitable)
- Insecure defaults

**Low:**
- Best practice violations
- Documentation issues
- Minor information disclosure

## Security Updates

### How We Communicate

Security updates are announced via:
1. GitHub Security Advisories
2. Release notes with `[SECURITY]` tag
3. CHANGELOG.md with Security section

### Applying Updates

```bash
# Check current version
cat CHANGELOG.md

# Pull latest security fixes
git pull origin main

# Reinstall skills
./install.sh

# Verify update
grep "## \[" CHANGELOG.md | head -1
```

## Responsible Disclosure

We believe in responsible disclosure and will:
- Work with you to understand the issue
- Keep you informed of progress
- Credit you in release notes (if desired)
- Coordinate public disclosure timing

Thank you for helping keep Claude AI Skills secure! 🔒

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OpenRouter Security](https://openrouter.ai/docs/security)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

---

**Last Updated:** 2025-01-14
