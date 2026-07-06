# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| >= 3.3  | :white_check_mark: |
| < 3.3   | :x:                |

## Reporting a Vulnerability

LexRAG is a legal research tool designed for professional use. Security is a top priority.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them directly to:

- **Email**: security@eulogik.com
- **PGP Key**: Available on request

### What to include

- Type of vulnerability
- Steps to reproduce
- Affected versions
- Potential impact
- Suggested fix (if any)

### What to expect

- **Acknowledgment** within 48 hours
- **Status update** within 5 business days
- **Fix timeline** communicated within 10 business days

## Best Practices for Deployment

1. **API Keys**: Never commit `.env` to version control. Use environment variables or a secrets manager.
2. **Network**: Run the server behind a reverse proxy (nginx, Caddy) with TLS in production.
3. **Authentication**: Add API key authentication for production deployments (see `LEXRAG_API_KEY`).
4. **Database**: Regularly backup `data/lexrag.db` and `qdrant_storage/`.
5. **Updates**: Keep dependencies updated: `pip install --upgrade -r requirements.txt`

## Hallucination & Accuracy Disclaimer

LexRAG uses Retrieval-Augmented Generation (RAG) to ground answers in legal documents. However:
- AI models may still produce inaccurate or incomplete responses
- Always verify AI-generated legal analysis against primary sources
- LexRAG is a research assistance tool, not a substitute for qualified legal counsel

## Responsible Disclosure

We believe in responsible disclosure. If you discover a vulnerability, please give us a reasonable time to fix it before public disclosure.
