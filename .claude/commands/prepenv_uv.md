---
allowed-tools: Bash(deactivate:*), Bash(rm: -rf .venv), Bash(uv venv), Bash(source: .venv/bin/activate), Bash(source: .venv/Scripts/activate)
description: Creates a fresh Python virtual environment (venv) and activates it.
---

## Context

The current state of your environment needs to be prepared for a new virtual environment setup.

## Your task

Your task is to implement the following steps:

1.  **Deactivate** any existing virtual environment to ensure a clean slate.
2.  **Remove** the `.venv` directory if it already exists, ensuring a fresh installation.
3.  **Create** a new virtual environment named `.venv` using uv module.
4.  **Activate** the newly created virtual environment.
5.  **Install** libraries using uv and pip based on requirements.txt

***

**Command Implementation (for Unix/Linux/macOS):**

```bash
deactivate 2>/dev/null || true # Safely deactivate any active venv, suppressing errors if none is active
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```