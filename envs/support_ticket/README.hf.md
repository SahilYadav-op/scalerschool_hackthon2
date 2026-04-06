---
title: Support Ticket Resolution Environment
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Support Ticket Resolution Environment

This is a production-grade environment that simulates a **customer support ticket resolution system**. 

## Features

- **FastAPI Backend**: Real-time ticket management API.
- **Support for Multiple Tasks**: From basic classification to complex triage.
- **Dockerized Deployment**: Ready for Hugging Face Spaces.

## Quick Start (Local)

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face.
2. Select **Docker** as the SDK.
3. Upload the following files/folders:
   - `Dockerfile`
   - `requirements.txt`
   - `models.py`
   - `server/`
   - `README.md` (this file)
4. Your API will be available at `https://<your-username>-<space-name>.hf.space`.

## API Endpoints

- `GET /health` - Health check.
- `GET /tasks` - List all available tasks.
- `POST /reset` - Reset the environment for a specific task.
- `POST /step` - Take an action in the environment.
- `GET /state` - Get the full internal state.
