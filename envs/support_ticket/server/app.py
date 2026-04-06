"""
FastAPI server for the Support Ticket Resolution Environment.

Exposes REST endpoints that mirror the Gym-style API:
  POST /reset   → initial observation
  POST /step    → observation, reward, done, info
  GET  /state   → full internal state
  GET  /tasks   → list of available tasks
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on the path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import Action, Observation, State
from server.environment import SupportTicketEnv
from server.tasks import ALL_TASKS, TASK_IDS


# ---------------------------------------------------------------------------
# Lifespan / app setup
# ---------------------------------------------------------------------------

_env: SupportTicketEnv | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = SupportTicketEnv()
    yield
    _env = None


app = FastAPI(
    title="Support Ticket Resolution Environment",
    description="OpenEnv-compatible environment for customer support ticket management.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str | None = Field(default=None, description="Task identifier.")
    seed: int | None = Field(default=None, description="Random seed (ignored, deterministic env).")


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Root endpoint for platform health checks and space probes."""
    return {"message": "Support Ticket Resolution Environment is running.", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Return metadata for all available tasks."""
    return {
        "tasks": [
            {
                "task_id": tid,
                "name": ALL_TASKS[tid].name,
                "difficulty": ALL_TASKS[tid].difficulty,
                "description": ALL_TASKS[tid].description,
                "max_steps": ALL_TASKS[tid].max_steps,
            }
            for tid in TASK_IDS
        ],
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest | None = None):
    """Reset the environment and return the initial observation."""
    global _env
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")

    try:
        task_id = req.task_id if req else None
        seed = req.seed if req else None
        obs = _env.reset(task_id=task_id, seed=seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """Execute an action and return (observation, reward, done, info)."""
    global _env
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")

    try:
        obs, reward, done, info = _env.step(req.action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
def get_state():
    """Return the full internal state of the environment."""
    global _env
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")

    return _env.state()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
