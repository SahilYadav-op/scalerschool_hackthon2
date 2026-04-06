"""
Python client for the Support Ticket Resolution Environment.

Provides a clean, local-facing API that talks to the remote FastAPI server.
Mirrors the Gym-style interface: reset(), step(), state().
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from models import Action, Observation, State


class SupportTicketClient:
    """
    HTTP client for the Support Ticket Resolution Environment.

    Usage:
        client = SupportTicketClient(base_url="http://localhost:8000")
        obs = client.reset(task_id="task_1_classify_prioritize")
        action = Action(action_type=ActionType.CLASSIFY_TICKET, ticket_id="T-101", category=TicketCategory.ACCOUNT)
        obs, reward, done, info = client.step(action)
        state = client.state()
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("ENV_BASE_URL", "http://localhost:8000")).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        """Reset the environment and return the initial observation."""
        payload: dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed

        resp = self._client.post("/reset", json=payload if payload else None)
        resp.raise_for_status()
        return Observation(**resp.json())

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute an action and return (observation, reward, done, info)."""
        payload = {"action": action.model_dump(mode="json", exclude_none=True)}
        resp = self._client.post("/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = Observation(**data["observation"])
        return obs, data["reward"], data["done"], data["info"]

    def state(self) -> State:
        """Return the full internal state."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return State(**resp.json())

    def list_tasks(self) -> dict[str, Any]:
        """Return metadata for all available tasks."""
        resp = self._client.get("/tasks")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict[str, Any]:
        """Check server health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
