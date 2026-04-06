"""
Pydantic models for the Support Ticket Resolution Environment.

Defines Action, Observation, and State types following the OpenEnv specification.
All models are typed, validated, and serializable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All valid actions an agent can take."""
    CLASSIFY_TICKET = "classify_ticket"
    ASSIGN_PRIORITY = "assign_priority"
    WRITE_RESPONSE = "write_response"
    ESCALATE_TICKET = "escalate_ticket"
    CLOSE_TICKET = "close_ticket"
    REOPEN_TICKET = "reopen_ticket"
    ADD_TAG = "add_tag"
    REQUEST_INFO = "request_info"
    MERGE_TICKETS = "merge_tickets"
    NOOP = "noop"


class TicketCategory(str, Enum):
    """Supported ticket categories."""
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    GENERAL = "general"


class Priority(str, Enum):
    """Ticket priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    """Lifecycle states for a ticket."""
    NEW = "new"
    OPEN = "open"
    PENDING_CUSTOMER = "pending_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    A single structured action the agent can perform.

    Only the fields relevant to the action_type need to be populated.
    """

    action_type: ActionType = Field(
        ...,
        description="The type of action to perform.",
    )
    ticket_id: Optional[str] = Field(
        default=None,
        description="Target ticket identifier.",
    )
    category: Optional[TicketCategory] = Field(
        default=None,
        description="Category for classify_ticket.",
    )
    priority: Optional[Priority] = Field(
        default=None,
        description="Priority for assign_priority.",
    )
    response_text: Optional[str] = Field(
        default=None,
        description="Response body for write_response / request_info.",
    )
    tag: Optional[str] = Field(
        default=None,
        description="Tag to add via add_tag.",
    )
    target_ticket_id: Optional[str] = Field(
        default=None,
        description="Secondary ticket id for merge_tickets.",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Extra structured metadata.",
    )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

class TicketView(BaseModel):
    """Serialisable snapshot of a single ticket shown to the agent."""

    ticket_id: str
    subject: str
    body: str
    category: Optional[TicketCategory]
    priority: Optional[Priority]
    status: TicketStatus
    tags: list[str]
    is_escalated: bool
    has_response: bool
    customer_replied_after_response: bool


class Observation(BaseModel):
    """
    What the agent sees after reset() or step().

    Mirrors the Gym-style observation dict but is strongly typed.
    """

    task_id: str = Field(..., description="Identifier of the current task.")
    episode_id: str = Field(..., description="Unique episode identifier.")
    step: int = Field(..., ge=0, description="Current step count (0-based).")
    max_steps: int = Field(..., ge=1, description="Maximum steps allowed.")
    tickets: list[TicketView] = Field(
        default_factory=list,
        description="Current state of all tickets.",
    )
    action_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Log of actions taken so far.",
    )
    last_reward: float = Field(
        default=0.0,
        description="Reward from the most recent step.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of rewards this episode.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has terminated.",
    )
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic / hint data.",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(BaseModel):
    """
    Full internal state returned by state().

    Contains everything the environment tracks (more than Observation).
    """

    task_id: str
    episode_id: str
    step: int
    max_steps: int
    tickets: list[dict[str, Any]]
    action_history: list[dict[str, Any]]
    cumulative_reward: float
    done: bool
    grader_score: Optional[float] = Field(
        default=None,
        description="Final grader score (set when done=True).",
    )
    info: dict[str, Any]
