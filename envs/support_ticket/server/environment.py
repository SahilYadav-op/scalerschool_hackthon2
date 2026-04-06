"""
Core environment logic for the Support Ticket Resolution Environment.

Implements the Gym-style API: reset(), step(), state().
Handles deterministic state transitions, reward computation, and episode lifecycle.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from models import Action, ActionType, Observation, State, TicketView
from server.graders import grade
from server.tasks import ALL_TASKS, TaskDef


class SupportTicketEnv:
    """
    A deterministic, reproducible support-ticket resolution environment.

    The agent interacts with a queue of customer support tickets and must
    classify, prioritise, respond to, escalate, merge, and close them
    efficiently.
    """

    def __init__(self, task_id: str | None = None):
        self._task_id: str | None = task_id
        self._task: TaskDef | None = None
        self._episode_id: str = ""
        self._step: int = 0
        self._tickets: list[dict[str, Any]] = []
        self._action_history: list[dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._merged_into: dict[str, str] = {}  # child -> parent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        """
        Reset the environment and return the initial observation.

        Args:
            task_id: Override the task for this episode.
            seed: Ignored (environment is deterministic), kept for Gym compat.

        Returns:
            Initial Observation.
        """
        tid = task_id or self._task_id
        if tid is None:
            tid = "task_1_classify_prioritize"
        if tid not in ALL_TASKS:
            raise ValueError(f"Unknown task_id={tid!r}. Available: {list(ALL_TASKS.keys())}")

        self._task_id = tid
        self._task = ALL_TASKS[tid]
        self._episode_id = f"{tid}-{uuid.uuid4().hex[:8]}"
        self._step = 0
        self._tickets = copy.deepcopy(self._task.tickets)
        self._action_history = []
        self._cumulative_reward = 0.0
        self._done = False
        self._merged_into = {}

        # In a real RL env, seed would initialize the RNG. 
        # This environment is deterministic by design.
        if seed is not None:
            pass 

        return self._make_observation(last_reward=0.0)

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Execute one action and return (observation, reward, done, info).

        Args:
            action: A validated Action model.

        Returns:
            observation, reward, done, info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        reward = self._compute_reward(action)
        self._cumulative_reward += reward
        self._apply_action(action)

        done = self._check_done()
        self._done = done

        info: dict[str, Any] = {}
        if done:
            assert self._task is not None
            assert self._task_id is not None
            grader_score = grade(
                task_id=self._task_id,
                tickets=self._tickets,
                expected=self._task.expected,
                step=self._step,
                max_steps=self._task.max_steps,
                action_history=self._action_history,
            )
            info["grader_score"] = grader_score

        obs = self._make_observation(last_reward=reward)
        return obs, reward, done, info

    def state(self) -> State:
        """
        Return the full internal state (more detailed than Observation).

        Returns:
            State model with all internal fields.
        """
        grader_score = None
        if self._task and self._task_id:
            grader_score = grade(
                task_id=self._task_id,
                tickets=self._tickets,
                expected=self._task.expected,
                step=self._step,
                max_steps=self._task.max_steps,
                action_history=self._action_history,
            )

        return State(
            task_id=self._task_id or "unknown",
            episode_id=self._episode_id,
            step=self._step,
            max_steps=self._task.max_steps if self._task else 0,
            tickets=copy.deepcopy(self._tickets),
            action_history=copy.deepcopy(self._action_history),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            grader_score=grader_score,
            info={
                "task_name": self._task.name if self._task else None,
                "task_difficulty": self._task.difficulty if self._task else None,
                "is_terminal": self._done
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        """Find a ticket by id, returning None if merged or missing."""
        for t in self._tickets:
            if t["ticket_id"] == ticket_id:
                return t
        return None

    def _make_observation(self, *, last_reward: float) -> Observation:
        """Build an Observation from current internal state."""
        ticket_views = []
        for t in self._tickets:
            tid = t["ticket_id"]
            if tid in self._merged_into:
                continue
            ticket_views.append(TicketView(
                ticket_id=tid,
                subject=t["subject"],
                body=t["body"],
                category=t.get("category"),
                priority=t.get("priority"),
                status=t["status"],
                tags=list(t.get("tags", [])),
                is_escalated=t.get("is_escalated", False),
                has_response=t.get("has_response", False),
                customer_replied_after_response=t.get("customer_replied_after_response", False),
            ))

        return Observation(
            task_id=self._task_id or "",
            episode_id=self._episode_id,
            step=self._step,
            max_steps=self._task.max_steps if self._task else 0,
            tickets=ticket_views,
            action_history=copy.deepcopy(self._action_history),
            last_reward=round(last_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            info={
                "task_name": self._task.name if self._task else None,
                "difficulty": self._task.difficulty if self._task else None,
                "remaining_steps": (self._task.max_steps - self._step) if self._task else 0,
            },
        )

    def _apply_action(self, action: Action) -> None:
        """Mutate internal state according to the action."""
        action_log: dict[str, Any] = {
            "step": self._step,
            "action_type": action.action_type.value,
            "ticket_id": action.ticket_id,
            "target_ticket_id": action.target_ticket_id,
        }

        at = action.action_type

        if at == ActionType.NOOP:
            action_log["detail"] = "no operation"

        elif at == ActionType.CLASSIFY_TICKET:
            ticket = self._find_ticket(action.ticket_id) if action.ticket_id else None
            if ticket and action.category:
                ticket["category"] = action.category.value
                action_log["detail"] = f"classified as {action.category.value}"
            else:
                action_log["detail"] = "invalid – missing ticket_id or category"

        elif at == ActionType.ASSIGN_PRIORITY:
            ticket = self._find_ticket(action.ticket_id) if action.ticket_id else None
            if ticket and action.priority:
                ticket["priority"] = action.priority.value
                action_log["detail"] = f"priority set to {action.priority.value}"
            else:
                action_log["detail"] = "invalid – missing ticket_id or priority"

        elif at == ActionType.WRITE_RESPONSE:
            ticket = self._find_ticket(action.ticket_id) if action.ticket_id else None
            if ticket and action.response_text:
                ticket["has_response"] = True
                ticket["status"] = "pending_customer"
                action_log["detail"] = "response sent"
            else:
                action_log["detail"] = "invalid – missing ticket_id or response_text"

        elif at == ActionType.ESCALATE_TICKET:
            ticket = self._find_ticket(action.ticket_id) if action.ticket_id else None
            if ticket:
                ticket["is_escalated"] = True
                ticket["status"] = "escalated"
                action_log["detail"] = "escalated"
            else:
                action_log["detail"] = "invalid – ticket not found"

        elif at == ActionType.CLOSE_TICKET:
            ticket = self._find_ticket(action.ticket_id) if action.ticket_id else None
            if ticket:
                ticket["status"] = "closed"
                action_log["detail"] = "closed"
            else:
                action_log["detail"] = "invalid – ticket not found"

        elif at == ActionType.REOPEN_TICKET:
            ticket = self._find_ticket(action.ticket_id) if action.ticket_id else None
            if ticket:
                ticket["status"] = "open"
                action_log["detail"] = "reopened"
            else:
                action_log["detail"] = "invalid – ticket not found"

        elif at == ActionType.ADD_TAG:
            ticket = self._find_ticket(action.ticket_id) if action.ticket_id else None
            if ticket and action.tag:
                if action.tag not in ticket.get("tags", []):
                    ticket.setdefault("tags", []).append(action.tag)
                action_log["detail"] = f"tag '{action.tag}' added"
            else:
                action_log["detail"] = "invalid – missing ticket_id or tag"

        elif at == ActionType.REQUEST_INFO:
            ticket = self._find_ticket(action.ticket_id) if action.ticket_id else None
            if ticket and action.response_text:
                ticket["has_response"] = True
                ticket["status"] = "pending_customer"
                action_log["detail"] = "info requested from customer"
            else:
                action_log["detail"] = "invalid – missing ticket_id or request text"

        elif at == ActionType.MERGE_TICKETS:
            src = self._find_ticket(action.ticket_id) if action.ticket_id else None
            dst = self._find_ticket(action.target_ticket_id) if action.target_ticket_id else None
            if src and dst and action.ticket_id and action.target_ticket_id and action.ticket_id != action.target_ticket_id:
                self._merged_into[action.ticket_id] = action.target_ticket_id
                src["status"] = "closed"
                dst.setdefault("tags", []).append(f"merged_from:{action.ticket_id}")
                action_log["detail"] = f"merged into {action.target_ticket_id}"
            else:
                action_log["detail"] = "invalid merge – missing or same tickets"

        self._action_history.append(action_log)

    def _compute_reward(self, action: Action) -> float:
        """
        Compute dense reward for the action.

        Positive:
          +0.15 for correct classification
          +0.10 for correct priority assignment
          +0.10 for writing a response
          +0.15 for correct escalation
          +0.10 for correct close
          +0.20 for correct merge
          +0.05 for adding a useful tag

        Negative:
          -0.05 for noop
          -0.10 for acting on non-existent ticket
          -0.05 for redundant action (already done)
          -0.15 for wrong escalation
          -0.10 for closing a ticket that shouldn't be closed yet
        """
        reward = 0.0
        at = action.action_type
        expected = self._task.expected if self._task else {}

        if at == ActionType.NOOP:
            return -0.05

        ticket = None
        if action.ticket_id:
            ticket = self._find_ticket(action.ticket_id)
            # Penalise acting on a ticket that has been merged away
            if action.ticket_id in self._merged_into:
                return -0.10

        if ticket is None and at not in (ActionType.MERGE_TICKETS, ActionType.NOOP):
            return -0.10

        if at == ActionType.CLASSIFY_TICKET:
            if ticket and action.category:
                tid = ticket["ticket_id"]
                if ticket.get("category") is not None:
                    reward -= 0.05
                elif expected.get("classifications", {}).get(tid) == action.category.value:
                    reward += 0.15
                else:
                    reward -= 0.05

        elif at == ActionType.ASSIGN_PRIORITY:
            if ticket and action.priority:
                tid = ticket["ticket_id"]
                if ticket.get("priority") is not None:
                    reward -= 0.05
                elif expected.get("priorities", {}).get(tid) == action.priority.value:
                    reward += 0.10
                else:
                    reward -= 0.05

        elif at == ActionType.WRITE_RESPONSE:
            if ticket and action.response_text:
                if ticket.get("has_response"):
                    reward -= 0.05
                else:
                    reward += 0.10

        elif at == ActionType.REQUEST_INFO:
            if ticket and action.response_text:
                if ticket.get("has_response"):
                    reward -= 0.05
                else:
                    reward += 0.08

        elif at == ActionType.ESCALATE_TICKET:
            if ticket:
                tid = ticket["ticket_id"]
                must_esc = set(expected.get("must_escalate", []))
                if ticket.get("is_escalated"):
                    reward -= 0.05
                elif tid in must_esc:
                    reward += 0.15
                else:
                    reward -= 0.15

        elif at == ActionType.CLOSE_TICKET:
            if ticket:
                tid = ticket["ticket_id"]
                must_close = set(expected.get("must_close", []))
                if ticket.get("status") == "closed":
                    reward -= 0.05
                elif tid in must_close:
                    reward += 0.10
                elif ticket.get("has_response") and ticket.get("category"):
                    reward += 0.03
                else:
                    reward -= 0.10

        elif at == ActionType.REOPEN_TICKET:
            if ticket:
                reward -= 0.05

        elif at == ActionType.ADD_TAG:
            if ticket and action.tag:
                if action.tag in ticket.get("tags", []):
                    reward -= 0.02
                else:
                    reward += 0.05

        elif at == ActionType.MERGE_TICKETS:
            if action.ticket_id and action.target_ticket_id:
                src_exists = self._find_ticket(action.ticket_id) is not None
                dst_exists = self._find_ticket(action.target_ticket_id) is not None
                if src_exists and dst_exists and action.ticket_id != action.target_ticket_id:
                    merge_groups = expected.get("merge_groups", [])
                    in_same_group = False
                    for group in merge_groups:
                        if action.ticket_id in group and action.target_ticket_id in group:
                            in_same_group = True
                            break
                    if in_same_group:
                        reward += 0.20
                    else:
                        reward += 0.05
                else:
                    reward -= 0.10

        return round(reward, 4)

    def _check_done(self) -> bool:
        """Episode terminates when max_steps reached or all tickets are terminal."""
        if self._step >= (self._task.max_steps if self._task else 0):
            return True

        terminal_statuses = {"closed", "resolved", "escalated"}
        all_terminal = all(
            t["status"] in terminal_statuses or t["ticket_id"] in self._merged_into
            for t in self._tickets
        )
        if all_terminal:
            return True

        # Also terminate if all tickets have been classified AND prioritised
        # and no further required actions exist (for Task 1)
        if self._task_id == "task_1_classify_prioritize":
            all_classified = all(t.get("category") is not None for t in self._tickets)
            all_prioritized = all(t.get("priority") is not None for t in self._tickets)
            if all_classified and all_prioritized:
                return True

        # For Task 3: terminate when all required actions are complete
        if self._task_id == "task_3_complex_triage" and self._task is not None:
            expected = self._task.expected
            must_escalate = set(expected.get("must_escalate", []))
            must_respond = set(expected.get("must_respond", []))
            merge_groups = expected.get("merge_groups", [])
            
            all_classified = all(t.get("category") is not None for t in self._tickets)
            all_prioritized = all(t.get("priority") is not None for t in self._tickets)
            all_escalated = all(
                t.get("is_escalated") for t in self._tickets 
                if t["ticket_id"] in must_escalate
            )
            all_responded = all(
                t.get("has_response") for t in self._tickets 
                if t["ticket_id"] in must_respond
            )
            all_merged = True
            for group in merge_groups:
                if len(group) < 2:
                    continue
                for secondary in group[1:]:
                    if secondary not in self._merged_into:
                        all_merged = False
                        break
            
            if all_classified and all_prioritized and all_escalated and all_responded and all_merged:
                return True

        return False
