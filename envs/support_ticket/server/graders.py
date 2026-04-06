"""
Graders for the Support Ticket Resolution Environment.

Each grader evaluates the final state of an episode and returns a score
in [0.0, 1.0].  Scores are deterministic and based on correctness,
efficiency, and completeness.
"""

from __future__ import annotations

from typing import Any, Callable

from server.tasks import TaskDef


def _safe_get(ticket: dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely retrieve a key from a ticket dict."""
    return ticket.get(key, default)


# ---------------------------------------------------------------------------
# Grader for Task 1 – Classify and Prioritise
# ---------------------------------------------------------------------------

def grade_task_1(tickets: list[dict[str, Any]], expected: dict[str, Any],
                 step: int, max_steps: int) -> float:
    """
    Scoring breakdown (total = 1.0):
      - Classification correctness : 0.40
      - Priority correctness         : 0.30
      - Coverage (all tickets acted) : 0.15
      - Efficiency (fewer steps)     : 0.15
    """
    exp_class = expected["classifications"]
    exp_prior = expected["priorities"]
    n_tickets = len(exp_class)

    # --- Classification score (0.40) ---
    correct_class = 0
    for t in tickets:
        tid = _safe_get(t, "ticket_id")
        cat = _safe_get(t, "category")
        if cat is not None and exp_class.get(tid) == cat:
            correct_class += 1
    class_score = (correct_class / n_tickets) * 0.40 if n_tickets else 0.0

    # --- Priority score (0.30) ---
    correct_prior = 0
    for t in tickets:
        tid = _safe_get(t, "ticket_id")
        pri = _safe_get(t, "priority")
        if pri is not None and exp_prior.get(tid) == pri:
            correct_prior += 1
    prior_score = (correct_prior / n_tickets) * 0.30 if n_tickets else 0.0

    # --- Coverage score (0.15) ---
    acted = sum(
        1 for t in tickets
        if _safe_get(t, "category") is not None or _safe_get(t, "priority") is not None
    )
    coverage_score = (acted / n_tickets) * 0.15 if n_tickets else 0.0

    # --- Efficiency score (0.15) ---
    # Perfect if done in <= n_tickets * 2 steps; degrades linearly
    ideal_steps = n_tickets * 2
    efficiency_ratio = min(1.0, ideal_steps / step) if step > 0 else 1.0
    efficiency_score = efficiency_ratio * 0.15

    total = class_score + prior_score + coverage_score + efficiency_score
    return round(min(1.0, max(0.0, total)), 4)


# ---------------------------------------------------------------------------
# Grader for Task 2 – Full Lifecycle
# ---------------------------------------------------------------------------

def grade_task_2(tickets: list[dict[str, Any]], expected: dict[str, Any],
                 step: int, max_steps: int) -> float:
    """
    Scoring breakdown (total = 1.0):
      - Classification correctness : 0.20
      - Priority correctness         : 0.15
      - Responses written            : 0.20
      - Escalations correct          : 0.15
      - Closures correct             : 0.15
      - Efficiency                   : 0.15
    """
    exp_class = expected["classifications"]
    exp_prior = expected["priorities"]
    must_escalate = set(expected.get("must_escalate", []))
    must_respond = set(expected.get("must_respond", []))
    must_close = set(expected.get("must_close", []))
    n_tickets = len(exp_class)

    # --- Classification (0.20) ---
    correct_class = sum(
        1 for t in tickets
        if _safe_get(t, "category") is not None
        and exp_class.get(_safe_get(t, "ticket_id")) == _safe_get(t, "category")
    )
    class_score = (correct_class / n_tickets) * 0.20 if n_tickets else 0.0

    # --- Priority (0.15) ---
    correct_prior = sum(
        1 for t in tickets
        if _safe_get(t, "priority") is not None
        and exp_prior.get(_safe_get(t, "ticket_id")) == _safe_get(t, "priority")
    )
    prior_score = (correct_prior / n_tickets) * 0.15 if n_tickets else 0.0

    # --- Responses (0.20) ---
    responded = sum(
        1 for t in tickets
        if _safe_get(t, "ticket_id") in must_respond and _safe_get(t, "has_response")
    )
    response_score = (responded / len(must_respond)) * 0.20 if must_respond else 0.0

    # --- Escalations (0.15) ---
    escalated_correct = sum(
        1 for t in tickets
        if _safe_get(t, "ticket_id") in must_escalate and _safe_get(t, "is_escalated")
    )
    # Penalise over-escalation
    over_escalated = sum(
        1 for t in tickets
        if _safe_get(t, "ticket_id") not in must_escalate and _safe_get(t, "is_escalated")
    )
    esc_score = max(0.0, (escalated_correct - over_escalated * 0.5) / len(must_escalate)) * 0.15 if must_escalate else 0.0

    # --- Closures (0.15) ---
    closed_correct = sum(
        1 for t in tickets
        if _safe_get(t, "ticket_id") in must_close and _safe_get(t, "status") == "closed"
    )
    close_score = (closed_correct / len(must_close)) * 0.15 if must_close else 0.0

    # --- Efficiency (0.15) ---
    ideal = expected.get("min_steps_to_complete", 16)
    efficiency_ratio = min(1.0, ideal / step) if step > 0 else 1.0
    efficiency_score = efficiency_ratio * 0.15

    total = class_score + prior_score + response_score + esc_score + close_score + efficiency_score
    return round(min(1.0, max(0.0, total)), 4)


# ---------------------------------------------------------------------------
# Grader for Task 3 – Complex Triage with Merging
# ---------------------------------------------------------------------------

def grade_task_3(tickets: list[dict[str, Any]], expected: dict[str, Any],
                 step: int, max_steps: int, action_history: list[dict[str, Any]]) -> float:
    """
    Scoring breakdown (total = 1.0):
      - Classification correctness : 0.15
      - Priority correctness         : 0.15
      - Merge correctness            : 0.20
      - Escalations correct          : 0.15
      - Responses written            : 0.15
      - Efficiency                   : 0.10
      - No redundant actions         : 0.10
    """
    exp_class = expected["classifications"]
    exp_prior = expected["priorities"]
    must_escalate = set(expected.get("must_escalate", []))
    must_respond = set(expected.get("must_respond", []))
    merge_groups = expected.get("merge_groups", [])
    n_tickets = len(exp_class)

    # --- Classification (0.15) ---
    correct_class = sum(
        1 for t in tickets
        if _safe_get(t, "category") is not None
        and exp_class.get(_safe_get(t, "ticket_id")) == _safe_get(t, "category")
    )
    class_score = (correct_class / n_tickets) * 0.15 if n_tickets else 0.0

    # --- Priority (0.15) ---
    correct_prior = sum(
        1 for t in tickets
        if _safe_get(t, "priority") is not None
        and exp_prior.get(_safe_get(t, "ticket_id")) == _safe_get(t, "priority")
    )
    prior_score = (correct_prior / n_tickets) * 0.15 if n_tickets else 0.0

    # --- Merge correctness (0.20) ---
    merge_actions = [
        a for a in action_history if a.get("action_type") == "merge_tickets"
    ]
    # Track which tickets from each group have been merged into the group's "primary"
    # Actually, any ticket in the group merged into any other ticket in the group is okay,
    # as long as the group eventually converges.
    correct_merges = 0
    total_needed_merges = 0
    
    for group in merge_groups:
        if len(group) < 2:
            continue
        total_needed_merges += (len(group) - 1)
        # Check how many secondary tickets in this group were merged into a target in this group
        merged_count = 0
        secondaries = set(group[1:])
        primary = group[0]
        
        for a in merge_actions:
            tid = a.get("ticket_id")
            target = a.get("target_ticket_id")
            if tid in secondaries and target in group:
                merged_count += 1
                secondaries.remove(tid)  # Only count each secondary once
        
        correct_merges += merged_count

    merge_score = (correct_merges / total_needed_merges) * 0.20 if total_needed_merges else 0.0

    # --- Escalations (0.15) ---
    escalated_correct = sum(
        1 for t in tickets
        if _safe_get(t, "ticket_id") in must_escalate and _safe_get(t, "is_escalated")
    )
    esc_score = (escalated_correct / len(must_escalate)) * 0.15 if must_escalate else 0.0

    # --- Responses (0.15) ---
    responded = sum(
        1 for t in tickets
        if _safe_get(t, "ticket_id") in must_respond and _safe_get(t, "has_response")
    )
    response_score = (responded / len(must_respond)) * 0.15 if must_respond else 0.0

    # --- Efficiency (0.10) ---
    # Perfect if completed in <= 19 steps (optimal for this task)
    ideal = 19
    efficiency_ratio = min(1.0, ideal / step) if step > 0 else 1.0
    efficiency_score = efficiency_ratio * 0.10

    # --- No redundant actions (0.10) ---
    noop_count = sum(1 for a in action_history if a.get("action_type") == "noop")
    redundant_penalty = min(0.10, noop_count * 0.02)
    redundancy_score = 0.10 - redundant_penalty

    total = class_score + prior_score + merge_score + esc_score + response_score + efficiency_score + redundancy_score
    return round(min(1.0, max(0.0, total)), 4)


# ---------------------------------------------------------------------------
# Grader dispatcher
# ---------------------------------------------------------------------------

GRADERS: dict[str, Callable[..., float]] = {
    "task_1_classify_prioritize": grade_task_1,
    "task_2_full_lifecycle": grade_task_2,
    "task_3_complex_triage": grade_task_3,
}


def grade(task_id: str, tickets: list[dict[str, Any]], expected: dict[str, Any],
          step: int, max_steps: int, action_history: list[dict[str, Any]] | None = None) -> float:
    """Dispatch to the correct grader and return a score strictly in (0.0, 1.0)."""
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task_id={task_id!r}")

    if task_id == "task_3_complex_triage":
        score = grader(tickets, expected, step, max_steps, action_history or [])
    else:
        score = grader(tickets, expected, step, max_steps)
        
    return min(0.9999, max(0.0001, score))
