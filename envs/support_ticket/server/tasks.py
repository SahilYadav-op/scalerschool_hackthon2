"""
Task definitions for the Support Ticket Resolution Environment.

Each task provides:
  - A set of initial tickets
  - Expected outcomes (ground-truth classifications, priorities, etc.)
  - Metadata used by the grader
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Task schema (internal, not exposed to agent)
# ---------------------------------------------------------------------------


class TaskDef:
    """Describes a single task / scenario."""

    def __init__(
        self,
        task_id: str,
        name: str,
        difficulty: str,
        description: str,
        max_steps: int,
        tickets: list[dict[str, Any]],
        expected: dict[str, Any],
    ):
        self.task_id = task_id
        self.name = name
        self.difficulty = difficulty
        self.description = description
        self.max_steps = max_steps
        self.tickets = tickets  # list of ticket dicts
        self.expected = expected  # ground-truth for grading


# ---------------------------------------------------------------------------
# Task 1 – Easy: Classify and prioritise a small batch of tickets
# ---------------------------------------------------------------------------

TASK_1 = TaskDef(
    task_id="task_1_classify_prioritize",
    name="Classify and Prioritise Incoming Tickets",
    difficulty="easy",
    description=(
        "A batch of 5 new support tickets has arrived. "
        "Classify each ticket into the correct category and assign an "
        "appropriate priority. Close tickets that are trivially resolved."
    ),
    max_steps=20,
    tickets=[
        {
            "ticket_id": "T-101",
            "subject": "Cannot access my account after password reset",
            "body": (
                "I reset my password using the link in the email but now I get "
                "an 'Invalid credentials' error every time I try to log in. "
                "I've tried clearing cookies and using a different browser."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-102",
            "subject": "Charged twice for monthly subscription",
            "body": (
                "My credit card was charged $29.99 twice this month for the "
                "Pro plan. I only have one active subscription. Please refund "
                "the duplicate charge immediately."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-103",
            "subject": "Feature request: dark mode for dashboard",
            "body": (
                "It would be great if the analytics dashboard had a dark mode. "
                "Many tools offer this now and it helps reduce eye strain."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-104",
            "subject": "API returns 500 error on /v2/users endpoint",
            "body": (
                "Since yesterday, every GET request to /v2/users returns a "
                "500 Internal Server Error. Our production integration is "
                "completely broken. This is urgent."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-105",
            "subject": "How do I change my email address?",
            "body": (
                "I want to update the email on my account. I couldn't find "
                "the setting. Can you point me to it?"
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
    ],
    expected={
        "classifications": {
            "T-101": "account",
            "T-102": "billing",
            "T-103": "feature_request",
            "T-104": "bug_report",
            "T-105": "account",
        },
        "priorities": {
            "T-101": "high",
            "T-102": "high",
            "T-103": "low",
            "T-104": "critical",
            "T-105": "low",
        },
        "min_classified": 5,
        "min_prioritized": 5,
    },
)


# ---------------------------------------------------------------------------
# Task 2 – Medium: Full lifecycle – respond, escalate, close
# ---------------------------------------------------------------------------

TASK_2 = TaskDef(
    task_id="task_2_full_lifecycle",
    name="Full Ticket Lifecycle Management",
    difficulty="medium",
    description=(
        "Manage 4 tickets through their full lifecycle: classify, prioritise, "
        "write appropriate responses, escalate where needed, and close "
        "resolved tickets. Some tickets require requesting additional "
        "information from the customer."
    ),
    max_steps=30,
    tickets=[
        {
            "ticket_id": "T-201",
            "subject": "Billing discrepancy on enterprise invoice #4521",
            "body": (
                "Our invoice #4521 shows 150 seats but we only purchased 120. "
                "We need a corrected invoice before our finance team can "
                "process payment. Our company is Acme Corp."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-202",
            "subject": "Data export feature missing CSV format",
            "body": (
                "The data export only supports JSON and XML. Our analytics "
                "pipeline requires CSV. Is there a workaround or is CSV "
                "support planned?"
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-203",
            "subject": "URGENT: Production database connection timeout",
            "body": (
                "Our production environment has been experiencing database "
                "connection timeouts every 5-10 minutes since the last "
                "deployment. Error: 'Connection pool exhausted'. "
                "Revenue impact estimated at $10k/hour."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-204",
            "subject": "SSO integration failing with Okta",
            "body": (
                "After upgrading to v3.2, our Okta SSO integration returns "
                "SAML assertion errors. We have 500+ users unable to log in. "
                "Rolling back is not an option due to other fixes in v3.2."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
    ],
    expected={
        "classifications": {
            "T-201": "billing",
            "T-202": "feature_request",
            "T-203": "bug_report",
            "T-204": "technical",
        },
        "priorities": {
            "T-201": "high",
            "T-202": "medium",
            "T-203": "critical",
            "T-204": "critical",
        },
        "must_escalate": ["T-203", "T-204"],
        "must_respond": ["T-201", "T-202", "T-203", "T-204"],
        "must_close": ["T-201", "T-202"],
        "min_steps_to_complete": 16,
    },
)


# ---------------------------------------------------------------------------
# Task 3 – Hard: Multi-step triage with dependencies and merging
# ---------------------------------------------------------------------------

TASK_3 = TaskDef(
    task_id="task_3_complex_triage",
    name="Complex Triage with Duplicate Detection and Escalation",
    difficulty="hard",
    description=(
        "Handle 6 tickets that include duplicates, correlated issues, and "
        "varying severity. The agent must: classify, prioritise, merge "
        "duplicates, escalate critical issues, respond appropriately, and "
        "close what can be closed — all within the step budget."
    ),
    max_steps=35,
    tickets=[
        {
            "ticket_id": "T-301",
            "subject": "Cannot upload files larger than 10MB",
            "body": (
                "Every time I try to upload a PDF larger than 10MB the "
                "progress bar freezes at 80% and then fails. Browser: Chrome "
                "120. OS: Windows 11."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-302",
            "subject": "File upload fails for large documents",
            "body": (
                "I'm getting upload failures for files over 10MB. The error "
                "message says 'Upload timeout'. This started after the latest "
                "update."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-303",
            "subject": "Security concern: API keys visible in client bundle",
            "body": (
                "I noticed that API keys are being included in the "
                "JavaScript bundle sent to browsers. This is a serious "
                "security vulnerability. Anyone can inspect the source and "
                "extract the keys."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-304",
            "subject": "Request: Slack integration for notifications",
            "body": (
                "Our team uses Slack extensively. It would be very helpful "
                "if we could receive ticket notifications directly in a Slack "
                "channel instead of only via email."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-305",
            "subject": "Upload timeout on large files – enterprise client",
            "body": (
                "We are an enterprise client (GlobalTech Inc.) and our users "
                "cannot upload files larger than 10MB. This is blocking our "
                "document management workflow. We have 200 affected users."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
        {
            "ticket_id": "T-306",
            "subject": "Password reset emails not being delivered",
            "body": (
                "Multiple users report not receiving password reset emails. "
                "We've checked spam folders. Our domain is globaltech.io. "
                "This started 2 days ago."
            ),
            "category": None,
            "priority": None,
            "status": "new",
            "tags": [],
            "is_escalated": False,
            "has_response": False,
            "customer_replied_after_response": False,
        },
    ],
    expected={
        "classifications": {
            "T-301": "bug_report",
            "T-302": "bug_report",
            "T-303": "technical",
            "T-304": "feature_request",
            "T-305": "bug_report",
            "T-306": "technical",
        },
        "priorities": {
            "T-301": "high",
            "T-302": "high",
            "T-303": "critical",
            "T-304": "low",
            "T-305": "high",
            "T-306": "high",
        },
        "merge_groups": [
            ["T-301", "T-302", "T-305"],
        ],
        "must_escalate": ["T-303"],
        "must_respond": ["T-301", "T-303", "T-304", "T-306"],
        "min_classified": 6,
        "min_prioritized": 6,
        "min_merged_groups": 1,
    },
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TASKS: dict[str, TaskDef] = {
    TASK_1.task_id: TASK_1,
    TASK_2.task_id: TASK_2,
    TASK_3.task_id: TASK_3,
}

TASK_IDS: list[str] = list(ALL_TASKS.keys())
