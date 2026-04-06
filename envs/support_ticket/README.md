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

A production-grade OpenEnv environment that simulates a **customer support ticket resolution system**. An AI agent manages a queue of incoming support tickets — classifying, prioritising, responding to, escalating, merging duplicates, and closing them efficiently.

## Real-World Relevance

Every software company deals with support tickets. This environment tests an agent's ability to:

- **Understand customer intent** from natural-language descriptions
- **Triage by urgency** (billing disputes vs. feature requests vs. production outages)
- **Detect duplicates** and merge related tickets
- **Escalate critical issues** to engineering
- **Communicate professionally** with customers
- **Work within step budgets** (efficiency matters)

## Project Structure

```
envs/support_ticket/
├── models.py                 # Pydantic models: Action, Observation, State
├── client.py                 # Python HTTP client (Gym-style API)
├── inference.py              # Baseline agent runner (all tasks)
├── openenv.yaml              # Environment configuration
├── README.md                 # This file
└── server/
    ├── app.py                # FastAPI REST server
    ├── environment.py        # Core env logic (reset, step, state)
    ├── tasks.py              # Task definitions with ticket data
    ├── graders.py            # Deterministic scoring functions
    ├── Dockerfile            # Container definition
    └── requirements.txt      # Python dependencies
```

## Action Space

| Action | Description | Parameters |
|---|---|---|
| `classify_ticket` | Assign a category to a ticket | `ticket_id`, `category` |
| `assign_priority` | Set priority level | `ticket_id`, `priority` |
| `write_response` | Send a reply to the customer | `ticket_id`, `response_text` |
| `escalate_ticket` | Escalate to engineering/management | `ticket_id` |
| `close_ticket` | Mark a ticket as resolved | `ticket_id` |
| `reopen_ticket` | Reopen a closed ticket | `ticket_id` |
| `add_tag` | Add a metadata tag | `ticket_id`, `tag` |
| `request_info` | Ask customer for more details | `ticket_id`, `response_text` |
| `merge_tickets` | Merge duplicate tickets | `ticket_id`, `target_ticket_id` |
| `noop` | No operation (penalised) | — |

### Categories
`billing`, `technical`, `account`, `feature_request`, `bug_report`, `general`

### Priorities
`low`, `medium`, `high`, `critical`

### Ticket Statuses
`new`, `open`, `pending_customer`, `resolved`, `closed`, `escalated`

## Observation Space

After each `reset()` or `step()`, the agent receives an `Observation` containing:

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Current task identifier |
| `episode_id` | str | Unique episode ID |
| `step` | int | Current step count (0-based) |
| `max_steps` | int | Maximum steps allowed |
| `tickets` | list[TicketView] | Current state of all visible tickets |
| `action_history` | list[dict] | Log of all actions taken |
| `last_reward` | float | Reward from the most recent step |
| `cumulative_reward` | float | Sum of rewards this episode |
| `done` | bool | Whether the episode has terminated |
| `info` | dict | Additional metadata (task name, difficulty, remaining steps) |

Each `TicketView` exposes: `ticket_id`, `subject`, `body`, `category`, `priority`, `status`, `tags`, `is_escalated`, `has_response`, `customer_replied_after_response`.

## Reward Design

Dense, evolving rewards (not binary):

| Action | Correct | Incorrect / Redundant |
|---|---|---|
| `classify_ticket` | +0.15 | -0.05 |
| `assign_priority` | +0.10 | -0.05 |
| `write_response` | +0.10 | -0.05 (duplicate) |
| `request_info` | +0.08 | -0.05 (duplicate) |
| `escalate_ticket` | +0.15 | -0.15 (unnecessary) |
| `close_ticket` | +0.10 | -0.10 (premature) |
| `merge_tickets` | +0.20 | +0.05 (partial) / -0.10 (invalid) |
| `add_tag` | +0.05 | -0.02 (duplicate) |
| `noop` | — | -0.05 |

## Task Descriptions

### Task 1 — Classify and Prioritise (Easy)

- **5 new tickets** arrive in the queue
- Agent must classify each into the correct category and assign appropriate priority
- **Max steps:** 20
- **Grader:** 40% classification + 30% priority + 15% coverage + 15% efficiency
- **Baseline score:** ~0.85-0.95

### Task 2 — Full Lifecycle Management (Medium)

- **4 tickets** requiring full lifecycle handling
- Agent must classify, prioritise, write responses, escalate critical issues, and close resolved tickets
- Some tickets must be escalated (production outage, SSO failure)
- **Max steps:** 30
- **Grader:** 20% classification + 15% priority + 20% responses + 15% escalations + 15% closures + 15% efficiency
- **Baseline score:** ~0.70-0.85

### Task 3 — Complex Triage with Merging (Hard)

- **6 tickets** including duplicates and correlated issues
- Agent must detect and merge duplicate tickets (T-301, T-302, T-305 are all about file upload)
- Must escalate security vulnerability (T-303)
- Must respond to and manage all tickets within step budget
- **Max steps:** 35
- **Grader:** 15% classification + 15% priority + 20% merges + 15% escalations + 15% responses + 10% efficiency + 10% no-redundancy
- **Baseline score:** ~0.60-0.80

## Setup

### Prerequisites

- Python 3.11+
- Docker (optional, for containerised deployment)

### Local Development

```bash
# Install dependencies
pip install fastapi uvicorn pydantic httpx

# Start the server
cd envs/support_ticket
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run the baseline agent (in another terminal)
python inference.py
```

### Docker

```bash
cd envs/support_ticket/server
docker build -t support-ticket-env .
docker run -p 8000:8000 support-ticket-env
```

### Hugging Face Space Deployment

1. Create a new HF Space (Docker template)
2. Push the `server/` directory contents
3. The API will be available at `https://<username>-support-ticket.hf.space`
4. Set `API_BASE_URL` env var in inference.py to point to the Space

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Reset environment (body: `{"task_id": "..."}`) |
| `POST` | `/step` | Execute action (body: `{"action": {...}}`) |
| `GET` | `/state` | Get full internal state |

## Expected Outputs

Running `inference.py` produces output in the required format:

```
============================================================
Support Ticket Resolution Environment — Baseline Agent
API Base URL : http://localhost:8000
Model        : gpt-4o-mini
============================================================

============================================================
[START] Task: task_1_classify_prioritize
============================================================
[STEP] step=1 | action=classify_ticket | ticket=T-101 | reward=+0.1500 | cumulative=+0.1500 | done=False
[STEP] step=2 | action=assign_priority | ticket=T-101 | reward=+0.1000 | cumulative=+0.2500 | done=False
...
[END] Task: task_1_classify_prioritize | Grader Score: 0.9250 | Total Steps: 12 | Cumulative Reward: +1.4500
```

## Baseline Scores

| Task | Expected Score Range |
|---|---|
| Task 1 (Easy) | 0.95 – 1.00 |
| Task 2 (Medium) | 0.90 – 1.00 |
| Task 3 (Hard) | 0.85 – 1.00 |
| **Average** | **0.92 – 1.00** |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | URL of the FastAPI server |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model name (for LLM-based agents) |
| `HF_TOKEN` | `""` | Hugging Face token (optional) |
