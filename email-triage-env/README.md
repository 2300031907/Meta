# Email Triage Environment

An [OpenEnv](https://openenv.dev)-compatible environment where an AI agent
triages a realistic email inbox across three progressively harder tasks.

## Tasks

| Task | Difficulty | Description | Scoring |
|------|-----------|-------------|---------|
| `task_email_labeling` | Easy | Label each email (urgent / social / newsletter / action_required / spam) | Fraction correct |
| `task_inbox_prioritization` | Medium | Assign priority 1–5 to each email | 1 − normalised MAE |
| `task_smart_reply` | Hard | Draft a professional reply under 100 words | Keyword + tone score |

## Action space
```json
{
  "action_type": "label | prioritize | reply",
  "email_id": "e1",
  "label": "urgent",          // task 1 only
  "priority": 2,              // task 2 only (1–5)
  "reply_text": "Hi, ..."     // task 3 only
}
```

## Observation space
```json
{
  "task_id": "task_email_labeling",
  "step": 1,
  "emails": [{ "id": "e1", "subject": "...", "sender": "...", "body": "...", "timestamp": "..." }],
  "instructions": "Label each email...",
  "context": {}
}
```

## Quick start
```bash
# Local
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

# Inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
export ENV_URL="http://localhost:7860"
python inference.py
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start new episode |
| POST | `/step` | Send an action |
| GET | `/state` | Get current state |

## Baseline scores

| Task | Difficulty | Score |
|------|-----------|-------|
| task_email_labeling | Easy | ~0.75 |
| task_inbox_prioritization | Medium | ~0.65 |
| task_smart_reply | Hard | ~0.50 |