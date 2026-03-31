from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from env.environment import EmailTriageEnv
from env.models import Action

app = FastAPI(title="EmailTriageEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# One env instance per task (simple; for production use session IDs)
_envs: dict[str, EmailTriageEnv] = {}

TASK_IDS = [
    "task_email_labeling",
    "task_inbox_prioritization",
    "task_smart_reply",
]

class ResetRequest(BaseModel):
    task_id: str = "task_email_labeling"

class StepRequest(BaseModel):
    task_id: str
    action: Action

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_IDS}

@app.post("/reset")
def reset(req: ResetRequest):
    if req.task_id not in TASK_IDS:
        raise HTTPException(400, f"Unknown task_id: {req.task_id}")
    env = EmailTriageEnv(task_id=req.task_id)
    _envs[req.task_id] = env
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(req: StepRequest):
    env = _envs.get(req.task_id)
    if not env:
        raise HTTPException(400, "Call /reset first")
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }

@app.get("/state")
def state(task_id: str = "task_email_labeling"):
    env = _envs.get(task_id)
    if not env:
        raise HTTPException(400, "Call /reset first")
    return env.state()