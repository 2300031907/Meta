from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    category: Optional[str] = None      # ground-truth label (hidden from agent)
    priority: Optional[int] = None      # ground-truth priority 1-5 (hidden)

class Observation(BaseModel):
    task_id: str
    step: int
    emails: List[Email]                 # current inbox slice shown to agent
    instructions: str                   # natural-language task description
    context: Dict[str, Any] = {}        # extra info (e.g. previous actions taken)

class Action(BaseModel):
    action_type: str                    # "label" | "prioritize" | "reply"
    email_id: str
    label: Optional[str] = None        # for action_type="label"
    priority: Optional[int] = None     # for action_type="prioritize"  (1=highest)
    reply_text: Optional[str] = None   # for action_type="reply"

class Reward(BaseModel):
    value: float                        # 0.0 – 1.0
    done: bool
    info: Dict[str, Any] = {}