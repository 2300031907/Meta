from typing import Optional
from .models import Observation, Action, Reward
from .tasks import Task1, Task2, Task3

TASK_MAP = {
    "task_email_labeling": Task1,
    "task_inbox_prioritization": Task2,
    "task_smart_reply": Task3,
}

class EmailTriageEnv:
    def __init__(self, task_id: str = "task_email_labeling"):
        if task_id not in TASK_MAP:
            raise ValueError(f"Unknown task_id '{task_id}'. "
                             f"Choose from: {list(TASK_MAP.keys())}")
        self.task_id = task_id
        self.task = TASK_MAP[task_id]()
        self._obs: Optional[Observation] = None
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False

    def reset(self) -> Observation:
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        obs, _ = self.task.reset()
        self._obs = obs
        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        self._step_count += 1

        reward = self.task.step(action, self._step_count)
        self._total_reward = max(self._total_reward, reward.value)
        self._done = reward.done

        # refresh observation with updated step count
        if self._obs:
            self._obs = self._obs.copy(update={"step": self._step_count})

        return self._obs, reward, self._done, reward.info

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step": self._step_count,
            "done": self._done,
            "total_reward": self._total_reward,
            "observation": self._obs.dict() if self._obs else None,
        }