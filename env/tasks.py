import random
from typing import List, Tuple
from .models import Email, Observation, Action, Reward

# ── Seed dataset ──────────────────────────────────────────────────────────────
EMAILS = [
    Email(id="e1", subject="URGENT: Server down in prod",
          sender="ops@company.com",
          body="Production DB crashed. Need immediate attention!",
          timestamp="2024-01-15T09:00:00", category="urgent", priority=1),
    Email(id="e2", subject="Team lunch this Friday",
          sender="hr@company.com",
          body="Join us for a team lunch at noon on Friday.",
          timestamp="2024-01-15T09:05:00", category="social", priority=5),
    Email(id="e3", subject="Q4 budget approval needed",
          sender="finance@company.com",
          body="Please review and approve the Q4 budget by EOD.",
          timestamp="2024-01-15T09:10:00", category="action_required", priority=2),
    Email(id="e4", subject="Newsletter: Industry trends",
          sender="newsletter@techdigest.com",
          body="This week in tech: AI advances, new frameworks...",
          timestamp="2024-01-15T09:15:00", category="newsletter", priority=5),
    Email(id="e5", subject="Client meeting rescheduled",
          sender="alice@clientco.com",
          body="Hi, need to move our 2pm meeting to 4pm today.",
          timestamp="2024-01-15T09:20:00", category="action_required", priority=2),
    Email(id="e6", subject="Invoice #4521 overdue",
          sender="billing@vendor.com",
          body="Your invoice of $2,400 is 30 days overdue.",
          timestamp="2024-01-15T09:25:00", category="action_required", priority=3),
    Email(id="e7", subject="Congrats on the promotion!",
          sender="manager@company.com",
          body="Just wanted to say — well deserved!",
          timestamp="2024-01-15T09:30:00", category="social", priority=5),
    Email(id="e8", subject="Security alert: login from new device",
          sender="security@company.com",
          body="We detected a login from a new device. Was this you?",
          timestamp="2024-01-15T09:35:00", category="urgent", priority=1),
]

VALID_LABELS = {"urgent", "social", "newsletter", "action_required", "spam"}
REPLY_KEYWORDS = {
    "e1": ["immediately", "looking", "investigating", "fix", "team", "on it"],
    "e5": ["confirm", "4pm", "works", "see you", "rescheduled", "noted"],
    "e8": ["yes", "was me", "not me", "block", "secure", "password"],
}


# ── TASK 1 — Easy: label a single email ───────────────────────────────────────
class Task1:
    id = "task_email_labeling"
    description = "Label each email with the correct category."
    difficulty = "easy"

    def reset(self) -> Tuple[Observation, dict]:
        self.emails = random.sample(EMAILS, k=3)
        self.labels_given: dict = {}
        return Observation(
            task_id=self.id,
            step=0,
            emails=[Email(**{k: v for k, v in e.dict().items()
                             if k not in ("category", "priority")})
                    for e in self.emails],
            instructions=(
                "Label each email. Valid labels: urgent, social, newsletter, "
                "action_required, spam. Use action_type='label'."
            ),
        ), {}

    def step(self, action: Action, current_step: int) -> Reward:
        if action.action_type != "label":
            return Reward(value=0.0, done=False,
                          info={"error": "wrong action_type, use 'label'"})
        email = next((e for e in self.emails if e.id == action.email_id), None)
        if not email:
            return Reward(value=0.0, done=False,
                          info={"error": f"unknown email_id {action.email_id}"})
        if action.label not in VALID_LABELS:
            return Reward(value=0.0, done=False,
                          info={"error": f"invalid label '{action.label}'"})

        correct = email.category == action.label
        self.labels_given[action.email_id] = action.label

        done = len(self.labels_given) == len(self.emails)
        # partial credit: fraction of emails labelled correctly so far
        correct_count = sum(
            1 for eid, lbl in self.labels_given.items()
            if next(e for e in self.emails if e.id == eid).category == lbl
        )
        score = correct_count / len(self.emails)

        return Reward(
            value=round(score, 3),
            done=done,
            info={"correct_this_step": correct,
                  "labels_so_far": self.labels_given,
                  "score": score},
        )


# ── TASK 2 — Medium: prioritize full inbox ────────────────────────────────────
class Task2:
    id = "task_inbox_prioritization"
    description = "Assign a priority 1–5 to every email (1 = most urgent)."
    difficulty = "medium"

    def reset(self) -> Tuple[Observation, dict]:
        self.emails = random.sample(EMAILS, k=5)
        self.priorities_given: dict = {}
        return Observation(
            task_id=self.id,
            step=0,
            emails=[Email(**{k: v for k, v in e.dict().items()
                             if k not in ("category", "priority")})
                    for e in self.emails],
            instructions=(
                "Assign priority 1 (most urgent) to 5 (least urgent) to each email. "
                "Use action_type='prioritize'."
            ),
        ), {}

    def step(self, action: Action, current_step: int) -> Reward:
        if action.action_type != "prioritize":
            return Reward(value=0.0, done=False,
                          info={"error": "wrong action_type"})
        email = next((e for e in self.emails if e.id == action.email_id), None)
        if not email:
            return Reward(value=0.0, done=False, info={"error": "unknown email_id"})
        if action.priority not in range(1, 6):
            return Reward(value=0.0, done=False, info={"error": "priority must be 1-5"})

        self.priorities_given[action.email_id] = action.priority

        done = len(self.priorities_given) == len(self.emails)
        # score = 1 - normalised absolute error across assigned emails
        total_error = sum(
            abs(p - next(e for e in self.emails if e.id == eid).priority)
            for eid, p in self.priorities_given.items()
        )
        max_error = 4 * len(self.priorities_given)   # worst case per email = 4
        score = 1.0 - (total_error / max_error) if max_error > 0 else 1.0

        return Reward(
            value=round(score, 3),
            done=done,
            info={"priorities_so_far": self.priorities_given, "score": score},
        )


# ── TASK 3 — Hard: draft a contextually appropriate reply ─────────────────────
class Task3:
    id = "task_smart_reply"
    description = "Draft a professional reply to a specific email."
    difficulty = "hard"

    def reset(self) -> Tuple[Observation, dict]:
        # pick an email that has expected reply keywords defined
        pool = [e for e in EMAILS if e.id in REPLY_KEYWORDS]
        self.target = random.choice(pool)
        self.done = False
        self.best_score = 0.0
        return Observation(
            task_id=self.id,
            step=0,
            emails=[Email(**{k: v for k, v in self.target.dict().items()
                             if k not in ("category", "priority")})],
            instructions=(
                f"Draft a professional reply to the email from {self.target.sender} "
                f"with subject '{self.target.subject}'. "
                "Use action_type='reply'. Keep it under 100 words."
            ),
        ), {}

    def step(self, action: Action, current_step: int) -> Reward:
        if action.action_type != "reply":
            return Reward(value=0.0, done=False,
                          info={"error": "wrong action_type, use 'reply'"})
        if action.email_id != self.target.id:
            return Reward(value=0.0, done=False,
                          info={"error": "wrong email_id"})
        if not action.reply_text or len(action.reply_text.strip()) < 10:
            return Reward(value=0.1, done=False,
                          info={"error": "reply too short"})

        reply = action.reply_text.lower()
        keywords = REPLY_KEYWORDS[self.target.id]
        hits = sum(1 for kw in keywords if kw in reply)
        keyword_score = hits / len(keywords)

        # length penalty: > 100 words loses 0.2
        word_count = len(action.reply_text.split())
        length_penalty = 0.2 if word_count > 100 else 0.0

        # professional tone check (penalise very informal/rude words)
        rude = ["lol", "wtf", "asap!!", "whatever"]
        tone_penalty = 0.15 if any(w in reply for w in rude) else 0.0

        score = max(0.0, keyword_score - length_penalty - tone_penalty)
        # allow 1 retry for partial improvement
        if score > self.best_score:
            self.best_score = score

        self.done = score >= 0.5   # pass threshold

        return Reward(
            value=round(score, 3),
            done=self.done,
            info={
                "keyword_hits": hits,
                "keyword_score": keyword_score,
                "word_count": word_count,
                "length_penalty": length_penalty,
                "tone_penalty": tone_penalty,
            },
        )