"""
Inference script — EmailTriageEnv baseline
Env vars needed: API_BASE_URL, MODEL_NAME, HF_TOKEN
"""
import os, json, time
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS    = 10

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

TASK_PROMPTS = {
    "task_email_labeling": (
        "You are an email triage agent. For each email in the list, call the "
        "'label' action with one of: urgent, social, newsletter, action_required, spam.\n"
        "Respond ONLY with a JSON object matching:\n"
        '{"action_type":"label","email_id":"<id>","label":"<category>"}\n'
        "Process one email at a time."
    ),
    "task_inbox_prioritization": (
        "You are an email triage agent. Assign priority 1 (most urgent) to 5 (least) "
        "to each email.\n"
        "Respond ONLY with JSON:\n"
        '{"action_type":"prioritize","email_id":"<id>","priority":<1-5>}\n'
        "Process one email at a time."
    ),
    "task_smart_reply": (
        "You are an email assistant. Draft a professional reply to the given email.\n"
        "Respond ONLY with JSON:\n"
        '{"action_type":"reply","email_id":"<id>","reply_text":"<your reply under 100 words>"}\n'
    ),
}

def run_task(task_id: str) -> float:
    # reset
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    system_prompt = TASK_PROMPTS[task_id]
    best_score = 0.0

    for step_num in range(MAX_STEPS):
        # build user message from observation
        user_msg = (
            f"Step {obs['step']}. Instructions: {obs['instructions']}\n"
            f"Emails:\n{json.dumps(obs['emails'], indent=2)}"
        )
        # call LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        # strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            action_dict = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [step {step_num+1}] bad JSON: {raw[:80]}")
            continue

        # step env
        step_resp = requests.post(
            f"{ENV_URL}/step",
            json={"task_id": task_id, "action": action_dict},
            timeout=30,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward_val = result["reward"]["value"]
        best_score = max(best_score, reward_val)
        done = result["done"]

        print(f"  [step {step_num+1}] action={action_dict.get('action_type')} "
              f"email={action_dict.get('email_id')} reward={reward_val:.3f}")

        obs = result["observation"]
        if done:
            print(f"  Done! Final score: {best_score:.3f}")
            break
        time.sleep(0.3)   # be polite

    return best_score

def main():
    tasks = [
        "task_email_labeling",
        "task_inbox_prioritization",
        "task_smart_reply",
    ]
    scores = {}
    for task_id in tasks:
        print(f"\n{'='*50}")
        print(f"Running task: {task_id}")
        score = run_task(task_id)
        scores[task_id] = score
        print(f"Score: {score:.3f}")

    print("\n" + "="*50)
    print("BASELINE RESULTS")
    print("="*50)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"{task_id:<35} {score:.3f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\nAverage score: {avg:.3f}")

if __name__ == "__main__":
    main()