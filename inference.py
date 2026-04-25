"""
Inference Script — OpenEnv: Distributed Cluster Triage
=======================================================
Mandatory STDOUT format (parsed by the automated validator):

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import re
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from agents.cluster_triage.environment import ClusterTriageEnv
from agents.cluster_triage.models import ClusterAction

load_dotenv()

# ── 1. Load Required Environment Variables ──────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

BENCHMARK = "cluster-triage"
MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.5

if not API_KEY:
    raise EnvironmentError("CRITICAL: HF_TOKEN or API_KEY environment variable is required.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def parse_model_action(response_text: str) -> ClusterAction:
    # 1. Strip out DeepSeek reasoning blocks!
    text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "action_type" in data:
                return ClusterAction(**data)
        except Exception:
            pass
    match = re.search(r'\[action\]\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            return ClusterAction(**data)
        except Exception:
            pass
    return ClusterAction(action_type="noop", target_id="none")


def run_task(env: ClusterTriageEnv, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01  # Safe default strictly > 0
    success = False

    try:
        observation = env.reset(task=task_id)
        history: List[str] = []

        for step in range(1, MAX_STEPS + 1):
            history_text = "\n".join(history) if history else "None."

            system_prompt = (
                "You are an automated DevOps system. You cannot speak. "
                "You can only output raw JSON commands. No explanations, no extra text."
            )

            user_prompt = f"""You are an SRE agent triaging a distributed cluster failure.

CURRENT CLUSTER STATE:
{observation.model_dump_json(indent=2)}

PREVIOUS ACTIONS (do NOT repeat failed actions):
{history_text}

RULES:
1. If there are any hanging jobs, kill ALL of them before doing anything else.
2. Never restart a node whose disk_usage is above 50%. Clear its storage first.
3. Clear nodes in order after all jobs are killed.
4. Only restart nodes after their disk has been cleared.
5. For nightmare: kill ALL 3 hydra jobs before clearing ANY storage.

Respond with EXACTLY ONE JSON object. No other text.
Valid action_type values: "kill_job", "restart_node", "clear_temp_storage", "noop"

EXAMPLE:
{{"action_type": "kill_job", "target_id": "job_rogue_99"}}
"""

            last_error = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=400
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                last_error = str(e)
                log_step(step=step, action="noop", reward=0.0, done=True, error=last_error)
                rewards.append(0.0)
                steps_taken = step
                break

            action = parse_model_action(response_text)
            action_str = f"{action.action_type}({action.target_id})"

            result = env.step(action)
            observation = result.observation
            reward = result.reward
            done = result.done
            msg = result.info.get("message", "")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            history.append(
                f"Step {step}: {action.action_type} on {action.target_id} -> reward={reward:.2f} | {msg}"
            )
            if action.action_type == "noop":
                history.append("WARNING: Last output was invalid JSON. Output ONLY a JSON object.")

            if done:
                break

        # ── THE FIX: Clamp the score strictly between (0, 1) ──
        success = observation.health_score >= 1.0
        raw_score = 1.0 if success else max(0.0, observation.health_score)
        
        # Force the score to be exactly 0.99 for a perfect run, and 0.01 for a total failure
        score = max(0.01, min(0.99, raw_score))

    except Exception as e:
        score = 0.01  # Cannot be 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    env = ClusterTriageEnv()
    tasks = ["easy", "medium", "hard", "very_hard", "nightmare"]
    for task_id in tasks:
        run_task(env, task_id)


if __name__ == "__main__":
    main()