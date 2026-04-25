"""
inference.py — OpenEnv: Split-Brain Collapse
=============================================
Mandatory STDOUT format (parsed by the automated validator):

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from agents.split_brain.environment import SplitBrainEnv

load_dotenv()

# ── 1. Load Required Environment Variables ──────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

BENCHMARK             = "split-brain"
SUCCESS_SCORE_THRESHOLD = 0.5

if not API_KEY:
    raise EnvironmentError("CRITICAL: HF_TOKEN or API_KEY environment variable is required.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def run_task(env: SplitBrainEnv, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score   = 0.01  # Safe default strictly > 0
    success = False

    max_steps = env.max_steps if hasattr(env, "max_steps") else 50

    try:
        obs = env.reset(task=task_id)

        for step in range(1, max_steps + 1):
            # The split_brain env provides context-aware multi-agent prompts
            system_prompt, user_prompt = env.get_llm_prompts()

            last_error = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=400,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                last_error  = str(e)
                log_step(step=step, action="noop", reward=0.0, done=True, error=last_error)
                rewards.append(0.0)
                steps_taken = step
                break

            action     = env._parse_action(response_text)
            action_str = f"{action.action_type}"
            if getattr(action, "target_id", None):
                action_str += f"({action.target_id})"
            elif getattr(action, "target_agent", None):
                action_str += f"→{action.target_agent}"

            result  = env.step(action)
            obs     = result.observation
            reward  = result.reward
            done    = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # Clamp score strictly between (0, 1)
        final_health = getattr(obs, "global_health", getattr(obs, "health_score", 0.0))
        success      = final_health >= 1.0
        raw_score    = 1.0 if success else max(0.0, final_health)
        score        = max(0.01, min(0.99, raw_score))

    except Exception:
        score   = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    env   = SplitBrainEnv()
    tasks = [
        "partition_basic",
        "replication_storm",
        "split_brain",
        "cascading_deadlock",
        "regional_wipeout",
    ]
    for task_id in tasks:
        env.reset(task=task_id)  # re-use same env instance
        run_task(env, task_id)


if __name__ == "__main__":
    main()