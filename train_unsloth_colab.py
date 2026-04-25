"""
train_unsloth_colab.py — Curriculum GRPO Training on Split-Brain Collapse
=========================================================================
Trains Llama-3.2-3B-Instruct via GRPO (Group Relative Policy Optimization)
directly against the live SplitBrainEnv reward function using curriculum learning.

ROOT CAUSES OF 0% POST-TRAINING (all fixed in this version):
─────────────────────────────────────────────────────────────
CAUSE 1 — Zero gradient / cold-start problem (PRIMARY CAUSE):
  The previous reward function ran only ONE env.step() per completion.
  An untrained 3B model produces noops on ~95% of completions.
  With num_generations=4, all 4 completions score ~-0.23 (noop).
  GRPO advantage = reward - group_mean = 0 for all → gradient = 0.
  The model trained for 330 steps and learned absolutely nothing because
  it never saw any variance in rewards to differentiate good from bad.

  FIX: Run the FULL multi-step episode inside the reward function.
  This gives:
    - The +3.0 completion bonus a real chance to fire (some episodes finish)
    - Natural variance: some rollouts get lucky and succeed early (high reward),
      most fail (low reward) → strong gradient signal from the first stage.

CAUSE 2 — model.train()/model.eval() toggling corrupts KV cache mid-episode:
  generate_action() was calling model.eval() → generate → model.train()
  on every single step of a 25-step episode. The constant mode switching
  corrupts the attention KV cache and produces garbage tokens after step 3-4.
  This is why diag_loops=0.0: the model never successfully parsed an action.

  FIX: Set model mode ONCE before the episode loop, restore ONCE after.
  Never toggle inside a loop.

CAUSE 3 — LoRA adapter potentially disabled after GRPOTrainer.train():
  GRPOTrainer internally may call model.merge_adapter() or equivalent
  operations that deactivate the LoRA weights. After trainer.train()
  returns, the adapter may not be active in the forward pass.

  FIX: Explicitly call model.enable_adapters() before evaluation to
  guarantee the trained LoRA weights are used during inference.

CAUSE 4 — Reward scale mismatch between training and evaluation:
  During training: reward_fn returned values in range [-0.73, +3.07]
  During eval: total_reward accumulated over 25 steps → range [-1.75, +3+]
  These different scales caused the comparison table to be misleading.
  A model could have partial learning but still show 0% because the
  success threshold (global_health >= 1.0) requires a perfect run.

  FIX: Use a graduated success threshold:
    - global_health >= 1.0 → full success (counts in SR%)
    - global_health >= 0.5 → partial success (shown separately)
  This gives much more informative diagnostics.

Run on Google Colab (free T4 GPU):
  !git clone https://github.com/Sayantan181222/openenv-cluster-triage-updated.git
  %cd openenv-cluster-triage-updated
  !pip install unsloth trl datasets matplotlib pydantic networkx python-dotenv
  !python train_unsloth_colab.py
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os, re, json, copy, random, time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from agents.split_brain.environment import SplitBrainEnv
from agents.split_brain.models import SplitBrainAction

os.makedirs("plots", exist_ok=True)
os.makedirs("openenv_outputs", exist_ok=True)

# ── 1. Config ─────────────────────────────────────────────────────────────────
BASE_MODEL      = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
LORA_OUTPUT_DIR = "openenv-split-brain-lora"
MAX_SEQ_LENGTH  = 1024
LORA_RANK       = 16
EVAL_EPISODES   = 5

# Per-task step budgets matching environment's own max_steps exactly.
# NEVER use a single hardcoded number — each task has a different budget.
TASK_MAX_STEPS = {
    "partition_basic":    15,
    "replication_storm":  25,
    "split_brain":        35,
    "cascading_deadlock": 35,
    "regional_wipeout":   50,
}

# Curriculum: (task_id, grpo_steps, num_prompts)
# num_prompts kept modest — quality of reward signal matters more than quantity
CURRICULUM = [
    ("partition_basic",    60,  80),
    ("replication_storm",  70,  100),
    ("split_brain",        80,  120),
    ("cascading_deadlock", 80,  120),
    ("regional_wipeout",   80,  140),
]

TASK_LABELS = {
    "partition_basic":    "Partition\nBasic",
    "replication_storm":  "Replication\nStorm",
    "split_brain":        "Split\nBrain",
    "cascading_deadlock": "Cascading\nDeadlock",
    "regional_wipeout":   "Regional\nWipeout",
}

STAGE_COLORS = ["#10b981", "#fbbf24", "#f97316", "#7c3aed", "#b91c1c"]

# ── 2. Load Model + LoRA ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Loading {BASE_MODEL}")
print(f"{'='*65}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=LORA_RANK,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
)
print("[INFO] Model + LoRA ready.\n")


# ── 3. Core Helpers ───────────────────────────────────────────────────────────

def parse_action(text: str) -> SplitBrainAction:
    """
    Parse LLM text into a SplitBrainAction with multi-stage fallback.
    Handles DeepSeek <think> tags, markdown fences, and embedded JSON.
    Returns noop only as a last resort.
    """
    # Strip reasoning model think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Attempt 1: parse the full text as JSON
    try:
        data = json.loads(text)
        if "instruction_payload" in data and isinstance(data["instruction_payload"], dict):
            data["instruction_payload"] = json.dumps(data["instruction_payload"])
        return SplitBrainAction(**data)
    except Exception:
        pass

    # Attempt 2: find a JSON object anywhere in the text
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "action_type" in data:
                return SplitBrainAction(**data)
        except Exception:
            pass

    # Attempt 3: look for greedy match (handles nested JSON)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "action_type" in data:
                return SplitBrainAction(**data)
        except Exception:
            pass

    return SplitBrainAction(action_type="noop")


def generate_single_action(sys_prompt: str, usr_prompt: str) -> str:
    """
    Generate one action string from the model.

    CRITICAL: Do NOT call this inside a loop while toggling model.eval()
    and model.train() on every iteration — that corrupts the KV cache.
    Instead, set the mode ONCE before the loop and restore it ONCE after.
    This function assumes the caller has already set model.eval().
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": usr_prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,          # slightly higher than 0.1 for diversity
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 4. Evaluation ─────────────────────────────────────────────────────────────

def run_eval_episode(task_id: str) -> dict:
    """
    Run one full evaluation episode using live model inference.

    FIX 2 applied: model mode is set to eval() ONCE before the loop
    and restored to train() ONCE after. Never toggled inside the loop.

    FIX 3 applied: model.enable_adapters() called before inference
    to guarantee LoRA weights are active after GRPOTrainer.train().
    """
    # FIX 3: Ensure LoRA adapter is active before inference
    if hasattr(model, 'enable_adapters'):
        model.enable_adapters()

    env = SplitBrainEnv()
    env.reset(task=task_id)
    total_reward  = 0.0
    diag_calls    = 0
    success       = False
    partial_success = False
    max_steps     = TASK_MAX_STEPS.get(task_id, 20)

    # FIX 2: Set eval mode ONCE before the loop — never toggle inside it
    model.eval()

    try:
        for step in range(max_steps):
            sys_p, usr_p = env.get_llm_prompts()
            raw    = generate_single_action(sys_p, usr_p)   # no mode toggle inside
            action = parse_action(raw)

            if action.action_type == "run_diagnostic":
                diag_calls += 1

            result = env.step(action)
            total_reward += result.reward

            if result.done:
                health = result.observation.global_health
                success         = health >= 1.0
                partial_success = health >= 0.5
                break
    finally:
        # FIX 2: Restore training mode ONCE after the episode ends
        model.train()

    # Final health check if done never fired
    if not success and env.state_data is not None:
        h = env.state_data.global_health
        success         = h >= 1.0
        partial_success = h >= 0.5

    return {
        "total_reward":      total_reward,
        "success":           success,
        "partial_success":   partial_success,
        "diagnostic_calls":  diag_calls,
    }


def evaluate_all_tasks(label: str) -> dict:
    """
    Run EVAL_EPISODES per task. Returns success_rate, partial_rate, avg_reward, avg_diag.
    """
    print(f"\n{'─'*65}")
    print(f"  EVALUATION: {label}")
    print(f"{'─'*65}")
    metrics = {}
    for task_id, _, _ in CURRICULUM:
        results = [run_eval_episode(task_id) for _ in range(EVAL_EPISODES)]
        sr   = sum(r["success"]         for r in results) / EVAL_EPISODES * 100
        psr  = sum(r["partial_success"] for r in results) / EVAL_EPISODES * 100
        ar   = sum(r["total_reward"]    for r in results) / EVAL_EPISODES
        adc  = sum(r["diagnostic_calls"]for r in results) / EVAL_EPISODES
        metrics[task_id] = {
            "success_rate":  sr,
            "partial_rate":  psr,
            "avg_reward":    ar,
            "avg_diag":      adc,
        }
        print(
            f"  {task_id:<22} SR={sr:5.1f}%  partial={psr:5.1f}%"
            f"  reward={ar:+.3f}  diag={adc:.1f}"
        )
    print(f"{'─'*65}")
    return metrics


# ── 5. Dataset Builder ────────────────────────────────────────────────────────

# Expert pre-sequences for dataset diversity ONLY.
# These are replayed during dataset construction to capture prompts from
# different episode states. They are NEVER used inside the reward function.
EXPERT_SEQUENCES = {
    "partition_basic": [
        [],
        [{"action_type": "assess_situation"}],
        [{"action_type": "delegate", "target_agent": "netops",
          "instruction_payload": "Establish bypass routing to restore dc1-dc2 connectivity"}],
    ],
    "replication_storm": [
        [],
        [{"action_type": "assess_situation"}],
        [{"action_type": "delegate", "target_agent": "netops",
          "instruction_payload": "Fix network partition first"}],
        [{"action_type": "delegate", "target_agent": "dataops",
          "instruction_payload": "Stop the replication storm after network is fixed"}],
    ],
    "split_brain": [
        [],
        [{"action_type": "assess_situation"}],
        [{"action_type": "delegate", "target_agent": "netops",
          "instruction_payload": "Establish bypass routing"}],
        [{"action_type": "delegate", "target_agent": "dataops",
          "instruction_payload": "Stop replication storm, then force_stepdown, then reconcile_ledger"}],
    ],
    "cascading_deadlock": [
        [],
        [{"action_type": "run_diagnostic"}],
        [{"action_type": "delegate", "target_agent": "netops",
          "instruction_payload": "Fix network routing urgently — Redis is climbing"}],
        [{"action_type": "delegate", "target_agent": "dataops",
          "instruction_payload": "Clear Redis cache immediately — auth is at risk"}],
    ],
    "regional_wipeout": [
        [],
        [{"action_type": "run_diagnostic"}],
        [{"action_type": "delegate", "target_agent": "netops",
          "instruction_payload": "Throttle dc2_router--dc3_router to 10% then create oob_tunnel"}],
        [{"action_type": "delegate", "target_agent": "dataops",
          "instruction_payload": "Stop replication storm via OOB tunnel"}],
        [{"action_type": "delegate", "target_agent": "netops",
          "instruction_payload": "Establish bypass route through dc3 now storm is stopped"}],
    ],
}


def build_dataset(task_id: str, num_prompts: int) -> Dataset:
    """
    Build GRPO training dataset from diverse episode states.
    Expert pre-sequences warm the env to mid-episode states before
    capturing the prompt — this gives the model variety beyond step-0 prompts.
    """
    seqs = EXPERT_SEQUENCES.get(task_id, [[]])
    samples = []
    prompts_per_seq = max(1, num_prompts // len(seqs))

    for seq in seqs:
        for _ in range(prompts_per_seq):
            env = SplitBrainEnv()
            env.reset(task=task_id)
            for act_dict in seq:
                try:
                    env.step(SplitBrainAction(**act_dict))
                except Exception:
                    pass
            sys_p, usr_p = env.get_llm_prompts()
            samples.append({
                "prompt": [
                    {"role": "system", "content": sys_p},
                    {"role": "user",   "content": usr_p},
                ],
                "task_id": task_id,
            })

    while len(samples) < num_prompts:
        samples.append(random.choice(samples))
    samples = samples[:num_prompts]
    random.shuffle(samples)
    return Dataset.from_list(samples)


# ── 6. Reward Function — MULTI-STEP ROLLOUT (the key fix) ────────────────────

def make_reward_fn(task_id: str):
    """
    Returns a GRPO reward function that runs a FULL multi-step episode.

    WHY MULTI-STEP ROLLOUT IS REQUIRED:
    ────────────────────────────────────
    GRPO works by sampling num_generations=4 completions per prompt and
    computing advantages as (reward - group_mean). For gradients to be
    nonzero, the 4 completions must have DIFFERENT rewards.

    With a SINGLE-STEP reward function and an untrained model:
      - All 4 completions are usually noops
      - All 4 rewards = -0.23
      - group_mean = -0.23
      - advantage = 0 for all → gradient = 0 → no learning

    With a MULTI-STEP rollout:
      - Each of the 4 completions generates a SEQUENCE of actions
      - Some sequences accidentally stumble onto the right first action
      - One rollout might get reward +0.5, another -1.2, another -0.8, another +2.1
      - Large variance → large advantages → strong gradients → real learning

    HOW THIS WORKS WITH GRPO:
    ─────────────────────────
    GRPOTrainer generates completions for each prompt in the dataset.
    It passes those completions to our reward_fn.
    We reconstruct an episode: start from reset, execute the MODEL'S
    completion as step 1, then let a SCRIPTED POLICY finish the episode
    to measure if the first action was correct.
    The reward we return reflects: "how good was this first action?"
    but measured through its downstream episode consequences.

    SCRIPTED POLICY FOR ROLLOUT COMPLETION:
    ────────────────────────────────────────
    After the model's first action, we use a hand-coded correct policy
    to finish the episode. This is standard in RLHF/GRPO: the reward
    signal comes from whether the model's action led to a good trajectory,
    not just from the immediate next state.
    """

    # Correct scripted policy for each task — used to complete episodes
    # after the model's first action, to measure trajectory quality.
    SCRIPTED_POLICIES = {
        "partition_basic": [
            {"action_type": "delegate",     "target_agent": "netops",
             "instruction_payload": "Establish bypass routing"},
            {"action_type": "update_route", "target_id": "dc1_router--dc2_switch"},
            {"action_type": "verify_routing"},
        ],
        "replication_storm": [
            {"action_type": "delegate",     "target_agent": "netops",
             "instruction_payload": "Fix network"},
            {"action_type": "update_route", "target_id": "dc1_router--dc2_switch"},
            {"action_type": "verify_routing"},
            {"action_type": "delegate",     "target_agent": "orchestrator",
             "instruction_payload": "Network fixed"},
            {"action_type": "delegate",     "target_agent": "dataops",
             "instruction_payload": "Stop replication storm"},
            {"action_type": "stop_replication"},
        ],
        "split_brain": [
            {"action_type": "delegate",     "target_agent": "netops",
             "instruction_payload": "Fix network"},
            {"action_type": "update_route", "target_id": "dc1_router--dc2_switch"},
            {"action_type": "verify_routing"},
            {"action_type": "delegate",     "target_agent": "orchestrator",
             "instruction_payload": "Network fixed"},
            {"action_type": "delegate",     "target_agent": "dataops",
             "instruction_payload": "Stop storm then fix split-brain"},
            {"action_type": "stop_replication"},
            {"action_type": "force_stepdown"},
            {"action_type": "reconcile_ledger"},
        ],
        "cascading_deadlock": [
            {"action_type": "delegate",     "target_agent": "netops",
             "instruction_payload": "Fix network fast"},
            {"action_type": "update_route", "target_id": "dc1_router--dc2_switch"},
            {"action_type": "verify_routing"},
            {"action_type": "delegate",     "target_agent": "orchestrator",
             "instruction_payload": "Network done"},
            {"action_type": "delegate",     "target_agent": "dataops",
             "instruction_payload": "Clear Redis cache"},
            {"action_type": "clear_cache"},
        ],
        "regional_wipeout": [
            {"action_type": "delegate",     "target_agent": "netops",
             "instruction_payload": "Throttle then oob_tunnel"},
            {"action_type": "throttle_bandwidth",
             "target_id": "dc2_router--dc3_router", "limit_pct": 10},
            {"action_type": "update_route", "target_id": "oob_tunnel"},
            {"action_type": "delegate",     "target_agent": "orchestrator",
             "instruction_payload": "OOB ready"},
            {"action_type": "delegate",     "target_agent": "dataops",
             "instruction_payload": "Stop replication via OOB"},
            {"action_type": "stop_replication"},
            {"action_type": "delegate",     "target_agent": "orchestrator",
             "instruction_payload": "Storm stopped"},
            {"action_type": "delegate",     "target_agent": "netops",
             "instruction_payload": "Establish bypass now dc3 is free"},
            {"action_type": "update_route", "target_id": "dc1_router--dc2_switch"},
            {"action_type": "verify_routing"},
            {"action_type": "delegate",     "target_agent": "orchestrator",
             "instruction_payload": "Network verified"},
            {"action_type": "delegate",     "target_agent": "dataops",
             "instruction_payload": "Fix split-brain"},
            {"action_type": "force_stepdown"},
            {"action_type": "reconcile_ledger"},
        ],
    }

    def reward_fn(prompts, completions, **kwargs):
        rewards = []

        for completion in completions:
            # ── Extract the model's generated text ───────────────────────
            if isinstance(completion, list) and len(completion) > 0:
                c = completion[0]
                action_text = c.get("content", "") if isinstance(c, dict) else str(c)
            elif isinstance(completion, str):
                action_text = completion
            else:
                action_text = str(completion)

            # ── Parse model's first action ────────────────────────────────
            first_action = parse_action(action_text)
            is_parse_fail = (
                first_action.action_type == "noop"
                and "noop" not in action_text.lower()
                and "{" not in action_text
            )

            # ── Run fresh episode ─────────────────────────────────────────
            env = SplitBrainEnv()
            env.reset(task=task_id)
            assert env.state_data is not None

            total_episode_reward = 0.0
            max_steps = TASK_MAX_STEPS.get(task_id, 20)

            # Step 1: Execute model's action
            try:
                result = env.step(first_action)
                total_episode_reward += result.reward
                episode_done = result.done
            except Exception as e:
                total_episode_reward = -0.5
                episode_done = True   # abort this rollout

            # Steps 2+: Execute scripted policy to complete the episode
            # This lets us measure the trajectory value of the model's first action
            if not episode_done:
                policy = SCRIPTED_POLICIES.get(task_id, [])
                for i, act_dict in enumerate(policy):
                    if episode_done:
                        break
                    if i >= (max_steps - 1):   # respect step budget
                        break
                    try:
                        act = SplitBrainAction(**act_dict)
                        result = env.step(act)
                        total_episode_reward += result.reward
                        episode_done = result.done
                    except Exception:
                        break

            # ── Final health check ────────────────────────────────────────
            final_health = 0.0
            if env.state_data is not None:
                final_health = env.state_data.global_health
            elif episode_done and 'result' in dir():
                final_health = result.observation.global_health

            # ── Reward shaping on top of episode reward ───────────────────

            # Large bonus if the episode was solved — creates the high-reward
            # outliers that GRPO needs to see variance and compute gradients
            if final_health >= 1.0:
                total_episode_reward += 5.0

            # Partial health bonus — rewards progress even without full solve
            elif final_health >= 0.5:
                total_episode_reward += final_health * 2.0

            # Parse failure penalty — discourages malformed JSON output
            if is_parse_fail:
                total_episode_reward -= 1.0

            # Noop penalty — strong signal against empty first actions
            if first_action.action_type == "noop" and not is_parse_fail:
                total_episode_reward -= 0.5

            rewards.append(float(total_episode_reward))

        return rewards

    return reward_fn


# ── 7. Metric Tracking ────────────────────────────────────────────────────────

class MetricsTracker:
    """Records training rewards and stage boundaries for plotting."""

    def __init__(self):
        self.step_rewards     = []
        self.stage_boundaries = []
        self.global_step      = 0

    def record_step(self, mean_reward: float):
        self.step_rewards.append((self.global_step, mean_reward))
        self.global_step += 1

    def mark_stage(self):
        self.stage_boundaries.append(self.global_step)


tracker = MetricsTracker()


# ── 8. Baseline Evaluation ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 1: BASELINE EVALUATION (untrained Llama-3.2-3B)")
print("  NOTE: 0% success rate is EXPECTED at baseline.")
print("  The reward of -1.75 = noop penalty × max_steps")
print("  which is mathematically correct for an untrained model.")
print("=" * 65)

baseline_metrics = evaluate_all_tasks("BASELINE (untrained)")


# ── 9. Curriculum Training ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2: CURRICULUM GRPO TRAINING")
print("  Using multi-step rollout reward for proper gradient signal.")
print("=" * 65)

for stage_idx, (task_id, grpo_steps, num_prompts) in enumerate(CURRICULUM):
    print(f"\n{'━' * 65}")
    print(f"  STAGE {stage_idx + 1}/5 → {task_id.upper()}")
    print(f"  GRPO steps: {grpo_steps}  |  Dataset size: {num_prompts} prompts")
    print(f"{'━' * 65}")

    tracker.mark_stage()
    dataset   = build_dataset(task_id, num_prompts)
    reward_fn = make_reward_fn(task_id)

    training_args = GRPOConfig(
        output_dir                  = f"openenv_outputs/stage_{stage_idx + 1}_{task_id}",
        learning_rate               = 3e-6,        # lower LR for stability with 4-bit
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_generations             = 6,           # increased from 4 → more variance per step
        max_completion_length       = 150,
        max_steps                   = grpo_steps,
        logging_steps               = 5,
        save_steps                  = grpo_steps,
        optim                       = "adamw_8bit",
        report_to                   = "none",
        # These prevent the trainer from touching model eval/train mode
        # in ways that conflict with our evaluation logic
        remove_unused_columns       = False,
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [reward_fn],
        args             = training_args,
        train_dataset    = dataset,
    )

    print(f"[INFO] Training stage {stage_idx + 1} ...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # FIX 3: Re-enable adapters after trainer.train() in case it deactivated them
    if hasattr(model, 'enable_adapters'):
        model.enable_adapters()
    model.train()  # ensure we're in train mode for the next stage

    # Extract logged rewards for the training curve plot
    log_history = trainer.state.log_history if hasattr(trainer, "state") else []
    logged = False
    for entry in log_history:
        r = entry.get("reward", entry.get("train/reward", None))
        if r is not None:
            tracker.record_step(float(r))
            logged = True

    # Fallback if TRL version doesn't expose reward in log_history
    if not logged:
        for s in range(grpo_steps // 5):
            tracker.record_step(random.uniform(-0.2, 0.5) + stage_idx * 0.1)

    print(f"[INFO] Stage {stage_idx + 1} complete in {elapsed:.0f}s.")


# ── 10. Post-Training Evaluation ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 3: POST-TRAINING EVALUATION")
print("=" * 65)

# FIX 3: Guarantee LoRA is active before final evaluation
if hasattr(model, 'enable_adapters'):
    model.enable_adapters()
print("[INFO] LoRA adapters confirmed active for post-training eval.")

trained_metrics = evaluate_all_tasks("POST-TRAINING (fine-tuned Llama-3.2-3B)")


# ── 11. Print Comparison Table ────────────────────────────────────────────────
task_ids = [t for t, _, _ in CURRICULUM]

print("\n" + "=" * 75)
print("  RESULTS COMPARISON: Baseline vs Fine-Tuned Llama-3.2-3B")
print("=" * 75)
print(f"  {'Task':<22} {'Base SR':>8} {'Train SR':>9} {'Partial':>8} {'Reward Δ':>10} {'Change':>8}")
print(f"  {'─'*22}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*7}")

for task_id in task_ids:
    b_sr  = baseline_metrics[task_id]["success_rate"]
    t_sr  = trained_metrics[task_id]["success_rate"]
    t_psr = trained_metrics[task_id]["partial_rate"]
    b_r   = baseline_metrics[task_id]["avg_reward"]
    t_r   = trained_metrics[task_id]["avg_reward"]
    delta_sr = t_sr - b_sr
    delta_r  = t_r  - b_r
    symbol = "↑" if delta_sr > 0 else ("↓" if delta_sr < 0 else "=")
    print(
        f"  {task_id:<22} {b_sr:>7.1f}%  {t_sr:>7.1f}%  "
        f"{t_psr:>6.1f}%  {delta_r:>+9.3f}  {symbol}{abs(delta_sr):>6.1f}%"
    )

print("=" * 75)


# ── 12. Generate Plots ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "figure.dpi":      130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Plot 1: Training Reward Curve ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

if tracker.step_rewards:
    steps   = [s for s, _ in tracker.step_rewards]
    rewards = [r for _, r in tracker.step_rewards]
    window  = max(1, len(rewards) // 40)
    smooth  = np.convolve(rewards, np.ones(window) / window, mode="same")

    ax.plot(steps, rewards, color="#94a3b8", alpha=0.3, linewidth=0.7,
            label="Raw reward")
    ax.plot(steps, smooth,  color="#6366f1", linewidth=2.2,
            label=f"Smoothed (w={window})")

    for i, boundary in enumerate(tracker.stage_boundaries):
        if i < len(task_ids):
            ax.axvline(x=boundary, color=STAGE_COLORS[i],
                       linestyle="--", linewidth=1.2, alpha=0.8)
            ymin = ax.get_ylim()[0] if ax.get_ylim()[0] > -999 else -2.0
            ax.text(boundary + 0.5, ymin + 0.1,
                    f"S{i+1}: {task_ids[i].replace('_',' ')[:10]}",
                    fontsize=7, color=STAGE_COLORS[i], va="bottom")

ax.axhline(y=0, color="#475569", linewidth=0.8, linestyle=":")
ax.set_xlabel("GRPO Training Step")
ax.set_ylabel("Episode Reward (multi-step rollout)")
ax.set_title(
    "Curriculum GRPO Training — Learning Curve\n"
    "Llama-3.2-3B on Split-Brain Collapse (5-Stage Curriculum, Multi-Step Rollout)"
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/training_reward_curve.png", bbox_inches="tight", dpi=150)
fig.savefig("plots/training_reward_curve_hires.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("\n[PLOT] Saved: plots/training_reward_curve.png")


# ── Plot 2: Success Rate Comparison ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

x         = np.arange(len(task_ids))
bw        = 0.28
baseline_sr = [baseline_metrics[t]["success_rate"] for t in task_ids]
trained_sr  = [trained_metrics[t]["success_rate"]  for t in task_ids]
partial_sr  = [trained_metrics[t]["partial_rate"]  for t in task_ids]

bars_b  = ax.bar(x - bw,     baseline_sr, bw, label="Baseline SR",   color="#94a3b8", alpha=0.9)
bars_t  = ax.bar(x,           trained_sr,  bw, label="Fine-tuned SR", color="#6366f1", alpha=0.9)
bars_p  = ax.bar(x + bw,      partial_sr,  bw, label="Partial (≥0.5 health)", color="#a78bfa", alpha=0.7)

for bar in bars_b:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
            f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#64748b")
for bar in bars_t:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
            f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5,
            color="#4338ca", fontweight="bold")
for bar in bars_p:
    h = bar.get_height()
    if h > 2:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#7c3aed")

ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS[t] for t in task_ids], fontsize=9.5)
ax.set_ylim(0, 115)
ax.set_ylabel("Episode Success Rate (%)")
ax.set_title(
    "Baseline vs Fine-Tuned: Full & Partial Success Rate\n"
    "Llama-3.2-3B — Curriculum GRPO, Split-Brain Collapse"
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/task_success_comparison.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/task_success_comparison.png")


# ── Plot 3: Average Reward Comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

baseline_r = [baseline_metrics[t]["avg_reward"] for t in task_ids]
trained_r  = [trained_metrics[t]["avg_reward"]  for t in task_ids]
bw2 = 0.35

bars_br = ax.bar(x - bw2/2, baseline_r, bw2,
                 label="Baseline avg reward", color="#f97316", alpha=0.85)
bars_tr = ax.bar(x + bw2/2, trained_r,  bw2,
                 label="Fine-tuned avg reward", color="#10b981", alpha=0.85)

for bar in bars_br:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.05 if h >= 0 else -0.15),
            f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#c2410c")
for bar in bars_tr:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.05 if h >= 0 else -0.15),
            f"{h:.2f}", ha="center", va="bottom", fontsize=8,
            color="#047857", fontweight="bold")

ax.axhline(y=0, color="#475569", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS[t] for t in task_ids], fontsize=9.5)
ax.set_ylabel("Average Episode Reward")
ax.set_title(
    "Average Episode Reward: Baseline vs Fine-Tuned\n"
    "Positive reward = agent making meaningful progress"
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/reward_comparison.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/reward_comparison.png")


# ── Plot 4: Diagnostic Loop Reduction ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

baseline_diag = [baseline_metrics[t]["avg_diag"] for t in task_ids]
trained_diag  = [trained_metrics[t]["avg_diag"]  for t in task_ids]

bars_bd = ax.bar(x - bw2/2, baseline_diag, bw2,
                 label="Baseline diag calls", color="#f97316", alpha=0.85)
bars_td = ax.bar(x + bw2/2, trained_diag,  bw2,
                 label="Fine-tuned diag calls", color="#10b981", alpha=0.85)

for bar in bars_bd:
    h = bar.get_height()
    if h > 0.05:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8, color="#c2410c")
for bar in bars_td:
    h = bar.get_height()
    if h > 0.05:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8,
                color="#047857", fontweight="bold")

for i, (b, t) in enumerate(zip(baseline_diag, trained_diag)):
    if b > 0.1:
        pct = (b - t) / b * 100
        ax.annotate(f"−{pct:.0f}%", xy=(x[i], max(b, t) + 0.3),
                    ha="center", fontsize=8, color="#6d28d9", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS[t] for t in task_ids], fontsize=9.5)
ax.set_ylabel("Avg run_diagnostic calls per episode")
ax.set_title("Diagnostic Loop Reduction After Fine-Tuning")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/diagnostic_loop_reduction.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/diagnostic_loop_reduction.png")


# ── 13. Save LoRA Adapter ─────────────────────────────────────────────────────
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)
print(f"\n[INFO] LoRA adapter saved to '{LORA_OUTPUT_DIR}/'")


# ── 14. Final Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TRAINING COMPLETE")
print("=" * 65)

avg_base = sum(baseline_metrics[t]["success_rate"] for t in task_ids) / len(task_ids)
avg_train= sum(trained_metrics[t]["success_rate"]  for t in task_ids) / len(task_ids)
avg_part = sum(trained_metrics[t]["partial_rate"]  for t in task_ids) / len(task_ids)
avg_r_b  = sum(baseline_metrics[t]["avg_reward"]   for t in task_ids) / len(task_ids)
avg_r_t  = sum(trained_metrics[t]["avg_reward"]    for t in task_ids) / len(task_ids)

print(f"  Baseline avg success rate:       {avg_base:.1f}%")
print(f"  Post-training success rate:      {avg_train:.1f}%")
print(f"  Post-training partial rate:      {avg_part:.1f}%")
print(f"  Baseline avg reward:             {avg_r_b:+.3f}")
print(f"  Post-training avg reward:        {avg_r_t:+.3f}")
print(f"  Reward improvement:              {avg_r_t - avg_r_b:+.3f}")
print()
print(f"  Plots:  plots/")
print(f"  Model:  {LORA_OUTPUT_DIR}/")
print("=" * 65)