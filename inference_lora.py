"""
inference_lora.py — Local LoRA Inference for Split-Brain Environment
=====================================================================
Loads the GRPO-trained LoRA adapter (openenv-split-brain-lora/) on top of
Llama-3.2-3B-Instruct and runs it against the Split-Brain environment.

Usage (requires GPU with >=6GB VRAM, or run on Colab):
    pip install unsloth peft transformers torch
    python inference_lora.py

This script demonstrates the improvement of the fine-tuned model over
the base model on the Split-Brain cascading_deadlock task (Task 4).
"""

import os
import json
import re
import torch
from typing import List, Optional

from agents.split_brain.environment import SplitBrainEnv
from agents.split_brain.models import SplitBrainAction

# ── 1. Load Model + LoRA Adapter ────────────────────────────────────────────

LORA_PATH = os.path.join(os.path.dirname(__file__), "openenv-split-brain-lora")
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_STEPS = 15

print(f"[INFO] Loading base model: {BASE_MODEL}")
print(f"[INFO] Applying LoRA adapter from: {LORA_PATH}")

try:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=1024,
        load_in_4bit=True,
        fast_inference=False,
    )
    # Load the trained LoRA weights on top
    model.load_adapter(LORA_PATH, adapter_name="split_brain_lora")
    FastLanguageModel.for_inference(model)
    print("[INFO] LoRA adapter loaded successfully via Unsloth.")
    USE_UNSLOTH = True

except ImportError:
    # Fallback: use raw transformers + PEFT (no Unsloth needed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("[INFO] Unsloth not available, falling back to transformers + PEFT...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    print("[INFO] LoRA adapter loaded successfully via PEFT.")
    USE_UNSLOTH = False


# ── 2. Local Generation Function ────────────────────────────────────────────

def generate_action(system_prompt: str, user_prompt: str) -> str:
    """Generate a single action using the local LoRA-tuned model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 3. Parse LLM Output into Action ─────────────────────────────────────────

def parse_action(text: str) -> SplitBrainAction:
    """Extract a JSON action from the model's output text."""
    # Strip thinking blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "action_type" in data:
                return SplitBrainAction(**data)
        except Exception:
            pass

    # Fallback: noop
    return SplitBrainAction(action_type="noop")


# ── 4. Run the Split-Brain Episode ──────────────────────────────────────────

def run_episode(task_id: str = "cascading_deadlock") -> dict:
    """Run a full episode on the Split-Brain environment using the LoRA model."""
    env = SplitBrainEnv()
    obs = env.reset(task=task_id)

    rewards: List[float] = []
    actions_taken: List[str] = []
    total_reward = 0.0

    print(f"\n{'='*70}")
    print(f"  SPLIT-BRAIN LORA INFERENCE — Task: {task_id}")
    print(f"  Model: {BASE_MODEL} + LoRA ({LORA_PATH})")
    print(f"{'='*70}\n")

    for step in range(1, MAX_STEPS + 1):
        # Get prompts from the environment (multi-agent aware)
        system_prompt, user_prompt = env.get_llm_prompts()
        actor = env.state_data.current_actor

        # Generate action with the LoRA model
        raw_text = generate_action(system_prompt, user_prompt)
        action = parse_action(raw_text)

        # Step the environment
        result = env.step(action)
        reward = result.reward
        done = result.done
        msg = result.info.get("message", "")

        rewards.append(reward)
        total_reward += reward
        actions_taken.append(action.action_type)

        print(f"Step {step:2d} [{actor}] {action.action_type}"
              f"{(' → ' + action.target_id) if action.target_id else ''}")
        print(f"  reward={reward:+.3f} | {msg}")

        if done:
            print(f"\n{'─'*70}")
            print(f"  ✅ EPISODE COMPLETE at step {step}")
            break
    else:
        print(f"\n{'─'*70}")
        print(f"  ⏱  MAX STEPS REACHED ({MAX_STEPS})")

    # Summary
    final_health = env.state_data.global_health
    success = final_health >= 1.0

    print(f"  Final Health: {final_health:.2f}")
    print(f"  Total Reward: {total_reward:.3f}")
    print(f"  Success: {'YES ✅' if success else 'NO ❌'}")
    print(f"  Actions: {' → '.join(actions_taken)}")
    print(f"{'='*70}\n")

    # Detect if the model got stuck in a loop
    diagnostic_count = actions_taken.count("run_diagnostic")
    if diagnostic_count > 2:
        print(f"  ⚠️  WARNING: Model ran diagnostic {diagnostic_count} times (loop detected)")
    elif diagnostic_count <= 1:
        print(f"  ✅ IMPROVEMENT: Model avoided the diagnostic loop!")

    return {
        "task": task_id,
        "success": success,
        "steps": len(rewards),
        "total_reward": total_reward,
        "final_health": final_health,
        "actions": actions_taken,
        "diagnostic_loops": diagnostic_count,
    }


# ── 5. Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run the task that was previously failing with the base 8B model
    result = run_episode("cascading_deadlock")

    print("\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)
    print(f"  Before (base Llama 3.1 8B, no LoRA):")
    print(f"    → Stuck in infinite run_diagnostic loop (10+ repeats)")
    print(f"    → Never executed update_route, verify_routing, etc.")
    print(f"    → Episode timed out with minimal reward")
    print(f"")
    print(f"  After (Llama 3.2 3B + GRPO LoRA):")
    print(f"    → Diagnostic loops: {result['diagnostic_loops']}")
    print(f"    → Actions taken: {' → '.join(result['actions'])}")
    print(f"    → Final health: {result['final_health']:.2f}")
    print(f"    → Success: {'YES ✅' if result['success'] else 'NO ❌'}")
    print("="*70)
