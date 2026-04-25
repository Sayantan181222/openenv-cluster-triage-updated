# Minimal Unsloth GRPO Training Script for OpenEnv
# ------------------------------------------------
# Upload this script to a Google Colab notebook to satisfy the Hackathon requirement.
# This script uses Unsloth and TRL's GRPOTrainer to train an LLM on your custom OpenEnv.

# 1. Install dependencies (Run this in a Colab cell first)
# !pip install unsloth trl openenv-core
# !pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import re
import json

# Import the Split-Brain Environment (where the 8B model struggled)
from agents.split_brain.environment import SplitBrainEnv
from agents.split_brain.models import SplitBrainAction

# --- 1. Load the Model via Unsloth ---
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct", # Changed to Llama-3.1-8B
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False, # <-- CHANGED: Disabled vLLM to fix the graph compile crash
    max_lora_rank=16,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
)

# --- 2. Define the OpenEnv Reward Function ---
def openenv_reward_function(prompts, completions, **kwargs):
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # 1. Initialize the specific Split-Brain environment
        env = SplitBrainEnv()
        obs = env.reset(task=4) # Task 4 is the one that was failing!
        
        # 2. Extract the JSON action from the LLM's completion
        action_text = completion[0]['content']
        match = re.search(r'\{.*?\}', action_text, re.DOTALL)
        
        step_reward = -0.5 # Default penalty for invalid JSON formatting
        if match:
            try:
                data = json.loads(match.group(0))
                # Add the active agent context since split_brain requires it
                if "agent" not in data:
                    data["agent"] = "netops" 
                
                action = SplitBrainAction(**data)
                
                # 3. Step the environment
                result = env.step(action)
                step_reward = result.reward
                
                # Bonus for breaking out of the loop and solving it
                if result.done:
                    step_reward += 5.0
                    
                # Heavy penalty for infinite diagnostic loop
                if action.action_type == "run_diagnostic":
                    step_reward -= 0.5 
                    
            except Exception:
                pass
                
        rewards.append(step_reward)
        
    return rewards

# --- 3. Create Training Dataset ---
dummy_env = SplitBrainEnv()
init_obs = dummy_env.reset(task=4)

system_prompt = "You are the netops agent. You MUST output EXACTLY ONE JSON object."
user_prompt = f"CURRENT STATE:\n{init_obs.model_dump_json(indent=2)}\n\nRECENT EVENTS: Orchestrator delegated to netops. DIAGNOSTIC: dc1_router--dc2_switch is degraded. Use update_route on dc1_router--dc2_switch.\n\nValid actions: update_route, verify_routing, throttle_bandwidth, delegate, noop, run_diagnostic.\nProvide your action in JSON:"

from datasets import Dataset
dataset = Dataset.from_list([
    {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
] * 100) # Duplicate the prompt 100 times to form a batch dataset

# --- 4. Train with GRPO ---
training_args = GRPOConfig(
    output_dir="openenv_outputs",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=50,
    logging_steps=5,
    optim="adamw_8bit",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[openenv_reward_function], # Inject OpenEnv logic here!
    args=training_args,
    train_dataset=dataset,
)

print("Starting OpenEnv GRPO Training...")
trainer.train()

# --- 5. Save the trained agent ---
model.save_pretrained("openenv-cluster-sre-lora")
print("Training Complete! Model saved.")
