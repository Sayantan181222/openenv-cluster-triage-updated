import torch
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import re
import json

# Import the Split-Brain Environment
from agents.split_brain.environment import SplitBrainEnv
from agents.split_brain.models import SplitBrainAction

# --- 1. Load the Model ---
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit", 
    max_seq_length=1024, 
    load_in_4bit=True,
    fast_inference=False,  # vLLM crashes on Colab T4 (compute cap 7.5)
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
        env = SplitBrainEnv()
        # FIX: Use the string name defined in the environment
        obs = env.reset(task="cascading_deadlock") 
        
        action_text = completion[0]['content']
        match = re.search(r'\{.*?\}', action_text, re.DOTALL)
        
        step_reward = -0.5 
        if match:
            try:
                data = json.loads(match.group(0))
                # Ensure the data matches the SplitBrainAction model
                action = SplitBrainAction(**data)
                
                result = env.step(action)
                step_reward = result.reward
                
                # FIX: Field name is global_health in models.py
                if result.done and result.observation.global_health >= 1.0:
                    step_reward += 5.0
                    
                if action.action_type == "run_diagnostic":
                    step_reward -= 0.5 
                    
            except Exception:
                pass
                
        rewards.append(step_reward)
        
    return rewards

# --- 3. Create Training Dataset ---
dummy_env = SplitBrainEnv()
init_obs = dummy_env.reset(task="cascading_deadlock")

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
] * 100)

# --- 4. Train with GRPO ---
training_args = GRPOConfig(
    output_dir="openenv_outputs",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    
    # ── AGGRESSIVE MEMORY SAVING ──
    num_generations=4,              # Reduced from 8 (saves 50% generation VRAM)
    max_prompt_length=512,          # Keep prompts short
    max_completion_length=256,      # Limit the JSON output length
    
    max_steps=50,
    logging_steps=5,
    optim="adamw_8bit",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[openenv_reward_function],
    args=training_args,
    train_dataset=dataset,
)

print("Starting OpenEnv GRPO Training...")
trainer.train()

model.save_pretrained("openenv-split-brain-lora")
print("Training Complete!")