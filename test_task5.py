from agents.split_brain.environment import SplitBrainEnv
from app import _extract_action, MODEL_NAME, llm_client
import json
import re

env = SplitBrainEnv()
obs = env.reset(task="regional_wipeout")

# We simulate Step 1 manually: run_diagnostic
print("STEP 1: run_diagnostic")
result = env.step({"action_type": "run_diagnostic"})
obs = result.observation
print(result.info["message"])

system_prompt, user_prompt = env.get_llm_prompts()
print("\n--- PROMPTS FOR STEP 2 ---")
# print("SYSTEM:", system_prompt)
# print("USER:", user_prompt)

completion = llm_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.1,
    max_tokens=400,
)
raw_text = completion.choices[0].message.content or ""
print("\n--- LLM RAW TEXT ---")
print(raw_text)

# Test parse
def parse_action(text: str):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            return f"Parse error: {e}"
    return "No JSON found"

print("\n--- PARSED ACTION ---")
print(parse_action(raw_text))

