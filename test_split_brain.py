from app import llm_client, MODEL_NAME
from agents.split_brain.environment import SplitBrainEnv
env = SplitBrainEnv()
env.reset("split_brain")
# Simulate netops finishing
env.bypass_established = True
env.routing_verified = True
env.state_data.dc1_dc2_connected = True
env.state_data.network_status = "healthy"
env.state_data.current_actor = "orchestrator"
env.state_data.delegation_context = "Bypass routing established and routing verified. Ready for next steps."
sys, usr = env.get_llm_prompts()
print(f"System: {sys}\nUser: {usr}")
completion = llm_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": sys},
        {"role": "user",   "content": usr},
    ],
    temperature=0.1,
    max_tokens=400,
)
print("RAW LLM OUTPUT:")
print(completion.choices[0].message.content)
action = env._parse_action(completion.choices[0].message.content)
print("PARSED ACTION:")
print(action)
