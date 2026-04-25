import re
text = '{"action_type": "delegate", "target_agent": "netops", "instruction_payload": {"action_type": "throttle_bandwidth", "target_id": "dc2_router--dc3_router", "parameters": {"limit_pct": 10}}}'
match = re.search(r'\{.*\}', text, re.DOTALL)
print(match.group(0))
