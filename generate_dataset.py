import os
import json
import random
import sys

# Ensure local directory is in path for imports
sys.path.append(os.getcwd())
from agents.split_brain.environment import SplitBrainEnv

class SREOracle:
    """Expert heuristic agent derived from environment.py rules."""
    def decide(self, env):
        obs = env.state_data 
        actor = obs.current_actor
        task = env.current_task
        history = obs.recent_events 
        
        if actor == "orchestrator":
            if not any("DIAGNOSTIC:" in ev for ev in history):
                return "Initial triage: identify root cause via diagnostics.", {"action_type": "run_diagnostic"}
            if task == "cascading_deadlock" and obs.auth_status == "offline":
                return "Emergency: Auth offline. Must clear Redis cache immediately.", \
                       {"action_type": "delegate", "target_agent": "dataops", "instruction_payload": "clear_cache"}
            if not env.routing_verified:
                return "Network partition detected. Delegating to NetOps.", \
                       {"action_type": "delegate", "target_agent": "netops", "instruction_payload": "fix_network"}
            if obs.newsql.split_brain_active:
                return "Network fixed. Delegating to DataOps to resolve split-brain.", \
                       {"action_type": "delegate", "target_agent": "dataops", "instruction_payload": "resolve_db"}
            return "Systems nominal. Running final check.", {"action_type": "assess_situation"}

        if actor == "netops":
            if task == "regional_wipeout":
                dc2_dc3 = env._find_edge("dc2_router--dc3_router")
                if dc2_dc3 and dc2_dc3.bandwidth_used > 100:
                    return "Throttling DC2-DC3 to 10% to enable OOB tunnel.", \
                           {"action_type": "throttle_bandwidth", "target_id": "dc2_router--dc3_router", "parameters": {"limit_pct": 10}}
                if not obs.oob_tunnel_active:
                    return "Establishing OOB tunnel management link.", {"action_type": "update_route", "target_id": "oob_tunnel"}
            if not env.routing_verified:
                return "Verifying routing to confirm DC connectivity.", {"action_type": "verify_routing"}
            return "Network work done. Reporting to Orchestrator.", \
                   {"action_type": "delegate", "target_agent": "orchestrator", "instruction_payload": "network_ready"}

        if actor == "dataops":
            if task == "cascading_deadlock" and obs.auth_status == "offline":
                return "Flushing Redis cache to restore Auth services.", {"action_type": "clear_cache"}
            if obs.hdfs.replication_storm_active:
                return "Stopping HDFS replication storm.", {"action_type": "stop_replication"}
            if obs.newsql.split_brain_active:
                return "Resolving leadership conflict via force stepdown.", {"action_type": "force_stepdown", "target_id": "dc2"}
            if not env.ledger_reconciled and env.routing_verified:
                return "Reconciling ledger now that network is stable.", {"action_type": "reconcile_ledger"}
            return "Data operations done. Reporting back.", \
                   {"action_type": "delegate", "target_agent": "orchestrator", "instruction_payload": "db_ready"}
        
        return "No action.", {"action_type": "noop"}

def generate_split_datasets():
    env = SplitBrainEnv()
    oracle = SREOracle()
    tasks = ["partition_basic", "replication_storm", "split_brain", "cascading_deadlock", "regional_wipeout"]
    
    # 1. Generate SFT Dataset
    sft_data = []
    print("🛠️  Generating sft_dataset.json...")
    while len(sft_data) < 500:
        env.reset(task=random.choice(tasks))
        for _ in range(25):
            sys_p, usr_p = env.get_llm_prompts()
            thought, action = oracle.decide(env)
            sft_data.append({
                "instruction": sys_p,
                "input": usr_p,
                "output": f"<think>\n{thought}\n</think>\n{json.dumps(action)}"
            })
            res = env.step(action)
            if res.done or len(sft_data) >= 500: break

    # 2. Generate RL Prompts
    rl_prompts = []
    print("🎯 Generating rl_prompts.json...")
    while len(rl_prompts) < 200:
        task = random.choice(tasks)
        env.reset(task=task)
        for _ in range(random.randint(0, 10)): # Start from random episode depths
            _, action = oracle.decide(env)
            env.step(action)
        sys_p, usr_p = env.get_llm_prompts()
        rl_prompts.append({
            "prompt": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            "task_id": task
        })
        if len(rl_prompts) >= 200: break

    with open("sft_dataset.json", "w") as f: json.dump(sft_data, f, indent=2)
    with open("rl_prompts.json", "w") as f: json.dump(rl_prompts, f, indent=2)
    print("\n✅ Dataset generation complete. Files saved locally.")

if __name__ == "__main__":
    generate_split_datasets()