"""
app.py — OpenEnv Cluster Triage Environment
============================================
Serves:
  • FastAPI HTTP API  — /reset, /step, /state, /health, /tasks
  • Agent endpoints   — /agents, /agent/step
  • Static HTML UI    — GET / → static/index.html
All on port 7860 for Hugging Face Spaces.

ARCHITECTURE:
  agents/__init__.py  → AGENT_REGISTRY (central config)
  agents/<name>/      → each agent's env_class + task metadata
  Add a new agent = new folder + one import + one dict entry
"""

import os
import json
import re
from dotenv import load_dotenv
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from openai import OpenAI

from agents import AGENT_REGISTRY
from agents.cluster_triage.models import ClusterAction, ResetRequest
from agents.split_brain.models import SplitBrainAction

load_dotenv()

# ── LLM Config ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
SPLIT_BRAIN_MODEL = os.getenv("SPLIT_BRAIN_MODEL", "")  # GRPO fine-tuned model for split_brain
API_KEY      = os.getenv("HF_TOKEN", "").strip().strip('"').strip("'")

if API_KEY:
    llm_client = OpenAI(
        base_url=API_BASE_URL, 
        api_key=API_KEY,
        timeout=300.0  
    )
    print(f"[INFO] LLM ready. Token: {API_KEY[:6]}... Model: {MODEL_NAME}")
    # llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    # print(f"[INFO] LLM ready. Token: {API_KEY[:6]}... Model: {MODEL_NAME}")
else:
    llm_client = None
    print("[WARN] HF_TOKEN not set — LLM calls will fail.")

# ── Active Environment State ─────────────────────────────────────────────────
# Tracks which agent is currently loaded. Reset swaps the env.
active_agent_id = None
active_env      = None


def get_or_create_env(agent_id: str):
    """Lazily create/switch the active environment when agent changes."""
    global active_agent_id, active_env
    agent = AGENT_REGISTRY.get(agent_id)
    if not agent:
        raise HTTPException(status_code=400, detail=f"Unknown agent '{agent_id}'. Available: {list(AGENT_REGISTRY.keys())}")
    if active_agent_id != agent_id:
        active_agent_id = agent_id
        active_env = agent["env_class"]()
    return active_env


# ── FastAPI App ───────────────────────────────────────────────────────────────
fastapi_app = FastAPI(
    title="OpenEnv: Distributed Cluster Triage",
    description=(
        "An OpenEnv-compliant RL environment simulating a 4-node enterprise "
        "data cluster. An AI agent acting as an SRE must triage infrastructure "
        "failures by issuing precise commands."
    ),
    version="1.0.0",
)


# ══════════════════════════════════════════════════════════════════════════════
# CORE RL API  (openenv validate + external agents)
# ══════════════════════════════════════════════════════════════════════════════

@fastapi_app.get("/health")
def health():
    """Health check — returns 200 OK."""
    return {
        "status": "ok",
        "env": "cluster-triage",
        "version": "1.0.0",
        "llm_configured": bool(API_KEY),
        "model": MODEL_NAME,
    }


@fastapi_app.post("/reset")
def reset(request: ResetRequest = None):
    """
    Reset the environment for a given task.
    Body: {"task": "easy", "agent_id": "cluster_triage"}
    """
    agent_id = getattr(request, "agent_id", None) or "cluster_triage"
    task     = (request.task if request else "easy")
    env      = get_or_create_env(agent_id)
    agent    = AGENT_REGISTRY[agent_id]
    valid    = [t["id"] for t in agent["tasks"]]
    if task not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Valid: {valid}")
    obs = env.reset(task=task)
    return JSONResponse(content=obs.model_dump())


@fastapi_app.post("/step")
async def step(request: Request):
    """Execute one raw action in the environment (generic — routes by active agent)."""
    if active_env is None:
        raise HTTPException(status_code=400, detail="No environment loaded. Call /reset first.")
    body = await request.json()
    if active_agent_id == "split_brain":
        action = SplitBrainAction(**body)
    else:
        action = ClusterAction(**body)
    result = active_env.step(action)
    return JSONResponse(content=result.model_dump())


@fastapi_app.get("/state")
def state():
    """Return the current cluster observation without advancing the episode."""
    if active_env is None:
        raise HTTPException(status_code=400, detail="No environment loaded. Call /reset first.")
    return JSONResponse(content=active_env.state().model_dump())


@fastapi_app.get("/tasks")
def list_tasks(agent_id: str = "cluster_triage"):
    """List all available tasks with difficulty metadata for a specific agent."""
    if agent_id not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail="Agent not found.")
    
    agent_meta = AGENT_REGISTRY[agent_id]
    return {"tasks": agent_meta["tasks"]}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT REGISTRY API  (frontend discovers agents + tasks)
# ══════════════════════════════════════════════════════════════════════════════

@fastapi_app.get("/agents")
def list_agents():
    """
    Return all registered agents with their tasks.
    The frontend sidebar auto-populates from this endpoint.
    Add new agents to agents/__init__.py — no HTML changes needed.
    """
    return [
        {
            "id":          agent["id"],
            "name":        agent["name"],
            "icon":        agent["icon"],
            "description": agent["description"],
            "tasks":       agent["tasks"],
        }
        for agent in AGENT_REGISTRY.values()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# LLM AGENT STEP  (used by the HTML UI's ▶ Agent Step button)
# ══════════════════════════════════════════════════════════════════════════════

class AgentStepRequest(BaseModel):
    agent_id: str = Field("cluster_triage", description="Which agent (environment) to step")
    task: Optional[str] = Field(None, description="Task hint (set via /reset)")


def _extract_action(response_text: str) -> ClusterAction:
    """Parse LLM output into a ClusterAction, stripping reasoning blocks."""
    text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "action_type" in data:
                return ClusterAction(**data)
        except Exception:
            pass
    return ClusterAction(action_type="noop", target_id="none")


@fastapi_app.post("/agent/step")
def agent_step(body: AgentStepRequest):
    """
    Run ONE LLM-powered step on the active environment.
    Supports both single-agent (cluster_triage) and multi-agent (split_brain) modes.
    For split_brain, the system prompt changes based on the current_actor.
    """
    if not llm_client:
        raise HTTPException(status_code=503, detail="HF_TOKEN not configured.")

    env = get_or_create_env(body.agent_id)
    obs = env.state()

    # Check episode completion — use health_score or global_health based on agent
    health = getattr(obs, 'global_health', None) or getattr(obs, 'health_score', 0)
    if health >= 1.0 or env.step_count >= env.max_steps:
        raise HTTPException(status_code=400, detail="Episode complete. Call /reset to start a new one.")

    # Get prompts — multi-agent envs provide their own prompts
    if hasattr(env, 'get_llm_prompts'):
        system_prompt, user_prompt = env.get_llm_prompts()
    else:
        system_prompt = (
            "You are an automated DevOps system. You cannot speak. "
            "You can only output raw JSON commands. No explanations, no extra text."
        )
        user_prompt = f"""You are an SRE agent triaging a distributed cluster failure.

CURRENT CLUSTER STATE:
{obs.model_dump_json(indent=2)}

RULES:
1. Kill ALL hanging jobs before clearing any storage.
2. Never restart a node whose disk_usage is above 50%. Clear its storage first.
3. For nightmare: kill ALL 3 hydra jobs before clearing ANY storage.

Respond with EXACTLY ONE JSON object. No other text.
Valid action_type values: "kill_job", "restart_node", "clear_temp_storage", "noop"

EXAMPLE:
{{"action_type": "kill_job", "target_id": "job_rogue_99"}}
"""

    # Auto-select model: use fine-tuned LoRA model for split_brain if available
    active_model = MODEL_NAME
    if body.agent_id == "split_brain" and SPLIT_BRAIN_MODEL:
        active_model = SPLIT_BRAIN_MODEL
        print(f"[INFO] Using fine-tuned model: {active_model}")

    try:
        completion = llm_client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw_text = completion.choices[0].message.content or ""
        print(f"[DEBUG LLM ({active_model})] {raw_text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {str(e)}")

    # Parse action based on agent type
    if body.agent_id == "split_brain":
        action = env._parse_action(raw_text)
    else:
        action = _extract_action(raw_text)

    result = env.step(action)
    new_obs = result.observation
    msg     = result.info.get("message", "")

    response = {
        "step":        env.step_count,
        "agent_id":    body.agent_id,
        "action":      action.model_dump(),
        "reward":      result.reward,
        "done":        result.done,
        "message":     msg,
        "observation": new_obs.model_dump(),
    }

    # Add multi-agent metadata for split_brain
    if body.agent_id == "split_brain":
        response["current_actor"] = result.info.get("current_actor", "orchestrator")
        response["delegation_log"] = result.info.get("delegation_log", [])

    return response


# ══════════════════════════════════════════════════════════════════════════════
# STATIC FILES — serves static/index.html at /
# Mount AFTER all API routes so API routes take priority.
# ══════════════════════════════════════════════════════════════════════════════
fastapi_app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="info")