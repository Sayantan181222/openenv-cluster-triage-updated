"""
app.py — OpenEnv Split-Brain Collapse Environment
==================================================
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
from dotenv import load_dotenv
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from openai import OpenAI

from agents import AGENT_REGISTRY
from agents.split_brain.models import SplitBrainAction

load_dotenv()

# ── LLM Config ──────────────────────────────────────────────────────────────
API_BASE_URL      = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME        = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
SPLIT_BRAIN_MODEL = os.getenv("SPLIT_BRAIN_MODEL", "")  # GRPO fine-tuned LoRA model
API_KEY           = os.getenv("HF_TOKEN", "").strip().strip('"').strip("'")

if API_KEY:
    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=300.0,
    )
    print(f"[INFO] LLM ready. Token: {API_KEY[:6]}... Model: {MODEL_NAME}")
else:
    llm_client = None
    print("[WARN] HF_TOKEN not set — LLM calls will fail.")

# ── Active Environment State ─────────────────────────────────────────────────
active_agent_id = None
active_env      = None


def get_or_create_env(agent_id: str):
    """Lazily create/switch the active environment when agent changes."""
    global active_agent_id, active_env
    agent = AGENT_REGISTRY.get(agent_id)
    if not agent:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent '{agent_id}'. Available: {list(AGENT_REGISTRY.keys())}",
        )
    if active_agent_id != agent_id:
        active_agent_id = agent_id
        active_env = agent["env_class"]()
    return active_env


# ── FastAPI App ───────────────────────────────────────────────────────────────
fastapi_app = FastAPI(
    title="OpenEnv: Split-Brain Collapse",
    description=(
        "An OpenEnv-compliant multi-agent RL environment simulating a "
        "three-datacenter split-brain network partition crisis. "
        "AI agents acting as orchestrator, netops, and dba must collaboratively "
        "resolve network partitions, replication storms, and cascading deadlocks."
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
        "status":       "ok",
        "env":          "split-brain",
        "version":      "1.0.0",
        "llm_configured": bool(API_KEY),
        "model":        MODEL_NAME,
    }


@fastapi_app.post("/reset")
async def reset(request: Request):
    """
    Reset the environment for a given task.
    Body: {"task": "partition_basic", "agent_id": "split_brain"}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    agent_id = body.get("agent_id", "split_brain")
    task     = body.get("task", "partition_basic")
    env      = get_or_create_env(agent_id)
    agent    = AGENT_REGISTRY[agent_id]
    valid    = [t["id"] for t in agent["tasks"]]
    if task not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Valid: {valid}",
        )
    obs = env.reset(task=task)
    return JSONResponse(content=obs.model_dump())


@fastapi_app.post("/step")
async def step(request: Request):
    """Execute one raw action in the environment."""
    if active_env is None:
        raise HTTPException(status_code=400, detail="No environment loaded. Call /reset first.")
    body   = await request.json()
    action = SplitBrainAction(**body)
    result = active_env.step(action)
    return JSONResponse(content=result.model_dump())


@fastapi_app.get("/state")
def state():
    """Return the current observation without advancing the episode."""
    if active_env is None:
        raise HTTPException(status_code=400, detail="No environment loaded. Call /reset first.")
    return JSONResponse(content=active_env.state().model_dump())


@fastapi_app.get("/tasks")
def list_tasks(agent_id: str = "split_brain"):
    """List all available tasks with difficulty metadata for a specific agent."""
    if agent_id not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return {"tasks": AGENT_REGISTRY[agent_id]["tasks"]}


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
    agent_id: str          = Field("split_brain", description="Which agent (environment) to step")
    task:     Optional[str] = Field(None, description="Task hint (set via /reset)")


@fastapi_app.post("/agent/step")
def agent_step(body: AgentStepRequest):
    """
    Run ONE LLM-powered step on the active Split-Brain environment.
    The environment provides its own multi-agent prompts via get_llm_prompts().
    """
    if not llm_client:
        raise HTTPException(status_code=503, detail="HF_TOKEN not configured.")

    env = get_or_create_env(body.agent_id)
    obs = env.state()

    # Check episode completion
    health = getattr(obs, "global_health", None) or getattr(obs, "health_score", 0)
    if health >= 1.0 or env.step_count >= env.max_steps:
        raise HTTPException(status_code=400, detail="Episode complete. Call /reset to start a new one.")

    # The split_brain env always exposes get_llm_prompts() for multi-agent routing
    system_prompt, user_prompt = env.get_llm_prompts()

    # Auto-select model: prefer fine-tuned LoRA model for cascading_deadlock
    active_model = MODEL_NAME
    if SPLIT_BRAIN_MODEL and getattr(env, "current_task", "") == "cascading_deadlock":
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

    action = env._parse_action(raw_text)
    result = env.step(action)
    new_obs = result.observation
    msg     = result.info.get("message", "")

    return {
        "step":           env.step_count,
        "agent_id":       body.agent_id,
        "action":         action.model_dump(),
        "reward":         result.reward,
        "done":           result.done,
        "message":        msg,
        "observation":    new_obs.model_dump(),
        "current_actor":  result.info.get("current_actor", "orchestrator"),
        "delegation_log": result.info.get("delegation_log", []),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STATIC FILES — serves static/index.html at /
# Mount AFTER all API routes so API routes take priority.
# ══════════════════════════════════════════════════════════════════════════════
fastapi_app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="info")