"""
agents/__init__.py — Central Agent Registry
============================================

EXTENSIBILITY:
  To add a new agent, create a new package under agents/:
    agents/my_new_agent/__init__.py   →  define AGENT_META, ENV_CLASS
  Then import and register it in this file.

The frontend sidebar auto-discovers all agents from GET /agents.
"""

from agents.split_brain import AGENT_META as sb_meta, ENV_CLASS as sb_env

# ══════════════════════════════════════════════════════════════════════════════
# AGENT_REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
# Each key maps an agent_id to its metadata + environment class.
#
# To add a new agent:
#   1. Create agents/your_agent/__init__.py with AGENT_META and ENV_CLASS
#   2. Import them here
#   3. Add a new entry to this dict
#
# The HTML sidebar and all API endpoints will discover it automatically.
# ══════════════════════════════════════════════════════════════════════════════
AGENT_REGISTRY = {
    "split_brain": {
        **sb_meta,
        "env_class": sb_env,
    },
    # ── Add new agents below ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────
}
