---
title: OpenEnv Split-Brain Collapse
emoji: 🧠
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
license: mit
---

# 🧠 OpenEnv: Split-Brain Collapse (Three-Datacenter Crisis)

## 📖 Overview & Motivation
Managing a distributed database cluster spanning multiple datacenters is one of the hardest challenges in Site Reliability Engineering. When network partitions occur between datacenters, competing nodes may each believe they are the authoritative primary — a **split-brain** state that leads to data divergence, replication storms, and cascading deadlocks that require precise, ordered multi-step intervention.

This OpenEnv simulates a three-datacenter enterprise infrastructure under a cascading split-brain crisis. It is a genuine multi-agent RL scenario. An AI team consisting of an **orchestrator**, a **netops** specialist, and a **dba** specialist must collaborate to restore connectivity, elect a single primary, reconcile diverged write logs, and recover the cluster before data loss becomes permanent.

---

## ⚙️ Architecture: How It Works
This environment is built as a complete, containerized Reinforcement Learning (RL) ecosystem:

1. **The Simulation Engine (`agents/split_brain/environment.py`):** An OpenEnv-compliant multi-agent state machine tracking datacenter health, network link status, replication lag, transaction deadlocks, and enforcing strict logic gates (e.g., you cannot promote a new primary before routing is verified; resolving a deadlock before re-routing re-triggers it).
2. **The Backend API (`FastAPI`):** Exposes standard programmatic RL endpoints (`/reset`, `/step`, `/state`, `/health`, `/tasks`) allowing external scripts and evaluators to interact with the environment headlessly.
3. **The Web Dashboard (`static/index.html`):** A dark-mode, multi-panel "SRE Command Center" that auto-discovers agents from `GET /agents` and lets humans interactively step through the simulation watching the LLM make decisions in real-time.
4. **The Agent (`inference.py` / `OpenAI Client`):** Uses the `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` model via the Hugging Face Serverless API. The environment injects context-aware multi-agent prompts (different system prompts for orchestrator, netops, dba) so the LLM knows which role it is playing each step.

---

## 🧠 Observation & Action Spaces
The environment strictly adheres to the OpenEnv Pydantic specifications.

### Observation Space
The agent receives a `SplitBrainObservation` detailing the full three-datacenter state:
* **`global_health`**: (Float 0.0 – 1.0) Continuous cluster stability metric across all three DCs.
* **`current_actor`**: Which sub-agent is active this step (`orchestrator`, `netops`, or `dba`).
* **`dc1_dc2_connected` / `dc2_dc3_connected`**: Network link status between datacenters.
* **`network_status`**: Overall network state (`partitioned`, `degraded`, `healthy`).
* **`replication_lag_ms`**: Database replication lag in milliseconds.
* **`primary_db`**: Which DC currently holds the authoritative primary database.
* **`datacenters`**: List of 3 `DatacenterStatus` objects — status, node count, load %.
* **`recent_alerts`**: System log messages and critical alerts.

### Action Space
The active sub-agent issues a `SplitBrainAction` JSON object:
* **`action_type`**: Command to execute — `run_diagnostic`, `update_route`, `verify_routing`, `throttle_bandwidth`, `restore_replica`, `promote_primary`, `force_sync`, `resolve_deadlock`, `failover_region`, `delegate`, `noop`.
* **`target_id`**: Resource to target (e.g. `dc2_router--dc3_router`, `replica_dc2`).
* **`target_agent`**: Sub-agent to delegate to (`netops` or `dba`) — only for `delegate` actions.
* **`instruction_payload`**: Nested action passed to the delegated sub-agent.

---

## 🎯 Task Descriptions & Difficulty
The environment features a 5-tier difficulty scale with continuous shaped rewards penalising wrong ordering and rewarding correct multi-step sequences.

* **🟢 BASIC (The Partition):** A single DC1–DC2 link has failed.
  * *Expected Sequence:* `run_diagnostic` → `update_route` → `verify_routing`.
* **🟡 STORM (Replication Storm):** A thundering-herd burst has saturated all inter-DC links.
  * *Expected Sequence:* `throttle_bandwidth` → `restore_replica` × N → `force_sync`.
* **🔴 SPLIT (The Split-Brain):** Both inter-DC links are down; two competing primaries have diverged.
  * *Expected Sequence:* Restore both links, isolate one primary, `promote_primary`, reconcile all replicas.
* **🟣 DEADLOCK (Cascading Deadlock):** A distributed transaction deadlock has frozen all write paths.
  * *Expected Sequence:* `update_route` → `verify_routing` → `resolve_deadlock` → `force_sync` (strict order).
* **☠️ WIPEOUT (Regional Wipeout):** DC3 offline, DC1/DC2 in split-brain. Full regional failover required.
  * *Expected Sequence:* `failover_region` → `promote_primary` → restore all replicas → `verify_routing`.

---

## 💻 Setup & Usage Instructions

You can run this project in three different ways depending on your needs.

### Method 1: Hugging Face Spaces (Interactive Web UI)
The easiest way to evaluate the environment is via the public Hugging Face Space. The UI acts as a step-by-step RL debugger.

1. **Access the Dashboard:** Open the Hugging Face Space URL.
2. **Select Threat Level:** Use the sidebar to choose a task scenario.
3. **Initialize the Environment:** Click the **🔄 Reset** button to boot the simulation.
4. **Deploy the Agent:** Click the **▶ AGENT STEP** button. The LLM evaluates the current state and makes exactly *one* move, routing to the correct sub-agent (orchestrator / netops / dba).
5. **Observe the Triage:** Watch the delegation log, step reward, and datacenter health panel update.
6. **Iterate:** Continue clicking **▶ AGENT STEP** until the terminal declares `CLUSTER RESTORED`.

### Method 2: Local Docker (Production Simulation)
Run the exact containerized environment that Hugging Face uses, locally on your machine.

1. Create a `.env` file in the root directory:
   ```env
   HF_TOKEN="your_huggingface_token"
   MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
   API_BASE_URL="https://router.huggingface.co/v1"
   SPLIT_BRAIN_MODEL=""  # optional: path to fine-tuned LoRA model
   ```
2. Build the Docker image:
   ```bash
   docker build -t split-brain-env .
   ```
3. Run the container:
   ```bash
   docker run -p 7860:7860 --env-file .env split-brain-env
   ```
4. Open your browser at: **`http://localhost:7860`**

### Method 3: Local Python (For Developers)
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure your `.env` file is configured (same as Docker step 1).
3. **Run the Interactive Web Dashboard & API:**
   ```bash
   python app.py
   ```
   *(Access the UI via `http://127.0.0.1:7860`).*
4. **Run the Automated Terminal Baseline:**
   ```bash
   python inference.py
   ```

---

## 🧬 GRPO Fine-Tuning with Unsloth (LoRA Adapter)

### The Problem
When running the **Cascading Deadlock** task (Task 4) with smaller models like Llama 3.1 8B, the agent gets stuck in an infinite `run_diagnostic` loop — it keeps repeating the same diagnostic action instead of progressing to `update_route`, `verify_routing`, and `resolve_deadlock`. Larger models (70B+) handle this correctly but are expensive to run.

### The Solution: GRPO Reinforcement Learning
We used **Group Relative Policy Optimization (GRPO)** via [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) to fine-tune `Llama-3.2-3B-Instruct` directly against the OpenEnv reward function. The training:

1. Feeds the model the exact stuck scenario (post-diagnostic delegation to netops)
2. Generates multiple candidate actions via sampling
3. Steps each action through the live `SplitBrainEnv.step()` function
4. Rewards correct actions (e.g., `update_route`) and **penalises** repeated diagnostics
5. The model learns to break out of loops and follow multi-step repair sequences

### Training Script
The training script is `train_unsloth_colab.py` — designed to run on **Google Colab** with a free T4 GPU:

```bash
# In Google Colab:
!git clone https://github.com/Sayantan181222/openenv-cluster-triage-updated.git
%cd openenv-cluster-triage-updated
!pip install -r requirements.txt
!pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git" trl datasets
!python train_unsloth_colab.py
```

The trained adapter is saved to `openenv-split-brain-lora/` (~93MB LoRA weights).

### Using the LoRA Adapter
Run the fine-tuned model against the Split-Brain environment:

```bash
python inference_lora.py
```

This loads `Llama-3.2-3B-Instruct` + the LoRA adapter and runs a full episode on the `cascading_deadlock` task, printing a before/after comparison.

### Results & Improvement

| Metric | Base 8B (No LoRA) | 3B + GRPO LoRA |
|---|---|---|
| Diagnostic Loops | 10+ (infinite) | ≤1 |
| Reached `update_route` | ❌ Never | ✅ Yes |
| Episode Completion | ❌ Timed out | ✅ Completes |
| Model Size | 8B parameters | 3B parameters |

The fine-tuned **3B model outperforms the base 8B model** by learning environment-specific action sequences through reinforcement learning.