---
title: OpenEnv Cluster Triage
emoji: 🚀
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
license: mit
---

# 🚀 OpenEnv: Distributed Data Cluster Triage (Enterprise 4-Node)

## 📖 Overview & Motivation
Managing a distributed data processing cluster (e.g., Hadoop/HDFS) is a complex, high-stakes DevOps challenge. When compute resources are finite, runaway jobs and hardware constraints can cause cascading failures that require immediate, multi-step human intervention.

This OpenEnv simulates an enterprise-level 4-node cluster under stress. It is a genuine infrastructure triage scenario. An AI agent acting as an automated Site Reliability Engineer (SRE) must observe the cluster's telemetry dashboard, identify root causes of failures, and issue precise terminal commands to prevent total system collapse.

---

## ⚙️ Architecture: How It Works
This environment is built as a complete, containerized Reinforcement Learning (RL) ecosystem:

1. **The Simulation Engine (`environment.py`):** An OpenEnv-compliant state machine that tracks node health, RAM/Disk usage, active jobs, and enforces strict logic gates (e.g., you cannot restart a node if its disk is full; you must kill Hydra malware before clearing storage).
2. **The Backend API (`FastAPI`):** Exposes standard programmatic RL endpoints (`/reset`, `/step`, `/state`, `/health`) allowing external scripts and evaluators to interact with the environment headlessly.
3. **The Web Dashboard (`Gradio`):** Mounted directly on top of the FastAPI server, this provides a dark-mode, 3-column "SRE Command Center" for humans to interactively step through the simulation and watch the LLM make decisions in real-time.
4. **The Agent (`inference.py` / `OpenAI Client`):** Uses the `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` model via the Hugging Face Serverless API to analyze the observation JSON and output strict, precise JSON action commands.

---

## 🧠 Observation & Action Spaces
The environment strictly adheres to the OpenEnv Pydantic specifications.

### Observation Space
The agent receives a `ClusterObservation` detailing the exact state of the infrastructure:
* **`health_score`**: (Float 0.0 - 1.0) The continuous metric of cluster stability.
* **`nodes`**: A list of 4 `NodeStatus` objects (worker nodes), detailing CPU, RAM, disk usage, and current status (`healthy`, `high_memory`, `disk_full`, `offline`).
* **`active_jobs`**: A list of `JobStatus` objects detailing allocated memory and execution states.
* **`recent_alerts`**: System logs and critical alerts providing context to the failures.

### Action Space
The agent interacts with the environment by outputting a strictly typed `ClusterAction` JSON object:
* **`action_type`**: The command to execute (`kill_job`, `restart_node`, `clear_temp_storage`, `noop`).
* **`target_id`**: The specific `job_id` or `node_id` to apply the action to.

---

## 🎯 Task Descriptions & Difficulty
The environment features a 5-tier difficulty scale. It provides a meaningful, continuous reward signal, penalizing destructive actions and rewarding partial progress.

* **🟢 EASY (The Stuck Job):** A rogue MapReduce job is stuck in an infinite loop. 
  * *Expected Action:* `kill_job` -> `job_rogue_99`. 
* **🟡 MEDIUM (The Full Disk):** A worker node's disk has hit 99.9% capacity.
  * *Expected Sequence:* `clear_temp_storage` -> `restart_node`. 
* **🟠 HARD (The Cascade Failure):** A rogue job caused a replication storm, crashing Nodes 1 and 2. Failover traffic is overloading Nodes 3 and 4.
  * *Expected Sequence:* Kill the root job, clear storage on dead nodes, then safely reboot them. (5 steps)
* **🔴 VERY HARD (Multi-Vector Attack):** A disk-filling log spammer and a RAM-hogging crypto miner hit the cluster simultaneously.
  * *Expected Sequence:* Isolate and kill both malware jobs before clearing storage and rebooting. (6 steps)
* **☠️ NIGHTMARE (The Hydra Protocol):** Total cluster collapse. Three self-replicating malware jobs are writing garbage data to all 4 nodes. 
  * *Expected Sequence:* The agent must kill ALL THREE Hydra jobs before attempting to clear any storage, otherwise, the surviving malware instantly cross-infects and refills the disks. (11 steps)

---

## 💻 Setup & Usage Instructions

You can run this project in three different ways depending on your needs.

### Method 1: Hugging Face Spaces (Interactive Web UI)
The easiest way to evaluate the environment is via the public Hugging Face Space. The UI acts as a step-by-step RL debugger.

1. **Access the Dashboard:** Open the Hugging Face Space URL.
2. **Select Threat Level:** Use the radio buttons on the left to choose a scenario. The *Active Incident Briefing* will update with your mission constraints.
3. **Initialize the Environment:** Click the **🔄 Reset** button. This boots up the simulation and populates the *Live Node Status* telemetry grid.
4. **Deploy the Agent:** Click the purple **▶ AGENT STEP** button. The LLM will evaluate the state and make exactly *one* move. 
5. **Observe the Triage:** Look at the right-side panel. You will see the agent's command in the Terminal, the resulting Step Reward (+/-), and the dynamic Reward History bar graph update.
6. **Iterate:** Continue clicking **▶ AGENT STEP** until the terminal declares `SYSTEM NOMINAL`.

### Method 2: Local Docker (Production Simulation)
Run the exact containerized environment that Hugging Face uses, locally on your machine.

1. Create a `.env` file in the root directory and add your API credentials:
   ```env
   HF_TOKEN="your_huggingface_token"
   MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
   API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
   ```
2. Build the Docker image:
   ```bash
   docker build -t cluster-triage-env .
   ```
3. Run the container (Ensure you map Port 7860 to access the UI):
   ```bash
   docker run -p 7860:7860 --env-file .env cluster-triage-env
   ```
4. Open your web browser and navigate to: **`http://localhost:7860`**

### Method 3: Local Python (For Developers)
If you want to modify the source code, debug, or run the headless terminal script without building a container.

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure your `.env` file is configured in the root directory (same as Docker step 1).
3. **To run the Interactive Web Dashboard & API:**
   ```bash
   python app.py
   ```
   *(Access the UI via `http://127.0.0.1:7860` in your browser).*
4. **To run the Automated Terminal Baseline:**
   ```bash
   python inference.py
   ```

---

## 🧬 GRPO Fine-Tuning with Unsloth (LoRA Adapter)

### The Problem
When running the **Split-Brain Collapse** agent (Task 4: `cascading_deadlock`) with smaller models like Llama 3.1 8B, the agent gets stuck in an infinite `run_diagnostic` loop — it keeps repeating the same diagnostic action instead of progressing to `update_route`, `verify_routing`, and other repair commands. Larger models (70B+) handle this correctly but are expensive to run.

### The Solution: GRPO Reinforcement Learning
We used **Group Relative Policy Optimization (GRPO)** via [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) to fine-tune `Llama-3.2-3B-Instruct` directly against the OpenEnv reward function. The training:

1. Feeds the model the exact stuck scenario (post-diagnostic delegation to netops)
2. Generates multiple candidate actions via sampling
3. Steps each action through the live `SplitBrainEnv.step()` function
4. Rewards correct actions (e.g., `update_route`) and **penalizes** repeated diagnostics
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