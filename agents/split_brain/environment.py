"""
Split-Brain Collapse Environment
=================================
Multi-agent DC partition recovery simulation with network, storage, and database layers.
3 agents (Orchestrator, NetOps, DataOps) must cooperate via delegation to restore health.
"""
import copy, json, re, random
from typing import Optional, Tuple, List, Dict, Any

try:
    import networkx as nx
except ImportError:
    nx = None

from agents.split_brain.models import (
    NetworkNode, NetworkEdge, HDFSCluster, DatabaseState, LedgerEntry,
    SplitBrainObservation, SplitBrainAction, SplitBrainStepResult,
)

# Valid actions per actor
ACTOR_ACTIONS = {
    "orchestrator": {"delegate", "assess_situation", "noop", "run_diagnostic"},
    "netops": {"update_route", "verify_routing", "throttle_bandwidth", "delegate", "noop", "run_diagnostic"},
    "dataops": {"tune_hdfs", "stop_replication", "force_stepdown", "reconcile_ledger", "clear_cache", "delegate", "noop", "run_diagnostic"},
}

# Role-specific LLM prompts
SYSTEM_PROMPTS = {
    "orchestrator": (
        "You are the Incident Commander (Orchestrator). You analyze the situation and delegate "
        "tasks to specialist agents. You CANNOT fix things directly — you must delegate.\n"
        "Your only actions: delegate, assess_situation, noop.\n"
        "When delegating, specify target_agent and instruction_payload."
    ),
    "netops": (
        "You are the Network Operations specialist. You fix network partitions using routing "
        "and bandwidth management. You understand Dijkstra's algorithm and failover routing.\n"
        "Your actions: update_route, verify_routing, throttle_bandwidth, delegate (back to orchestrator), noop, run_diagnostic.\n"
        "CRITICAL: Once verify_routing succeeds, your network job is DONE. Do NOT try to fix severed links. "
        "Delegate back to orchestrator IMMEDIATELY with a status report.\n"
        "REGIONAL WIPEOUT EXCEPTION: The dc1↔dc2 primary link is permanently severed — verify_routing WILL FAIL. "
        "Your job is ONLY to: (1) throttle dc2_router--dc3_router to 10%, (2) establish the oob_tunnel. "
        "Once oob_tunnel_active is True in the state, your work is DONE. Delegate back to orchestrator immediately."
    ),
    "dataops": (
        "You are the Data Operations specialist. You manage HDFS storage and NewSQL databases. "
        "You handle replication storms, split-brain resolution, and ledger reconciliation.\n"
        "Your actions: tune_hdfs, stop_replication, force_stepdown, reconcile_ledger, delegate (back to orchestrator), noop.\n"
        "CRITICAL: Do NOT reconcile_ledger unless routing has been verified by NetOps first."
    ),
}


class SplitBrainEnv:
    def __init__(self):
        self.current_task = None
        self.state_data: Optional[SplitBrainObservation] = None
        self.step_count = 0
        self.max_steps = 35
        # Milestones
        self.bypass_established = False
        self.routing_verified = False
        self.storm_killed = False
        self.stepdown_done = False
        self.ledger_reconciled = False
        # Delayed penalty
        self.corruption_timer = -1
        self.corruption_triggered = False
        # Logs
        self.delegation_log: List[Dict] = []

    def _reset_trackers(self):
        self.step_count = 0
        self.bypass_established = False
        self.routing_verified = False
        self.storm_killed = False
        self.stepdown_done = False
        self.ledger_reconciled = False
        self.corruption_timer = -1
        self.corruption_triggered = False
        self.delegation_log = []

    # ── TOPOLOGY BUILDERS ────────────────────────────────────────────────────

    def _build_nodes(self) -> List[NetworkNode]:
        nodes = []
        dcs = ("dc1", "dc2", "dc3") if self.current_task == "regional_wipeout" else ("dc1", "dc2")
        for dc in dcs:
            nodes.append(NetworkNode(
                node_id=f"{dc}_router", datacenter=dc, node_type="router",
                status="online", cpu_usage=35.0, bandwidth_used=200.0, bandwidth_capacity=1000.0))
            nodes.append(NetworkNode(
                node_id=f"{dc}_switch", datacenter=dc, node_type="switch",
                status="online", cpu_usage=20.0, bandwidth_used=150.0, bandwidth_capacity=1000.0))
            nodes.append(NetworkNode(
                node_id=f"{dc}_server_a", datacenter=dc, node_type="server",
                status="online", cpu_usage=55.0, bandwidth_used=300.0, bandwidth_capacity=1000.0))
            nodes.append(NetworkNode(
                node_id=f"{dc}_server_b", datacenter=dc, node_type="server",
                status="online", cpu_usage=50.0, bandwidth_used=250.0, bandwidth_capacity=1000.0))
        return nodes

    def _build_edges(self, partitioned=True) -> List[NetworkEdge]:
        internal = [
            ("dc1_router", "dc1_switch", 1.0), ("dc1_switch", "dc1_server_a", 0.5),
            ("dc1_switch", "dc1_server_b", 0.5), ("dc2_router", "dc2_switch", 1.0),
            ("dc2_switch", "dc2_server_a", 0.5), ("dc2_switch", "dc2_server_b", 0.5),
        ]
        if self.current_task == "regional_wipeout":
            internal.extend([
                ("dc3_router", "dc3_switch", 1.0), ("dc3_switch", "dc3_server_a", 0.5),
                ("dc3_switch", "dc3_server_b", 0.5)
            ])
            
        edges = []
        for src, tgt, lat in internal:
            edges.append(NetworkEdge(
                edge_id=f"{src}--{tgt}", source=src, target=tgt,
                status="healthy", latency_ms=lat, bandwidth_used=150.0, bandwidth_capacity=1000.0))
        
        if self.current_task == "regional_wipeout":
            edges.append(NetworkEdge(
                edge_id="dc1_router--dc3_router", source="dc1_router", target="dc3_router",
                status="healthy", latency_ms=10.0, bandwidth_used=0.0, bandwidth_capacity=1000.0))
            edges.append(NetworkEdge(
                edge_id="dc2_router--dc3_router", source="dc2_router", target="dc3_router",
                status="congested", latency_ms=10.0, bandwidth_used=1000.0, bandwidth_capacity=1000.0))
            edges.append(NetworkEdge(
                edge_id="dc1_router--dc2_router", source="dc1_router", target="dc2_router",
                status="severed", latency_ms=5.0, bandwidth_used=0.0, bandwidth_capacity=10000.0))
        else:
            edges.append(NetworkEdge(
                edge_id="dc1_router--dc2_router", source="dc1_router", target="dc2_router",
                status="severed" if partitioned else "healthy",
                latency_ms=5.0, bandwidth_used=0.0, bandwidth_capacity=10000.0))
            edges.append(NetworkEdge(
                edge_id="dc1_router--dc2_switch", source="dc1_router", target="dc2_switch",
                status="congested" if partitioned else "healthy",
                latency_ms=25.0, bandwidth_used=950.0, bandwidth_capacity=1000.0))
            edges.append(NetworkEdge(
                edge_id="dc1_switch--dc2_router", source="dc1_switch", target="dc2_router",
                status="congested" if partitioned else "healthy",
                latency_ms=30.0, bandwidth_used=920.0, bandwidth_capacity=1000.0))
        return edges

    def _build_ledger(self, conflicts=0) -> List[LedgerEntry]:
        entries = []
        for i in range(min(conflicts, 8)):
            entries.append(LedgerEntry(
                key=f"txn_{1000+i}", dc1_value=f"val_dc1_{random.randint(100,999)}",
                dc2_value=f"val_dc2_{random.randint(100,999)}", conflict=True,
                timestamp_dc1=1700000000.0 + i, timestamp_dc2=1700000000.0 + i + 0.5))
        return entries

    # ── RESET ────────────────────────────────────────────────────────────────

    def reset(self, task: str = "partition_basic") -> SplitBrainObservation:
        self.current_task = task
        self._reset_trackers()
        nodes = self._build_nodes()
        edges = self._build_edges(partitioned=True)

        if task == "partition_basic":
            self.max_steps = 15
            hdfs = HDFSCluster(io_bandwidth_used_pct=20.0, replication_storm_active=False)
            db = DatabaseState(dc1_role="leader", dc2_role="follower", split_brain_active=False)
            alerts = ["Symptom: High latency on DB-West."]
            health = 0.35
        elif task == "replication_storm":
            self.max_steps = 25
            hdfs = HDFSCluster(io_bandwidth_used_pct=100.0, replication_storm_active=True,
                               under_replicated_blocks=400)
            db = DatabaseState(dc1_role="leader", dc2_role="follower", split_brain_active=False)
            alerts = ["Symptom: High latency on DB-West.", "Symptom: I/O wait times exceeding thresholds."]
            health = 0.15
        elif task == "split_brain":
            self.max_steps = 35
            hdfs = HDFSCluster(io_bandwidth_used_pct=100.0, replication_storm_active=True,
                               under_replicated_blocks=400)
            db = DatabaseState(dc1_role="leader", dc2_role="leader", split_brain_active=True,
                               conflicting_entries=15, ledger_sample=self._build_ledger(15))
            alerts = [
                "Symptom: High latency on DB-West.",
                "Symptom: I/O wait times exceeding thresholds.",
                "Symptom: Checkout service returning stale data."
            ]
            health = 0.05
        elif task == "cascading_deadlock":
            self.max_steps = 35
            hdfs = HDFSCluster(io_bandwidth_used_pct=20.0, replication_storm_active=False)
            db = DatabaseState(dc1_role="leader", dc2_role="follower", split_brain_active=False)
            alerts = [
                "Symptom: API latency increased from 40ms to 900ms.",
                "Symptom: Database slow query log is growing."
            ]
            health = 0.40
        elif task == "regional_wipeout":
            self.max_steps = 50
            hdfs = HDFSCluster(io_bandwidth_used_pct=100.0, replication_storm_active=True,
                               under_replicated_blocks=400)
            db = DatabaseState(dc1_role="leader", dc2_role="leader", split_brain_active=True,
                               conflicting_entries=15, ledger_sample=self._build_ledger(15))
            alerts = [
                "Symptom: Lost all telemetry from DC-Beta.",
                "Symptom: DC-Gamma is reporting extreme storage I/O alerts.",
                "Symptom: Global transaction ledger is drifting."
            ]
            health = 0.05
        else:
            self.max_steps = 15
            hdfs = HDFSCluster(io_bandwidth_used_pct=20.0, replication_storm_active=False)
            db = DatabaseState()
            alerts = ["INFO: Systems nominal."]
            health = 1.0

        self.state_data = SplitBrainObservation(
            global_health=health, current_actor="orchestrator", step=0, max_steps=self.max_steps,
            network_status="partitioned", dc1_dc2_connected=False, routing_verified=False,
            network_nodes=nodes, network_edges=edges, hdfs=hdfs, newsql=db,
            recent_events=alerts, delegation_context=None,
            auth_status="online",
            redis_cache_usage=55.0 if task == "cascading_deadlock" else 0.0
        )
        return self.state()

    def state(self) -> SplitBrainObservation:
        return copy.deepcopy(self.state_data)

    # ── PATHFINDING ──────────────────────────────────────────────────────────

    def _build_nx_graph(self):
        if nx is None:
            return None
        G = nx.Graph()
        for n in self.state_data.network_nodes:
            G.add_node(n.node_id)
        for e in self.state_data.network_edges:
            if e.status in ("healthy", "bypass", "congested"):
                avail = max(0.1, e.bandwidth_capacity - e.bandwidth_used)
                w = e.latency_ms + (100.0 / avail)
                G.add_edge(e.source, e.target, weight=w, edge_id=e.edge_id)
        return G

    def _check_connectivity(self) -> Tuple[bool, Optional[List[str]]]:
        G = self._build_nx_graph()
        if G is None:
            return False, None
        try:
            path = nx.dijkstra_path(G, "dc1_router", "dc2_router", weight="weight")
            return True, path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False, None

    # ── ACTION PARSING ───────────────────────────────────────────────────────

    def _parse_action(self, action_input) -> SplitBrainAction:
        if isinstance(action_input, SplitBrainAction):
            return action_input
        if isinstance(action_input, dict):
            # Convert nested dict instruction_payloads to strings to avoid type validation errors
            if "instruction_payload" in action_input and isinstance(action_input["instruction_payload"], dict):
                action_input["instruction_payload"] = json.dumps(action_input["instruction_payload"])
            try:
                return SplitBrainAction(**action_input)
            except Exception as e:
                print(f"[DEBUG PARSE ERROR dict] {e}")
                return SplitBrainAction(action_type="noop")
        if isinstance(action_input, str):
            text = re.sub(r'<think>.*?</think>', '', action_input, flags=re.DOTALL).strip()
            text = text.replace("```json", "").replace("```", "").strip()
            
            # First try direct JSON parse
            try:
                data = json.loads(text)
                if "instruction_payload" in data and isinstance(data["instruction_payload"], dict):
                    data["instruction_payload"] = json.dumps(data["instruction_payload"])
                return SplitBrainAction(**data)
            except Exception as e:
                print(f"[DEBUG PARSE ERROR direct] {e}")
            
            # Fallback regex parsing
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if "action_type" in data:
                        return SplitBrainAction(**data)
                except Exception:
                    pass
        return SplitBrainAction(action_type="noop")

    # ── STEP ─────────────────────────────────────────────────────────────────

    def step(self, action_input) -> SplitBrainStepResult:
        if self.state_data is None:
            self.reset()
        self.step_count += 1
        self.state_data.step = self.step_count
        action = self._parse_action(action_input)
        reward = 0.0
        done = False
        msg = "No effect."
        actor = self.state_data.current_actor

        # Validate actor permissions
        valid_actions = ACTOR_ACTIONS.get(actor, set())
        if action.action_type not in valid_actions:
            reward = -0.10
            msg = f"INVALID: '{action.action_type}' not allowed for {actor}. Valid: {sorted(valid_actions)}"
            self._add_event(msg)
            return self._make_result(reward, done, msg)

        # Dispatch by action type
        if action.action_type == "delegate":
            reward, msg = self._handle_delegate(action)
        elif action.action_type == "assess_situation":
            reward, msg = self._handle_assess()
        elif action.action_type == "update_route":
            reward, msg = self._handle_update_route(action)
        elif action.action_type == "verify_routing":
            reward, msg = self._handle_verify_routing()
        elif action.action_type == "throttle_bandwidth":
            reward, msg = self._handle_throttle(action)
        elif action.action_type == "stop_replication":
            reward, msg = self._handle_stop_replication()
        elif action.action_type == "tune_hdfs":
            reward, msg = self._handle_tune_hdfs(action)
        elif action.action_type == "force_stepdown":
            reward, msg = self._handle_force_stepdown(action)
        elif action.action_type == "reconcile_ledger":
            reward, msg = self._handle_reconcile()
        elif action.action_type == "run_diagnostic":
            reward, msg = self._handle_run_diagnostic()
        elif action.action_type == "clear_cache":
            reward, msg = self._handle_clear_cache()
        elif action.action_type == "noop":
            reward = -0.05
            msg = "WARN: No action taken."

        # Tick delayed corruption timer
        corruption_penalty = self._tick_corruption()
        reward += corruption_penalty

        # Task 4: Cascading Deadlock logic
        # Redis only climbs while the root cause (network) is unfixed.
        # Once routing is verified, the app retries stop and Redis stabilizes.
        if self.current_task == "cascading_deadlock":
            if not self.routing_verified and self.state_data.auth_status == "online":
                self.state_data.redis_cache_usage = min(100.0, self.state_data.redis_cache_usage + 20.0)
            
            if self.state_data.redis_cache_usage >= 100.0 and self.state_data.auth_status == "online":
                self.state_data.auth_status = "offline"
                interrupt_msg = "CRITICAL: Redis OOM. Auth offline. User session drop rate spiked by 800%."
                self.state_data.recent_events.append(f"[Step {self.step_count}] {interrupt_msg}")
            
            if self.state_data.auth_status == "offline":
                reward -= 0.05  # Bleed penalty
        
        # Time penalty for all tasks to encourage efficiency
        reward -= 0.02

        # Update health
        self.state_data.global_health = self._calc_health()

        # Check win
        if self.state_data.global_health >= 1.0:
            done = True
            msg += " 🎉 GLOBAL HEALTH RESTORED TO 1.0 — INCIDENT RESOLVED."

        # Check max steps
        if not done and self.step_count >= self.max_steps:
            done = True
            msg += f" TIMEOUT: Max steps ({self.max_steps}) reached."

        self._add_event(msg)
        return self._make_result(reward, done, msg)

    def _make_result(self, reward, done, msg) -> SplitBrainStepResult:
        return SplitBrainStepResult(
            observation=self.state(), reward=reward, done=done,
            info={"message": msg, "step": self.step_count,
                  "current_actor": self.state_data.current_actor,
                  "delegation_log": self.delegation_log[-5:]})

    def _add_event(self, msg: str):
        self.state_data.recent_events.append(f"[Step {self.step_count}] {msg}")
        if len(self.state_data.recent_events) > 20:
            self.state_data.recent_events = self.state_data.recent_events[-20:]

    # ── ACTION HANDLERS ──────────────────────────────────────────────────────

    def _handle_delegate(self, action: SplitBrainAction) -> Tuple[float, str]:
        target = action.target_agent
        payload = action.instruction_payload or "No specific instructions."
        actor = self.state_data.current_actor

        if not target or target == actor:
            return -0.05, f"INVALID: Must delegate to a different agent. Got '{target}'."

        # Check delegation ordering for orchestrator
        reward = 0.05
        if actor == "orchestrator":
            if target == "dataops" and not self.bypass_established and not self.routing_verified:
                reward = -0.10
                msg = f"PENALTY: Delegating to DataOps before network is fixed. Network must come first."
            else:
                msg = f"Orchestrator delegates to {target}: '{payload}'"
        else:
            msg = f"{actor} reports back to {target}: '{payload}'"

        self.delegation_log.append({
            "step": self.step_count, "from": actor, "to": target, "message": payload})
        self.state_data.current_actor = target
        self.state_data.delegation_context = payload
        return reward, msg

    def _handle_assess(self) -> Tuple[float, str]:
        issues = []
        if not self.bypass_established:
            issues.append("Network: PRIMARY LINK SEVERED, backup routes congested")
        elif not self.routing_verified:
            issues.append("Network: Bypass active but routing NOT verified")
        else:
            issues.append("Network: HEALTHY — routing verified")

        if self.state_data.hdfs.replication_storm_active:
            issues.append(f"HDFS: REPLICATION STORM — I/O at {self.state_data.hdfs.io_bandwidth_used_pct:.0f}%")
        else:
            issues.append("HDFS: Stable")

        if self.state_data.newsql.split_brain_active:
            issues.append(f"NewSQL: SPLIT-BRAIN — {self.state_data.newsql.conflicting_entries} conflicts")
        elif self.stepdown_done and not self.ledger_reconciled:
            issues.append("NewSQL: Stepdown done, ledger needs reconciliation")
        else:
            issues.append("NewSQL: Healthy")

        if self.current_task == "cascading_deadlock":
            issues.append(f"Redis Cache: {self.state_data.redis_cache_usage:.0f}% used")
            issues.append(f"Auth: {self.state_data.auth_status}")

        if self.current_task == "regional_wipeout":
            issues.append(f"OOB Tunnel: {'ACTIVE' if self.state_data.oob_tunnel_active else 'NOT ESTABLISHED'}")
            issues.append("Topology: 3-DC (DC-Alpha, DC-Beta, DC-Gamma)")

        summary = "SITUATION ASSESSMENT:\n" + "\n".join(f"  • {i}" for i in issues)
        return 0.05, summary

    def _handle_run_diagnostic(self) -> Tuple[float, str]:
        if random.random() < 0.2:
            return -0.05, "ERROR: Diagnostic tools timeout. Node unresponsive."
        actor = self.state_data.current_actor
        if self.current_task == "cascading_deadlock":
            if actor == "netops":
                return 0.10, "DIAGNOSTIC: dc1_router--dc2_switch is degraded causing packet loss. Use update_route on dc1_router--dc2_switch to establish bypass."
            if actor == "dataops":
                if self.state_data.auth_status == "offline":
                    return 0.10, "DIAGNOSTIC: Redis cache OOM (Out of Memory). Requires clear_cache immediately."
                return 0.05, f"DIAGNOSTIC: DB queries slow, retries piling up in Redis cache. (Usage: {self.state_data.redis_cache_usage:.0f}%)"
            return 0.05, "DIAGNOSTIC: Need specialist to check network and data planes."
        elif self.current_task == "regional_wipeout":
            if actor == "netops":
                if not self.state_data.oob_tunnel_active:
                    dc2_dc3 = self._find_edge("dc2_router--dc3_router")
                    bw = dc2_dc3.bandwidth_used if dc2_dc3 else 1000
                    return 0.10, (f"DIAGNOSTIC: CYCLIC DEPENDENCY DETECTED. "
                                  f"dc1↔dc2 link is SEVERED. dc2→dc3 replication storm saturating dc3 at {bw:.0f}Mbps. "
                                  f"Cannot route through dc3 until storm traffic is reduced. "
                                  f"SOLUTION: First throttle_bandwidth on dc2_router--dc3_router to 10%, "
                                  f"then update_route with target_id 'oob_tunnel' to create management link to dc2, "
                                  f"then delegate to dataops to stop_replication via the tunnel.")
                else:
                    return 0.10, "DIAGNOSTIC: OOB tunnel is active. DC-Beta is reachable via management link. Storm can be stopped."
            if actor == "dataops":
                if self.state_data.oob_tunnel_active:
                    return 0.10, "DIAGNOSTIC: OOB tunnel active. You can now reach DC-Beta. Use stop_replication to kill the storm."
                return 0.05, "DIAGNOSTIC: Cannot reach DC-Beta. No management path exists. Need NetOps to establish OOB tunnel first."
            return 0.05, "DIAGNOSTIC: 3-DC topology. DC-Alpha(dc1) partitioned from DC-Beta(dc2). DC-Beta flooding DC-Gamma(dc3) with replication storm. You MUST now delegate to NetOps to run_diagnostic and resolve this."
        else:
            return self._handle_assess()

    def _handle_clear_cache(self) -> Tuple[float, str]:
        if self.current_task == "cascading_deadlock":
            if self.state_data.auth_status == "offline":
                self.state_data.redis_cache_usage = 0.0
                self.state_data.auth_status = "online"
                return 0.95, "Redis cache flushed. Auth services restored."
            else:
                self.state_data.redis_cache_usage = 0.0
                return 0.05, "Redis cache flushed preemptively."
        return 0.0, "INFO: Cache cleared."

    def _handle_update_route(self, action: SplitBrainAction) -> Tuple[float, str]:
        if random.random() < 0.2:
            return -0.05, "ERROR: SSH Timeout. Node unresponsive."
        eid = action.target_id
        
        if self.current_task == "regional_wipeout" and eid == "oob_tunnel":
            if self.state_data.oob_tunnel_active:
                return 0.0, "INFO: OOB tunnel is already active. Delegate to dataops to stop_replication."
            dc2_dc3 = self._find_edge("dc2_router--dc3_router")
            if dc2_dc3 and dc2_dc3.bandwidth_used > 100:
                return -0.05, "FAIL: DC-Gamma routers dropping packets due to 100% bandwidth saturation."
            self.state_data.oob_tunnel_active = True
            self.state_data.dc1_dc2_connected = True
            self.bypass_established = True
            return 0.15, "OOB TUNNEL ESTABLISHED: Low-bandwidth management link to DC-Beta active."
            
        edge = self._find_edge(eid)
        if not edge:
            return -0.05, f"WARN: Edge '{eid}' not found. Valid: {self._inter_dc_edge_ids()}"

        if self.current_task == "regional_wipeout" and eid == "dc2_router--dc3_router":
            if self.state_data.hdfs.replication_storm_active:
                return -0.05, "FAIL: Cannot establish bypass. DC-Gamma is completely saturated by the replication storm."

        if edge.status == "severed":
            return -0.05, f"FAIL: Edge '{eid}' is physically severed — cannot route through it."

        if edge.status == "congested":
            edge.status = "bypass"
            edge.bandwidth_used = min(edge.bandwidth_used * 0.4, edge.bandwidth_capacity * 0.5)
            edge.latency_ms = max(edge.latency_ms * 0.6, 2.0)
            if not self.bypass_established:
                self.bypass_established = True
                self.state_data.dc1_dc2_connected = True
                self.state_data.network_status = "degraded"
                return 0.15, f"BYPASS ESTABLISHED on '{eid}'. DC1↔DC2 connectivity restored (degraded)."
            return 0.05, f"Additional bypass activated on '{eid}'."

        if edge.status == "bypass":
            return 0.0, f"INFO: '{eid}' is already in bypass mode."

        return 0.0, f"INFO: Edge '{eid}' is {edge.status}, no update needed."

    def _handle_verify_routing(self) -> Tuple[float, str]:
        connected, path = self._check_connectivity()
        if connected and path:
            if not self.routing_verified:
                self.routing_verified = True
                self.state_data.routing_verified = True
                self.state_data.network_status = "healthy" if self.bypass_established else "degraded"
                path_str = " → ".join(path)
                return 0.20, f"ROUTING VERIFIED ✓ Path: {path_str}"
            return 0.0, f"INFO: Routing already verified. Path: {' → '.join(path)}"
        return 0.0, "FAIL: No route from dc1_router to dc2_router. Establish bypass first."

    def _handle_throttle(self, action: SplitBrainAction) -> Tuple[float, str]:
        eid = action.target_id
        edge = self._find_edge(eid)
        if not edge:
            return -0.05, f"WARN: Edge '{eid}' not found."
        limit = (action.parameters or {}).get("limit_pct", 50)

        # Regional Wipeout: The OOB tunnel requires bandwidth_used <= 100.
        # bandwidth_used = capacity * (limit/100), so limit must be <= 10%.
        # Give explicit corrective feedback if the LLM uses a bad value.
        if self.current_task == "regional_wipeout" and eid == "dc2_router--dc3_router":
            if limit > 10:
                edge.bandwidth_used = edge.bandwidth_capacity * (limit / 100.0)
                return -0.05, (
                    f"Bandwidth on '{eid}' throttled to {limit}%, but this is NOT ENOUGH. "
                    f"Current bandwidth: {edge.bandwidth_used:.0f}Mbps. "
                    f"OOB tunnel requires bandwidth BELOW 100Mbps. "
                    f"You MUST set limit_pct to 10 or lower. "
                    f"Use: {{\"action_type\": \"throttle_bandwidth\", \"target_id\": \"{eid}\", \"parameters\": {{\"limit_pct\": 10}}}}"
                )

        edge.bandwidth_used = edge.bandwidth_capacity * (limit / 100.0)
        return 0.05, f"Bandwidth on '{eid}' throttled to {limit}%."

    def _handle_stop_replication(self) -> Tuple[float, str]:
        hdfs = self.state_data.hdfs
        if not hdfs.replication_storm_active:
            if not self.routing_verified:
                return 0.0, "INFO: Replication storm already halted. Routing not yet verified — delegate back to orchestrator so NetOps can verify_routing first."
            elif not self.stepdown_done:
                return 0.0, "INFO: Replication storm already halted. Proceed to force_stepdown on dc2."
            elif not self.ledger_reconciled:
                return 0.0, "INFO: Stepdown already complete. Proceed to reconcile_ledger now."
            else:
                return 0.0, "INFO: All DataOps tasks complete. Delegate back to orchestrator."

        if self.current_task == "regional_wipeout" and not self.state_data.oob_tunnel_active:
            return -0.05, "FAIL: Cannot reach DC-Beta. Primary link severed and no OOB tunnel exists."

        if random.random() < 0.2:
            return -0.05, "ERROR: SSH Timeout. Node unresponsive."

        hdfs.replication_storm_active = False
        hdfs.io_bandwidth_used_pct = 20.0
        self.storm_killed = True
        return 0.15, "REPLICATION STORM HALTED ✓ I/O bandwidth released. Next: force_stepdown on dc2."


    def _handle_tune_hdfs(self, action: SplitBrainAction) -> Tuple[float, str]:
        hdfs = self.state_data.hdfs
        params = action.parameters or {}
        new_rf = params.get("replication_factor", 2)
        old_rf = hdfs.replication_factor
        hdfs.target_replication_factor = new_rf
        hdfs.replication_factor = new_rf
        if hdfs.replication_storm_active:
            hdfs.io_bandwidth_used_pct = max(20.0, hdfs.io_bandwidth_used_pct * 0.5)
            if hdfs.io_bandwidth_used_pct <= 30:
                hdfs.replication_storm_active = False
                self.storm_killed = True
        hdfs.under_replicated_blocks = max(0, hdfs.under_replicated_blocks - 200)
        return 0.10, f"HDFS tuned: replication {old_rf}→{new_rf}. I/O now at {hdfs.io_bandwidth_used_pct:.0f}%."

    def _handle_force_stepdown(self, action: SplitBrainAction) -> Tuple[float, str]:
        if random.random() < 0.2:
            return -0.05, "ERROR: SSH Timeout. Node unresponsive."
        db = self.state_data.newsql
        if not db.split_brain_active:
            if self.stepdown_done and not self.ledger_reconciled:
                return 0.0, "INFO: Stepdown already complete. Proceed to reconcile_ledger now."
            return 0.0, "INFO: No split-brain — database is healthy. Delegate back to orchestrator."
        if not self.state_data.dc1_dc2_connected:
            return -0.15, "FAIL: Cannot force stepdown — DCs not connected. Fix network first."
        target_dc = action.target_id or "dc2"
        if target_dc == "dc2":
            db.dc2_role = "follower"
        else:
            db.dc1_role = "follower"
        db.split_brain_active = False
        self.stepdown_done = True
        return 0.15, f"STEPDOWN COMPLETE ✓ {target_dc} demoted to follower. Ledger reconciliation needed."

    def _handle_reconcile(self) -> Tuple[float, str]:
        db = self.state_data.newsql
        if db.split_brain_active:
            return -0.15, "FAIL: Cannot reconcile — split-brain still active. Force stepdown first."
        if not self.stepdown_done:
            return -0.10, "FAIL: No stepdown performed. Force stepdown first."
        if self.ledger_reconciled:
            return 0.0, "INFO: Ledger already reconciled."

        self.ledger_reconciled = True
        db.conflicting_entries = 0
        db.ledger_sample = []

        if not self.routing_verified:
            # Silent trap — start corruption timer
            self.corruption_timer = 10
            return 0.25, "LEDGER RECONCILED (⚠ routing unverified — sync may be unreliable)."

        return 0.25, "LEDGER RECONCILED ✓ All conflicts resolved via Last-Writer-Wins."

    # ── DELAYED CORRUPTION ───────────────────────────────────────────────────

    def _tick_corruption(self) -> float:
        if self.corruption_timer < 0:
            return 0.0
        self.corruption_timer -= 1
        if self.corruption_timer <= 0 and not self.corruption_triggered:
            self.corruption_triggered = True
            self.ledger_reconciled = False
            db = self.state_data.newsql
            db.conflicting_entries = 20
            db.ledger_sample = self._build_ledger(20)
            self._add_event("💀 DELAYED CORRUPTION: Ledger re-corrupted! Reconciliation failed due to unverified routing.")
            return -0.50
        return 0.0

    # ── HEALTH CALCULATION ───────────────────────────────────────────────────

    def _calc_health(self) -> float:
        components = []
        # Network component
        if self.routing_verified:
            components.append(1.0)
        elif self.bypass_established:
            components.append(0.5)
        else:
            components.append(0.0)

        if self.current_task in ("replication_storm", "split_brain", "regional_wipeout"):
            if self.storm_killed:
                components.append(1.0)
            elif self.state_data.hdfs.io_bandwidth_used_pct < 50:
                components.append(0.5)
            else:
                components.append(0.0)

        if self.current_task in ("split_brain", "regional_wipeout"):
            if self.ledger_reconciled and not self.corruption_triggered:
                components.append(1.0)
            elif self.stepdown_done:
                components.append(0.4)
            else:
                components.append(0.0)

        # Task 4: Redis/Auth must be healthy
        if self.current_task == "cascading_deadlock":
            if self.state_data.auth_status == "online" and self.state_data.redis_cache_usage <= 10:
                components.append(1.0)
            elif self.state_data.auth_status == "online":
                components.append(0.5)
            else:
                components.append(0.0)

        # Task 5: OOB tunnel must be established
        if self.current_task == "regional_wipeout":
            if self.state_data.oob_tunnel_active:
                components.append(1.0)
            else:
                components.append(0.0)

        return sum(components) / len(components) if components else 1.0

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _find_edge(self, edge_id: Optional[str]) -> Optional[NetworkEdge]:
        if not edge_id:
            return None
        for e in self.state_data.network_edges:
            if e.edge_id == edge_id:
                return e
        return None

    def _inter_dc_edge_ids(self) -> List[str]:
        return [e.edge_id for e in self.state_data.network_edges
                if e.source.split("_")[0] != e.target.split("_")[0]]

    # ── TASK-SPECIFIC PROMPT RULES ────────────────────────────────────────────

    def _task_specific_rules(self) -> str:
        if self.current_task == "cascading_deadlock":
            return (
                "\n\nTASK-SPECIFIC CONTEXT (CASCADING DEADLOCK):\n"
                "- This is a 2-phase incident: a network degradation AND a Redis cache time-bomb.\n"
                "- The Redis cache usage is climbing every step. When it hits 100%, Auth goes OFFLINE.\n"
                "- You MUST fix the network first, BUT if auth_status becomes 'offline', you must\n"
                "  IMMEDIATELY delegate to dataops to run clear_cache before doing anything else.\n"
                "- The incident is only resolved when: network is fixed AND auth is online AND redis is cleared.\n"
                "- After clearing the cache, resume fixing the network if not already done."
            )
        elif self.current_task == "regional_wipeout":
            return (
                "\n\nTASK-SPECIFIC CONTEXT (REGIONAL WIPEOUT - 3 DATA CENTERS):\n"
                "- Topology: DC-Alpha(dc1), DC-Beta(dc2), DC-Gamma(dc3).\n"
                "- dc1↔dc2 primary link is SEVERED (partitioned).\n"
                "- dc2 is sending a replication storm to dc3, saturating dc3 bandwidth at 100%.\n"
                "- CYCLIC DEPENDENCY: You cannot route through dc3 (saturated) and you cannot\n"
                "  reach dc2 to stop the storm (link severed). Standard routing WILL FAIL.\n"
                "- Valid inter-DC edges: dc1_router--dc3_router, dc2_router--dc3_router, dc1_router--dc2_router\n"
                "- Special target_id: 'oob_tunnel' (creates a management link, but only works AFTER dc3 bandwidth is throttled below 100)\n"
                "\n"
                "EXACT STEP-BY-STEP SEQUENCE FOR NETOPS:\n"
                "  Step A: throttle_bandwidth on dc2_router--dc3_router with limit_pct=10 (MUST be 10, NOT 50 or 90!)\n"
                "          JSON: {\"action_type\": \"throttle_bandwidth\", \"target_id\": \"dc2_router--dc3_router\", \"parameters\": {\"limit_pct\": 10}}\n"
                "  Step B: update_route with target_id 'oob_tunnel' (only works after Step A)\n"
                "          JSON: {\"action_type\": \"update_route\", \"target_id\": \"oob_tunnel\"}\n"
                "  Step C: verify_routing\n"
                "  Step D: delegate back to orchestrator\n"
                "\n"
                "EXACT STEP-BY-STEP SEQUENCE FOR DATAOPS:\n"
                "  Step E: stop_replication (only works after OOB tunnel is active)\n"
                "  Step F: force_stepdown on dc2\n"
                "  Step G: reconcile_ledger\n"
                "  Step H: delegate back to orchestrator\n"
                "\nCRITICAL INSTRUCTION FOR ALL AGENTS: You MUST strictly obey the DELEGATION CONTEXT. "
                "Do exactly what the delegation context tells you to do, and nothing else. "
                "Do not skip steps. Do not guess future steps."
            )
        return ""

    # ── LLM PROMPTS ─────────────────────────────────────────────────────────

    def get_llm_prompts(self):
        """Return (system_prompt, user_prompt) for the current actor."""
        actor = self.state_data.current_actor
        sys_prompt = SYSTEM_PROMPTS.get(actor, SYSTEM_PROMPTS["orchestrator"])
        obs = self.state_data

        ctx = f"\nDELEGATION CONTEXT: {obs.delegation_context}" if obs.delegation_context else ""
        user_prompt = f"""You are the {actor.upper()} agent handling a datacenter incident.
{ctx}

CURRENT STATE:
{obs.model_dump_json(indent=2)}

RULES:
1. ORCHESTRATOR SEQUENCE OF REPAIRS:
   - First, MUST run_diagnostic to identify the exact issue if it is unknown. (NEVER run this more than once. If it is already in recent_events, proceed to Step 2).
   - Second, delegate to netops to fix the network partition and verify routing.
   - Third, once the network is verified, delegate to dataops to halt the replication storm.
   - Fourth, delegate to dataops to resolve the split-brain (force stepdown) and reconcile the ledger.
   - EMERGENCY: If a critical system like Auth goes offline (auth_status=offline), you must IMMEDIATELY preempt current operations, delegate to dataops to clear_cache, and then resume normal operations.
2. Network MUST be fixed before database operations.
3. Kill replication storms before reconciling data.
4. NEVER reconcile_ledger unless routing is verified.
5. When your work is done, delegate back to orchestrator.
6. DO NOT output noop if there is an active incident. Always take action or delegate.
7. If a command fails with SSH Timeout or Node unresponsive, RETRY the same command.
{self._task_specific_rules()}

Respond with EXACTLY ONE JSON object. No other text.
Valid actions for {actor}: {sorted(ACTOR_ACTIONS[actor])}

EXAMPLES:
{{"action_type": "run_diagnostic"}}
{{"action_type": "clear_cache"}}
{{"action_type": "delegate", "target_agent": "netops", "instruction_payload": "Establish bypass routing"}}
{{"action_type": "delegate", "target_agent": "dataops", "instruction_payload": "Stop replication storm and reconcile ledger"}}
{{"action_type": "update_route", "target_id": "dc1_router--dc2_switch"}}
{{"action_type": "throttle_bandwidth", "target_id": "dc2_router--dc3_router", "parameters": {{"limit_pct": 10}}}}
{{"action_type": "update_route", "target_id": "oob_tunnel"}}
{{"action_type": "stop_replication"}}
{{"action_type": "force_stepdown", "target_id": "dc2"}}
"""
        return sys_prompt, user_prompt
