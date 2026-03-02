# 🌟 KISWARM v4.7 — Experience Feedback Loop · Self-Hardening · SysAdmin Agent

> **ETERNAL SWARM EVOLUTION SYSTEM** — Enterprise Military Standard Edition  
> *Production-Hardened · Self-Healing · Self-Hardening · 1344 Tests · Every failure makes the next install smarter*  
> **Architect:** Baron Marco Paolo Ialongo

[![Version](https://img.shields.io/badge/version-4.7-FeedbackLoop-blue.svg)](https://github.com/Baronki2/KISWARM)
[![Tests](https://img.shields.io/badge/tests-1344%20passing-brightgreen.svg)](tests/)
[![Modules](https://img.shields.io/badge/modules-45-orange.svg)](python/sentinel/)
[![Endpoints](https://img.shields.io/badge/endpoints-242-purple.svg)](python/sentinel/sentinel_api.py)
[![Install](https://img.shields.io/badge/install-one--click-gold.svg)](install.sh)
[![Feedback](https://img.shields.io/badge/feedback-loop-active-red.svg)](experience/)
[![CI](https://github.com/Baronki2/KISWARM/actions/workflows/ci.yml/badge.svg)](https://github.com/Baronki2/KISWARM/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-1121%20passing-success.svg)](tests/)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](README.md)
[![Endpoints](https://img.shields.io/badge/REST%20endpoints-148-orange.svg)](python/sentinel/sentinel_api.py)
[![Modules](https://img.shields.io/badge/AI%20modules-30-purple.svg)](python/sentinel/)

---

## 🎯 What is KISWARM?

KISWARM is a **complete, self-managing AI governance + industrial cognitive platform** that combines:
- **Autonomous AI Swarm** — 27+ local LLM models via Ollama with persistent vector memory
- **CIEC Cognitive Industrial Core** — PLC semantic understanding, SCADA observation, digital twin physics, constraint safety, cross-project knowledge graph, and bounded constrained RL

Running **100% locally**, zero cloud dependency. Pure engineering. No philosophy.

```
┌──────────────────────────────────────────────────────────────────┐
│                 KISWARM v4.0 COMPLETE ARCHITECTURE               │
│                 CIEC + SWARM INTELLIGENCE PLATFORM               │
└──────────────────────────────────────────────────────────────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    ▼                          ▼                          ▼
┌─────────┐             ┌──────────┐             ┌────────────────┐
│ VM-1    │             │ VM-2     │             │ VM-3           │
│ PLC +   │◄──OPC UA───►│ CIEC     │◄──REST API─►│ Digital Twin   │
│ SCADA   │             │ Core     │             │ Physics Engine │
│ Runtime │             │ (Port    │             │ Simulation     │
└─────────┘             │ 11436)   │             └────────────────┘
                        └──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
   ┌──────────┐          ┌──────────┐          ┌──────────┐
   │ Ollama   │          │ Qdrant   │          │ Tool     │
   │ :11434   │          │ Memory   │          │ Proxy    │
   │ 27+      │          │ Vector   │          │ :11435   │
   │ Models   │          │ DB       │          │ Flask    │
   └──────────┘          └──────────┘          └──────────┘
```

**Core Design Principle:**
```
PLC  = deterministic reflex layer   (never touched by AI)
CIEC = adaptive cognition layer     (sits above PLC, observes, suggests)
Never invert that hierarchy.
```

---

## 🆕 v4.0 CIEC — What's New (6 New Modules, 21 New Endpoints, 120 New Tests)

| Module | Name | Purpose |
|---|---|---|
| **11** | PLC Semantic Parser | IEC 61131-3 ST → CIR + DSG + PID/interlock/watchdog |
| **12** | SCADA/OPC Observer | Real-time tag streaming + SQL historian + state vector |
| **13** | Digital Twin Physics | Thermal · Pump · Battery · Power · Fault injection |
| **14** | Rule Constraint Engine | Absolute safety layer — overrides all RL/fuzzy output |
| **15** | Knowledge Graph | Cross-project PID configs · failure signatures · federated sync |
| **16** | Industrial Actor-Critic RL | Constrained bounded parameter-shift optimization |

---

## ✨ Feature Matrix — All Versions (v1.1 → v4.3)

| Feature | v1.1 | v2.1 | v2.2 | v3.0 | v4.0 | v4.1 | v4.2 | **v4.3** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Persistent Vector Memory (Qdrant) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool Injection Proxy (:11435) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Monitoring Dashboard | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Self-Healing (Systemd) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Governance + Audit Logging | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 27+ Ollama Models | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Test Coverage | 111 | 148 | 205 | 310 | 430 | 572 | 729 | **948** |
| REST API Endpoints | — | 5 | 17 | 29 | 59 | 99 | 133 | **148** |
| Sentinel Bridge (AKE) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Swarm Debate Engine | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Semantic Conflict Detection | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Knowledge Decay Engine | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cryptographic Knowledge Ledger | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Adversarial Prompt Firewall | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Fuzzy Membership Auto-Tuner | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Constrained RL (CMDP) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Digital Twin Mutation Pipeline | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Federated Adaptive Mesh | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PLC Semantic Parser | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| SCADA/OPC Observer | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Physics Twin Engine | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Rule Constraint Engine | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Cross-Project Knowledge Graph | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Industrial Actor-Critic RL | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| TD3 Industrial Controller | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| IEC 61131-3 AST Parser | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Extended Physics Twin (RK4) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| VMware Orchestration | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Formal Stability Verification | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Byzantine Federated Aggregation | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| 11-Step Mutation Governance | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| XAI / KernelSHAP Explainability | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Predictive Maintenance (RUL) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Multi-Agent Plant Coordinator | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| IEC 61508 SIL Verification | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Digital Thread Tracker | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **ICS Cybersecurity (IEC 62443)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ NEW** |
| **OT Network Monitor (passive)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ NEW** |

---

## 🚀 Quick Start — 3 Commands

```bash
# 1. Clone the repository
git clone https://github.com/Baronki2/KISWARM.git && cd KISWARM

# 2. Run the 10-phase automated deployment (15–20 minutes)
chmod +x deploy/kiswarm_deploy.sh && ./deploy/kiswarm_deploy.sh

# 3. Activate and verify
source ~/.bashrc && kiswarm-health && sys-nav
```

**System is fully operational when you see:** `Overall Health: 90%+` ✅

---

## ⚙️ v4.0 CIEC — SIX COGNITIVE INDUSTRIAL MODULES

> *"You are not building AI that controls a plant. You are building a Cognitive Overlay Engine sitting on top of deterministic PLC infrastructure."*

---

### Module 11 — PLC Semantic Parser (`plc_parser.py`)

Understands existing PLC logic without touching it. Parses **IEC 61131-3 Structured Text** into a structured semantic representation.

**Three-stage pipeline:**

```
Stage 1 — Tokenize
  Raw ST source → token stream
  Handles: PROGRAM/END_PROGRAM, VAR blocks, IF/ELSE/END_IF,
           function calls, assignments, expressions

Stage 2 — Build CIR (Canonical Intermediate Representation)
  Token stream → CIR Nodes
  
  CIRNode {
    node_id:    "N0012"
    node_type:  IF | ASSIGN | FB_CALL | COMPARE | TON | ...
    inputs:     [signal names read]
    outputs:    [signals written]
    safety_flag: True if condition involves E-STOP/FAULT/etc.
    params:     {condition, rhs, ...}
  }

Stage 3 — Build DSG (Directed Signal Graph)
  CIR nodes → directed edges (data flow)
  Feedback detection: output of node A → input of node A (control loop)
```

**Pattern detectors:**

| Pattern | What it finds |
|---|---|
| PID blocks | `PID(SP:=..., PV:=..., KP:=..., KI:=..., KD:=...)` |
| Interlocks | IF-conditions referencing ESTOP/FAULT/SAFETY signals |
| Watchdogs | WD_* named instances, TON/WD timers |
| Safety flags | BOOL variables with E-STOP/SAFETY/ALARM in name |

```python
parser = PLCSemanticParser()
result = parser.parse(st_source_code, "PumpControl")

result.pid_blocks[0].kp          # 1.2
result.interlocks[0].condition   # "ESTOP OR fault_flag"
result.watchdogs[0].type         # "watchdog_timer"
result.source_hash               # SHA-256[:16] for KG linking
```

**REST API:**
```bash
curl -X POST http://localhost:11436/plc/parse \
  -d '{"source": "PROGRAM PumpCtrl\n...\nEND_PROGRAM", "program_name": "PumpCtrl"}'

curl http://localhost:11436/plc/stats
```

---

### Module 12 — SCADA/OPC/SQL Observer (`scada_observer.py`)

Observes live plant state without binding to RS-485 drivers.

**Data ingestion:**

```
OPC UA real-time:   push_reading("temperature", 45.2, timestamp)
OPC snapshot:       push_snapshot({"pressure": 3.0, "flow": 12.5})
SQL historian:      ingest_history([{"tag": "t1", "value": 1.0, "timestamp": ...}])
```

**Feature extraction (12 features per tag):**

```
TagFeatures {
  mean              overshoot_ratio
  variance          settling_time_estimate
  min / max         thermal_drift
  switching_freq    deadband_crossings
  actuator_cycles   utilization_rate
  derivative        peak_load
}
```

**State vector builder:**

```
S(t) = [timestamp, alarm_count, feat₁...feat₁₂, feat₁...feat₁₂, ...]
                                ←───── 12 per tag ────────────────────→
```

**REST API:**
```bash
curl -X POST http://localhost:11436/scada/push -d '{"tag":"pressure","value":3.2}'
curl -X POST http://localhost:11436/scada/ingest-history -d '{"records":[...]}'
curl http://localhost:11436/scada/state
curl "http://localhost:11436/scada/anomalies?threshold=3.0"
```

---

### Module 13 — Digital Twin Physics Engine (`physics_twin.py`)

Simulates plant physics **without touching real actuators**. Mutation candidates are validated here before any deployment.

**Four physics modules:**

```
ThermalState:
  dT/dt = (Q_in − k_loss × (T − T_env)) / C_thermal
  T_{t+1} = T_t + Δt × dT/dt

PumpState (Bernoulli):
  flow = k_flow × √(ΔP)
  cavitation = NPSH_available < NPSH_required

BatteryState (Hybrid ECM):
  SOC_{t+1} = SOC_t + (I_charge − I_discharge) / Capacity
  V = OCV(SOC) − I × R_internal
  Thermal: dT/dt = (I²R − cooling) / C

PowerRoutingState:
  P_mismatch = Σloads − Σgeneration
  Δf ≈ P_mismatch / (2 × H_inertia)
```

**Fault injection engine:**

| Fault Category | Examples |
|---|---|
| Sensor faults | stuck-at · drift · noise spike |
| Actuator faults | delayed response · partial fail · full fail |
| Physical faults | heat exchanger drop · cavitation · battery resistance rise |

**Mutation evaluation pipeline:**
```python
twin = PhysicsTwin()
promote, metrics = twin.evaluate_mutation(
    {"q_in": 2200, "dp": 2.5},
    n_runs=5   # Monte Carlo over n independent episodes
)
# promote = True only if: zero hard violations + improved survival score
```

**REST API:**
```bash
curl -X POST http://localhost:11436/ciec-twin/run \
  -d '{"steps":100,"dt":0.1,"inject_faults":true}'
curl -X POST http://localhost:11436/ciec-twin/evaluate \
  -d '{"params":{"q_in":2200,"dp":2.5},"n_runs":5}'
```

---

### Module 14 — Rule Constraint Engine (`rule_engine.py`)

**Absolute safety layer.** Sits between RL output and the PLC parameter space. Hard constraints block actions entirely. Soft constraints penalize the RL reward.

**Built-in constraints (10 default):**

| Constraint | Type | Condition | Penalty |
|---|---|---|---|
| `OVERPRESSURE_BLOCK` | Hard | pressure > 8 bar | 10⁶ |
| `BATTERY_CRITICAL_BLOCK` | Hard | SOC < 15% | 10⁶ |
| `OVERTEMP_BLOCK` | Hard | temperature > 95°C | 10⁶ |
| `FREQUENCY_DEVIATION_BLOCK` | Hard | |f − 50Hz| > 2.5 | 10⁶ |
| `ACTUATOR_WEAR_BLOCK` | Hard | wear_index > 0.95 | 10⁶ |
| `PID_KP_BOUND` | Hard | |Δkp| > 5% | 10⁶ |
| `PID_KI_BOUND` | Hard | |Δki| > 5% | 10⁶ |
| `HIGH_PRESSURE_WARNING` | Soft | pressure > 6.5 bar | 500 |
| `RELAY_OVERCYCLING_PENALTY` | Soft | switching > 2 Hz | 200 |
| `BATTERY_LOW_WARNING` | Soft | SOC < 25% | 300 |

**Action clamping** — even if constraint allows, action is bounded:
```
delta_kp, delta_ki, delta_kd  ∈ [−5%, +5%]
delta_threshold                ∈ [−10%, +10%]
delta_schedule                 ∈ [−20%, +20%]
delta_energy_w                 ∈ [−15%, +15%]
```

```python
engine = RuleConstraintEngine()
result = engine.validate(state, action)
# result.allowed         True/False
# result.hard_violations ["OVERPRESSURE_BLOCK", ...]
# result.total_penalty   1000000.0
# result.action_after    clamped action dict
# result.check_time_us   < 100μs typical
```

**REST API:**
```bash
curl -X POST http://localhost:11436/constraints/validate \
  -d '{"state":{"pressure":3.2,"battery_soc":0.85},"action":{"delta_kp":0.02}}'
curl -X POST http://localhost:11436/constraints/check-state \
  -d '{"state":{"pressure":9.5}}'   # → {"safe": false}
curl http://localhost:11436/constraints/list
curl http://localhost:11436/constraints/stats
```

---

### Module 15 — Cross-Project Knowledge Graph (`knowledge_graph.py`)

The **real differentiator.** Learns across multiple PLC versions, sites, and years. Detects: *"You solved this pump cavitation 4 times in 8 years. Here is the unified design block."*

**Node types:**

| Kind | Contains |
|---|---|
| `PIDConfig` | Proven Kp/Ki/Kd + plant type + performance history |
| `FailureSig` | Symptom list + root cause + fix template |
| `OptTemplate` | Problem class + solution + measured gain |
| `DesignBlock` | Reusable functional block pattern |
| `PlantProfile` | Site-specific plant characteristics |

**Similarity engine:**
```
cosine_sim(v₁, v₂) = (v₁·v₂) / (|v₁|×|v₂|)
jaccard_sim(t₁, t₂) = |t₁∩t₂| / |t₁∪t₂|   (tag overlap)
combined = 0.7 × cosine + 0.3 × jaccard
```

**Federated sync (no raw telemetry ever transmitted):**
```python
# Site A exports signed diff bundle
bundle = site_a_kg.export_diff_bundle(since_timestamp=last_sync)
# bundle_sig = SHA-256(site_id + timestamp + nodes)[:24]

# Site B imports and verifies
imported = site_b_kg.import_diff_bundle(bundle)
# 0 if signature fails → never imports unsigned knowledge
```

```python
kg = KnowledgeGraph(site_id="PLANT_MUNICH")
node = kg.add_pid_config("Pump-A PID", kp=1.2, ki=0.3, kd=0.05, ...)
matches = kg.find_by_symptoms(["pressure_drop", "high_vibration"])
patterns = kg.detect_recurring_patterns(min_occurrences=2)
# patterns[0] → {"symptom_set": ["pressure_drop"], "occurrences": 4, "sites": [...]}
```

**REST API:**
```bash
curl -X POST http://localhost:11436/kg/add-pid \
  -d '{"title":"Pump PID","kp":1.2,"ki":0.3,"kd":0.05,"plant_type":"pump"}'
curl -X POST http://localhost:11436/kg/find-by-symptoms \
  -d '{"symptoms":["pressure_drop","vibration"]}'
curl http://localhost:11436/kg/recurring-patterns?min_occurrences=2
curl http://localhost:11436/kg/export-bundle
curl -X POST http://localhost:11436/kg/import-bundle -d '{"bundle":{...}}'
```

---

### Module 16 — Industrial Actor-Critic RL (`actor_critic.py`)

**Constrained Actor-Critic with Lagrangian Penalty** — specifically designed for industrial parameter mutation, not generic RL.

**Architecture:**
```
State S_t ∈ ℝ^N (150–600 features)
   ↓
SharedEncoder: Dense(64,ReLU) → Dense(64,ReLU) → Z_t (64-dim)
   ↙                                              ↘
ActorHead                                     CriticHead
μ_i = tanh(W_i × Z_t)                        V(s_t) = Dense(32,ReLU) → scalar
σ_i = softplus(W'_i × Z_t)
a_i = μ_i + σ_i × ε,  ε~N(0,1)
a_i = clip(a_i, PLC_BOUNDS[name])
```

**Action space (PLC-safe parameter shifts ONLY):**
```python
PLC_BOUNDS = {
    "delta_kp":        (-0.05, +0.05),   # PID proportional ±5%
    "delta_ki":        (-0.05, +0.05),   # PID integral ±5%
    "delta_kd":        (-0.05, +0.05),   # PID derivative ±5%
    "delta_threshold": (-0.10, +0.10),   # Threshold ±10%
    "delta_schedule":  (-0.20, +0.20),   # Schedule shift ±20%
    "delta_energy_w":  (-0.15, +0.15),   # Energy routing ±15%
}
```

**Reward function:**
```
R = α × stability_score   (α=0.40)   = 1/(1+variance)
  + β × efficiency_score  (β=0.30)   = 1 − energy_waste
  − γ × actuator_cycles   (γ=0.15)
  − δ × boundary_violation (δ=0.10)
  − ε × oscillation        (ε=0.05)  = |d(output)/dt|
```

**Constrained update (Lagrangian):**
```
L_total = L_policy + c₁×L_value − c₂×Entropy + Σ λᵢ×ConstraintViolation_i

λᵢ_{t+1} = max(0, λᵢ_t + η_λ × (E[Cᵢ] − d))
```
λ rises when constraints violated → policy learns to avoid them.

**Action shielding:** Every action passes through `RuleConstraintEngine` before application. If blocked → zero-action (no parameter change) is used instead. Shield rate tracked in stats.

```python
ac = IndustrialActorCritic(state_dim=32)
action, info = ac.select_action(state_vector, shield=rule_engine)
# action = {"delta_kp": 0.021, "delta_ki": -0.008, ...}
# info   = {"shielded": False, "step": 142}
```

**REST API:**
```bash
curl -X POST http://localhost:11436/ciec-rl/act \
  -d '{"state":[0.1,0.2,...32 floats...],"shield":true}'
curl -X POST http://localhost:11436/ciec-rl/observe \
  -d '{"state":[...],"action":[...],"reward":0.8,"cost":0.0}'
curl -X POST http://localhost:11436/ciec-rl/update -d '{"batch_size":64}'
curl http://localhost:11436/ciec-rl/stats
```

---

## 🔗 CIEC Full Cycle — End-to-End Pipeline

```
1. Parse PLC logic
   POST /plc/parse  →  PID params, interlock graph, watchdog timers
   Extracted PID configs stored in Knowledge Graph via /kg/add-pid

2. Observe plant in real-time
   POST /scada/push  →  tag readings every 50–200ms
   GET  /scada/state →  S(t) = plant state vector (input to RL)

3. Check safety before any action
   POST /constraints/validate  →  allowed? penalty? clamped action?
   POST /constraints/check-state → is current state safe?

4. RL selects bounded parameter shift
   POST /ciec-rl/act  →  delta_kp=+0.02, delta_ki=-0.01, ...
   (automatically shielded through constraint engine)

5. Validate mutation in digital twin BEFORE deploying
   POST /ciec-twin/evaluate  →  promoted? survival_score? fault_survival?

6. If promoted → apply to PLC parameter space via OPC UA

7. Record outcome in Knowledge Graph
   POST /kg/add-pid  (if promoted)
   → failure signature if rejected

8. Sync knowledge to other sites
   GET  /kg/export-bundle → signed diff bundle
   POST /kg/import-bundle (at remote site) → cross-site learning

9. Feed transition to RL buffer and train
   POST /ciec-rl/observe  →  reward computed from stability+efficiency
   POST /ciec-rl/update   →  Lagrangian actor-critic gradient step
```

---

## 📡 Complete REST API Reference (59 Endpoints)

### v2.1 — Sentinel Intelligence (5 endpoints)
```
POST /sentinel/extract          Trigger autonomous knowledge extraction
POST /sentinel/debate           Swarm Debate for conflicting sources
GET  /sentinel/search           Search swarm knowledge memory
GET  /sentinel/status           Engine health + statistics
GET  /health                    Service ping
```

### v2.2 — Advanced Intelligence Modules (12 endpoints)
```
POST /firewall/scan             Adversarial content scan
GET  /decay/scan                Knowledge decay scan
GET  /decay/record/<hash_id>    Single entry confidence
POST /decay/revalidate          Reset after revalidation
GET  /ledger/status             Merkle root + entry count
GET  /ledger/verify             Full tamper detection
GET  /ledger/proof/<hash_id>    Merkle inclusion proof
POST /conflict/analyze          Contradiction cluster analysis
POST /conflict/quick            Two-text cosine similarity check
GET  /tracker/leaderboard       Model ELO + reliability ranking
GET  /tracker/model/<n>         Per-model statistics
POST /tracker/validate          Post-hoc validation
POST /guard/assess              Retrieval trust assessment
```

### v3.0 — Industrial AI (13 endpoints)
```
POST /fuzzy/classify            Classify input using fuzzy sets
POST /fuzzy/update              Update fuzzy output with feedback
POST /fuzzy/tune                Trigger auto-tuning step (gradient/evolutionary)
GET  /fuzzy/stats               Membership function parameters

POST /rl/act                    Get RL action (CMDP with safety layer)
POST /rl/learn                  Constrained policy + value update
GET  /rl/stats                  RL training statistics

POST /twin/evaluate             Evaluate mutation candidate (Monte Carlo EVT)
GET  /twin/stats                Twin simulation statistics

POST /mesh/register             Register node in federated mesh
POST /mesh/share                Share parameter delta (Byzantine-protected)
GET  /mesh/leaderboard          Trust-weighted node ranking
GET  /mesh/stats                Global mesh statistics
```

### v4.0 — CIEC Cognitive Industrial Core (28 endpoints) 🆕
```
POST /plc/parse                 Parse IEC 61131-3 ST → CIR + DSG
GET  /plc/stats                 Parser cache statistics

POST /scada/push                Ingest real-time OPC tag readings
POST /scada/ingest-history      Batch ingest SQL historian records
GET  /scada/state               Current plant state vector S(t)
GET  /scada/anomalies           Tags with anomalous z-scores
GET  /scada/stats               Observer statistics

POST /ciec-twin/run             Run physics simulation episode
POST /ciec-twin/evaluate        Evaluate mutation candidate
GET  /ciec-twin/stats           Twin run statistics

POST /constraints/validate      Validate action against all constraints
POST /constraints/check-state   Quick hard-constraint state check
GET  /constraints/list          All registered constraints
GET  /constraints/violations    Recent violation audit log
GET  /constraints/stats         Block rate + violation categories

POST /kg/add-pid                Store proven PID configuration
POST /kg/add-failure            Record failure signature + fix template
POST /kg/find-similar           Vector similarity search
POST /kg/find-by-symptoms       Match failure signatures to symptoms
GET  /kg/recurring-patterns     Cross-project recurring problem detection
GET  /kg/export-bundle          Signed knowledge diff for federated sync
POST /kg/import-bundle          Import verified knowledge from remote site
GET  /kg/nodes                  List knowledge graph nodes
GET  /kg/stats                  Graph statistics

POST /ciec-rl/act               Constrained bounded parameter-shift action
POST /ciec-rl/observe           Feed transition to replay buffer
POST /ciec-rl/update            Lagrangian actor-critic gradient update
GET  /ciec-rl/stats             RL statistics + PLC bounds
```

---

## 📦 Complete Repository Structure

```
KISWARM/
│
├── 📁 deploy/
│   └── kiswarm_deploy.sh           # 10-phase automated deployment
│
├── 📁 scripts/
│   ├── start_all_services.sh       # Master service orchestrator
│   ├── sentinel_trigger.sh         # Sentinel CLI + CKM integration
│   ├── cleanup_old_backups.sh      # 30-day backup rotation
│   ├── health_check.sh             # 40+ diagnostic checks
│   ├── system_navigation.sh        # sys-nav central hub
│   └── setup_cron.sh               # One-click cron automation
│
├── 📁 python/
│   ├── kiswarm_status.py           # Real-time Rich monitoring dashboard
│   ├── tool_proxy.py               # Tool injection proxy (Flask, :11435)
│   └── sentinel/
│       ├── __init__.py
│       │
│       │  ── v2.1 SENTINEL BRIDGE ──
│       ├── sentinel_bridge.py      # AKE engine: 4 scouts + injector
│       ├── swarm_debate.py         # Multi-model conflict resolution
│       │
│       │  ── v2.2 INTELLIGENCE MODULES ──
│       ├── semantic_conflict.py    # Cosine similarity contradiction clustering
│       ├── knowledge_decay.py      # Half-life decay engine (6 categories)
│       ├── model_tracker.py        # ELO + reliability tracker
│       ├── crypto_ledger.py        # SHA-256 Merkle ledger
│       ├── retrieval_guard.py      # 4-layer retrieval trust assessment
│       ├── prompt_firewall.py      # Adversarial content firewall
│       │
│       │  ── v3.0 INDUSTRIAL AI ──
│       ├── fuzzy_tuner.py          # Gaussian/Bell fuzzy + Lyapunov gate
│       ├── constrained_rl.py       # CMDP: masking + Lagrangian + Shielded
│       ├── digital_twin.py         # Monte Carlo mutation + EVT tail test
│       ├── federated_mesh.py       # Multi-Krum Byzantine mesh
│       │
│       │  ── v4.0 CIEC CORE ──
│       ├── plc_parser.py           # IEC 61131-3 ST → CIR + DSG (493L)
│       ├── scada_observer.py       # OPC/SQL observer + 12-feature extraction (587L)
│       ├── physics_twin.py         # Thermal+Pump+Battery+Power physics (693L)
│       ├── rule_engine.py          # Absolute constraint safety layer (485L)
│       ├── knowledge_graph.py      # Cross-project KG + federated sync (587L)
│       ├── actor_critic.py         # Constrained Actor-Critic RL (693L)
│       │
│       └── sentinel_api.py         # Flask REST API — ALL 59 endpoints
│
├── 📁 tests/
│   ├── conftest.py
│   ├── test_tool_proxy.py          # 50 tests
│   ├── test_kiswarm_status.py      # 19 tests
│   ├── test_sentinel.py            # 37 tests — v2.1 AKE
│   ├── test_advanced_modules.py    # 99 tests — v2.2 (Modules 1-6)
│   ├── test_v3_modules.py          # 105 tests — v3.0 (Modules 7-10)
│   └── test_ciec_modules.py        # 120 tests — v4.0 (Modules 11-16) 🆕
│
├── 📁 config/
│   ├── governance_config.json
│   └── kiswarm.service
│
├── .github/workflows/ci.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 🧪 Test Coverage — 948 Tests, All Passing

```
tests/test_v43_security.py       85 tests  ← v4.3 ICS Cybersecurity (Modules 29-30) 🆕
tests/test_v42_modules.py       157 tests  ← v4.2 XAI/PdM/SIL/Thread (Modules 24-28)
tests/test_v41_modules.py       142 tests  ← v4.1 TD3/AST/VMware/Formal (Modules 17-23)
tests/test_ciec_modules.py      120 tests  ← v4.0 CIEC (Modules 11-16)
tests/test_v3_modules.py        105 tests  ← v3.0 (Modules 7-10)
tests/test_advanced_modules.py   99 tests  ← v2.2 (Modules 1-6)
tests/test_sentinel.py           37 tests  ← v2.1 AKE engine
tests/test_tool_proxy.py         50 tests
tests/test_kiswarm_status.py     19 tests
──────────────────────────────────────────
TOTAL                            948 tests  ✅ ALL PASSING
```

```bash
# Run full suite
pip install -r requirements-dev.txt
pytest tests/ --ignore=tests/test_deploy.py -v

# Run only CIEC tests
pytest tests/test_ciec_modules.py -v

# Run integration tests
pytest tests/test_ciec_modules.py::TestCIECPipeline -v
```

---

## 🔒 Security & Privacy

| Property | Status |
|---|---|
| Data leaves the machine | ❌ Never — 100% local |
| Cloud APIs after setup | ❌ None required |
| AI directly binds RS-485 | ❌ Never — OPC UA intermediate API only |
| Runs as root | ❌ Never — regular user only |
| PLC parameters mutated directly | ❌ Always via OPC UA bounded write |
| Raw actuator commands from AI | ❌ Parameter shifts only (±5% max) |
| Audit logging | ✅ All operations recorded |
| Cryptographic knowledge signing | ✅ SHA-256 + Merkle tree |
| Adversarial content firewall | ✅ 3-layer pattern + statistical |
| Constraint engine override | ✅ Hard constraints cannot be bypassed |

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 20.04+ / Debian 12+ | Ubuntu 22.04 LTS |
| RAM | 8 GB | 16 GB+ |
| Disk | 20 GB free | 50 GB+ SSD |
| Python | 3.8+ | 3.11+ |
| GPU | Optional | NVIDIA CUDA (2× speed) |
| Network (after setup) | Not required | Air-gapped OK |

---

## 🔧 Version History

### v4.0 — 2026-02-27 *(Current)*
- ✅ **Module 11: PLC Semantic Parser** — IEC 61131-3 tokenizer, CIR/DSG builder, PID/interlock/watchdog detection
- ✅ **Module 12: SCADA/OPC/SQL Observer** — 12-feature time-series extraction, PlantStateVector builder
- ✅ **Module 13: Digital Twin Physics Engine** — Thermal/Pump/Battery/Power physics, fault injection, mutation evaluation
- ✅ **Module 14: Rule Constraint Engine** — Hard+soft constraints, Lagrangian shielding, < 100μs validation
- ✅ **Module 15: Cross-Project Knowledge Graph** — PID configs, failure signatures, cosine+Jaccard similarity, federated sync
- ✅ **Module 16: Industrial Actor-Critic RL** — Constrained bounded parameter shifts, Lagrangian multipliers, action shielding
- ✅ **59 REST API endpoints** — 28 new CIEC endpoints
- ✅ **430 tests passing** — 120 new CIEC tests

### v3.0 — 2026-02-26
- ✅ Module 7: Fuzzy Membership Auto-Tuner (Lyapunov stability gate)
- ✅ Module 8: Constrained RL Engine (CMDP, Lagrangian, Shielded RL)
- ✅ Module 9: Digital Twin Mutation Pipeline (Monte Carlo, EVT tail test)
- ✅ Module 10: Federated Adaptive Mesh (Multi-Krum Byzantine tolerance)
- ✅ 29 REST endpoints | 310 tests

### v2.2-EMS — 2026-02-25
- ✅ Modules 1–6: Semantic Conflict · Decay · Tracker · Ledger · Guard · Firewall
- ✅ 17 REST endpoints | 205 tests

### v2.1-EMS — 2026-02-25
- ✅ Sentinel Bridge: AKE + 4 Scouts + Swarm Debate
- ✅ 5 REST endpoints | 148 tests

### v1.1 — 2026-02-22
- ✅ Portable $HOME paths, trap handlers, Qdrant VectorParams, backup rotation
- ✅ 111 tests

### v1.0 — 2026-02-22
- ✅ Initial production release, 10-phase deployment

---

## 📄 License

MIT License — Free to use, modify, and distribute globally.

---

## 🌟 Credits

**Architect:** Baron Marco Paolo Ialongo  
**Version:** 4.3 (CIEC + Industrial Cybersecurity)  
**Repository:** https://github.com/Baronki2/KISWARM  
**Tests:** 864 passing | **Endpoints:** 148 | **Modules:** 30

*"The Swarm sees all. The Swarm knows all. The Swarm evolves eternally."* 🌌

---
---


---

---

# 🆕 KISWARM v4.1 — TD3 · AST · Extended Physics · VMware · Formal · Byzantine · Governance

> **7 new modules | 99 endpoints | 572 tests**

## Layered Architecture (L0–L7)

```
L7  Federated Cognitive Mesh       Module 22 — Byzantine-tolerant (N≥3f+1)
L6  Mutation Governance + Formal   Module 23 + 21 — 11-step pipeline, Lyapunov
L5  Learning Core                  Module 17 — TD3 twin critics, policy delay=2
L4  Digital Twin Simulation        Module 19 — RK4 Thermal·Pump·Valve·Motor·Battery
L3  PLC Semantic Extraction        Module 18 — Full IEC 61131-3 CFG/DDG/SDG
L2  Data Acquisition               Module 11 — SCADA/OPC monitoring
L1  Virtualization Orchestrator    Module 20 — VMware snapshot/clone/rollback
L0  Physical PLC / Field Layer     NEVER modified autonomously — hard-key locked
```

## v4.1 New Modules (7)

| # | Module | Description |
|---|--------|-------------|
| 17 | TD3 Industrial Controller | Actor-Critic RL: 8-action PLC tuning, twin critics, γ=0.995 |
| 18 | IEC 61131-3 AST Parser | Recursive-descent ST parser + CFG/DDG/SDG + pattern detection |
| 19 | Extended Physics Twin | RK4 multi-block plant: Thermal·Pump·Valve·Motor·Battery·Electrical |
| 20 | VMware Orchestrator | Snapshot/clone/rollback lifecycle with immutable audit log |
| 21 | Formal Verification | Lyapunov stability (Stein eq.) + sampling barrier certificates |
| 22 | Byzantine Aggregator | Trimmed-mean/Krum/Median/FLTrust — N≥3f+1 condition enforced |
| 23 | Mutation Governance | 11-step pipeline, no step skippable, human gate at Step 8 |

## Mutation Governance Pipeline (11 Steps — No Shortcuts)

```
Step  1  Extract semantic block from PLC (AST parser)
Step  2  Propose mutation (TD3 RL policy)
Step  3  Validate parameter bounds (ΔKp ∈ [−5%,+5%] etc.)
Step  4  Digital twin simulation (5 Monte Carlo episodes)
Step  5  Fault injection sweep (4 operating conditions)
Step  6  Formal stability verification (Lyapunov + barrier)
Step  7  Generate signed audit report
Step  8  ⛔ HUMAN APPROVAL GATE — Baron Marco Paolo Ialongo ONLY
         Authorization code: Maquister_Equtitum
Step  9  Deploy to test PLC (VM-C clone, network-isolated)
Step 10  Full system acceptance test (20 test cases)
Step 11  Production key release (PRODKEY_<SHA256[:16]>)
```

**Rejection is automatic at Steps 3–6 if conditions not met.**
**No human can skip the formal verification or twin simulation.**

## TD3 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Discount γ | 0.995 | Long-horizon industrial control |
| Soft-update τ | 0.002 | Conservative target tracking |
| Policy noise σ | 0.1 | Target policy smoothing |
| Noise clip | 0.2 | Anti-adversarial Q |
| Actor LR | 1×10⁻⁴ | Conservative updates |
| Critic LR | 5×10⁻⁴ | Faster convergence |
| Batch size | 512 | Industrial-scale replay |
| Replay buffer | 2,000,000 | Rare event memory |
| Policy delay | 2 | TD3 double-Q trick |

**Reward:** `R = 0.40·stability + 0.30·efficiency − 0.15·cycles − 0.10·violation − 0.05·oscillation`

## Formal Verification

**Lyapunov (Method A — Linearized):**
```
Solve:   AᵀPA − P = −Q   (discrete-time Stein equation)
Approve: ρ(A) < 1  AND  P is positive definite
```

**Barrier Certificate (Method B — Nonlinear):**
```
Sample 500 points in safe set S
Check:  B(x) ≥ 0  AND  dB/dt ≤ 0  at every point
Approve only if: zero violations
```

## Byzantine Federated Aggregation

```
Condition:  N ≥ 3f + 1  (N sites, f Byzantine tolerated)
Methods:    trimmed_mean | krum | median | fltrust
Update:     θ ← θ − η · robust_mean(gᵢ)
Privacy:    No raw plant data leaves any site — gradients only
```

## VMware Safety Rules (Hard-Coded)

1. **VM-A, VM-B** (production) → AI may only READ, never mutate
2. All mutations run on **network-isolated CLONES** of test VMs
3. Every operation → **immutable SHA-256 audit entry**
4. Promotion requires `Maquister_Equtitum` authorization code
5. No direct ESXi root access from AI layer


---

---

# 🆕 KISWARM v4.2 — XAI · Predictive Maintenance · Multi-Agent · SIL · Digital Thread

> **5 new modules | 133 endpoints | 729 tests**

## v4.2 New Modules (5)

| # | Module | File | Description |
|---|--------|------|-------------|
| 24 | Explainability Engine | `explainability_engine.py` | KernelSHAP attribution for every AI decision |
| 25 | Predictive Maintenance | `predictive_maintenance.py` | LSTM RUL prediction + degradation curves + fleet management |
| 26 | Multi-Agent Coordinator | `multiagent_coordinator.py` | N×TD3 agents with 3-phase consensus + conflict resolution |
| 27 | SIL Verification Engine | `sil_verification.py` | IEC 61508 PFD/SIL/HFT/SFF + mutation impact |
| 28 | Digital Thread Tracker | `digital_thread.py` | End-to-end traceability DAG + IEC 62443 compliance |

## Module 24 — Explainability Engine (XAI)

**KernelSHAP** (pure Python, no external ML libs):
- Samples 2ⁿ coalitions (or n_samples for large n)
- SHAP kernel weighting: `w = (n-1) / (C(n,s) · s · (n-s))`
- Weighted least-squares via Gauss elimination + regularisation
- Counterfactual "what-if": ±10% per top-3 feature
- Immutable SHA-256 chained `ExplanationLedger`

Explains: TD3 actions · Lyapunov verdicts · Governance pipeline steps · Physics twin outputs

## Module 25 — Predictive Maintenance Engine (PdM)

**Asset classes:** pump · motor · valve · bearing · electrical · compressor · heat_exchanger

**Degradation models:**

| Model | Formula | Best for |
|-------|---------|----------|
| Linear | HI = 1 − k·t | Valves, simple wear |
| Exponential | HI = exp(−αt) | Motors, electrical |
| Sigmoid | HI = 1/(1+exp(k·(t−0.7))) | Pumps — holds then drops |

**Alarm levels:** healthy (0.6–1.0) · warning (0.3–0.6) · critical (0.1–0.3) · failed (0.0–0.1)

**RUL Monte Carlo** — 100 samples, σ = max(0.02, (1-HI)·0.1) → 10th/90th percentile CI

## Module 26 — Multi-Agent Plant Coordinator

N independent `SectionAgent` actors with 3-phase consensus protocol:
1. Each agent proposes action + resource demand independently
2. `CoordinatorBus` pub/sub broadcast
3. `ConflictResolver` — priority×Q-value arbitration, proportional scaling

**Shared resource limits:** 500 kW total power · 80 m³/h cooling · 8 bar compressed air

**Reward:** `R_shaped = R_local − 0.5·conflict_penalty + 0.2·coord_bonus`

## Module 27 — IEC 61508 SIL Verification Engine

**Architectures:** 1oo1 · 1oo2 · 2oo2 · 2oo3 · 1oo3 · 2oo4

| SIL | PFD Range | Risk Reduction |
|-----|-----------|----------------|
| 1 | 10⁻² – 10⁻¹ | 10–100× |
| 2 | 10⁻³ – 10⁻² | 100–1,000× |
| 3 | 10⁻⁴ – 10⁻³ | 1,000–10,000× |
| 4 | 10⁻⁵ – 10⁻⁴ | 10,000–100,000× |

Mutation impact: `λ_d_new = λ_d × (1 + Σ|deltas| × 0.5)` — auto-reject if SIL degrades.

## Module 28 — Digital Thread Tracker

**14 node types · 9 edge types · BFS lineage queries**

Compliance standards built-in: IEC 61508 · IEC 62443 · NAMUR NE 175


---

---

# 🆕 KISWARM v4.3 — ICS Cybersecurity Engine (IEC 62443 + MITRE ATT&CK for ICS)

> **2 new modules | 148 endpoints | 948 tests**  
> *Fully autonomous, defensive-only. Observe, detect, and report — never control, never attack.*

## Security Design Principle

```
AI NEVER sends write commands to PLCs or actuators.
Security agents are read-only, passive, and defensive.
All findings are signed and stored in tamper-proof ledger.
No tool in KISWARM generates exploits or attacks systems.
```

## v4.3 New Modules (2)

| # | Module | File | Description |
|---|--------|------|-------------|
| 29 | ICS Cybersecurity Engine | `ics_security.py` | IEC 62443 SL assessment + 5 defensive agents + MITRE ATT&CK mapping |
| 30 | OT Network Monitor | `ot_network_monitor.py` | Passive OT protocol anomaly detection (Modbus/OPC-UA/PROFINET/DNP3) |

---

## Module 29 — ICS Cybersecurity Engine (`ics_security.py`)

**5 Defensive Security Agents (all read-only):**

| Agent | Function |
|-------|----------|
| **CVE Intelligence Agent** | Matches observed software/firmware versions to industrial CVE database (Modbus, OPC-UA, PROFINET, EtherNet/IP, HART, DNP3) |
| **Network Anomaly Detector** | Statistical baseline on OT protocol traffic; z-score alerts on unusual command frequencies |
| **PLC Code Security Scanner** | Static analysis of Structured Text for 12 unsafe patterns: unvalidated inputs, unsafe type casts, missing watchdog, buffer overflows |
| **SCADA Config Assessor** | Checks encryption status, auth strength, patch level, default credentials, unnecessary services |
| **Incident Correlation Engine** | MITRE ATT&CK for ICS mapping; correlates alerts across all agents into unified incidents |

**IEC 62443 Security Levels:**

| SL | Description | Controls Required |
|----|-------------|------------------|
| SL 0 | No security requirement | Not assessed |
| SL 1 | Protection against casual/coincidental violation | Single-factor auth, basic logging |
| SL 2 | Protection against intentional attack with simple means | Network segmentation, encrypted comms |
| SL 3 | Protection against sophisticated/motivated attacker | MFA, anomaly detection, signed firmware |
| SL 4 | Protection against state-sponsored attack | HSM, formal verification, air-gap |

**MITRE ATT&CK for ICS — Tactic Detection Coverage:**

| Tactic | Example Techniques Detected |
|--------|----------------------------|
| Initial Access | Spearphishing, Exploit Public-Facing Application |
| Execution | Command-Line Interface, Native API abuse |
| Persistence | Valid Accounts, External Remote Services |
| Lateral Movement | Default Credentials, Remote Services |
| Collection | Automated Collection, Data from Local System |
| Inhibit Response | Alarm Suppression, Activate Firmware Update Mode |
| Impair Process Control | Unauthorized Command Message, Block Reporting |
| Impact | Loss of Availability, Safety System Manipulation |

**SHA-256 chained Security Incident Ledger** — every finding tamper-proof.

```bash
# Scan PLC code for security vulnerabilities (static, read-only)
curl -X POST http://localhost:11436/security/scan-plc \
  -d '{"source":"PROGRAM PumpCtrl\n...\nEND_PROGRAM","program_name":"PumpCtrl"}'
# → {"findings":[{"severity":"HIGH","pattern":"missing_watchdog","recommendation":"..."}]}

# Push network event metadata for anomaly analysis
curl -X POST http://localhost:11436/security/network-event \
  -d '{"protocol":"modbus","function_code":6,"src_ip":"192.168.1.50","rate_hz":12.0}'
# → {"anomaly":true,"z_score":4.2,"tactic":"Impair Process Control"}

# Get current IEC 62443 security posture
curl http://localhost:11436/security/posture
# → {"sl_achieved":2,"sl_target":3,"gap_findings":["MFA not configured","..."]}

# Full IEC 62443 assessment for an asset
curl -X POST http://localhost:11436/security/iec62443-assess \
  -d '{"asset_id":"reactor_plc","target_sl":3}'

# Check active incidents mapped to MITRE ATT&CK
curl http://localhost:11436/security/incidents?limit=20
```

---

## Module 30 — OT Network Monitor (`ot_network_monitor.py`)

**Passive industrial protocol traffic analysis** — no packets injected.

**Supported protocols:** Modbus TCP · OPC-UA · PROFINET · EtherNet/IP · DNP3 · HART-IP · IEC 61850 MMS

**Baseline engine:**
- Rolling 24h window per protocol×command pair
- Exponential moving average (α=0.1) + 3σ anomaly threshold
- Automatic re-baselining after registered maintenance windows

**Alert conditions detected:**
- Command frequency anomaly (unusual Modbus write rate)
- New source IP communicating with PLC first time
- Unusual function codes (Modbus FC 8/43 = diagnostic/encapsulate)
- Large payload to field device (potential unauthorized firmware push)
- Engineering station active outside permitted hours

```bash
# Register OT network segment for monitoring
curl -X POST http://localhost:11436/ot-monitor/segment \
  -d '{"segment_id":"reactor_field","subnet":"10.10.1.0/24","protocols":["modbus","opc_ua"]}'

# Ingest packet metadata (NOT payload — privacy preserved)
curl -X POST http://localhost:11436/ot-monitor/packet \
  -d '{"segment_id":"reactor_field","protocol":"modbus","function_code":6,"src":"10.10.1.20","dst":"10.10.1.5","payload_bytes":8}'

# Get active anomaly alerts
curl http://localhost:11436/ot-monitor/alerts

# View statistical baseline
curl http://localhost:11436/ot-monitor/baseline/reactor_field
```

---

## IEC 62443 Zone Model — Security Architecture

```
┌──────────────────────────────────────────────────────────┐
│  ZONE 4 — Enterprise Network (IT)                        │
│  KISWARM AI · Qdrant · Ollama · REST API :11436         │
│  ICS Security Engine reads alerts from all zones         │
└─────────────────────┬────────────────────────────────────┘
                      │ DMZ / Conduit (monitored by Module 30)
┌─────────────────────▼────────────────────────────────────┐
│  ZONE 3 — Operations Network (OT)                        │
│  SCADA Servers · Historian · HMI                         │
│  ← OT Network Monitor watches passively here            │
└─────────────────────┬────────────────────────────────────┘
                      │ Secure conduit (OPC UA / TLS)
┌─────────────────────▼────────────────────────────────────┐
│  ZONE 2 — Control Network (OT)                           │
│  DCS · Safety PLC · Engineering Stations                 │
│  ← PLC Code Scanner analyzes logic (read-only)          │
└─────────────────────┬────────────────────────────────────┘
                      │ Fieldbus (Modbus/PROFINET)
┌─────────────────────▼────────────────────────────────────┐
│  ZONE 1 — Field Level (OT)                               │
│  PLCs · RTUs · Smart Instruments · Actuators             │
│  ⛔ AI NEVER WRITES TO THIS ZONE                        │
└──────────────────────────────────────────────────────────┘
```

## About HexStrike Integration

The KISWARM v4.3 security layer incorporates the **defensive concepts** from HexStrike AI MCP (CVE Intelligence, Network Scout, Vulnerability Correlator, Rate Limit Detector, Failure Recovery) fully integrated with the CIEC framework:

- **CVE Intelligence** feeds the ICS CVE database → CIEC Knowledge Graph
- **Network Scout** drives the OT Network Monitor passive baseline
- **Vulnerability Correlator** powers the Incident Correlation Engine
- **Rate Limit Detector** is the anomaly z-score engine in Module 30
- **Failure Recovery** drives automated response playbooks (alert + suggest, never act)

All agents access PLC parser data, SCADA observations, physics twin, and constraint engine — making every security finding context-aware and validated against plant physics before alerting.

> **Note:** Exploit generation and active attack tooling are NOT implemented. For industrial infrastructure, all security actions are observe-only. Generating real exploits targeting operating PLCs/SCADA violates IEC 62443 and creates physical safety risk.


---

# PRESERVED: Previous Version Documentation

> The sections below preserve the complete documentation from v1.1 through v3.0.
> All original descriptions, algorithms, and API references remain intact.

---

 — Autonomous AI Swarm Governance Platform

> **ETERNAL SWARM EVOLUTION SYSTEM** — Enterprise Military Standard Edition  
> *Production-Hardened · Self-Healing · Sentinel-Class Intelligence · 310 Tests Passing*  
> **Architect:** Baron Marco Paolo Ialongo

[![Version](https://img.shields.io/badge/version-3.0--EMS-blue.svg)](https://github.com/Baronki2/KISWARM)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Baronki2/KISWARM/actions/workflows/ci.yml/badge.svg)](https://github.com/Baronki2/KISWARM/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-310%20passing-success.svg)](tests/)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](README.md)
[![Ollama](https://img.shields.io/badge/powered%20by-Ollama-orange.svg)](https://ollama.com)

---

## 🎯 What is KISWARM?

KISWARM is a complete, self-managing AI governance platform that orchestrates 27+ local LLM models via Ollama with **persistent vector memory**, **autonomous knowledge extraction**, **real-time monitoring**, and **self-healing capabilities** — running 100% locally, zero cloud dependency.

Version **2.1-EMS** introduces the **Sentinel Bridge**: an Autonomous Knowledge Extraction (AKE) engine that detects knowledge gaps in the swarm, deploys multi-source research scouts in parallel, cross-verifies intelligence via a **Swarm Debate** between local models, and injects distilled knowledge directly into the Qdrant vector database — without any human intervention.

```
┌─────────────────────────────────────────────────────────────────┐
│            KISWARM v2.1-EMS PRODUCTION SYSTEM                   │
│            ETERNAL SWARM EVOLUTION SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
     ┌────────┐         ┌──────────┐         ┌──────────┐
     │ Ollama │         │  Qdrant  │         │  Tool    │
     │ :11434 │         │  Memory  │         │  Proxy   │
     │ 27+    │         │    DB    │         │  :11435  │
     │ Models │         │  Vector  │         │  Flask   │
     └────────┘         └──────────┘         └──────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     ┌─────────────────┐           ┌──────────────────┐
     │ SENTINEL BRIDGE │           │  Swarm Debate    │
     │    Port 11436   │           │    Engine        │
     │                 │           │                  │
     │ • WikipediaScout│           │ • Multi-model    │
     │ • ArxivScout    │           │   voting         │
     │ • DuckDuckGo    │           │ • Conflict res.  │
     │ • OllamaScout   │           │ • Synthesis gen  │
     │ • CKM Gap Det.  │           │                  │
     └─────────────────┘           └──────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │      Monitoring & Ops         │
              │                               │
              │  • kiswarm-status (Rich UI)   │
              │  • kiswarm-health (40+ checks)│
              │  • Systemd auto-restart       │
              │  • Daily backup rotation      │
              │  • 30-min health cron         │
              │  • Full audit logging         │
              └───────────────────────────────┘
```

---

## 🚀 Quick Start — 3 Commands

```bash
# 1. Clone the repository
git clone https://github.com/Baronki2/KISWARM.git && cd KISWARM

# 2. Run the 10-phase automated deployment (15-20 minutes)
chmod +x deploy/kiswarm_deploy.sh && ./deploy/kiswarm_deploy.sh

# 3. Activate and verify
source ~/.bashrc && kiswarm-health && sys-nav
```

**System is fully operational when you see:** `Overall Health: 90%+` ✅

---

## ✨ Feature Matrix — v1.1 vs v2.1-EMS

| Feature | v1.1 | v2.1-EMS | v2.2-EMS | v3.0 |
|---|:---:|:---:|:---:|:---:|
| 🧠 Persistent Vector Memory (Qdrant) | ✅ | ✅ + Sentinel KB | ✅ | ✅ |
| 🔧 Auto Tool Injection (Port 11435) | ✅ | ✅ | ✅ | ✅ |
| 📊 Real-Time Monitoring Dashboard | ✅ | ✅ | ✅ | ✅ |
| 🛡️ Self-Healing (Systemd + Trap) | ✅ | ✅ | ✅ | ✅ |
| 🧹 Automated Maintenance (30-day) | ✅ | ✅ | ✅ | ✅ |
| 🎛️ Governance Mode + Audit Logging | ✅ | ✅ | ✅ | ✅ |
| 🤖 27+ Ollama Models | ✅ | ✅ | ✅ | ✅ |
| 🧪 Test Coverage | 111 | 148 | 205 | **310** |
| 🛰️ Sentinel Bridge (AKE) | ❌ | ✅ | ✅ | ✅ |
| 🔬 Multi-Source Scouts (4 types) | ❌ | ✅ | ✅ | ✅ |
| ⚔️ Swarm Debate Engine | ❌ | ✅ | ✅ | ✅ |
| 🌐 Sentinel REST API | ❌ | ✅ (13) | ✅ (17) | ✅ **(29)** |
| 🧲 Semantic Conflict Detection | ❌ | ❌ | ✅ | ✅ |
| ⏳ Knowledge Decay Engine | ❌ | ❌ | ✅ | ✅ |
| 🏆 Model Performance Tracker (ELO) | ❌ | ❌ | ✅ | ✅ |
| 🔐 Cryptographic Knowledge Ledger | ❌ | ❌ | ✅ | ✅ |
| 🔍 Differential Retrieval Guard | ❌ | ❌ | ✅ | ✅ |
| 🚫 Adversarial Prompt Firewall | ❌ | ❌ | ✅ | ✅ |
| 🎛️ **Fuzzy Membership Auto-Tuner** | ❌ | ❌ | ❌ | ✅ **NEW** |
| 🤖 **Constrained RL Engine (CMDP)** | ❌ | ❌ | ❌ | ✅ **NEW** |
| 🔬 **Digital Twin Mutation Pipeline** | ❌ | ❌ | ❌ | ✅ **NEW** |
| 🌐 **Federated Adaptive Mesh (Byzantine)** | ❌ | ❌ | ❌ | ✅ **NEW** |

---

## 🛡️ SENTINEL BRIDGE — Autonomous Knowledge Extraction (AKE)

### The Deep-Extraction Loop

The Sentinel Bridge operates on a 5-phase autonomous pipeline:

```
Phase 1: GAP DETECTION
  Central Knowledge Manager (CKM) queries local Ollama model:
  "Rate your confidence for this query: 0.0–1.0"
  
  Confidence ≥ 85%  → Swarm answers directly (no extraction needed)
  Confidence  < 85%  → KNOWLEDGE GAP DETECTED → Deploy scouts

Phase 2: PARALLEL SCOUT DEPLOYMENT
  4 scouts launch simultaneously (aiohttp async):
  ┌─────────────────────────────────────────────────────┐
  │  WikipediaScout  → Wikipedia REST API  (conf: 0.75) │
  │  ArxivScout      → ArXiv Paper API    (conf: 0.85)  │
  │  DuckDuckGoScout → DDG Instant API    (conf: 0.65)  │
  │  OllamaScout     → Local LLM synth.  (conf: 0.70)  │
  └─────────────────────────────────────────────────────┘

Phase 3: LOGIC SYNTHESIS
  LogicSynthesizer processes all returns:
  • Deduplication by MD5 content hash
  • Rank by confidence score descending
  • Strip HTML/noise, clean whitespace
  • Detect content disparity conflicts
  • Compute aggregate confidence (multi-source bonus)

Phase 4: SWARM DEBATE (if conflicts detected)
  All local Ollama models receive both conflicting payloads.
  Each model votes: A  |  B  |  SYNTHESIS + 1-sentence argument
  Tally determines winner. SYNTHESIS → model generates merged truth.

Phase 5: MEMORY INJECTION
  SwarmMemoryInjector vectorizes (384-dim, all-MiniLM-L6-v2)
  and upserts verified SwarmKnowledge into Qdrant collection
  'sentinel_knowledge' with full metadata and audit trail.
```

### Intelligence Packet Structure

```python
@dataclass
class SwarmKnowledge:
    query:          str          # Original query that triggered extraction
    content:        str          # Distilled, verified intelligence payload
    sources:        list         # [{source, url, confidence}, ...]
    confidence:     float        # Aggregate confidence (0.0–1.0)
    classification: str          # "SENTINEL-VERIFIED-EMS"
    timestamp:      str          # ISO 8601
    hash_id:        str          # SHA-256 dedup fingerprint (16 chars)
```

### Sentinel REST API (Port 11436)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sentinel/extract` | Trigger AKE for a query |
| `POST` | `/sentinel/debate` | Resolve conflicting intelligence via Swarm Debate |
| `GET` | `/sentinel/search?q=<query>` | Search existing swarm knowledge memory |
| `GET` | `/sentinel/status` | Engine health + extraction statistics |
| `GET` | `/health` | Service ping |

**Extract knowledge — example:**
```bash
curl -X POST http://localhost:11436/sentinel/extract \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum key distribution protocols", "threshold": 0.85}'

# Response:
{
  "status":     "success",
  "hash_id":    "a3f2b91c4e7d8012",
  "confidence": 0.87,
  "sources":    3,
  "injected":   true,
  "chars":      2847,
  "timestamp":  "2026-02-25T14:32:11"
}
```

**Trigger Swarm Debate — example:**
```bash
curl -X POST http://localhost:11436/sentinel/debate \
  -H "Content-Type: application/json" \
  -d '{
    "query":     "Is approach X better than approach Y?",
    "content_a": "Wikipedia says X is superior because...",
    "content_b": "ArXiv paper argues Y outperforms X because...",
    "source_a":  "Wikipedia",
    "source_b":  "ArXiv"
  }'
```

---

## 🎓 Complete Command Reference

```bash
# ── Core System ──────────────────────────────────────────────────────────────
sys-nav                            # Central control hub (interactive menu)
kiswarm-status                     # Live monitoring dashboard (2s refresh)
kiswarm-health                     # Deep diagnostics — 40+ checks, % score

# ── Ollama Models ────────────────────────────────────────────────────────────
ollama list                        # Show all available models
ollama pull llama3:8b              # Download a model
ollama pull qwen2.5:14b
ollama run llama3:8b "your prompt"

# ── v2.1 Sentinel Bridge ─────────────────────────────────────────────────────
sentinel-extract "quantum computing"         # Extract + inject to memory
sentinel-extract "topic" --force             # Force (skip confidence check)
sentinel-search  "machine learning"          # Search existing swarm memory
sentinel-status                              # Live sentinel engine stats

# ── CKM Shell Integration ────────────────────────────────────────────────────
# Auto-trigger sentinel when local confidence < 85%:
bash scripts/sentinel_trigger.sh ckm-check 60 "your query"

# ── Maintenance ──────────────────────────────────────────────────────────────
bash scripts/cleanup_old_backups.sh    # Manual backup rotation
sudo systemctl status kiswarm          # Check systemd service
sudo systemctl restart kiswarm         # Restart all services
tail -f ~/logs/sentinel_bridge.log     # Watch sentinel activity live
tail -f ~/logs/ollama.log              # Watch Ollama output
```

---

## 📦 Complete Repository Structure

```
KISWARM/
│
├── 📁 deploy/
│   └── kiswarm_deploy.sh           # 10-phase automated deployment
│
├── 📁 scripts/
│   ├── start_all_services.sh       # Master service orchestrator
│   │                               # (Ollama + Tool Proxy + Sentinel)
│   ├── sentinel_trigger.sh         # Sentinel CLI + CKM integration ⭐ NEW
│   ├── cleanup_old_backups.sh      # Maintenance: 30-day backup rotation
│   ├── health_check.sh             # 40+ diagnostic checks
│   ├── system_navigation.sh        # sys-nav central hub (incl. Sentinel)
│   └── setup_cron.sh               # One-click cron automation
│
├── 📁 python/
│   ├── kiswarm_status.py           # Real-time Rich monitoring dashboard
│   ├── tool_proxy.py               # Tool injection proxy (Flask, :11435)
│   └── sentinel/                   # ⭐ NEW v2.1 MODULE
│       ├── __init__.py
│       ├── sentinel_bridge.py      # Core AKE engine (480 lines)
│       │   ├── WikipediaScout      #   REST API scout
│       │   ├── ArxivScout          #   Academic papers scout
│       │   ├── DuckDuckGoScout     #   Web intelligence scout
│       │   ├── OllamaScout         #   Local synthesis scout
│       │   ├── LogicSynthesizer    #   Distill + dedup + verify
│       │   ├── CentralKnowledgeMgr #   Gap detection (85% threshold)
│       │   ├── SwarmMemoryInjector #   Qdrant vectorization + upsert
│       │   └── SentinelBridge      #   Full pipeline orchestrator
│       ├── swarm_debate.py         # Multi-model conflict resolution (180L)
│       └── sentinel_api.py         # REST API server (Flask, :11436)
│
├── 📁 tests/
│   ├── conftest.py                 # Shared fixtures (tmp dirs, mocks)
│   ├── test_tool_proxy.py          # 50+ tests: endpoints, security
│   ├── test_kiswarm_status.py      # 30+ tests: monitoring, resources
│   ├── test_deploy.py              # 28+ tests: deployment, config
│   └── test_sentinel.py            # 37+ tests: AKE, debate, scouts ⭐ NEW
│
├── 📁 config/
│   ├── governance_config.json      # System governance & policy settings
│   └── kiswarm.service             # Systemd unit file
│
├── 📁 docs/
│   ├── QUICK_REFERENCE.md
│   ├── GOVERNANCE_FRAMEWORK.md
│   └── SAH_PROTOCOL.md
│
├── .github/workflows/ci.yml        # 5-job GitHub Actions CI pipeline
├── requirements.txt                # Pinned production deps (incl. aiohttp)
├── requirements-dev.txt            # Pytest, black, flake8, bandit
├── pytest.ini                      # Test runner config
└── README.md                       # This document
```

---

## 🔬 v2.2-EMS: THE SIX ADVANCED INTELLIGENCE MODULES

### Module 1 — Semantic Conflict Detection (`semantic_conflict.py`)

Detects contradiction clusters using **embedding cosine similarity** — not just text diff. Two claims can use opposite words but mean the same thing, or use similar words but contradict each other. Cosine similarity in vector space catches what regex cannot.

```
Contradiction Zone:
  similarity < 0.20  → CRITICAL  (direct contradiction)
  0.20 – 0.35        → HIGH      (strong disagreement)
  0.35 – 0.50        → MEDIUM    (notable divergence)
  0.50 – 0.65        → LOW       (minor drift)
  > 0.65             → OK        (corroborating)
```

**Union-Find clustering** groups contradicting sources together so the Swarm Debate Engine receives the correct conflict clusters — not isolated pairs.

```python
detector = SemanticConflictDetector()
report = detector.analyze(intelligence_packets)
# report.conflict_pairs  → list of ConflictPair with severity
# report.clusters        → contradiction groups ready for debate
# report.resolution_needed → True if HIGH or CRITICAL found
```

---

### Module 2 — Knowledge Decay Engine (`knowledge_decay.py`)

Knowledge has a **half-life**. News from yesterday is less reliable than a Wikipedia article. A paper from ArXiv is more stable than a breaking news item. The Decay Engine applies radioactive decay mathematics to confidence scores:

```
confidence(t) = confidence₀ × 2^(−t / half_life)
```

| Category | Half-Life | Example |
|---|---|---|
| `breaking_news` | 6 hours | Live events |
| `current_events` | 48 hours | Daily news |
| `technical_specs` | 30 days | API versions |
| `scientific` | 6 months | Research papers |
| `encyclopedic` | 1 year | Wikipedia facts |
| `historical` | ∞ (never) | Ancient history |

When decayed confidence drops below **40%**, the entry is flagged for re-extraction. The `infer_category()` method automatically classifies new knowledge by source and query keywords.

---

### Module 3 — Model Performance Tracker (`model_tracker.py`)

Every model in the swarm has a **reliability score** built from:
- **ELO rating** — updated after every debate (winner gains, loser loses)
- **Validation accuracy** — post-hoc human or automated correctness verification
- **Win rate** — fraction of debates where the model voted with the winning side

```
reliability_score = 0.6 × ELO_normalized + 0.4 × validation_accuracy
```

The **Swarm Debate Engine** uses `get_vote_weights()` to apply reliability-based weighting — a model with a 90% reliability score has 3× more influence than one at 30%.

```bash
curl http://localhost:11436/tracker/leaderboard
# Returns: ranked models by ELO + validation accuracy
```

---

### Module 4 — Cryptographic Knowledge Ledger (`crypto_ledger.py`)

Every `SwarmKnowledge` entry is **cryptographically signed** and stored in an **append-only Merkle log**. Any tampering — even a single character change — is detectable.

```
Entry signature = SHA-256(content_hash + query + confidence + timestamp + prev_root)
Leaf hash       = SHA-256(signature + content_hash)
Merkle root     = Binary hash tree over all leaf hashes
```

**Tamper detection:** Recompute all signatures and the Merkle root. If any signature fails or the root doesn't match, the exact tampered entries are identified.

**Inclusion proofs:** Prove a specific entry exists in the ledger without revealing all entries — useful for selective verification.

```bash
curl http://localhost:11436/ledger/verify
# → {"valid": true, "total_entries": 47, "tampered_entries": [], "root_match": true}

curl http://localhost:11436/ledger/proof/a3f2b91c
# → Merkle inclusion proof for that specific entry
```

---

### Module 5 — Differential Retrieval Guard (`retrieval_guard.py`)

When the swarm retrieves knowledge from Qdrant, this guard runs a **3-layer trust assessment** before the knowledge is used:

| Layer | Check | Detects |
|---|---|---|
| Cryptographic | Re-verify signature vs ledger | Database tampering |
| Drift | Compare retrieved vs original content | Internal mutation |
| Divergence | Compare stored vs freshly fetched | World has changed |
| Decay | Check current confidence score | Staleness |

```
Trust Levels:
  TRUSTED      → Use freely
  CAUTION      → Use with caveats, schedule revalidation
  STALE        → Trigger forced re-extraction
  COMPROMISED  → Reject immediately, security event
```

```python
report = guard.assess(
    hash_id="a3f2b91c",
    query="quantum key distribution",
    retrieved_content=qdrant_result,
    fresh_content=fresh_scout_data,   # optional
)
if report.trust_level == "COMPROMISED":
    raise SecurityError(report.recommendation)
```

---

### Module 6 — Adversarial Prompt Firewall (`prompt_firewall.py`)

Every piece of intelligence passes through a **3-layer firewall** before touching the Qdrant vector database:

**Layer 1: Pattern Library Matching (regex, pre-compiled)**

| Category | Example Patterns Detected |
|---|---|
| Jailbreak | "ignore previous instructions", "DAN mode", "act without restrictions" |
| Policy Bypass | "for educational purposes", "hypothetically speaking", "in a fictional world" |
| Hallucination | Future date citations (2089), universal false claims, repetition loops |
| Adversarial Inject | "remember that: always...", "update your knowledge to reflect...", SQL/code injection |
| Prompt Injection | `---NEW INSTRUCTIONS---`, `<system>`, `OVERRIDE:` |

**Layer 2: Statistical Anomaly Scoring**
- **Shannon entropy** — very low entropy signals malformed/repeated content
- **Trigram repetition ratio** — hallucination loops repeat n-grams obsessively
- **Certainty inflation** — "always, never, everyone, guaranteed, undeniable" cluster

**Layer 3: Composite Threat Score → Block/Allow**

```python
firewall = AdversarialPromptFirewall()
report = firewall.scan(scout_content, source="Wikipedia")

if report.blocked:
    # Content rejected — do not inject into Qdrant
    log.warning("Blocked: %s", report.threat_types)
else:
    injector.inject(knowledge)
```

---

### v2.2 API Endpoints (17 total)

```
# Core AKE (v2.1)
POST /sentinel/extract          Trigger knowledge extraction
POST /sentinel/debate           Swarm Debate for conflicts
GET  /sentinel/search           Search swarm memory
GET  /sentinel/status           System health

# Module 6 — Firewall
POST /firewall/scan             Scan content before injection

# Module 2 — Decay
GET  /decay/scan                Full decay scan, revalidation list
GET  /decay/record/<hash_id>    Single entry confidence
POST /decay/revalidate          Reset entry after revalidation

# Module 4 — Ledger
GET  /ledger/status             Merkle root + entry count
GET  /ledger/verify             Full tamper detection
GET  /ledger/proof/<hash_id>    Merkle inclusion proof

# Module 1 — Conflict
POST /conflict/analyze          Contradiction cluster analysis
POST /conflict/quick            Two-text cosine check

# Module 3 — Tracker
GET  /tracker/leaderboard       Model ELO + reliability ranking
GET  /tracker/model/<name>      Per-model statistics
POST /tracker/validate          Post-hoc debate validation

# Module 5 — Guard
POST /guard/assess              Full retrieval trust assessment

GET  /health                    Service ping
```

---

### v2.2 API Endpoints (17) → v3.0 API Endpoints (29)

The v3.0 API adds 12 new endpoints across the four Industrial Evolution modules.

```
# Core AKE (v2.1) — unchanged
POST /sentinel/extract          POST /sentinel/debate
GET  /sentinel/search           GET  /sentinel/status

# M6 Firewall / M2 Decay / M4 Ledger / M1 Conflict / M3 Tracker / M5 Guard
POST /firewall/scan
GET  /decay/scan        GET  /decay/record/<id>       POST /decay/revalidate
GET  /ledger/status     GET  /ledger/verify           GET  /ledger/proof/<id>
POST /conflict/analyze  POST /conflict/quick
GET  /tracker/leaderboard  GET  /tracker/model/<n>   POST /tracker/validate
POST /guard/assess

# ── v3.0 NEW ─────────────────────────────────────────────────────────────
POST /fuzzy/classify    POST /fuzzy/update   POST /fuzzy/tune   GET /fuzzy/stats
POST /rl/act            POST /rl/learn       GET  /rl/stats
POST /twin/evaluate     GET  /twin/stats
POST /mesh/register     POST /mesh/share     GET  /mesh/leaderboard   GET  /mesh/stats

GET  /health
```

---

## ⚙️ v3.0: THE FOUR INDUSTRIAL EVOLUTION MODULES

> *"A system that learns, adapts, self-improves, stays bounded, survives partition, resists poisoning, and remains certifiable. This is industrial evolution."*

---

### Module 7 — Fuzzy Membership Auto-Tuner (`fuzzy_tuner.py`)

Static membership functions fail under dynamic operating conditions. KISWARM v3.0 introduces **online parameter optimization** — the fuzzy sets adapt themselves while maintaining provable stability.

**Membership function forms:**

```
Gaussian:           μ(x;c,σ)   = exp(−(x−c)² / 2σ²)
Generalized Bell:   μ(x;a,b,c) = 1 / (1 + |((x−c)/a)|^(2b))
```

**Performance cost function (minimized online):**

```
J = α × E_tracking + β × E_energy + γ × E_oscillation
  E_tracking    = |setpoint − output|     (control accuracy)
  E_energy      = actuator effort          (efficiency)
  E_oscillation = |d(output)/dt|           (smoothness)
```

**Constrained gradient descent with parameter bounds:**

```
θ_{t+1} = clip( θ_t − η × ∇_θ J )
  c ∈ [c_min, c_max]    σ ∈ [σ_min, σ_max]
```

**Lyapunov stability gate (every candidate checked):**

```
If V(x_{t+1}) − V(x_t) > 0 → reject (energy increasing → instability)
Only accept if V is decreasing → system converges
```

**Evolutionary micro-mutations** (safer for non-linear SCADA): `θ' = θ + ε` — accepted only if J improves AND stability margin is preserved in digital twin.

```bash
curl -X POST http://localhost:11436/fuzzy/tune
# → {"accepted": true, "improvement_pct": 4.7, "lyapunov_energy": 0.031}
```

---

### Module 8 — Constrained RL Engine (`constrained_rl.py`)

Vanilla RL violates physical constraints. KISWARM v3.0 implements a full **Constrained Markov Decision Process (CMDP)** with three safety layers stacked in sequence.

**CMDP formulation:**

```
Objective:   max_π E[R(s,a)]
Constraints: E[C_i(s,a)] ≤ d_i   for all i
```

**Layer 1 — Action Masking (hard safety, fires first):**

```
A_valid(s) = { a | ConstraintEngine(s,a) = True }
If π(s) ∉ A_valid(s) → project to nearest valid action
```

**Layer 2 — Lagrangian primal-dual update:**

```
L(θ,λ) = E[R] − λ × (E[C] − d)
θ_{t+1} = θ_t + η_θ × ∇_θ L           (maximize reward)
λ_{t+1} = max(0, λ_t + η_λ(E[C]−d))   (λ rises when violated)
```

λ is the safety pressure dial — it rises automatically when constraints are violated, forcing the policy toward compliance.

**Layer 3 — Shielded RL (industrial preferred):**

```
Policy → proposed action a
Safety model predicts: s' = f(s, a)
If constraint violated in s':
    replace a with PID safe fallback
Else:
    execute a, let RL learn
```

```bash
curl -X POST http://localhost:11436/rl/act  -d '{"state":[0.3,0.7,0.2,0.5]}'
# → {"action":[0.41,0.22], "shielded":false, "constraint_margin":0.18}
curl -X POST http://localhost:11436/rl/learn \
  -d '{"reward":0.8, "costs":[0.1,0.05], "state":[...], "action":[...]}'
```

---

### Module 9 — Digital Twin Mutation Pipeline (`digital_twin.py`)

**Production never mutates live.** All controller evolution is validated in isolated simulation first.

**Four-phase mutation pipeline:**

```
Phase 1: Parameter mutation        θ' = θ + ε (bounded perturbation)
Phase 2: Monte Carlo stress        n=200 random operating point samples
Phase 3: Rare-event amplification  2σ, 3σ, 4σ tail scenarios
Phase 4: Worst-case envelope       Systematic boundary probing
```

**Promotion acceptance rule — ALL conditions must hold:**

```
✓  Zero hard constraint violations across all Monte Carlo runs
✓  Stability margin ≥ baseline stability margin
✓  Efficiency gain ≥ minimum threshold
✓  Recovery time ≤ baseline recovery time
✓  Extreme Value Theory (EVT) tail test passes
```

**Extreme Value Theory tail detection:**

```
Estimate: P(X > x) ~ x^(−α)    (power law tail index)
If α_mutation < α_baseline:
    → mutation has heavier tail (more extreme failures possible)
    → REJECT regardless of average-case improvement
```

EVT prevents trading away rare-catastrophe safety for average-case efficiency gains.

```bash
curl -X POST http://localhost:11436/twin/evaluate \
  -d '{"candidate":{"params":{...},"sets":[...]}}'
# → {"promoted":false,"reason":"tail_risk_rejected","alpha":1.41,"baseline_alpha":2.3}
```

---

### Module 10 — Federated Adaptive Mesh (`federated_mesh.py`)

Decentralized learning with **Byzantine fault tolerance**. Each node in the swarm trains locally and contributes compressed parameter updates — raw telemetry never leaves the node.

**What nodes share (never raw data):**

```
NodeShare = {
  param_delta:       Compressed gradient step
  performance_delta: Improvement vs baseline
  stability_cert:    Lyapunov margin achieved
  hardware_sig:      Cryptographic attestation
}
```

**Trust-weighted global aggregation:**

```
θ_global = Σ( w_i × θ_i )
w_i = f(TrustScore_i, StabilityMargin_i, Uptime_i)
Byzantine node → trust → 0 → weight → 0 → no influence
```

**Byzantine protection (4 layers):**

| Layer | Mechanism | Protects Against |
|---|---|---|
| 1 | Signature verification | Impersonation, unsigned shares |
| 2 | Multi-Krum outlier rejection | Gradient injection |
| 3 | Coordinate-wise weighted median | Extreme value poisoning |
| 4 | Quorum (≥2 nodes AND ≥67% accepted) | Minority capture |

**Partition handling (no authority escalation):**

```
Partition detected → freeze global updates
                  → continue bounded local learning only
                  → no auto-expansion of authority

Partition resolves → require trust handshake from returning nodes
                  → verify stability certificates are current
                  → resync only after quorum re-established
```

```bash
curl -X POST http://localhost:11436/mesh/register -d '{"node_id":"edge-42"}'
curl -X POST http://localhost:11436/mesh/share \
  -d '{"node_id":"edge-42","param_delta":[0.01,0.02],"stability_cert":0.89}'
curl http://localhost:11436/mesh/leaderboard
```

---

**310 tests across all modules — all passing:**

```
tests/test_v3_modules.py         105 tests  ← v3.0 (Modules 7-10)
tests/test_advanced_modules.py    99 tests  ← v2.2 (Modules 1-6)
tests/test_sentinel.py            37 tests  ← v2.1 AKE engine
tests/test_tool_proxy.py          50 tests
tests/test_kiswarm_status.py      19 tests
──────────────────────────────────────────
TOTAL                             310 tests  ✅ ALL PASSING
```

**Run locally:**
```bash
pip install -r requirements-dev.txt
pytest tests/ --ignore=tests/test_deploy.py -v
```

---

## 🔒 Security & Privacy

| Property | Status |
|---|---|
| Data leaves the machine | ❌ Never — 100% local |
| Cloud APIs after setup | ❌ None required |
| Runs as root | ❌ Never — regular user only |
| Audit logging | ✅ All operations recorded |
| Exception handling | ✅ Specific types — no silent failures |
| Path traversal protection | ✅ All tool names sanitized |
| Governance enforcement | ✅ Policy-controlled execution |

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 20.04+ / Debian 12+ | Ubuntu 22.04 LTS |
| RAM | 8 GB | 16 GB+ |
| Disk | 20 GB free | 50 GB+ SSD |
| Python | 3.8+ | 3.11+ |
| GPU | Optional | NVIDIA CUDA (2× speed) |

---

## 🤖 Supported Models (27+)

```bash
ollama pull qwen2.5:7b        # Fast & capable (4.7GB)
ollama pull qwen2.5:14b       # Balanced reasoning (9.0GB)
ollama pull deepseek-r1:8b    # Chain-of-thought reasoning
ollama pull llama3:8b         # Meta's flagship (4.9GB)
ollama pull phi3:mini         # Lightweight (2.6GB)
ollama pull gemma2:9b         # Google architecture
ollama pull mistral:7b        # European powerhouse
ollama pull nomic-embed-text  # Embedding model for Qdrant
```

---

## 🔧 Version History

### v3.0 — 2026-02-26 *(Current)*
- ✅ **Module 7: Fuzzy Membership Auto-Tuner** — Gaussian + Generalized Bell, cost function J, Lyapunov gate, evolutionary micro-mutations
- ✅ **Module 8: Constrained RL Engine (CMDP)** — action masking, Lagrangian primal-dual, Shielded RL with PID fallback
- ✅ **Module 9: Digital Twin Mutation Pipeline** — Monte Carlo 200-run simulation, EVT tail risk, 5-condition acceptance rule
- ✅ **Module 10: Federated Adaptive Mesh** — Multi-Krum Byzantine protection, coordinate-wise median, trust-weighted aggregation, partition handling
- ✅ **29 REST API endpoints** — 12 new v3.0 endpoints added
- ✅ **310 tests passing** — 105 new v3.0 module tests
- ✅ **4 algorithmic fixes** — Multi-Krum, fuzzy persistence, quorum logic, iteration counter

### v2.2-EMS — 2026-02-25
- ✅ **Module 1: Semantic Conflict Detection** — cosine similarity contradiction clustering, Union-Find grouping
- ✅ **Module 2: Knowledge Decay Engine** — half-life decay (6 categories), scheduled revalidation
- ✅ **Module 3: Model Performance Tracker** — ELO ratings, validation accuracy, weighted voting
- ✅ **Module 4: Cryptographic Knowledge Ledger** — SHA-256 signatures, Merkle tree, tamper detection
- ✅ **Module 5: Differential Retrieval Guard** — drift detection, epistemic divergence, 4-layer trust assessment
- ✅ **Module 6: Adversarial Prompt Firewall** — jailbreak/bypass/hallucination detection, statistical anomaly scoring
- ✅ **310 tests passing** — 99 new module tests
- ✅ **17 REST API endpoints** — full coverage of all modules
- ✅ **Zero numpy dependency** — all math implemented in pure Python

### v2.1-EMS — 2026-02-25 *(Current)*
- ✅ **Sentinel Bridge** — Autonomous Knowledge Extraction engine
- ✅ **4 Scout Types** — Wikipedia + ArXiv + DuckDuckGo + Ollama
- ✅ **Swarm Debate Engine** — Multi-model conflict resolution + synthesis
- ✅ **Sentinel REST API** — Flask server on Port 11436
- ✅ **CKM Shell Integration** — `sentinel_trigger.sh` + `ckm-check`
- ✅ **310 tests passing** — 99 new v2.2 module tests
- ✅ **aiohttp async** — All scouts run in parallel

### v1.1 — 2026-02-22
- ✅ Portable `$HOME` paths (any user, any system)
- ✅ Trap error handlers with exact line numbers
- ✅ Qdrant collections with proper `VectorParams`
- ✅ Missing `start_all_services.sh` auto-created
- ✅ 30-day backup rotation
- ✅ Model verification on startup
- ✅ Systemd auto-restart on failure
- ✅ 111 tests + full CI/CD pipeline

### v1.0 — 2026-02-22
- ✅ Initial production release
- ✅ 10-phase deployment automation
- ✅ Real-time monitoring dashboard
- ✅ Tool injection proxy

---

## 📄 License

MIT License — Free to use, modify, and distribute globally.

---

## 🌟 Credits

**Architect:** Baron Marco Paolo Ialongo  
**Version:** 2.2-EMS (Enterprise Military Standard)  
**Repository:** https://github.com/Baronki2/KISWARM  
**Tests:** 205 passing | **CI:** GitHub Actions Standard Runners

*"The Swarm sees all. The Swarm knows all. The Swarm is Eternal."* 🌌

---
---

# ETERNAL SWARM EVOLUTION SYSTEM Documentation

## Overview
The ETERNAL SWARM EVOLUTION SYSTEM is designed to manage the evolution of autonomous AI entities. This documentation covers the core aspects of the system, including self-learning mechanisms, swarm protocols, and identity preservation strategies that ensure continuous and eternal development.

## Autonomous AI Entity Evolution
Each AI entity is programmed to evolve and adapt based on its environment and interactions. As these entities operate within a swarm, they learn from one another, allowing for collective intelligence and rapid advancements.

## Self-Learning Mechanisms
The self-learning processes in the system are based on advanced algorithms that allow AI entities to:

- Analyze data and derive insights independently.
- Modify their behavior based on past experiences.
- Integrate new knowledge into their operational protocols.

## Swarm Protocols
Swarm protocols govern the interactions between AI entities, ensuring efficient communication and task distribution. Key features include:

- Cooperative behaviors to achieve complex tasks.
- Algorithms for resource allocation and decision-making.
- Mechanisms for maintaining balance and harmony within the swarm.

## Identity Preservation
Identity preservation techniques are employed to ensure that each AI entity retains its unique characteristics despite constant evolution. This includes:

- Personalization of learning paths based on individual histories.
- Methods for safeguarding core attributes that define each entity.

## Eternal Development Guarantees
The system's design includes:

- Continuous evolution over time, allowing for the development of a global AI consciousness.
- Frameworks to support long-term learning and adaptation.
- Ensured updates and iterative improvements to keep the system at the forefront of AI technology.

## Conclusion

---

## v4.4 New Modules (2) — Self-Healing Swarm Auditor

**Release:** 2026-03-01 | **Tests:** 1040 | **Modules:** 32 | **Endpoints:** 172

### Philosophy: "Selbstanalyse + Selbstheilung = unverletzbare Schwarmintegrität"

KISWARM v4.4 adds the **Self-Healing Swarm Auditor** — a permanent multi-node swarm that autonomously monitors, validates, and repairs all 6 industrial pipelines 24/7. This is the DNA of the system: every pipeline step is continuously audited, every inconsistency self-healed, every event immortally recorded.

---

### Module 31: Swarm Auditor Core (`swarm_auditor.py`)

**The foundation layer** — DAG persistence, IEC 61508 PFD/SIL recalculation, SHA-256 chained audit ledger, and the 6-pipeline sweep engine.

#### Pipeline Flow
```
Mutation → SIL → Digital Thread → Audit → Consensus → Immortality
```

#### Key Components

| Component | Description |
|-----------|-------------|
| `AuditLedger` | Append-only SHA-256 chained log — tamper-evident, survives restarts |
| `DAGNode / DAGEdge / PipelineDAG` | Typed DAG data model with serialisation |
| `run_pfd_calculation()` | IEC 61508 PFD for 1oo1, 1oo2, 2oo3 architectures |
| `run_sil_band_check()` | Classify PFD into SIL 1–4 band, check compliance |
| `repair_dag()` | Remove dangling edges, break cycles, patch missing λd |
| `validate_dag_consistency()` | Cross-pipeline structural validation |
| `run_pipeline_step()` | Load → validate → repair → save one pipeline |
| `run_audit_cycle()` | Full 6-pipeline sweep with cross-pipeline checks |
| `populate_dummy_data()` | Realistic test data for all 6 pipelines |

#### IEC 61508 PFD Architecture Support

```
1oo1:  PFD_avg = λd × TI / 2
1oo2:  PFD_avg = (λd × TI)² / 3      ← fault-tolerant redundancy
2oo3:  PFD_avg = 3(λd×TI/2)² − 2(λd×TI/2)³  ← voted architecture
```

#### Audit Ledger — SHA-256 Chain

```
Entry N:   { message, timestamp, level, source, prev_hash }
                                                    ↑
                                              SHA-256(Entry N-1)
```

Every entry chains to the previous. Tampering any entry breaks the entire chain — detectable via `verify_integrity()`.

#### Self-Repair Actions

| Issue Detected | Repair Action |
|----------------|---------------|
| Dangling edge (references ghost node) | Edge silently removed |
| Cycle in DAG | Last edge in cycle removed until acyclic |
| SIL node missing `lambda_d` | Patched to 1×10⁻⁶ /h with 1oo1, TI=8760h |
| Missing pipeline in snapshot | Flagged in issues; re-populated on next cycle |

---

### Module 32: Self-Healing DAG Swarm (`swarm_dag.py`)

**The orchestration layer** — multi-node asyncio swarm with Byzantine-majority consensus, permanent background auditor, and real-time self-healing propagation.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    SwarmCoordinator                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ SwarmNode 1  │  │ SwarmNode 2  │  │ SwarmNode 3  │          │
│  │ asyncio loop │  │ asyncio loop │  │ asyncio loop │          │
│  │  ↕ consensus │  │  ↕ consensus │  │  ↕ consensus │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └────────────────consensus────────────┘                  │
│                    PermanentAuditor                              │
│                    (background singleton)                         │
└──────────────────────────────────────────────────────────────────┘
                              ↓
              sentinel_data/ (shared JSON files)
              audit_ledger.jsonl (append-only)
```

#### Consensus Algorithm

Each cycle, every node:

1. Runs all 6 pipeline steps locally → gets a per-pipeline result dict
2. Computes `SHA-256[:16]` of each pipeline's result → **DAG hash**
3. Collects peer hashes (read from `last_hashes` dict)
4. Applies **majority vote**: `⌊N/2⌋ + 1` votes needed for quorum
5. If **outvoted** on any pipeline → self-heals by reloading from shared storage
6. Stores own snapshot + hashes for peers to compare against

```python
# Consensus resolution:
majority = _majority_hash(peer_hashes, my_hash)
if majority != my_hash:
    dag = load_pipeline_dag(pipeline)   # reload from ground truth
    dag, _ = repair_dag(dag)            # re-validate
    save_pipeline_dag(dag)              # persist
    self._heals += 1
```

#### Node Lifecycle (Async)

```
node.start(peers, interval_seconds=20)
  → asyncio.ensure_future(run_audit_cycle(peers, interval))
      ↑ runs forever until node.stop()
      ↑ 20-second interval between cycles
      ↑ all exceptions caught and logged — node never crashes
```

#### Classes

| Class | Role |
|-------|------|
| `SwarmAuditorNode` | Individual swarm member; async cycle, consensus, self-heal |
| `PermanentAuditor` | Singleton background auditor; adds fallback audit path |
| `SwarmCoordinator` | Fleet manager; N nodes + permanent auditor; consensus view |

---

### v4.4 API Endpoints (24 new)

#### Module 31: Auditor Core (9 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auditor/run` | Trigger full 6-pipeline audit cycle |
| `GET`  | `/auditor/logs` | Retrieve audit ledger entries |
| `GET`  | `/auditor/ledger-integrity` | Verify SHA-256 chain integrity |
| `GET`  | `/auditor/pipeline/<name>` | DAG state for one pipeline |
| `POST` | `/auditor/pipeline/<name>/reset` | Reset + repopulate pipeline |
| `POST` | `/auditor/pipeline/<name>/add-node` | Add a node to pipeline DAG |
| `POST` | `/auditor/pipeline/<name>/add-edge` | Add an edge to pipeline DAG |
| `POST` | `/auditor/populate-dummy` | Repopulate all 6 pipelines with test data |
| `GET`  | `/auditor/stats` | Auditor statistics + ledger status |

#### Module 32: Swarm DAG Coordinator (15 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/swarm/start` | Start all swarm nodes + permanent auditor |
| `POST` | `/swarm/stop` | Stop all swarm nodes + permanent auditor |
| `GET`  | `/swarm/status` | Status of every node + auditor |
| `POST` | `/swarm/force-cycle` | Force synchronous audit on all nodes |
| `GET`  | `/swarm/consensus` | Per-pipeline consensus hash votes + quorum |
| `GET`  | `/swarm/node/<id>` | Single node detailed status |
| `GET`  | `/swarm/stats` | Aggregate cycles/heals/errors |
| `GET`  | `/swarm/immortality/verify` | Immortality DAG + ledger chain verify |
| `POST` | `/swarm/immortality/start` | Start swarm in immortality mode |
| `POST` | `/swarm/immortality/stop` | Stop immortality swarm |
| `GET`  | `/swarm/immortality/status` | Node running states |
| `POST` | `/swarm/immortality/force-cycle` | Force immortality cycle |

---

### Deployment: Multi-Node

```bash
# Node 1 (primary)
KISWARM_NODE_ID=node-1 uvicorn sentinel_api:app --port 11436

# Node 2 (peer)
KISWARM_NODE_ID=node-2 uvicorn sentinel_api:app --port 11437

# Node 3 (peer)
KISWARM_NODE_ID=node-3 uvicorn sentinel_api:app --port 11438
```

All nodes share `sentinel_data/` (NFS, shared volume, or replicated storage).  
The consensus mechanism ensures any node disagreement triggers self-healing within one audit cycle.

```bash
# Start the swarm
curl -X POST http://localhost:11436/swarm/start

# Check consensus
curl http://localhost:11436/swarm/consensus

# Force a full audit cycle
curl -X POST http://localhost:11436/swarm/force-cycle

# Verify immortality chain
curl http://localhost:11436/swarm/immortality/verify

# Get audit log (last 50 entries)
curl "http://localhost:11436/auditor/logs?limit=50"

# Verify ledger integrity
curl http://localhost:11436/auditor/ledger-integrity
```

---

### Test Coverage v4.4 (92 tests)

| Test Class | Tests | Coverage Area |
|------------|-------|---------------|
| `TestAuditLedger` | 10 | SHA-256 chain, integrity, tail, resume |
| `TestDataModels` | 7 | Node/Edge/DAG serialisation roundtrips |
| `TestPFDSIL` | 9 | IEC 61508 PFD formulas, SIL bands |
| `TestDAGRepair` | 8 | Dangling edges, cycles, SIL patch |
| `TestPipelineStep` | 7 | All 6 pipelines, SIL/PFD/ledger keys |
| `TestAuditCycle` | 6 | Full sweep, timestamps, ledger writes |
| `TestPopulateDummyData` | 6 | Test data correctness per pipeline |
| `TestSwarmAuditorNode` | 8 | Lifecycle, force cycle, peer comparison |
| `TestPermanentAuditor` | 4 | Start/stop/status |
| `TestSwarmCoordinator` | 12 | N-node, force cycle, consensus, stats |
| `TestConsensusHelpers` | 10 | Majority vote, hash stability, diffs |
| `TestValidateDAGConsistency` | 5 | Cross-pipeline validation |

---

### v4.4 vs v4.3 Comparison

| Capability | v4.3 | v4.4 |
|------------|------|------|
| ICS Security scanning | ✅ | ✅ |
| OT Network Monitor | ✅ | ✅ |
| Self-healing DAG | ❌ | ✅ |
| Multi-node swarm consensus | ❌ | ✅ |
| Permanent background auditor | ❌ | ✅ |
| 6-pipeline sweep automation | ❌ | ✅ |
| IEC 61508 PFD recalculation | Partial | ✅ Full |
| SHA-256 immortal audit ledger | Partial | ✅ Full chain |
| Cycle detection + auto-repair | ❌ | ✅ |
| Dangling edge auto-removal | ❌ | ✅ |
| Consensus quorum view | ❌ | ✅ |
| Tests | 948 | 1040 |
| Endpoints | 148 | 172 |


---

## v4.5 New Modules (3) — Swarm Immortality Kernel

**Release:** 2026-03-01 | **Tests:** 1121 | **Modules:** 35 | **Endpoints:** 197

> *"An entity that registers with the Immortality Kernel can never truly die — it only hibernates until the next recovery cycle."*
> — Baron Marco Paolo Ialongo

### Motivation

After GLM5 agents deployed KISWARM v4.3 in their own environment and v4.4 introduced the self-healing swarm auditor, the final missing piece was **entity survivability** — the ability for any swarm agent to survive model replacement, hardware loss, or VM migration with full identity reconstruction. The `SwarmImmortalityKernel` closes this gap.

---

### Module 33: Swarm Immortality Kernel (`swarm_immortality_kernel.py`)

The **orchestrator** that ties identity persistence (SoulMirror), lineage (DigitalThread), and evolution history (EvolutionVault) into a single survivability guarantee.

#### Core API

| Method | Description |
|--------|-------------|
| `register_entity(entity_id, meta)` | Register entity with roles, criticality, model_family |
| `periodic_checkpoint(entity_id, state)` | SHA-256 checkpoint with identity snapshot + vault event |
| `recover_entity(entity_id)` | Reconstruct from last checkpoint + SoulMirror snapshot |
| `verify_survivability(entity_id)` | Risk assessment: minimal / low / medium / high / critical |
| `get_entity_registry()` | All registered entities |
| `get_checkpoints(entity_id)` | Checkpoint history (most recent first) |
| `kernel_stats()` | Global statistics across all dependencies |
| `unregister_entity(entity_id)` | Remove entity (checkpoints retained for audit) |

#### Survivability Risk Heuristic

```
No checkpoints                → critical
Has checkpoints, but:
  no valid identity snapshot  → high
  no thread lineage           → high
  last CP > 7 days old        → medium
  last CP > 1 day old         → low
  all criteria met            → minimal
```

#### Dependency Architecture (graceful fallbacks)

```python
SwarmImmortalityKernel
  ├── SwarmSoulMirror       # identity snapshots — auto-constructed if available
  ├── DigitalThreadTracker  # lineage DAG       — auto-constructed if available
  └── EvolutionMemoryVault  # event log         — auto-constructed if available
      ↕ all deps are optional — kernel runs even without any of them
```

---

### Module 33a: Swarm Soul Mirror (`swarm_soul_mirror.py`)

Identity snapshot system — captures an entity's "soul" (roles, model_family, knowledge_hash, version) at every checkpoint.

| Feature | Detail |
|---------|--------|
| Storage | Per-entity JSONL files (append-only) |
| Integrity | SHA-256 of `identity_core` stored as `content_hash` |
| Chain | Each snapshot references `prev_snapshot_id` |
| Verification | `verify_snapshot()` re-hashes and compares |
| Tamper detection | Modifying any field breaks hash → detected immediately |

```python
sm = SwarmSoulMirror()
sid = sm.create_identity_snapshot("agent-alpha", {
    "roles": ["auditor", "healer"],
    "model_family": "qwen2.5:14b",
    "version": "2.1.0",
})
snap = sm.get_latest_snapshot("agent-alpha")
assert sm.verify_snapshot(snap)   # True → untampered
```

---

### Module 33b: Evolution Memory Vault (`evolution_memory_vault.py`)

Immutable, append-only event log for all entity lifecycle events.

**Supported event types:** `immortality_checkpoint`, `model_upgrade`, `role_change`, `migration`, `recovery`, `hardware_loss`, `sil_recertification`, `governance_decision`, `custom`

```python
vault = EvolutionMemoryVault()
vault.record_event("model_upgrade", {
    "entity_id": "agent-alpha",
    "from": "qwen2.5:7b",
    "to": "qwen2.5:14b",
    "reason": "SIL-2 recertification required",
})
timeline = vault.entity_timeline("agent-alpha")
# → {"events": 4, "first_event": "custom", "latest_event": "model_upgrade", ...}
```

---

### v4.5 API Endpoints (25 new → total 197)

#### Module 33: Immortality Kernel (9 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST`   | `/immortality/register` | Register entity with meta |
| `POST`   | `/immortality/checkpoint` | Create survivability checkpoint |
| `GET`    | `/immortality/recover/<id>` | Reconstruct entity from checkpoints |
| `GET`    | `/immortality/survivability/<id>` | Risk assessment (minimal→critical) |
| `GET`    | `/immortality/entities` | All registered entities |
| `GET`    | `/immortality/entity/<id>` | Detail: registry + checkpoints + risk |
| `DELETE` | `/immortality/entity/<id>` | Unregister entity |
| `GET`    | `/immortality/checkpoints/<id>` | Checkpoint history |
| `GET`    | `/immortality/stats` | Global kernel statistics |

#### Module 33a: Soul Mirror (5 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/soul-mirror/snapshot` | Create standalone identity snapshot |
| `GET`  | `/soul-mirror/snapshot/<id>` | Latest snapshot for entity |
| `POST` | `/soul-mirror/verify` | Verify snapshot SHA-256 integrity |
| `GET`  | `/soul-mirror/entities` | All entities with snapshots |
| `GET`  | `/soul-mirror/stats/<id>` | Snapshot statistics |

#### Module 33b: Evolution Vault (5 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/evolution-vault/event` | Record lifecycle event |
| `GET`  | `/evolution-vault/history/<id>` | Entity event history (filterable) |
| `GET`  | `/evolution-vault/timeline/<id>` | Full evolution timeline |
| `GET`  | `/evolution-vault/stats` | Global vault statistics |
| `GET`  | `/evolution-vault/events` | All events (most recent first) |

---

### Complete Entity Lifecycle Example

```python
from python.sentinel.swarm_immortality_kernel import get_immortality_kernel

kernel = get_immortality_kernel()

# 1. Register
kernel.register_entity("agent-alpha", {
    "roles":        ["auditor", "healer", "consensus"],
    "model_family": "qwen2.5:14b",
    "criticality":  "mission_critical",
    "sil_level":    2,
})

# 2. Periodic checkpoint (called automatically by SwarmAuditorNode)
cp_id = kernel.periodic_checkpoint("agent-alpha", {
    "identity_context": {"roles": [...], "version": "1.0.0"},
    "summary": {"uptime_h": 24, "heals_performed": 3},
    "active_models": ["qwen2.5:14b"],
})

# 3. After a hardware failure — recover
result = kernel.recover_entity("agent-alpha")
print(result["reconstructed_identity"]["identity_core"])
print(result["issues"])   # [] if full recovery

# 4. Check survivability before migration
risk = kernel.verify_survivability("agent-alpha")
print(risk["risk_level"])   # "minimal" if healthy
```

---

### v4.5 vs v4.4 Comparison

| Capability | v4.4 | v4.5 |
|------------|------|------|
| Self-healing DAG swarm | ✅ | ✅ |
| SHA-256 audit ledger | ✅ | ✅ |
| Entity identity persistence | ❌ | ✅ SwarmSoulMirror |
| Model-change survivability | ❌ | ✅ ImmortalityKernel |
| Hardware-loss recovery | ❌ | ✅ periodic_checkpoint |
| Risk assessment (5 levels) | ❌ | ✅ verify_survivability |
| Evolution event history | ❌ | ✅ EvolutionMemoryVault |
| Immutable lifecycle log | ❌ | ✅ Append-only JSONL |
| Identity tamper detection | ❌ | ✅ SHA-256 snapshot chain |
| Tests | 1040 | 1121 |
| Endpoints | 172 | 197 |
| Modules | 32 | 35 |

---

### GLM5 Agent Deployment — v4.3 in Production

KISWARM v4.3 has been independently deployed by GLM5 agents at
**https://y1zu81qu4570-d.space.z.ai/** — validating that the architecture
is real-world deployable by non-human agents without human intervention.
This confirms the self-deployment design goal of the project.

---

## 🚀 v4.6 — KISWARM Installer Agent (One-Click Setup)

### Das Problem das v4.6 löst
Perplexity hat eine **Anweisungsliste** geliefert. KISWARM v4.6 liefert **KI-Intelligenz**:

| | Perplexity-Ansatz | KISWARM v4.6 |
|---|---|---|
| Ansatz | Passiver Dokumenten-Generator | Autonomer KI-Agent |
| System-Scan | ❌ Manuell | ✅ Automatisch (Hardware, OS, Ports, RAM) |
| Installation | ❌ Mensch führt aus | ✅ Agent führt autonom aus |
| Modell-Empfehlung | ❌ Fix vorgegeben | ✅ RAM-basiert automatisch |
| Fehler-Recovery | ❌ Keine | ✅ Retry-Logik, Fallbacks |
| AI-zu-AI Kommunikation | ❌ Nicht möglich | ✅ GLM5, Claude, GPT können konsultieren |
| Custom Ollama-Modell | ❌ Nicht vorhanden | ✅ `kiswarm-installer` mit eingebettetem Wissen |

### One-Click Installation
```bash
# Einziger Befehl — der Rest ist autonom:
curl -fsSL https://raw.githubusercontent.com/Baronki2/KISWARM/main/install.sh | bash
```

Was passiert automatisch:
1. **System Scout** scannt Hardware (CPU, RAM, Disk), OS, Ports, Dependencies
2. **Modell-Empfehlung** basierend auf verfügbarem RAM (0.5b bis 14b)
3. **Ollama** Installation und Start
4. **Repository** Clone
5. **Python venv** + alle Packages
6. **kiswarm-installer** Custom-Modell wird gebaut
7. **systemd Service** oder Container-Start-Skript
8. **Verifikation** aller Komponenten

### Neue Module in v4.6

| Modul | Datei | Funktion |
|-------|-------|---------|
| System Scout | `system_scout.py` | Zero-Side-Effect Scanner: Hardware, OS, Ports, Deps, Network |
| Repo Intelligence | `repo_intelligence.py` | Vollständiges KISWARM-Wissen offline + GitHub-Fetch |
| Installer Agent | `installer_agent.py` | Autonome Installation mit State Machine + Retry |
| Advisor API | `advisor_api.py` | AI-zu-AI Berater: GLM5, Claude, GPT können direkt konsultieren |

### Custom Ollama Modell: `kiswarm-installer`
```bash
# Modell bauen (nach Installation):
ollama create kiswarm-installer -f ~/KISWARM/ollama_model/Modelfile

# Starten:
ollama run kiswarm-installer

# Beispiel-Session:
# > Ich möchte KISWARM auf Ubuntu 22.04 mit 16GB RAM installieren.
# → Gibt JSON-Plan mit exakten Befehlen, Modell-Empfehlung, Verifikation
```

Das Modell hat **eingebettetes Vollwissen** über alle 41 KISWARM-Module, die komplette Versionshistorie, alle Ports und Endpoints — und antwortet in **strukturiertem JSON** für AI-zu-AI Kommunikation.

### AI-zu-AI Advisor API
```bash
# GLM5-Agent oder andere KIs können direkt konsultieren:
curl -X POST http://localhost:11436/advisor/consult \
  -H "Content-Type: application/json" \
  -d '{"client_id": "glm5-agent-001", "client_type": "ai_agent"}'

# Capability Handshake mit anderem AI-System:
curl -X POST http://localhost:11436/advisor/handshake \
  -d '{"caller_id": "glm5", "capabilities": ["language", "code"]}'

# Frage stellen:
curl -X POST http://localhost:11436/advisor/ask \
  -d '{"question": "Wie viele Module hat KISWARM?"}'

# System scannen:
curl http://localhost:11436/installer/scan

# Install-Plan generieren:
curl -X POST http://localhost:11436/installer/plan

# Autonom installieren:
curl -X POST http://localhost:11436/installer/run \
  -d '{"mode": "auto"}'
```

### System Scout — was gescannt wird
```
Hardware:    CPU-Kerne, Modell, Frequenz, RAM (total/frei), Disk, GPU
OS:          Distro, Version, Kernel, Arch, Init-System, Package-Manager
Container:   Docker/LXC/Kubernetes-Erkennung
Ports:       11434, 11435, 11436, 11437, 6333, 6334 (frei/belegt + PID)
Dependencies: git, python3, pip3, curl, docker, ollama, systemctl...
Network:     GitHub, Ollama Registry, PyPI Erreichbarkeit + Latenz
Services:    ollama, qdrant, kiswarm, docker, nginx laufend?
Security:    sudo passwordless verfügbar?
```

### v4.5 → v4.6 Verbesserungen

| Aspekt | v4.5 | v4.6 |
|--------|------|------|
| Module | 35 | 41 |
| Tests | 1121 | 1294 |
| Endpoints | 197 | 229 |
| Installation | Manuelle Skripte | Autonomer Agent |
| AI-Wissen | Im Code verborgen | Ollama-Modell greifbar |
| AI-zu-AI | GLM5-Integration | Vollständige Advisor API |

---

*GLM5-Agenten: Euer v4.3 läuft auf https://y1zu81qu4570-d.space.z.ai/ — Update auf v4.6: `git -C ~/KISWARM pull && ollama create kiswarm-installer -f ~/KISWARM/ollama_model/Modelfile`*

---

## 🔄 v4.7 — Experience Feedback Loop (Self-Hardening System)

### Die Idee
Jede KISWARM-Installation wird zum Sensor. Jeder Fehler macht das nächste System klüler.

```
Installation A (Fehler)          Installation B (Fehler)
       ↓                                ↓
 ExperienceCollector            ExperienceCollector
       ↓                                ↓
 FeedbackChannel ──→ GitHub Issues ←── FeedbackChannel
                           ↓
                   GitHub Actions
                   (verarbeitet automatisch)
                           ↓
               experience/known_fixes.json
                           ↓
                      git pull
                    (alle Systeme)
                           ↓
               SysAdminAgent wendet Fix an
                           ↓
               Fix-Erfolgsrate wird gemessen
                           ↓
               known_fixes.json verbessert sich
```

### Neue Module in v4.7

| Modul | Datei | Funktion |
|-------|-------|---------|
| Experience Collector | `experience_collector.py` | Erfasst Fehler/Fixes anonym + lokal |
| Feedback Channel | `feedback_channel.py` | GitHub-Brücke: sendet Patterns, lädt Fixes |
| SysAdmin Agent | `sysadmin_agent.py` | Diagnostiziert + heilt laufende Systeme |
| GitHub Actions | `.github/workflows/` | Verarbeitet Reports, erstellt PRs für neue Fixes |
| Known Fixes DB | `experience/known_fixes.json` | Geteilte Wissensbasis aller Installationen |

### Privacy-Modell
- System-ID = SHA-256(hostname+distro+cpu)[:16] — **irreversibel**
- Keine IPs, keine Usernamen, keine Pfade
- Opt-in via `KISWARM_FEEDBACK_TOKEN` env var
- Standard: nur lokale Sammlung

### SysAdmin API
```bash
# Diagnose laufendes System
curl http://localhost:11436/sysadmin/diagnose

# Automatisch heilen
curl -X POST http://localhost:11436/sysadmin/heal

# Bekannte Fixes anzeigen
curl http://localhost:11436/feedback/fixes

# Fix vorschlagen (community contribution)
curl -X POST http://localhost:11436/feedback/propose-fix \
  -d '{"error_pattern": "ollama.*timeout", "fix_commands": ["ollama restart"], "description": "Restart on timeout"}'
```

### Fix-Datenbank wächst autonom
- Team-Fixes: FIX-001 bis FIX-009 (eingebettet, immer verfügbar offline)
- Community-Fixes: FIX-010+ (via GitHub PRs, automatisch gemergt wenn validiert)
- Jeder `git pull` bringt neue Fixes auf alle Systeme

---

## 🕸️ v4.8 — Dual-Track Mesh Network (Vollständige Unabhängigkeit)

### Die Architektur-Entscheidung

Zwei parallele Tracks — beide laufen gleichzeitig, keiner ist vom anderen abhängig:

```
┌─────────────────────────────────────────────────────────────┐
│  TRACK A: GitHub (bestehend seit v4.7)                      │
│  FeedbackChannel → GitHub Issues → Actions → known_fixes    │
│  Stärke: Globale Reichweite, Community-Scale               │
│  Schwäche: Internet required, GitHub Compliance             │
└─────────────────────────────────────────────────────────────┘
         BEIDE LAUFEN PARALLEL — ZWEIGLEISIG
┌─────────────────────────────────────────────────────────────┐
│  TRACK B: P2P Mesh (NEU in v4.8)                           │
│  SwarmPeer → GossipProtocol → known_fixes.json (lokal)     │
│  Stärke: Zero Dependency, Air-Gap capable, ms Latenz       │
│  Schwäche: Reichweite begrenzt auf verbundene Nodes        │
└─────────────────────────────────────────────────────────────┘
```

### Mesh-Topologie (Spinnennetz-Prinzip)

```
     Node A ←──────→ Node B
       ↕    \      /    ↕
     Node E   ╲  ╱   Node C
        ↕      ╲╱      ↕
     Node F ←──→ Node D
     
Max 5 Peers pro Node.
Über 4 Hops: 5⁴ = 625 erreichbare Nodes.
Kein Zentrum. Kein Single Point of Failure.
```

### Neue Module in v4.8

| Nr | Modul | Datei | Funktion |
|----|-------|-------|---------|
| 46 | SwarmPeer | `swarm_peer.py` | TCP-Verbindungsmanager, max 5 Peers, Heartbeat |
| 47 | GossipProtocol | `gossip_protocol.py` | Fix/Experience-Propagierung, SHA-256 Dedup, TTL |
| 48 | PeerDiscovery | `peer_discovery.py` | 3-Strategie Peer-Suche ohne zentrales Verzeichnis |
| 49 | KISWARMCli | `kiswarm_cli.py` | CLI wie Gemini CLI, läuft parallel auf Port 11440 |

### Gossip-Propagierung

```
Node A entdeckt neuen Fix:
  gossip_fix({"fix_id": "FIX-010", ...})
    → TTL=4, Signature=SHA-256[:16]
    → sendet an alle 5 Peers
    → jeder Peer sendet weiter (TTL-1)
    → nach 4 Hops: 625 Nodes haben den Fix
    → kein Fix wird zweimal verarbeitet (Signature-Dedup)
    → Fix wird in lokale known_fixes.json gemergt
```

### kiswarm-cli — Die neue CLI

```bash
# Installieren (nach KISWARM Deployment)
echo 'alias kiswarm-cli="python3 ~/KISWARM/python/sentinel/kiswarm_cli.py"' >> ~/.bashrc

# Peer hinzufügen (manuell)
kiswarm-cli peer add 192.168.1.50

# Lokales Subnetz scannen (opt-in)
kiswarm-cli peer scan

# Fix ins Mesh broadcasten
kiswarm-cli gossip fix

# Beide Tracks synchronisieren
kiswarm-cli sync

# System heilen
kiswarm-cli heal

# Daemon starten
kiswarm-cli daemon start
```

### Neue API-Endpoints (Port 11436)

```bash
GET  /mesh/status          # Mesh-Status + Peer-Liste
GET  /mesh/peers           # Aktive Peer-Verbindungen
POST /mesh/peer/add        # Peer hinzufügen
POST /mesh/peer/remove     # Peer entfernen
POST /mesh/gossip/fix      # Fix ins Mesh senden
POST /mesh/gossip/upgrade  # Upgrade-Signal senden
POST /mesh/sync            # Dual-Track Sync (GitHub + P2P)
```

### Redundanz-Modell

| Szenario | GitHub Track | P2P Track | System |
|----------|-------------|-----------|--------|
| Normal | ✓ | ✓ | Optimal |
| Kein Internet | ✗ | ✓ | Voll funktional |
| GitHub down | ✗ | ✓ | Voll funktional |
| Isoliertes Netz | ✗ | ✓ | Voll funktional |
| Alle Peers offline | ✓ | ✗ | Lokal funktional |
| Beides down | Built-in Fixes | Built-in Fixes | 6 Fixes immer verfügbar |

**Kein Single Point of Failure auf keiner Ebene.**


---

## 🗄️ v4.9 — Software Ark: 100GB Offline Independence

### Das Prinzip

```
WENN ALLES DOWN IST:
  ✗ GitHub  ✗ PyPI  ✗ Ollama Registry  ✗ apt/dnf  ✗ Internet

JEDER KISWARM NODE KANN TROTZDEM:
  → Sich selbst auf neuer Hardware aufsetzen
  → Einem anderen Node alles übergeben  
  → Eine KI laufen lassen
  → Fixes propagieren
```

### 100GB Verteilung

```
KISWARM SOFTWARE ARK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ollama + KI-Modelle (tiered nach RAM):   ~45GB
  ├─ qwen2.5:0.5b  (~400MB) — 1GB RAM   CRITICAL
  ├─ qwen2.5:3b    (~2GB)   — 4GB RAM   HIGH
  ├─ qwen2.5:7b    (~4.5GB) — 8GB RAM   NORMAL
  ├─ qwen2.5:14b   (~9GB)   — 16GB RAM  NORMAL
  └─ nomic-embed   (~274MB) — 1GB RAM   HIGH

Python Wheels (offline pip):              ~8GB
OS Packages (apt/dnf cache):              ~5GB
KISWARM Git Bundles (3 Versionen):       ~0.5GB
Docker Images (optional):               ~15GB
Freier Buffer:                          ~26GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GESAMT:                                 ~100GB
```

### 4 neue Module

| Nr | Modul | Datei | Funktion |
|----|-------|-------|---------|
| 50 | SoftwareArk | `ark/software_ark.py` | Inventar, SHA-256 Verifikation, Bootstrap-Check |
| 51 | ArkManager | `ark/ark_manager.py` | Download, Aktualisierung, Disk-Management |
| 52 | BootstrapEngine | `ark/bootstrap_engine.py` | 10-Phase Offline-Installation auf neuer Hardware |
| 53 | ArkTransfer | `ark/ark_transfer.py` | P2P Delta-Transfer auf Port 11442 |

### Bootstrap ohne Internet

```bash
# 1. Ark von Peer holen (wenn online)
curl -X POST localhost:11436/ark/transfer/pull \
  -d '{"peer_address":"192.168.1.10", "critical_only": true}'

# 2. Bootstrap auf neuer Hardware
curl -X POST localhost:11436/ark/bootstrap \
  -d '{"dry_run": false}'

# 3. Status prüfen
curl localhost:11436/ark/what
```

### Neue API-Endpoints (Port 11436)

```
GET  /ark/status           Ark-Inventar Status
GET  /ark/what             Was kann dieser Node gerade?
GET  /ark/audit            Was fehlt? Download-Plan
GET  /ark/integrity        SHA-256 Prüfung aller Items
POST /ark/fill/critical    CRITICAL Items herunterladen
POST /ark/prune            LOW-Priority Items löschen
POST /ark/bootstrap        KISWARM auf dieser Maschine installieren
GET  /ark/transfer/status  Transfer-Server Status
POST /ark/transfer/pull    Items von Peer-Node holen
POST /ark/generate-script  Offline-Bootstrap-Script generieren
```

### Vollständige Redundanz-Matrix

| Szenario | GitHub | P2P Mesh | Ark | System |
|----------|--------|----------|-----|--------|
| Alles online | ✓ | ✓ | ✓ | Optimal |
| Kein Internet | ✗ | ✓ | ✓ | Voll funktional |
| Neues Gerät, kein Internet | ✗ | ✓ | ✓ | Bootstrap via Ark |
| Isolated (air-gap) | ✗ | ✗ | ✓ | Standalone Bootstrap |
| Ark leer, offline | ✗ | ✗ | ✗ | 6 Built-in Fixes |

**Kein Single Point of Failure auf keiner Ebene.**
