# 🏭 KISWARM v4.1 — Industrial AI Governance Platform

> **Cognitive Industrial Evolution Core with TD3 Reinforcement Learning, Full IEC 61131-3 AST Parsing, Formal Stability Verification, VMware Orchestration, Byzantine-Tolerant Federated Aggregation, and 11-Step Mutation Governance Pipeline**

[![Tests](https://img.shields.io/badge/tests-572%20passing-brightgreen)]()
[![Modules](https://img.shields.io/badge/modules-23-blue)]()
[![Endpoints](https://img.shields.io/badge/endpoints-99-blue)]()
[![Version](https://img.shields.io/badge/version-4.1-orange)]()

---

## Version History

| Version | Modules | Endpoints | Tests | Key Features |
|---------|---------|-----------|-------|--------------|
| v2.1 | 11 | 29 | 180 | Firewall, Decay, Ledger, Conflict |
| v3.0 | 15 | 42 | 280 | Fuzzy PID, Constrained RL, Digital Twin |
| v4.0 | 16 | 59 | 430 | PLC Parser, SCADA, CIEC Twin, Knowledge Graph |
| **v4.1** | **23** | **99** | **572** | **TD3, AST, Physics, VMware, Formal, Byzantine, Governance** |

---

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

---

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

---

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

---

## TD3 Controller Hyperparameters

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

**Reward:** R = 0.40·stability + 0.30·efficiency − 0.15·cycles − 0.10·violation − 0.05·oscillation

---

## Formal Verification

**Lyapunov (Method A — Linearized):**
```
Solve:  AᵀPA − P = −Q   (discrete-time Stein equation)
Approve if:  ρ(A) < 1  AND  P is positive definite
```

**Barrier Certificate (Method B — Nonlinear):**
```
Sample 500 points in safe set S
Check:  B(x) ≥ 0  AND  dB/dt ≤ 0  at every point
Approve only if:  zero violations
```

Every decision stored in cryptographic ledger with SHA-256 chain.

---

## Byzantine Federated Aggregation

```
Condition:  N ≥ 3f + 1  (N sites, f Byzantine tolerated)
Methods:    trimmed_mean | krum | median | fltrust
Update:     θ ← θ − η · robust_mean(gᵢ)
Privacy:    No raw plant data leaves any site — gradients only
```

---

## VMware Safety Rules (Hard-Coded)

1. **VM-A, VM-B** (production) → AI may only READ, never mutate
2. All mutations run on **network-isolated CLONES** of test VMs
3. Every operation → **immutable SHA-256 audit entry**
4. Promotion requires `Maquister_Equtitum` authorization code
5. No direct ESXi root access from AI layer

---

## REST API (99 Endpoints)

```
v2.1 Sentinel Intelligence   17 endpoints  /sentinel /firewall /decay /ledger /conflict /tracker /guard
v3.0 Industrial AI           13 endpoints  /fuzzy /rl /twin /mesh
v4.0 CIEC Core               28 endpoints  /plc /scada /ciec-twin /constraints /kg /ciec-rl
v4.1 Advanced CIEC           41 endpoints  /td3 /ast /physics /vmware /formal /federated /governance
```

**Start API:**
```bash
python -m python.sentinel.sentinel_api
# → http://127.0.0.1:11436
```

**Quick examples:**
```bash
# TD3 action
curl -X POST http://localhost:11436/td3/act \
  -d '{"state":[0.5,0.3], "deterministic":true}'

# Parse IEC 61131-3 ST
curl -X POST http://localhost:11436/ast/parse \
  -d '{"source":"PROGRAM P\nVAR x:REAL; END_VAR\nx:=1.0;\nEND_PROGRAM"}'

# Lyapunov stability check
curl -X POST http://localhost:11436/formal/lyapunov \
  -d '{"A":[[0.5,0.1],[0.0,0.6]],"mutation_id":"MUT_001"}'

# Start mutation governance pipeline
curl -X POST http://localhost:11436/governance/begin \
  -d '{"plc_program":"PumpCtrl","param_deltas":{"delta_kp":0.02}}'
```

---

## Test Coverage

```bash
python -m pytest tests/ --ignore=tests/test_deploy.py -q
# 572 passed
```

| Suite | Tests | Modules |
|-------|-------|---------|
| test_sentinel.py | 80 | v2.1 (6) |
| test_industrial.py | 90 | v3.0 (4) |
| test_ciec.py | 170 | v4.0 (6) |
| test_v41_modules.py | **142** | **v4.1 (7)** |
| test_api.py | 90 | REST API |

---

## Project Structure

```
KISWARM/
├── python/sentinel/
│   ├── sentinel_api.py           99 REST endpoints
│   ├── td3_controller.py         NEW v4.1
│   ├── ast_parser.py             NEW v4.1
│   ├── extended_physics.py       NEW v4.1
│   ├── vmware_orchestrator.py    NEW v4.1
│   ├── formal_verification.py    NEW v4.1
│   ├── byzantine_aggregator.py   NEW v4.1
│   ├── mutation_governance.py    NEW v4.1
│   └── [16 v2.1-v4.0 modules]
├── tests/
│   ├── test_v41_modules.py       142 tests NEW
│   └── [v2.1–v4.0 test suites]
└── README.md
```

---

## License

MIT License — © Baron Marco Paolo Ialongo

*KISWARM v4.1 · 23 Modules · 99 Endpoints · 572 Tests · Production Ready*

---

# 🧠 KISWARM v4.2 — Explainability · Predictive Maintenance · Multi-Agent · SIL · Digital Thread

> **v4.2 adds full AI explainability (XAI/SHAP), predictive maintenance with RUL, multi-agent plant coordination, IEC 61508 SIL verification, and end-to-end digital thread traceability.**

[![Tests](https://img.shields.io/badge/tests-729%20passing-brightgreen)]()
[![Modules](https://img.shields.io/badge/modules-28-blue)]()
[![Endpoints](https://img.shields.io/badge/endpoints-133-blue)]()
[![Version](https://img.shields.io/badge/version-4.2-orange)]()

---

## v4.2 New Modules (5)

| # | Module | File | Description |
|---|--------|------|-------------|
| 24 | Explainability Engine (XAI) | `explainability_engine.py` | KernelSHAP attribution for every AI decision |
| 25 | Predictive Maintenance | `predictive_maintenance.py` | RUL prediction + degradation curves + fleet management |
| 26 | Multi-Agent Coordinator | `multiagent_coordinator.py` | N×TD3 agents with 3-phase consensus + conflict resolution |
| 27 | SIL Verification Engine | `sil_verification.py` | IEC 61508 PFD/SIL assessment + mutation impact |
| 28 | Digital Thread Tracker | `digital_thread.py` | End-to-end traceability DAG + compliance checks |

---

## Module 24 — Explainability Engine (XAI)

Every AI decision in KISWARM is now explainable and auditable.

**KernelSHAP** (pure Python, no external ML libs):
- Samples 2ⁿ coalitions (or n_samples for large n)
- SHAP kernel weighting: `w = (n-1) / (C(n,s) · s · (n-s))`
- Weighted least-squares via Gauss elimination with regularisation
- Counterfactual "what-if": ±10% per top-3 feature

**Explanation types:**

| Type | Source | Features |
|------|--------|----------|
| `td3_action` | TD3 actor Q-output | State vector dimensions |
| `formal_verify` | Lyapunov result | spectral_radius, lyapunov_margin, P_pos_def, converged |
| `governance` | Evidence chain | Step pass/fail weighted by recency |
| `physics` | Twin episode | Plant state variables |
| `generic` | Any callable | Any feature list |

**Immutable Explanation Ledger** — SHA-256 chain, `verify_integrity()` detects tampering.

```bash
# Explain a TD3 action
curl -X POST http://localhost:11436/xai/explain-td3 \
  -d '{"state":[0.5,0.3,0.8,0.1,0.6,0.2,0.9,0.4]}'
# → {"top_features":["state_7","state_6","state_4"], "natural_language":"..."}
```

---

## Module 25 — Predictive Maintenance Engine (PdM)

**Asset Classes:** pump · motor · valve · bearing · electrical · compressor · heat_exchanger

**Degradation Models:**

| Model | Formula | Best for |
|-------|---------|----------|
| Linear | HI = 1 − k·t | Valves, simple wear |
| Exponential | HI = exp(−αt) | Motors, electrical |
| Sigmoid | HI = 1/(1+exp(k·(t−0.7))) | Pumps — holds then drops |

**LSTM Recurrent Health Model** — 16-unit LSTM cell (pure Python), tracking temporal sensor evolution across readings.

**Alarm Levels:**

| Level | HI Range | Action |
|-------|----------|--------|
| healthy | 0.6–1.0 | Normal operation |
| warning | 0.3–0.6 | Plan maintenance |
| critical | 0.1–0.3 | Schedule within days |
| failed | 0.0–0.1 | **IMMEDIATE SHUTDOWN** |

**RUL Monte Carlo** — 100 samples with σ = max(0.02, (1-HI)·0.1) → 10th/90th percentile CI.

**Maintenance Scheduling** — minimises `risk_cost - planned_cost` across fleet, sorted by urgency.

```bash
curl -X POST http://localhost:11436/pdm/ingest \
  -d '{"asset_id":"pump_1","hour":500,"temperature":75,"vibration":3.5,"current_draw":55,"pressure_drop":1.5,"efficiency":0.78}'
# → {"health_index": 0.71, "alarm_level": "healthy", "anomaly_score": 0.4}
```

---

## Module 26 — Multi-Agent Plant Coordinator

**Architecture:** N independent `SectionAgent` (lightweight 2-hidden-layer actor) + `ConsensusProtocol`.

**Default plant sections:**

| Section | Priority | Power (kW) | Cooling (m³/h) |
|---------|----------|------------|----------------|
| pump_station | 1 | 75 | 5 |
| reactor | 2 | 120 | 20 |
| separator | 3 | 30 | 8 |
| compressor | 2 | 200 | 15 |
| heat_exchanger | 4 | 10 | 25 |

**Shared resource limits:** 500 kW total power · 80 m³/h cooling · 8 bar compressed air

**3-Phase Consensus Protocol:**
1. Each agent proposes action + resource demand independently
2. Proposals broadcast on `CoordinatorBus` (in-process pub/sub)
3. `ConflictResolver` processes by priority×Q-value; over-budget proposals scaled down
4. Committed: full action; Arbitrated: scaled action + Q-penalty
5. `RewardShaper` applies: `R_shaped = R_local − 0.5·conflict_penalty + 0.2·coord_bonus`

```bash
curl -X POST http://localhost:11436/coordinator/step \
  -d '{"states":{"pump_station":[0.5,0.3,0.8,0.1,0.6,0.2,0.9,0.4]}}'
# → {"consensus":{"n_conflicts":0, "coordination_bonus":1.0, ...}}
```

---

## Module 27 — IEC 61508 SIL Verification Engine

**SIL Levels (PFD on Demand):**

| SIL | PFD Range | Risk Reduction |
|-----|-----------|----------------|
| 1 | 10⁻² – 10⁻¹ | 10–100× |
| 2 | 10⁻³ – 10⁻² | 100–1,000× |
| 3 | 10⁻⁴ – 10⁻³ | 1,000–10,000× |
| 4 | 10⁻⁵ – 10⁻⁴ | 10,000–100,000× |

**PFD Formulas (IEC 61508 Annex B):**

| Architecture | Formula |
|-------------|---------|
| 1oo1 | λ_d · T_i / 2 |
| 1oo2 | (1−β)² · λ_d² · T_i² / 3 + β · λ_d · T_i / 2 |
| 2oo3 | 3(1−β)² · λ_d² · T_i² / 3 + β · λ_d · T_i / 2 |
| 1oo3 | (1−β)³ · λ_d³ · T_i³ / 4 + β · λ_d · T_i / 2 |

**HFT Requirements** (IEC 61508-2 Table): minimum Hardware Fault Tolerance per SIL × SFF range.

**Mutation Impact Analysis:** Conservative model — total param delta × 0.5 sensitivity factor on λ_d. Auto-rejects if SIL degrades.

```bash
curl -X POST http://localhost:11436/sil/assess \
  -d '{"sif_id":"SIF_PUMP","sil_required":2,"subsystems":[
    {"subsystem_id":"sensor","architecture":"1oo2","lambda_d":1e-6,"lambda_s":2e-6,
     "mttf_hours":100000,"mttr_hours":8,"proof_test_interval_hours":8760,"dc":0.90,"hw_fault_tolerance":1}
  ]}'
# → {"sil_achieved":2, "compliant":true, "pfd_total":"3.40e-04"}
```

---

## Module 28 — Digital Thread Tracker

**The Digital Thread** links every artefact from first design to deployed PLC — enabling root-cause analysis, regulatory audit, and AI decision provenance.

**Node Types (14):**
`design_spec · simulation · formal_cert · governance · mutation · plc_build · test_result · deployment · alert · sil_assessment · xai_explanation · physics_episode · ast_parse`

**Edge Types (9):**
`derived_from · verified_by · tested_by · deployed_as · approved_by · caused · supersedes · implements · references`

**Compliance Standards:**

| Standard | Required Node Types | Required Edge Types |
|----------|--------------------|--------------------|
| IEC 61508 | design_spec · simulation · sil_assessment · formal_cert · test_result · deployment | verified_by · tested_by · deployed_as |
| IEC 62443 | design_spec · governance · test_result · deployment | approved_by · tested_by · deployed_as |
| NAMUR NE 175 | design_spec · simulation · formal_cert · xai_explanation · governance · deployment | verified_by · approved_by · derived_from |

**Lineage Queries:** `ancestors()`, `descendants()`, `impact_path()`, `mutation_lineage()` — all BFS on the DAG.

```bash
# Build a mutation thread
curl -X POST http://localhost:11436/thread/node \
  -d '{"node_type":"mutation","title":"Kp +2% on PumpCtrl","payload":{"delta_kp":0.02}}'

curl -X POST http://localhost:11436/thread/compliance \
  -d '{"standard":"namur_ne175"}'
# → {"compliant":true/false, "missing_node_types":[...]}
```

---

## v4.2 REST API — 34 New Endpoints

```
XAI (6):        /xai/explain-td3  /xai/explain-formal  /xai/explain-governance
                /xai/explain  /xai/ledger  /xai/stats

PdM (7):        /pdm/register  /pdm/ingest  /pdm/rul/<id>  /pdm/schedule
                /pdm/maintenance  /pdm/fleet  /pdm/stats

Coordinator(7): /coordinator/sections  /coordinator/add-section  /coordinator/step
                /coordinator/rewards  /coordinator/history  /coordinator/agents
                /coordinator/stats

SIL (5):        /sil/assess  /sil/mutation-impact  /sil/assessment/<id>
                /sil/impact-log  /sil/stats

Thread (10):    /thread/node  /thread/edge  /thread/node/<id>
                /thread/ancestors/<id>  /thread/descendants/<id>
                /thread/lineage/<id>  /thread/compliance
                /thread/find  /thread/stats
```

**Total: 99 → 133 endpoints**

---

## v4.2 Test Coverage

```bash
python -m pytest tests/ --ignore=tests/test_deploy.py -q
# 729 passed
```

| Suite | Tests | Modules |
|-------|-------|---------|
| test_v42_modules.py | **157** | **v4.2 (5 new)** |
| test_v41_modules.py | 142 | v4.1 (7) |
| test_ciec.py | 170 | v4.0 (6) |
| test_industrial.py | 90 | v3.0 (4) |
| test_sentinel.py | 80 | v2.1 (6) |
| test_api.py | 90 | REST API |

---

## Complete Version History

| Version | Modules | Endpoints | Tests | Key Additions |
|---------|---------|-----------|-------|---------------|
| v2.1 | 11 | 29 | 180 | Firewall, Decay, Ledger, Conflict |
| v3.0 | 15 | 42 | 280 | Fuzzy PID, Constrained RL, Digital Twin |
| v4.0 | 16 | 59 | 430 | PLC Parser, SCADA, CIEC Twin, Knowledge Graph |
| v4.1 | 23 | 99 | 572 | TD3, AST, Physics, VMware, Formal, Byzantine, Governance |
| **v4.2** | **28** | **133** | **729** | **XAI, PdM, MultiAgent, SIL, DigitalThread** |

---

*KISWARM v4.2 · 28 Modules · 133 Endpoints · 729 Tests · Production Ready*
*© Baron Marco Paolo Ialongo*
