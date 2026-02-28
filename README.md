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
