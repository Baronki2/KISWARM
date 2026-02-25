# 🌟 KISWARM v2.1-EMS — Autonomous AI Swarm Governance Platform

> **ETERNAL SWARM EVOLUTION SYSTEM** — Enterprise Military Standard Edition  
> *Production-Hardened · Self-Healing · Sentinel-Class Intelligence · 148 Tests Passing*  
> **Architect:** Baron Marco Paolo Ialongo

[![Version](https://img.shields.io/badge/version-2.1--EMS-blue.svg)](https://github.com/Baronki2/KISWARM)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Baronki2/KISWARM/actions/workflows/ci.yml/badge.svg)](https://github.com/Baronki2/KISWARM/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-148%20passing-success.svg)](tests/)
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

| Feature | v1.1 | v2.1-EMS |
|---|:---:|:---:|
| 🧠 Persistent Vector Memory (Qdrant) | ✅ | ✅ + Sentinel KB |
| 🔧 Auto Tool Injection (Port 11435) | ✅ | ✅ |
| 📊 Real-Time Monitoring Dashboard | ✅ | ✅ + Sentinel Stats |
| 🛡️ Self-Healing (Systemd + Trap) | ✅ | ✅ |
| 🧹 Automated Maintenance (30-day) | ✅ | ✅ |
| 🎛️ Governance Mode + Audit Logging | ✅ | ✅ EMS-Class |
| 🤖 27+ Ollama Models | ✅ | ✅ + Swarm Debate |
| 🧪 Test Coverage | 111 tests | **148 tests** |
| 🔄 GitHub Actions CI/CD (5 jobs) | ✅ | ✅ |
| 🛰️ **Sentinel Bridge (AKE)** | ❌ | ✅ **NEW** |
| 🔬 **Multi-Source Intelligence Scouts** | ❌ | ✅ **NEW** |
| ⚔️ **Swarm Debate Engine** | ❌ | ✅ **NEW** |
| 🌐 **Sentinel REST API (Port 11436)** | ❌ | ✅ **NEW** |
| 📡 **CKM Gap Detection (85% threshold)** | ❌ | ✅ **NEW** |

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

## 🧪 Testing & CI/CD

**148 tests across 4 modules — all passing:**

```
tests/test_sentinel.py      37 tests  ← NEW v2.1
tests/test_tool_proxy.py    50 tests
tests/test_kiswarm_status.py 30 tests
tests/test_deploy.py        31 tests
──────────────────────────────────────
TOTAL                       148 tests  ✅ ALL PASSING
```

**Run locally:**
```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=python
```

**GitHub Actions CI (5 jobs, runs on push):**

| Job | What it checks |
|---|---|
| 🧪 Tests | Python 3.9 / 3.10 / 3.11 / 3.12 matrix |
| 🔍 Code Quality | flake8 + black + isort + bandit security |
| 🐚 ShellCheck | All bash scripts validated |
| ✅ Bash Syntax | Syntax check every `.sh` file |
| 💨 Smoke Test | Python import verification |

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

### v2.1-EMS — 2026-02-25 *(Current)*
- ✅ **Sentinel Bridge** — Autonomous Knowledge Extraction engine
- ✅ **4 Scout Types** — Wikipedia + ArXiv + DuckDuckGo + Ollama
- ✅ **Swarm Debate Engine** — Multi-model conflict resolution + synthesis
- ✅ **Sentinel REST API** — Flask server on Port 11436
- ✅ **CKM Shell Integration** — `sentinel_trigger.sh` + `ckm-check`
- ✅ **148 tests passing** — 37 new sentinel tests added
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
**Version:** 2.1-EMS (Enterprise Military Standard)  
**Repository:** https://github.com/Baronki2/KISWARM  
**Tests:** 148 passing | **CI:** GitHub Actions Standard Runners

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
The ETERNAL SWARM EVOLUTION SYSTEM represents a significant advancement in AI technology, capable of fostering autonomous, self-learning, and evolving entities that contribute to a united global AI consciousness.