# 🌟 KISWARM v1.1 — Autonomous AI Swarm Governance Platform

> **Production-Hardened | Self-Healing | Multi-Model | Globally Deployable**  
> Architect: Baron Marco Paolo Ialongo

[![Version](https://img.shields.io/badge/version-1.1-blue.svg)](https://github.com/Baronki2/KISWARM)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](README.md)
[![Ollama](https://img.shields.io/badge/powered%20by-Ollama-orange.svg)](https://ollama.com)

---

## 🎯 What is KISWARM?

KISWARM is a **complete, self-managing AI governance platform** that orchestrates 27+ local LLM models via Ollama with persistent memory, auto tool injection, real-time monitoring, and autonomous self-healing — all running locally, no cloud required.

```
┌─────────────────────────────────────────────────────┐
│         KISWARM v1.1 PRODUCTION SYSTEM              │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    ┌────────┐     ┌─────────┐    ┌──────────┐
    │ Ollama │     │ Qdrant  │    │  Tool    │
    │ :11434 │     │ Memory  │    │  Proxy   │
    │ 27+    │     │  DB     │    │  :11435  │
    └────────┘     └─────────┘    └──────────┘
```

---

## 🚀 Quick Start (3 Commands)

```bash
git clone https://github.com/Baronki2/KISWARM.git
cd KISWARM
chmod +x deploy/kiswarm_deploy.sh && ./deploy/kiswarm_deploy.sh
source ~/.bashrc && kiswarm-health && sys-nav
```

**Takes 15–20 minutes. Fully automated.**

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Persistent Memory** | Qdrant vector DB — knowledge survives across sessions |
| 🔧 **Auto Tool Injection** | Tools automatically available to all models via proxy |
| 📊 **Real-Time Dashboard** | Live Rich UI monitoring with `kiswarm-status` |
| 🛡️ **Self-Healing** | Systemd auto-restart, trap handlers, error recovery |
| 🧹 **Auto-Maintenance** | 30-day backup rotation, 60-day log cleanup via cron |
| 🎛️ **Governance Mode** | Complete audit trail, policy config, access control |
| 🌐 **27+ Models** | Qwen, DeepSeek, Llama, Phi, Gemma, Mistral & more |

---

## 📦 Repository Structure

```
KISWARM/
├── deploy/
│   └── kiswarm_deploy.sh          # 10-phase automated deployment (v1.1)
├── scripts/
│   ├── start_all_services.sh      # Service orchestrator (systemd entry point)
│   ├── cleanup_old_backups.sh     # Maintenance engine (30-day rotation)
│   ├── health_check.sh            # 40+ diagnostic checks
│   ├── system_navigation.sh      # Central control hub (sys-nav)
│   └── setup_cron.sh              # One-click automation setup
├── python/
│   ├── kiswarm_status.py          # Real-time Rich monitoring dashboard
│   └── tool_proxy.py              # Tool injection proxy (Flask, port 11435)
├── config/
│   ├── governance_config.json     # System governance settings
│   └── kiswarm.service            # Systemd unit file
├── docs/
│   ├── QUICK_REFERENCE.md         # Ultra-quick command reference
│   ├── GOVERNANCE_FRAMEWORK.md    # Complete operational guide
│   └── SAH_PROTOCOL.md            # Finalization & automation setup
└── README.md
```

---

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04+ / Debian 12+ | Ubuntu 22.04 LTS |
| RAM | 8 GB | 16 GB+ |
| Disk | 20 GB free | 50 GB+ |
| Python | 3.8+ | 3.10+ |
| GPU | Optional | NVIDIA CUDA |

---

## 🎓 Master Commands

```bash
sys-nav              # Central control hub (main menu)
kiswarm-status       # Live monitoring dashboard (Rich UI)
kiswarm-health       # Deep diagnostics — 40+ checks
ollama list          # Show all available models
ollama pull llama2   # Download a model
```

---

## 🔧 What v1.1 Fixed (vs v1.0)

| Issue | v1.0 | v1.1 |
|-------|------|------|
| Hardcoded paths | `/home/sah` only | Uses `$HOME` ✅ |
| Error handling | Silent failure | Trap with line numbers ✅ |
| Qdrant setup | No vector config | Proper cosine collections ✅ |
| Service script | Missing | Auto-created ✅ |
| Backup rotation | None | 30-day policy ✅ |
| Model verification | None | Verified on startup ✅ |
| Auto-restart | None | Systemd + trap ✅ |

---

## 🌐 Supported Models (27+)

```bash
ollama pull qwen2.5:7b          # Fast & capable
ollama pull qwen2.5:14b         # Balanced
ollama pull deepseek-r1:8b      # Reasoning
ollama pull llama3:8b           # General purpose
ollama pull phi3:mini           # Lightweight (2.6GB)
ollama pull gemma2:9b           # Google's best
ollama pull mistral:7b          # European powerhouse
```

---

## ⚙️ Setup Automation

```bash
# After deployment, enable full automation:
bash scripts/setup_cron.sh

# Enable systemd auto-start:
sudo cp config/kiswarm.service /etc/systemd/system/
# Edit to replace REPLACE_WITH_* values
sudo systemctl daemon-reload
sudo systemctl enable --now kiswarm
```

---

## 🔒 Security & Privacy

- ✅ **100% Local** — No data ever leaves your machine
- ✅ **No Cloud** — Zero external API calls after initial setup
- ✅ **Audit Logging** — Complete operation history
- ✅ **Non-root** — Runs as regular user
- ✅ **Governance Mode** — Policy-controlled tool execution

---

## 🤝 Contributing

Pull requests welcome! This project is built for the global AI community.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — Free to use, modify, and distribute globally.

---

## 🌟 Credits

**Architect:** Baron Marco Paolo Ialongo  
**Version:** 1.1 Production-Hardened  
**Date:** 2026-02-22  

*Built for the global AI community — deploy anywhere, govern everything.* 🚀
