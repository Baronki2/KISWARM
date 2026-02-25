"""KISWARM v2.2 — Sentinel Bridge Package — All 6 Advanced Modules"""

# Core AKE
from .sentinel_bridge import SentinelBridge, SwarmKnowledge, IntelligencePacket
from .swarm_debate import SwarmDebateEngine, DebateVerdict

# Module 1 — Semantic Conflict Detection
from .semantic_conflict import SemanticConflictDetector, ConflictReport, ConflictPair, cosine_similarity

# Module 2 — Knowledge Decay Engine
from .knowledge_decay import KnowledgeDecayEngine, DecayRecord, DecayScanReport, HALF_LIVES

# Module 3 — Model Performance Tracker
from .model_tracker import ModelPerformanceTracker, ModelRecord, LeaderboardEntry

# Module 4 — Cryptographic Knowledge Ledger
from .crypto_ledger import CryptographicKnowledgeLedger, LedgerEntry, TamperReport, merkle_root

# Module 5 — Differential Retrieval Guard
from .retrieval_guard import DifferentialRetrievalGuard, RetrievalGuardReport, DriftDetector, DivergenceDetector

# Module 6 — Adversarial Prompt Firewall
from .prompt_firewall import AdversarialPromptFirewall, FirewallReport, ThreatType

__all__ = [
    # Core
    "SentinelBridge", "SwarmKnowledge", "IntelligencePacket",
    "SwarmDebateEngine", "DebateVerdict",
    # Module 1
    "SemanticConflictDetector", "ConflictReport", "ConflictPair", "cosine_similarity",
    # Module 2
    "KnowledgeDecayEngine", "DecayRecord", "DecayScanReport", "HALF_LIVES",
    # Module 3
    "ModelPerformanceTracker", "ModelRecord", "LeaderboardEntry",
    # Module 4
    "CryptographicKnowledgeLedger", "LedgerEntry", "TamperReport", "merkle_root",
    # Module 5
    "DifferentialRetrievalGuard", "RetrievalGuardReport", "DriftDetector", "DivergenceDetector",
    # Module 6
    "AdversarialPromptFirewall", "FirewallReport", "ThreatType",
]
