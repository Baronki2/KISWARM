"""KISWARM v3.0 — Sentinel Bridge Package — 10 Modules (v2.2 + v3.0 Industrial AI)"""

# ── Core AKE ──────────────────────────────────────────────────────────────────
from .sentinel_bridge import SentinelBridge, SwarmKnowledge, IntelligencePacket
from .swarm_debate import SwarmDebateEngine, DebateVerdict

# ── v2.2: Module 1 — Semantic Conflict Detection ─────────────────────────────
from .semantic_conflict import (
    SemanticConflictDetector, ConflictReport, ConflictPair, cosine_similarity,
)

# ── v2.2: Module 2 — Knowledge Decay Engine ──────────────────────────────────
from .knowledge_decay import (
    KnowledgeDecayEngine, DecayRecord, DecayScanReport, HALF_LIVES,
)

# ── v2.2: Module 3 — Model Performance Tracker ───────────────────────────────
from .model_tracker import ModelPerformanceTracker, ModelRecord, LeaderboardEntry

# ── v2.2: Module 4 — Cryptographic Knowledge Ledger ──────────────────────────
from .crypto_ledger import (
    CryptographicKnowledgeLedger, LedgerEntry, TamperReport, merkle_root,
)

# ── v2.2: Module 5 — Differential Retrieval Guard ────────────────────────────
from .retrieval_guard import (
    DifferentialRetrievalGuard, RetrievalGuardReport,
    DriftDetector, DivergenceDetector,
)

# ── v2.2: Module 6 — Adversarial Prompt Firewall ─────────────────────────────
from .prompt_firewall import AdversarialPromptFirewall, FirewallReport, ThreatType

# ── v3.0: Module 7 — Fuzzy Membership Auto-Tuning ────────────────────────────
from .fuzzy_tuner import (
    FuzzyAutoTuner, GaussianParams, BellParams, FuzzyBounds, CostWeights,
    LyapunovMonitor, gaussian_membership, generalized_bell_membership, compute_cost,
)

# ── v3.0: Module 8 — Constrained Reinforcement Learning ──────────────────────
from .constrained_rl import (
    ConstrainedRLAgent, SwarmState, SwarmAction,
    ConstraintEngine, ConstraintConfig, SafetyShield,
    LagrangeManager, LinearPolicy,
)

# ── v3.0: Module 9 — Digital Twin Mutation Pipeline ──────────────────────────
from .digital_twin import (
    DigitalTwin, AcceptanceReport, SimulationResult,
    PhysicsModel, ScenarioGenerator, ExtremeValueAnalyzer,
)

# ── v3.0: Module 10 — Federated Adaptive Mesh Protocol ───────────────────────
from .federated_mesh import (
    FederatedMeshCoordinator, FederatedMeshNode,
    ByzantineAggregator, PartitionHandler,
    NodeShare, NodeRecord, AggregationReport,
    compute_attestation, verify_attestation,
)

__all__ = [
    "SentinelBridge","SwarmKnowledge","IntelligencePacket","SwarmDebateEngine","DebateVerdict",
    "SemanticConflictDetector","ConflictReport","ConflictPair","cosine_similarity",
    "KnowledgeDecayEngine","DecayRecord","DecayScanReport","HALF_LIVES",
    "ModelPerformanceTracker","ModelRecord","LeaderboardEntry",
    "CryptographicKnowledgeLedger","LedgerEntry","TamperReport","merkle_root",
    "DifferentialRetrievalGuard","RetrievalGuardReport","DriftDetector","DivergenceDetector",
    "AdversarialPromptFirewall","FirewallReport","ThreatType",
    "FuzzyAutoTuner","GaussianParams","BellParams","FuzzyBounds","CostWeights",
    "LyapunovMonitor","gaussian_membership","generalized_bell_membership","compute_cost",
    "ConstrainedRLAgent","SwarmState","SwarmAction","ConstraintEngine","ConstraintConfig",
    "SafetyShield","LagrangeManager","LinearPolicy",
    "DigitalTwin","AcceptanceReport","SimulationResult",
    "PhysicsModel","ScenarioGenerator","ExtremeValueAnalyzer",
    "FederatedMeshCoordinator","FederatedMeshNode","ByzantineAggregator","PartitionHandler",
    "NodeShare","NodeRecord","AggregationReport","compute_attestation","verify_attestation",
]
