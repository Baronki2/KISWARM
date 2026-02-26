"""
KISWARM v3.0 — SENTINEL API SERVER
====================================
REST interface covering all 10 modules.
Port: 11436

Endpoints (29 total):
  --- Core AKE ---
  POST /sentinel/extract         Trigger knowledge extraction
  POST /sentinel/debate          Swarm Debate for conflicts
  GET  /sentinel/search          Search swarm memory
  GET  /sentinel/status          Full system status

  --- v2.2 Modules ---
  POST /firewall/scan            M6: Adversarial pattern scan
  GET  /decay/scan               M2: Decay scan + revalidation list
  GET  /decay/record/<hash_id>   M2: Single entry confidence
  POST /decay/revalidate         M2: Mark entry revalidated
  GET  /ledger/status            M4: Merkle summary
  GET  /ledger/verify            M4: Full tamper detection
  GET  /ledger/proof/<hash_id>   M4: Inclusion proof
  POST /conflict/analyze         M1: Contradiction clusters
  POST /conflict/quick           M1: Two-text cosine check
  GET  /tracker/leaderboard      M3: ELO + reliability ranking
  GET  /tracker/model/<name>     M3: Per-model stats
  POST /tracker/validate         M3: Post-hoc validation
  POST /guard/assess             M5: Retrieval trust assessment

  --- v3.0 Industrial AI Modules ---
  POST /fuzzy/classify           M7: Fuzzy membership classification
  POST /fuzzy/update             M7: Feed outcome for online tuning
  POST /fuzzy/tune               M7: Run auto-tuning cycle
  GET  /fuzzy/stats              M7: Tuner state + Lyapunov energy
  POST /rl/act                   M8: Get constrained RL action
  POST /rl/learn                 M8: Feed reward + costs for learning
  GET  /rl/stats                 M8: Policy stats + Lagrange multipliers
  POST /twin/evaluate            M9: Digital twin mutation evaluation
  GET  /twin/stats               M9: Promotion/rejection history
  POST /mesh/share               M10: Submit node parameter share
  POST /mesh/register            M10: Register new mesh node
  GET  /mesh/leaderboard         M10: Node trust leaderboard
  GET  /mesh/stats               M10: Global mesh state

  GET  /health                   Service health ping

Author: KISWARM Project (Baron Marco Paolo Ialongo)
Version: 3.0
"""

import asyncio
import datetime
import json
import logging
import os
import sys
import subprocess
from pathlib import Path

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "flask", "flask-cors"], check=True)
    from flask import Flask, jsonify, request
    from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.sentinel_bridge import SentinelBridge, IntelligencePacket
from sentinel.swarm_debate import SwarmDebateEngine
from sentinel.semantic_conflict import SemanticConflictDetector
from sentinel.knowledge_decay import KnowledgeDecayEngine
from sentinel.model_tracker import ModelPerformanceTracker
from sentinel.crypto_ledger import CryptographicKnowledgeLedger
from sentinel.retrieval_guard import DifferentialRetrievalGuard
from sentinel.prompt_firewall import AdversarialPromptFirewall
# v3.0
from sentinel.fuzzy_tuner import FuzzyAutoTuner
from sentinel.constrained_rl import ConstrainedRLAgent, SwarmState, SwarmAction
from sentinel.digital_twin import DigitalTwin
from sentinel.federated_mesh import (
    FederatedMeshCoordinator, FederatedMeshNode,
    NodeShare, compute_attestation,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

LOG_DIR = os.path.join(os.environ.get("KISWARM_HOME", os.path.expanduser("~")), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SENTINEL-API] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "sentinel_api.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("sentinel_api")

# ── Singletons ────────────────────────────────────────────────────────────────
_bridge    = SentinelBridge()
_debate    = SwarmDebateEngine()
_conflict  = SemanticConflictDetector()
_decay     = KnowledgeDecayEngine()
_tracker   = ModelPerformanceTracker()
_ledger    = CryptographicKnowledgeLedger()
_guard     = DifferentialRetrievalGuard(ledger=_ledger, decay_engine=_decay)
_firewall  = AdversarialPromptFirewall()
# v3.0
_fuzzy     = FuzzyAutoTuner()
_rl        = ConstrainedRLAgent()
_twin      = DigitalTwin()
_mesh      = FederatedMeshCoordinator(param_dim=8)

_start = datetime.datetime.now()
_stats = {
    "extractions": 0, "debates": 0, "searches": 0,
    "firewall_scans": 0, "firewall_blocked": 0,
    "decay_scans": 0, "ledger_verifications": 0, "guard_assessments": 0,
    # v3.0
    "fuzzy_classifies": 0, "fuzzy_tune_cycles": 0,
    "rl_actions": 0, "rl_learn_steps": 0,
    "twin_evaluations": 0, "twin_accepted": 0,
    "mesh_rounds": 0, "mesh_shares": 0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE AKE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    uptime = (datetime.datetime.now() - _start).total_seconds()
    return jsonify({
        "status":   "active",
        "service":  "KISWARM-SENTINEL-BRIDGE",
        "version":  "3.0",
        "port":     11436,
        "modules":  10,
        "endpoints": 29,
        "uptime":   round(uptime, 1),
        "stats":    _stats,
        "timestamp": datetime.datetime.now().isoformat(),
    })


@app.route("/sentinel/extract", methods=["POST"])
def extract():
    data  = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"status": "error", "error": "query required"}), 400

    # v3.0: firewall-check query before AKE
    fw = _firewall.scan_query(query)
    if fw.blocked:
        return jsonify({"status": "blocked", "firewall": {
            "threat_score": fw.threat_score, "threats": [t.value for t in fw.threat_types],
        }}), 403

    force     = bool(data.get("force", False))
    threshold = float(data.get("threshold", 0.85))
    _bridge.ckm.threshold = threshold

    try:
        result = run_async(_bridge.run(query, force=force))
        _stats["extractions"] += 1
        return jsonify(result)
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/sentinel/debate", methods=["POST"])
def debate():
    data = request.get_json() or {}
    for f in ["query", "content_a", "content_b"]:
        if not data.get(f):
            return jsonify({"status": "error", "error": f"{f} required"}), 400

    # Firewall both content payloads
    for key in ["content_a", "content_b"]:
        fw = _firewall.scan(data[key])
        if fw.blocked:
            return jsonify({"status": "blocked", "field": key,
                            "threat_score": fw.threat_score}), 403

    try:
        verdict = run_async(_debate.debate(
            query=data["query"],
            content_a=data["content_a"], content_b=data["content_b"],
            source_a_name=data.get("source_a", "Source A"),
            source_b_name=data.get("source_b", "Source B"),
        ))
        _stats["debates"] += 1
        return jsonify({
            "status":          "success",
            "winning_content": verdict.winning_content,
            "confidence":      verdict.confidence,
            "vote_tally":      verdict.vote_tally,
            "dissenting":      verdict.dissenting_models,
            "synthesis":       verdict.synthesis,
            "timestamp":       verdict.timestamp,
        })
    except Exception as exc:
        logger.error("Debate failed: %s", exc)
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/sentinel/search")
def search():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 5))
    if not query:
        return jsonify({"status": "error", "error": "q required"}), 400
    try:
        results = _bridge.memory.search(query, top_k=top_k)
        _stats["searches"] += 1
        return jsonify({"status": "success", "query": query, "results": results, "total": len(results)})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/sentinel/status")
def sentinel_status():
    uptime = (datetime.datetime.now() - _start).total_seconds()
    return jsonify({
        "system":    "KISWARM-SENTINEL-v3.0",
        "status":    "operational",
        "uptime":    round(uptime, 1),
        "stats":     _stats,
        "threshold": _bridge.ckm.threshold,
        "scouts":    [s.__class__.__name__ for s in _bridge.scouts],
        "qdrant":    _bridge.memory.client is not None,
        "ledger":    {"entries": _ledger.size, "root": _ledger.current_root[:16] + "..."},
        "mesh":      _mesh.get_stats(),
        "timestamp": datetime.datetime.now().isoformat(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# v2.2 MODULE ENDPOINTS (preserved)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/firewall/scan", methods=["POST"])
def firewall_scan():
    data    = request.get_json() or {}
    content = data.get("content", "")
    source  = data.get("source", "unknown")
    if not content:
        return jsonify({"status": "error", "error": "content required"}), 400
    report = _firewall.scan(content, source=source)
    _stats["firewall_scans"] += 1
    if report.blocked:
        _stats["firewall_blocked"] += 1
    return jsonify({
        "status":       "success",
        "blocked":      report.blocked,
        "threat_level": report.threat_level.value,
        "threat_score": report.threat_score,
        "threats":      [t.value for t in report.threat_types],
        "matches":      [{"pattern": m.pattern_name, "severity": m.severity.value} for m in report.matches],
        "statistical":  report.statistical,
        "recommendation": report.recommendation,
    })


@app.route("/decay/scan")
def decay_scan():
    report = _decay.scan()
    _stats["decay_scans"] += 1
    return jsonify({
        "status":             "success",
        "needs_revalidation": report.needs_revalidation,
        "retired":            report.retired,
        "healthy":            report.healthy,
        "total":              report.total,
    })


@app.route("/decay/record/<hash_id>")
def decay_record(hash_id: str):
    conf = _decay.get_confidence(hash_id)
    return jsonify({"status": "success", "hash_id": hash_id, "confidence": conf})


@app.route("/decay/revalidate", methods=["POST"])
def decay_revalidate():
    data    = request.get_json() or {}
    hash_id = data.get("hash_id", "")
    new_conf = float(data.get("confidence", 0.9))
    if not hash_id:
        return jsonify({"status": "error", "error": "hash_id required"}), 400
    _decay.mark_revalidated(hash_id, new_conf)
    return jsonify({"status": "success", "hash_id": hash_id, "new_confidence": new_conf})


@app.route("/ledger/status")
def ledger_status():
    return jsonify({
        "status":  "success",
        "entries": _ledger.size,
        "root":    _ledger.current_root,
        "valid":   _ledger.size > 0,
    })


@app.route("/ledger/verify")
def ledger_verify():
    report = _ledger.verify_integrity()
    _stats["ledger_verifications"] += 1
    return jsonify({
        "status":          "success",
        "valid":           report.valid,
        "total_entries":   report.total_entries,
        "tampered_entries": report.tampered_indices,
        "root_match":      report.root_match,
    })


@app.route("/ledger/proof/<hash_id>")
def ledger_proof(hash_id: str):
    proof = _ledger.get_proof(hash_id)
    if proof is None:
        return jsonify({"status": "error", "error": "entry not found"}), 404
    return jsonify({"status": "success", "hash_id": hash_id, "proof": proof})


@app.route("/conflict/analyze", methods=["POST"])
def conflict_analyze():
    data    = request.get_json() or {}
    packets = data.get("packets", [])
    if not packets:
        return jsonify({"status": "error", "error": "packets required"}), 400
    ips = [IntelligencePacket(
        content=p.get("content", ""),
        source=p.get("source", "unknown"),
        confidence=float(p.get("confidence", 0.5)),
    ) for p in packets]
    report = _conflict.analyze(ips)
    return jsonify({
        "status":           "success",
        "total_pairs":      report.total_pairs,
        "conflict_pairs":   len(report.conflict_pairs),
        "resolution_needed": report.resolution_needed,
        "clusters":         len(report.clusters),
        "severity":         report.severity_label,
        "pairs": [{
            "source_a":  p.source_a, "source_b": p.source_b,
            "similarity": p.similarity, "severity": p.severity,
        } for p in report.conflict_pairs],
    })


@app.route("/conflict/quick", methods=["POST"])
def conflict_quick():
    data = request.get_json() or {}
    a, b = data.get("text_a", ""), data.get("text_b", "")
    if not a or not b:
        return jsonify({"status": "error", "error": "text_a and text_b required"}), 400
    sim, severity = _conflict.quick_check(a, b)
    return jsonify({"status": "success", "similarity": sim, "severity": severity})


@app.route("/tracker/leaderboard")
def tracker_leaderboard():
    board = _tracker.leaderboard()
    return jsonify({
        "status":      "success",
        "leaderboard": [{"rank": e.rank, "model": e.model_id, "elo": e.elo,
                         "reliability": e.reliability_score, "debates": e.total_debates,
                         "win_rate": e.win_rate} for e in board],
    })


@app.route("/tracker/model/<path:model_name>")
def tracker_model(model_name: str):
    rec = _tracker.get_model(model_name)
    if rec is None:
        return jsonify({"status": "error", "error": "model not found"}), 404
    return jsonify({"status": "success", "model": rec.model_id, "elo": rec.elo,
                    "reliability_score": rec.reliability_score,
                    "debates": rec.total_debates, "win_rate": rec.win_rate,
                    "validation_accuracy": rec.validation_accuracy})


@app.route("/tracker/validate", methods=["POST"])
def tracker_validate():
    data = request.get_json() or {}
    debate_id = data.get("debate_id", "")
    correct_winner = data.get("correct_winner", "")
    if not debate_id or not correct_winner:
        return jsonify({"status": "error", "error": "debate_id and correct_winner required"}), 400
    _tracker.validate_debate(debate_id, correct_winner)
    return jsonify({"status": "success", "debate_id": debate_id, "validated_winner": correct_winner})


@app.route("/guard/assess", methods=["POST"])
def guard_assess():
    data = request.get_json() or {}
    hash_id   = data.get("hash_id", "")
    query     = data.get("query", "")
    retrieved = data.get("retrieved_content", "")
    fresh     = data.get("fresh_content", None)
    if not hash_id or not query or not retrieved:
        return jsonify({"status": "error", "error": "hash_id, query, retrieved_content required"}), 400
    report = _guard.assess(hash_id=hash_id, query=query, retrieved_content=retrieved, fresh_content=fresh)
    _stats["guard_assessments"] += 1
    return jsonify({
        "status":       "success",
        "trust_level":  report.trust_level,
        "trust_score":  report.trust_score,
        "recommendation": report.recommendation,
        "flags":        report.flags,
        "decay_confidence": report.decay_confidence,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# v3.0 MODULE 7 — FUZZY MEMBERSHIP AUTO-TUNING
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/fuzzy/classify", methods=["POST"])
def fuzzy_classify():
    """
    Classify a confidence value through fuzzy membership functions.
    Body: { "confidence": 0.72 }
    Returns: { "label": "HIGH", "membership": 0.85, "all_memberships": {...} }
    """
    data = request.get_json() or {}
    x    = float(data.get("confidence", 0.5))
    label, mu = _fuzzy.classify(x)
    all_mu    = _fuzzy.all_memberships(x)
    _stats["fuzzy_classifies"] += 1
    return jsonify({
        "status":          "success",
        "confidence":      x,
        "label":           label,
        "membership":      mu,
        "all_memberships": all_mu,
    })


@app.route("/fuzzy/update", methods=["POST"])
def fuzzy_update():
    """
    Feed a validation outcome to the online tuner.
    Body: { "confidence": 0.72, "actual_quality": true, "actuation": 0.1 }
    """
    data       = request.get_json() or {}
    confidence = float(data.get("confidence", 0.5))
    quality    = bool(data.get("actual_quality", True))
    actuation  = float(data.get("actuation", 0.0))
    _fuzzy.update(confidence, quality, actuation)
    return jsonify({"status": "success", "recorded": True,
                    "buffer_size": len(_fuzzy._errors)})


@app.route("/fuzzy/tune", methods=["POST"])
def fuzzy_tune():
    """
    Run one auto-tuning cycle (evolutionary micro-mutations + Lyapunov check).
    Body: { "n_mutations": 5 }
    """
    data        = request.get_json() or {}
    n_mutations = int(data.get("n_mutations", 5))
    report      = _fuzzy.tune_cycle(n_mutations=n_mutations)
    _stats["fuzzy_tune_cycles"] += 1
    return jsonify({"status": "success", "tune_report": report})


@app.route("/fuzzy/stats")
def fuzzy_stats():
    """Get fuzzy tuner state: parameters, Lyapunov energy, iteration count."""
    return jsonify({"status": "success", "fuzzy": _fuzzy.get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# v3.0 MODULE 8 — CONSTRAINED REINFORCEMENT LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/rl/act", methods=["POST"])
def rl_act():
    """
    Get the RL agent's recommended action for the current swarm state.
    Body: {
      "queue_depth": 0.6, "memory_pressure": 0.5,
      "model_availability": 0.9, "extraction_latency": 0.2,
      "scout_success_rate": 0.8, "debate_load": 0.1
    }
    """
    data  = request.get_json() or {}
    state = SwarmState(
        knowledge_queue_depth = float(data.get("queue_depth", 0.5)),
        memory_pressure       = float(data.get("memory_pressure", 0.3)),
        model_availability    = float(data.get("model_availability", 0.9)),
        extraction_latency    = float(data.get("extraction_latency", 0.2)),
        scout_success_rate    = float(data.get("scout_success_rate", 0.8)),
        debate_load           = float(data.get("debate_load", 0.1)),
    )
    action = _rl.act(state)
    _stats["rl_actions"] += 1
    return jsonify({
        "status":             "success",
        "scout_priority":     action.scout_priority,
        "extraction_rate":    action.extraction_rate,
        "debate_threshold":   action.debate_threshold,
        "cache_eviction_rate": action.cache_eviction_rate,
        "episode":            _rl._episode,
        "lambdas":            _rl.lagrange.lambdas,
    })


@app.route("/rl/learn", methods=["POST"])
def rl_learn():
    """
    Feed reward + constraint costs back to the RL agent for learning.
    Body: {
      "state": {...}, "action": {...},
      "reward": 0.8,
      "costs": [0.6, 0.3, 0.2],  // [memory, latency, model_load]
      "shielded": false
    }
    """
    data    = request.get_json() or {}
    s_data  = data.get("state", {})
    a_data  = data.get("action", {})
    reward  = float(data.get("reward", 0.0))
    costs   = data.get("costs", [0.3, 0.2, 0.1])
    shielded = bool(data.get("shielded", False))

    state  = SwarmState(
        knowledge_queue_depth = float(s_data.get("queue_depth", 0.5)),
        memory_pressure       = float(s_data.get("memory_pressure", 0.3)),
        model_availability    = float(s_data.get("model_availability", 0.9)),
        extraction_latency    = float(s_data.get("extraction_latency", 0.2)),
        scout_success_rate    = float(s_data.get("scout_success_rate", 0.8)),
        debate_load           = float(s_data.get("debate_load", 0.1)),
    )
    action = SwarmAction(
        scout_priority      = float(a_data.get("scout_priority", 0.5)),
        extraction_rate     = float(a_data.get("extraction_rate", 0.5)),
        debate_threshold    = float(a_data.get("debate_threshold", 0.3)),
        cache_eviction_rate = float(a_data.get("cache_eviction_rate", 0.1)),
    )

    _rl.learn(state=state, action=action, reward=reward,
              costs=costs, shielded=shielded)
    _stats["rl_learn_steps"] += 1
    return jsonify({"status": "success", "episode": _rl._episode,
                    "lambdas": _rl.lagrange.lambdas})


@app.route("/rl/stats")
def rl_stats():
    """Get RL agent state: episode count, mean reward, Lagrange multipliers."""
    return jsonify({"status": "success", "rl": _rl.get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# v3.0 MODULE 9 — DIGITAL TWIN MUTATION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/twin/evaluate", methods=["POST"])
def twin_evaluate():
    """
    Evaluate a candidate parameter set through the Digital Twin pipeline.
    Runs 75+ scenarios (Monte Carlo + rare-event + adversarial).
    Applies acceptance rules + EVT tail-risk analysis.

    Body: {
      "scout": 0.6, "rate": 0.55, "threshold": 0.3, "eviction": 0.12,
      "label": "candidate_v3",
      "set_as_baseline": false
    }
    """
    data      = request.get_json() or {}
    scout     = float(data.get("scout",     0.5))
    rate      = float(data.get("rate",      0.5))
    threshold = float(data.get("threshold", 0.3))
    eviction  = float(data.get("eviction",  0.1))
    label     = data.get("label", "api_candidate")
    as_baseline = bool(data.get("set_as_baseline", False))

    if as_baseline:
        _twin.set_baseline(scout, rate, threshold, eviction)
        return jsonify({"status": "success", "action": "baseline_set"})

    report = _twin.evaluate(scout, rate, threshold, eviction, label=label)
    _stats["twin_evaluations"] += 1
    if report.accepted:
        _stats["twin_accepted"] += 1

    return jsonify({
        "status":                   "success",
        "accepted":                 report.accepted,
        "rejection_reasons":        report.rejection_reasons,
        "n_scenarios":              report.n_scenarios,
        "hard_violations":          report.hard_violations,
        "stability_margin":         report.stability_margin_mean,
        "stability_baseline":       report.stability_margin_baseline,
        "efficiency_gain_pct":      round(report.efficiency_gain * 100, 2),
        "recovery_time":            report.recovery_time_mean,
        "recovery_time_baseline":   report.recovery_time_baseline,
        "tail_heavier":             report.tail_heavier,
        "tail_index_baseline":      report.tail_index_baseline,
        "tail_index_candidate":     report.tail_index_candidate,
        "adversarial_violations":   report.adversarial_violations,
        "timestamp":                report.timestamp,
    })


@app.route("/twin/stats")
def twin_stats():
    """Get digital twin promotion/rejection history and rates."""
    return jsonify({"status": "success", "twin": _twin.get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# v3.0 MODULE 10 — FEDERATED ADAPTIVE MESH
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/mesh/register", methods=["POST"])
def mesh_register():
    """
    Register a new node in the federated mesh.
    Body: { "node_id": "ollama_node_01", "initial_trust": 0.8 }
    """
    data    = request.get_json() or {}
    node_id = data.get("node_id", "")
    trust   = float(data.get("initial_trust", 0.8))
    if not node_id:
        return jsonify({"status": "error", "error": "node_id required"}), 400
    _mesh.register_node(node_id, trust)
    return jsonify({"status": "success", "node_id": node_id,
                    "registered_nodes": len(_mesh._nodes)})


@app.route("/mesh/share", methods=["POST"])
def mesh_share():
    """
    Submit a node's parameter share for the next aggregation round.
    Triggers aggregation when at least 2 shares are received.

    Body: {
      "node_id": "ollama_node_01",
      "param_delta": [0.01, -0.02, ...],
      "perf_delta": 0.05,
      "stability_cert": 0.85,
      "uptime": 0.99
    }
    """
    data     = request.get_json() or {}
    node_id  = data.get("node_id", "")
    if not node_id:
        return jsonify({"status": "error", "error": "node_id required"}), 400

    import time
    ts    = float(data.get("timestamp", time.time()))
    delta = data.get("param_delta", [0.0] * 8)
    stab  = float(data.get("stability_cert", 0.8))
    perf  = float(data.get("perf_delta", 0.0))
    upt   = float(data.get("uptime", 1.0))

    share = NodeShare(
        node_id=node_id,
        param_delta=delta,
        perf_delta=perf,
        stability_cert=stab,
        uptime=upt,
        timestamp=ts,
    )
    share.sign()   # compute attestation

    # Auto-aggregate: collect shares and run round
    if not hasattr(app, "_pending_shares"):
        app._pending_shares = []
    app._pending_shares.append(share)

    _stats["mesh_shares"] += 1

    # Run aggregation round if we have at least 2 shares
    if len(app._pending_shares) >= 2:
        report = _mesh.aggregate_round(app._pending_shares)
        app._pending_shares = []
        _stats["mesh_rounds"] += 1
        return jsonify({
            "status":         "success",
            "aggregation":    "completed",
            "participating":  report.participating,
            "rejected":       report.rejected,
            "quorum":         report.quorum_reached,
            "global_params":  _mesh.global_params,
        })

    return jsonify({
        "status":         "success",
        "aggregation":    "pending",
        "shares_queued":  len(app._pending_shares),
    })


@app.route("/mesh/leaderboard")
def mesh_leaderboard():
    """Node trust leaderboard ranked by reliability weight."""
    board = _mesh.node_leaderboard()
    return jsonify({"status": "success", "nodes": board})


@app.route("/mesh/stats")
def mesh_stats():
    """Global mesh state: round count, node count, global params."""
    return jsonify({"status": "success", "mesh": _mesh.get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(_):
    return jsonify({
        "status":  "error",
        "error":   "Endpoint not found",
        "version": "3.0",
        "all_endpoints": [
            "POST /sentinel/extract", "POST /sentinel/debate",
            "GET  /sentinel/search",  "GET  /sentinel/status",
            "POST /firewall/scan",
            "GET  /decay/scan",       "GET  /decay/record/<hash_id>",  "POST /decay/revalidate",
            "GET  /ledger/status",    "GET  /ledger/verify",           "GET  /ledger/proof/<hash_id>",
            "POST /conflict/analyze", "POST /conflict/quick",
            "GET  /tracker/leaderboard", "GET  /tracker/model/<name>", "POST /tracker/validate",
            "POST /guard/assess",
            "POST /fuzzy/classify",  "POST /fuzzy/update",  "POST /fuzzy/tune",  "GET  /fuzzy/stats",
            "POST /rl/act",          "POST /rl/learn",       "GET  /rl/stats",
            "POST /twin/evaluate",   "GET  /twin/stats",
            "POST /mesh/share",      "POST /mesh/register",
            "GET  /mesh/leaderboard", "GET  /mesh/stats",
            "GET  /health",
        ],
    }), 404


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  KISWARM v3.0 — SENTINEL BRIDGE API                    ║")
    logger.info("║  Port: 11436  |  Modules: 10  |  Endpoints: 29        ║")
    logger.info("║  Industrial AI: Fuzzy · CRL · Twin · FedMesh          ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    app.run(host="127.0.0.1", port=11436, debug=False, threaded=True)
