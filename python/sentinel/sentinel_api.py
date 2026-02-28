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
# CIEC v4.0 — LAZY MODULE SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

def _lazy(attr, factory):
    if not hasattr(app, attr):
        setattr(app, attr, factory())
    return getattr(app, attr)

def _plc():
    from sentinel.plc_parser import PLCSemanticParser
    return _lazy("_plc_parser", PLCSemanticParser)

def _scada():
    from sentinel.scada_observer import SCADAObserver
    return _lazy("_scada_obs", SCADAObserver)

def _ptwin():
    from sentinel.physics_twin import PhysicsTwin
    return _lazy("_physics_twin", PhysicsTwin)

def _rules():
    from sentinel.rule_engine import RuleConstraintEngine
    return _lazy("_rule_eng", RuleConstraintEngine)

def _kg():
    from sentinel.knowledge_graph import KnowledgeGraph
    return _lazy("_know_graph", KnowledgeGraph)

def _acrl():
    from sentinel.actor_critic import IndustrialActorCritic
    return _lazy("_actor_crit", lambda: IndustrialActorCritic(state_dim=32))


# ─── MODULE 11: PLC SEMANTIC PARSER ──────────────────────────────────────────

@app.route("/plc/parse", methods=["POST"])
def plc_parse():
    """Parse IEC 61131-3 ST source into CIR + DSG + PID/interlock/watchdog.
    Body: {"source": "<ST code>", "program_name": "optional"}
    """
    d = request.get_json(force=True) or {}
    if not d.get("source"):
        return jsonify({"status": "error", "error": "source required"}), 400
    result = _plc().parse(d["source"], d.get("program_name", "UNKNOWN"))
    return jsonify({"status": "success", "result": result.to_dict()})

@app.route("/plc/stats")
def plc_stats():
    """PLC parser cache and performance statistics."""
    return jsonify({"status": "success", "stats": _plc().get_stats()})


# ─── MODULE 12: SCADA / OPC / SQL OBSERVATION LAYER ──────────────────────────

@app.route("/scada/push", methods=["POST"])
def scada_push():
    """Ingest real-time tag readings (OPC UA callback).
    Body: {"tag": "temperature", "value": 45.2}
      OR  {"snapshot": {"tag1": v1, "tag2": v2}}
    """
    d = request.get_json(force=True) or {}
    if "snapshot" in d:
        _scada().push_snapshot(d["snapshot"])
        return jsonify({"status": "success", "ingested": len(d["snapshot"])})
    tag, value = d.get("tag"), d.get("value")
    if tag is None or value is None:
        return jsonify({"status": "error", "error": "tag and value required"}), 400
    _scada().push_reading(tag, float(value))
    return jsonify({"status": "success", "ingested": 1})

@app.route("/scada/ingest-history", methods=["POST"])
def scada_ingest_history():
    """Batch ingest SQL historian records.
    Body: {"records": [{"tag": "t1", "value": 1.0, "timestamp": 1700000000.0}]}
    """
    d = request.get_json(force=True) or {}
    records = d.get("records", [])
    if not records:
        return jsonify({"status": "error", "error": "records array required"}), 400
    count = _scada().ingest_history(records)
    return jsonify({"status": "success", "ingested": count})

@app.route("/scada/state")
def scada_state():
    """Build and return current plant state vector S(t)."""
    sv = _scada().build_state_vector()
    return jsonify({"status": "success", "state": sv.to_dict()})

@app.route("/scada/anomalies")
def scada_anomalies():
    """Return tags showing anomalous values (z-score based)."""
    threshold = float(request.args.get("threshold", 3.0))
    anomalies = _scada().get_anomalies(std_threshold=threshold)
    return jsonify({"status": "success", "anomalies": anomalies, "count": len(anomalies)})

@app.route("/scada/stats")
def scada_stats():
    """SCADA observer statistics and subscribed tag list."""
    return jsonify({"status": "success", "stats": _scada().get_stats()})


# ─── MODULE 13: DIGITAL TWIN PHYSICS ENGINE ──────────────────────────────────

@app.route("/ciec-twin/run", methods=["POST"])
def ciec_twin_run():
    """Run a physics simulation episode (thermal + pump + battery + power).
    Body: {"steps": 100, "dt": 0.1, "q_in": 2000, "dp": 2.0,
           "i_charge": 10, "i_disch": 8, "inject_faults": false}
    """
    d = request.get_json(force=True) or {}
    result = _ptwin().run(
        steps=int(d.get("steps", 100)), dt=float(d.get("dt", 0.1)),
        q_in=float(d.get("q_in", 2000.0)), dp=float(d.get("dp", 2.0)),
        i_charge=float(d.get("i_charge", 10.0)), i_disch=float(d.get("i_disch", 8.0)),
        inject_faults=bool(d.get("inject_faults", False)),
    )
    return jsonify({"status": "success", "result": result.to_dict()})

@app.route("/ciec-twin/evaluate", methods=["POST"])
def ciec_twin_evaluate():
    """Evaluate a mutation candidate against the digital twin.
    Body: {"params": {...mutation params...}, "n_runs": 3}
    Returns: {"promoted": bool, "metrics": {...fitness scores...}}
    """
    d = request.get_json(force=True) or {}
    promote, metrics = _ptwin().evaluate_mutation(
        d.get("params", {}), n_runs=int(d.get("n_runs", 3))
    )
    return jsonify({"status": "success", "promoted": promote, "metrics": metrics})

@app.route("/ciec-twin/stats")
def ciec_twin_stats():
    """Physics twin simulation statistics."""
    return jsonify({"status": "success", "stats": _ptwin().get_stats()})


# ─── MODULE 14: RULE CONSTRAINT ENGINE ───────────────────────────────────────

@app.route("/constraints/validate", methods=["POST"])
def constraints_validate():
    """Validate proposed parameter action against all safety constraints.
    Body: {"state": {"pressure": 3.0, "battery_soc": 0.8}, "action": {"delta_kp": 0.02}}
    Returns: {"allowed": bool, "total_penalty": float, "hard_violations": [...]}
    """
    d = request.get_json(force=True) or {}
    result = _rules().validate(d.get("state", {}), d.get("action", {}))
    return jsonify({"status": "success", "validation": result.to_dict()})

@app.route("/constraints/check-state", methods=["POST"])
def constraints_check_state():
    """Quick hard-constraint safety check on current plant state.
    Body: {"state": {"pressure": 3.0, ...}}
    """
    d = request.get_json(force=True) or {}
    return jsonify({"status": "success",
                    "safe": _rules().is_safe_state(d.get("state", {}))})

@app.route("/constraints/list")
def constraints_list():
    """List all registered hard and soft constraints."""
    return jsonify({"status": "success",
                    "constraints": _rules().get_constraints()})

@app.route("/constraints/violations")
def constraints_violations():
    """Recent constraint violation audit log."""
    n = int(request.args.get("n", 50))
    hist = _rules().get_violation_history(n)
    return jsonify({"status": "success", "violations": hist, "count": len(hist)})

@app.route("/constraints/stats")
def constraints_stats():
    """Constraint engine statistics: check count, block rate, violations by category."""
    return jsonify({"status": "success", "stats": _rules().get_stats()})


# ─── MODULE 15: CROSS-PROJECT KNOWLEDGE GRAPH ────────────────────────────────

@app.route("/kg/add-pid", methods=["POST"])
def kg_add_pid():
    """Store a proven PID configuration in the cross-project knowledge graph.
    Body: {"title": "...", "kp": 1.2, "ki": 0.3, "kd": 0.05,
           "sample_time": 0.1, "output_min": 0, "output_max": 100,
           "plant_type": "pump", "site_id": "PLANT_A", "project_id": "2024"}
    """
    d = request.get_json(force=True) or {}
    node = _kg().add_pid_config(
        title=d.get("title", "PID Config"),
        kp=float(d.get("kp", 1.0)), ki=float(d.get("ki", 0.1)),
        kd=float(d.get("kd", 0.01)), sample_time=float(d.get("sample_time", 0.1)),
        output_min=float(d.get("output_min", 0.0)), output_max=float(d.get("output_max", 100.0)),
        plant_type=d.get("plant_type", "generic"),
        site_id=d.get("site_id", ""), project_id=d.get("project_id", ""),
        tags=d.get("tags"),
    )
    return jsonify({"status": "success", "node_id": node.node_id})

@app.route("/kg/add-failure", methods=["POST"])
def kg_add_failure():
    """Record a failure signature and proven fix template.
    Body: {"title": "...", "symptoms": [...], "root_cause": "...",
           "fix_template": {...}, "site_id": "...", "project_id": "..."}
    """
    d = request.get_json(force=True) or {}
    node = _kg().add_failure_signature(
        title=d.get("title", "Failure"), symptoms=d.get("symptoms", []),
        root_cause=d.get("root_cause", ""), fix_template=d.get("fix_template", {}),
        site_id=d.get("site_id", ""), project_id=d.get("project_id", ""),
    )
    return jsonify({"status": "success", "node_id": node.node_id})

@app.route("/kg/find-similar", methods=["POST"])
def kg_find_similar():
    """Query knowledge graph for nodes similar to current context.
    Body: {"vector": [1.2, 0.3, 0.05], "tags": ["pump"], "kind": "PIDConfig", "top_k": 5}
    """
    d = request.get_json(force=True) or {}
    matches = _kg().find_similar(
        query_vector=d.get("vector", []), query_tags=d.get("tags", []),
        kind_filter=d.get("kind"), top_k=int(d.get("top_k", 5)),
        min_similarity=float(d.get("min_similarity", 0.1)),
    )
    return jsonify({"status": "success", "matches": [m.to_dict() for m in matches]})

@app.route("/kg/find-by-symptoms", methods=["POST"])
def kg_find_by_symptoms():
    """Match failure signatures to observed symptoms.
    Body: {"symptoms": ["pressure_drop", "high_vibration"], "top_k": 5}
    """
    d = request.get_json(force=True) or {}
    matches = _kg().find_by_symptoms(d.get("symptoms", []),
                                      top_k=int(d.get("top_k", 5)))
    return jsonify({"status": "success", "matches": [m.to_dict() for m in matches]})

@app.route("/kg/recurring-patterns")
def kg_recurring_patterns():
    """Detect problems solved multiple times across projects/sites.
    This is the 'you solved this pump cavitation 4 times in 8 years' detector.
    """
    min_occ  = int(request.args.get("min_occurrences", 2))
    patterns = _kg().detect_recurring_patterns(min_occurrences=min_occ)
    return jsonify({"status": "success", "patterns": patterns, "count": len(patterns)})

@app.route("/kg/export-bundle")
def kg_export_bundle():
    """Export signed knowledge diff bundle for federated multi-site sync."""
    since  = float(request.args.get("since", 0.0))
    bundle = _kg().export_diff_bundle(since_timestamp=since)
    return jsonify({"status": "success", "bundle": bundle})

@app.route("/kg/import-bundle", methods=["POST"])
def kg_import_bundle():
    """Import a verified knowledge diff bundle from another site."""
    d        = request.get_json(force=True) or {}
    bundle   = d.get("bundle", d)
    imported = _kg().import_diff_bundle(bundle)
    return jsonify({"status": "success", "imported": imported})

@app.route("/kg/nodes")
def kg_nodes():
    """List knowledge graph nodes (optionally filtered by kind)."""
    nodes = _kg().list_nodes(kind=request.args.get("kind"),
                              limit=int(request.args.get("limit", 50)))
    return jsonify({"status": "success", "nodes": nodes, "count": len(nodes)})

@app.route("/kg/stats")
def kg_stats():
    """Knowledge graph statistics: node count, edge count, sites, projects."""
    return jsonify({"status": "success", "stats": _kg().get_stats()})


# ─── MODULE 16: INDUSTRIAL ACTOR-CRITIC RL ───────────────────────────────────

@app.route("/ciec-rl/act", methods=["POST"])
def ciec_rl_act():
    """Get a constrained bounded parameter-shift action from the CIEC RL policy.
    Body: {"state": [0.1, 0.2, ...32 floats], "deterministic": false, "shield": true}
    Returns: {"action": {"delta_kp": 0.02, ...}, "info": {"shielded": false, ...}}
    Note: shield=true routes action through Rule Constraint Engine first.
    """
    d             = request.get_json(force=True) or {}
    state         = d.get("state", [0.0] * 32)
    deterministic = bool(d.get("deterministic", False))
    ce            = _rules() if bool(d.get("shield", True)) else None
    action, info  = _acrl().select_action(state, deterministic=deterministic,
                                           constraint_check=ce)
    return jsonify({"status": "success", "action": action, "info": info})

@app.route("/ciec-rl/observe", methods=["POST"])
def ciec_rl_observe():
    """Feed a transition into the constrained RL replay buffer.
    Body: {"state": [...], "action": [...], "reward": 1.0,
           "next_state": [...], "done": false, "cost": 0.0}
    """
    d = request.get_json(force=True) or {}
    if not d.get("state"):
        return jsonify({"status": "error", "error": "state required"}), 400
    _acrl().observe(state=d["state"], action=d.get("action", []),
                    reward=float(d.get("reward", 0.0)),
                    next_state=d.get("next_state", d["state"]),
                    done=bool(d.get("done", False)),
                    cost=float(d.get("cost", 0.0)))
    return jsonify({"status": "success", "buffer_size": len(_acrl().buffer)})

@app.route("/ciec-rl/update", methods=["POST"])
def ciec_rl_update():
    """Trigger one constrained actor-critic gradient update with Lagrangian penalty."""
    d      = request.get_json(force=True) or {}
    result = _acrl().update(batch_size=int(d.get("batch_size", 64)))
    return jsonify({"status": "success", "update": result})

@app.route("/ciec-rl/stats")
def ciec_rl_stats():
    """CIEC RL statistics: steps, episodes, shield rate, lambda values, PLC bounds."""
    return jsonify({"status": "success", "stats": _acrl().get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# v4.1 CIEC INDUSTRIAL EVOLUTION — LAZY INIT
# ═══════════════════════════════════════════════════════════════════════════════

_td3_inst   = None
_ast_inst   = None
_ephys_inst = None
_vmw_inst   = None
_fv_inst    = None
_byz_inst   = None
_mgov_inst  = None

def _td3():
    global _td3_inst
    if _td3_inst is None:
        from python.sentinel.td3_controller import TD3IndustrialController
        _td3_inst = TD3IndustrialController(state_dim=256)
    return _td3_inst

def _ast():
    global _ast_inst
    if _ast_inst is None:
        from python.sentinel.ast_parser import IEC61131ASTParser
        _ast_inst = IEC61131ASTParser()
    return _ast_inst

def _ephys():
    global _ephys_inst
    if _ephys_inst is None:
        from python.sentinel.extended_physics import ExtendedPhysicsTwin
        _ephys_inst = ExtendedPhysicsTwin()
    return _ephys_inst

def _vmw():
    global _vmw_inst
    if _vmw_inst is None:
        from python.sentinel.vmware_orchestrator import VMwareOrchestrator
        _vmw_inst = VMwareOrchestrator()
    return _vmw_inst

def _fv():
    global _fv_inst
    if _fv_inst is None:
        from python.sentinel.formal_verification import FormalVerificationEngine
        _fv_inst = FormalVerificationEngine()
    return _fv_inst

def _byz():
    global _byz_inst
    if _byz_inst is None:
        from python.sentinel.byzantine_aggregator import ByzantineFederatedAggregator
        _byz_inst = ByzantineFederatedAggregator()
    return _byz_inst

def _mgov():
    global _mgov_inst
    if _mgov_inst is None:
        from python.sentinel.mutation_governance import MutationGovernanceEngine
        _mgov_inst = MutationGovernanceEngine()
    return _mgov_inst


# ═══════════════════════════════════════════════════════════════════════════════
# v4.1 MODULE 17 — TD3 INDUSTRIAL CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/td3/act", methods=["POST"])
def td3_act():
    """Select TD3 action for a given state vector."""
    data  = request.get_json() or {}
    state = data.get("state", [0.0] * 256)
    det   = data.get("deterministic", False)
    noise = data.get("exploration_noise", 0.05)
    action, info = _td3().select_action(state, deterministic=det,
                                         exploration_noise=noise)
    return jsonify({"status": "success", "action": action, "info": info})

@app.route("/td3/observe", methods=["POST"])
def td3_observe():
    """Push a transition into the replay buffer."""
    data = request.get_json() or {}
    _td3().observe(
        state      = data.get("state",      [0.0] * 256),
        action     = data.get("action",     [0.0] * 8),
        reward     = float(data.get("reward", 0.0)),
        next_state = data.get("next_state", [0.0] * 256),
        done       = bool(data.get("done",   False)),
        cost       = float(data.get("cost",  0.0)),
    )
    return jsonify({"status": "success", "buffer_size": len(_td3().buffer)})

@app.route("/td3/update", methods=["POST"])
def td3_update():
    """Perform one TD3 gradient update."""
    data   = request.get_json() or {}
    result = _td3().update(batch_size=data.get("batch_size"))
    return jsonify({"status": "success", "result": result})

@app.route("/td3/reward", methods=["POST"])
def td3_reward():
    """Compute CIEC reward from performance metrics."""
    data = request.get_json() or {}
    r = _td3().compute_reward(
        stability_score    = float(data.get("stability_score",    0.8)),
        efficiency_score   = float(data.get("efficiency_score",   0.7)),
        actuator_cycles    = float(data.get("actuator_cycles",    0.1)),
        boundary_violation = float(data.get("boundary_violation", 0.0)),
        oscillation        = float(data.get("oscillation",        0.05)),
    )
    return jsonify({"status": "success", "reward": round(r, 6)})

@app.route("/td3/stats", methods=["GET"])
def td3_stats():
    return jsonify({"status": "success", "stats": _td3().get_stats()})

@app.route("/td3/checkpoint", methods=["GET"])
def td3_checkpoint():
    return jsonify({"status": "success", "checkpoint": _td3().checkpoint()})


# ═══════════════════════════════════════════════════════════════════════════════
# v4.1 MODULE 18 — IEC 61131-3 FULL AST PARSER
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/ast/parse", methods=["POST"])
def ast_parse():
    """Parse IEC 61131-3 ST source → AST + CFG + DDG + SDG."""
    data   = request.get_json() or {}
    source = data.get("source", "PROGRAM empty END_PROGRAM")
    name   = data.get("program_name", "UNKNOWN")
    result = _ast().parse(source, name)
    return jsonify({"status": "success", "result": result.to_dict()})

@app.route("/ast/detect-patterns", methods=["POST"])
def ast_detect():
    """Parse + return detected PID blocks, interlocks, dead code."""
    data   = request.get_json() or {}
    source = data.get("source", "PROGRAM empty END_PROGRAM")
    result = _ast().parse(source)
    return jsonify({
        "status":      "success",
        "pid_blocks":  [{"name": p.name, "kp": p.kp, "ki": p.ki, "kd": p.kd}
                         for p in result.pid_blocks],
        "interlocks":  [{"name": i.name, "condition": i.condition,
                          "safety": i.safety} for i in result.interlocks],
        "dead_code":   [{"description": d.description, "line": d.line}
                         for d in result.dead_code],
        "var_count":   result.var_count,
        "stmt_count":  result.stmt_count,
    })

@app.route("/ast/cfg", methods=["POST"])
def ast_cfg():
    """Return Control Flow Graph for a given ST program."""
    data   = request.get_json() or {}
    source = data.get("source", "PROGRAM empty END_PROGRAM")
    result = _ast().parse(source)
    return jsonify({
        "status":    "success",
        "cfg_nodes": len(result.cfg),
        "cfg":       {k: {"kind": v.kind, "successors": v.successors,
                           "stmts": v.stmts}
                      for k, v in result.cfg.items()},
    })

@app.route("/ast/stats", methods=["GET"])
def ast_stats():
    return jsonify({"status": "success", "stats": _ast().get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# v4.1 MODULE 19 — EXTENDED PHYSICS TWIN
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/ephys/step", methods=["POST"])
def ephys_step():
    """Advance all physics blocks by dt seconds."""
    data    = request.get_json() or {}
    inputs  = data.get("inputs", {})
    dt      = float(data.get("dt", 0.1))
    method  = data.get("method", "rk4")
    ss      = _ephys().step(inputs, dt, method)
    return jsonify({
        "status":         "success",
        "step":           ss.step,
        "t":              ss.t,
        "state":          ss.state,
        "hard_violation": ss.hard_violation,
        "fault_active":   ss.fault_active,
    })

@app.route("/ephys/episode", methods=["POST"])
def ephys_episode():
    """Run a complete simulation episode with optional fault injection."""
    data    = request.get_json() or {}
    n_steps = int(data.get("n_steps", 100))
    dt      = float(data.get("dt", 0.1))
    result  = _ephys().run_episode(n_steps=n_steps, dt=dt)
    return jsonify({"status": "success", "result": result})

@app.route("/ephys/evaluate-mutation", methods=["POST"])
def ephys_evaluate_mutation():
    """Evaluate parameter mutation across Monte Carlo episodes."""
    data         = request.get_json() or {}
    param_deltas = data.get("param_deltas", {})
    n_runs       = int(data.get("n_runs", 5))
    promoted, metrics = _ephys().evaluate_mutation(param_deltas, n_runs)
    return jsonify({"status": "success", "promoted": promoted, "metrics": metrics})

@app.route("/ephys/pump", methods=["POST"])
def ephys_pump():
    """Direct pump algebraic computation (Q, H, P)."""
    from python.sentinel.extended_physics import PumpBlock
    data   = request.get_json() or {}
    rpm    = float(data.get("RPM", 1450))
    params = data.get("params", {})
    result = PumpBlock().compute(rpm, params)
    return jsonify({"status": "success", "result": result})

@app.route("/ephys/battery-voltage", methods=["POST"])
def ephys_battery_voltage():
    """Compute battery terminal voltage."""
    from python.sentinel.extended_physics import BatteryBlock
    data  = request.get_json() or {}
    soc   = float(data.get("SOC",   0.8))
    I     = float(data.get("I",     10.0))
    R_int = float(data.get("R_int", 0.05))
    v     = BatteryBlock().compute_voltage(soc, I, R_int)
    return jsonify({"status": "success", "voltage": round(v, 4)})

@app.route("/ephys/stats", methods=["GET"])
def ephys_stats():
    return jsonify({"status": "success", "stats": _ephys().get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# v4.1 MODULE 20 — VMWARE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/vmw/vms", methods=["GET"])
def vmw_list():
    return jsonify({"status": "success", "vms": _vmw().list_vms()})

@app.route("/vmw/vm/<vm_id>", methods=["GET"])
def vmw_get(vm_id):
    vm = _vmw().get_vm(vm_id)
    if not vm:
        return jsonify({"status": "error", "error": "VM not found"}), 404
    return jsonify({"status": "success", "vm": vm})

@app.route("/vmw/snapshot", methods=["POST"])
def vmw_snapshot():
    data = request.get_json() or {}
    result = _vmw().create_snapshot(
        vm_id    = data.get("vm_id", "VM-C"),
        snap_name= data.get("snap_name", f"snap_{int(time.time())}"),
        actor    = data.get("actor", "api"),
    )
    return jsonify({"status": "success" if result.get("ok") else "error",
                    "result": result})

@app.route("/vmw/revert", methods=["POST"])
def vmw_revert():
    data = request.get_json() or {}
    result = _vmw().revert_snapshot(
        vm_id     = data.get("vm_id"),
        snap_name = data.get("snap_name"),
        actor     = data.get("actor", "api"),
    )
    return jsonify({"status": "success" if result.get("ok") else "error",
                    "result": result})

@app.route("/vmw/clone", methods=["POST"])
def vmw_clone():
    data = request.get_json() or {}
    result = _vmw().clone_vm(
        src_vm_id      = data.get("src_vm_id", "VM-C"),
        clone_name     = data.get("clone_name"),
        isolate_network= bool(data.get("isolate_network", True)),
        actor          = data.get("actor", "api"),
    )
    return jsonify({"status": "success" if result.get("ok") else "error",
                    "result": result})

@app.route("/vmw/mutation/begin", methods=["POST"])
def vmw_mutation_begin():
    data = request.get_json() or {}
    mid  = _vmw().begin_mutation(
        source_vm    = data.get("source_vm", "VM-C"),
        param_deltas = data.get("param_deltas", {}),
        actor        = data.get("actor", "api"),
    )
    return jsonify({"status": "success", "mutation_id": mid})

@app.route("/vmw/mutation/promote", methods=["POST"])
def vmw_mutation_promote():
    data   = request.get_json() or {}
    result = _vmw().promote_mutation(
        mutation_id  = data.get("mutation_id"),
        approval_code= data.get("approval_code", ""),
    )
    return jsonify({"status": "success" if result.get("ok") else "error",
                    "result": result})

@app.route("/vmw/audit", methods=["GET"])
def vmw_audit():
    limit = int(request.args.get("limit", 50))
    return jsonify({"status": "success", "audit": _vmw().get_audit_log(limit)})

@app.route("/vmw/stats", methods=["GET"])
def vmw_stats():
    return jsonify({"status": "success", "stats": _vmw().get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# v4.1 MODULE 21 — FORMAL VERIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/fv/lyapunov", methods=["POST"])
def fv_lyapunov():
    """Verify Lyapunov stability of a linearized system matrix A."""
    data = request.get_json() or {}
    A    = data.get("A")
    if not A:
        # Default: 2x2 stable system
        A = [[0.9, 0.1], [-0.05, 0.85]]
    mid    = data.get("mutation_id", "api_check")
    result = _fv().verify_linearized(A, mutation_id=mid)
    return jsonify({"status": "success", "result": result.to_dict()})

@app.route("/fv/barrier", methods=["POST"])
def fv_barrier():
    """Verify barrier certificate via sampling."""
    import math
    data     = request.get_json() or {}
    safe_set = data.get("safe_set", [[-1.0, 1.0], [-1.0, 1.0]])
    n_samples= int(data.get("n_samples", 200))
    mid      = data.get("mutation_id", "api_barrier")
    # Use quadratic B(x) = sum(xi^2) - bound as default
    bound    = float(data.get("barrier_bound", 1.5))
    def B(x): return bound - sum(xi*xi for xi in x)
    def f(x): return [-0.1*xi for xi in x]   # stable attractor
    result = _fv().verify_barrier(B, f, safe_set, n_samples, mutation_id=mid)
    return jsonify({"status": "success", "result": result.to_dict()})

@app.route("/fv/full", methods=["POST"])
def fv_full():
    """Full verification: Lyapunov + optional barrier."""
    data = request.get_json() or {}
    A    = data.get("A", [[0.9, 0.0], [0.0, 0.85]])
    mid  = data.get("mutation_id", "api_full")
    result = _fv().verify_full(A, mutation_id=mid)
    return jsonify({"status": "success", "result": result.to_dict()})

@app.route("/fv/ledger", methods=["GET"])
def fv_ledger():
    limit = int(request.args.get("limit", 50))
    return jsonify({
        "status":  "success",
        "ledger":  _fv().ledger.get_all(limit),
        "intact":  _fv().ledger.verify_integrity(),
    })

@app.route("/fv/stats", methods=["GET"])
def fv_stats():
    return jsonify({"status": "success", "stats": _fv().get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# v4.1 MODULE 22 — BYZANTINE FEDERATED AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/byz/register-site", methods=["POST"])
def byz_register():
    data   = request.get_json() or {}
    result = _byz().register_site(
        site_id  = data.get("site_id", "site_1"),
        metadata = data.get("metadata", {}),
    )
    return jsonify({"status": "success", "result": result})

@app.route("/byz/aggregate", methods=["POST"])
def byz_aggregate():
    """Aggregate gradient updates from multiple sites."""
    from python.sentinel.byzantine_aggregator import SiteUpdate
    data    = request.get_json() or {}
    raw_upd = data.get("updates", [])
    method  = data.get("method", "trimmed_mean")
    updates = [
        SiteUpdate(
            site_id    = u.get("site_id", f"site_{i}"),
            gradient   = u.get("gradient", [0.0] * 8),
            param_dim  = u.get("param_dim", 8),
            step       = int(u.get("step", 0)),
            performance= float(u.get("performance", 0.0)),
            n_samples  = int(u.get("n_samples", 1)),
        )
        for i, u in enumerate(raw_upd)
    ]
    result = _byz().aggregate(updates, method=method)
    return jsonify({"status": "success", "result": result.to_dict()})

@app.route("/byz/export-params", methods=["GET"])
def byz_export():
    return jsonify({"status": "success", "params": _byz().export_global_params()})

@app.route("/byz/leaderboard", methods=["GET"])
def byz_leaderboard():
    return jsonify({"status": "success",
                    "leaderboard": _byz().get_site_leaderboard()})

@app.route("/byz/anomalies", methods=["GET"])
def byz_anomalies():
    limit = int(request.args.get("limit", 50))
    return jsonify({"status": "success",
                    "anomalies": _byz().get_anomaly_log(limit)})

@app.route("/byz/stats", methods=["GET"])
def byz_stats():
    return jsonify({"status": "success", "stats": _byz().get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# v4.1 MODULE 23 — MUTATION GOVERNANCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/gov/begin", methods=["POST"])
def gov_begin():
    """Begin a new 11-step mutation pipeline."""
    data   = request.get_json() or {}
    mid    = _mgov().begin_mutation(
        plc_program  = data.get("plc_program",  "PLC_PROG"),
        param_deltas = data.get("param_deltas", {}),
    )
    return jsonify({"status": "success", "mutation_id": mid})

@app.route("/gov/step", methods=["POST"])
def gov_step():
    """Execute a specific pipeline step (1-7, 9-10)."""
    data    = request.get_json() or {}
    mid     = data.get("mutation_id")
    step_id = int(data.get("step_id", 1))
    context = data.get("context", {})
    result  = _mgov().run_step(mid, step_id, context=context)
    return jsonify({"status": "success" if result.get("passed") else "error",
                    "result": result})

@app.route("/gov/approve", methods=["POST"])
def gov_approve():
    """Step 8: Human approval gate — Baron Marco Paolo Ialongo only."""
    data   = request.get_json() or {}
    result = _mgov().approve(
        mutation_id  = data.get("mutation_id"),
        approval_code= data.get("approval_code", ""),
    )
    return jsonify({
        "status": "success" if result.get("approved") else "error",
        "result": result,
    })

@app.route("/gov/release-key", methods=["POST"])
def gov_release_key():
    """Step 11: Release production deployment key."""
    data   = request.get_json() or {}
    result = _mgov().release_production_key(data.get("mutation_id"))
    return jsonify({
        "status": "success" if result.get("deployed") else "error",
        "result": result,
    })

@app.route("/gov/mutation/<mutation_id>", methods=["GET"])
def gov_get_mutation(mutation_id):
    m = _mgov().get_mutation(mutation_id)
    if not m:
        return jsonify({"status": "error", "error": "Not found"}), 404
    return jsonify({"status": "success", "mutation": m})

@app.route("/gov/mutation/<mutation_id>/evidence", methods=["GET"])
def gov_get_evidence(mutation_id):
    ev = _mgov().get_full_evidence(mutation_id)
    if ev is None:
        return jsonify({"status": "error", "error": "Not found"}), 404
    return jsonify({"status": "success", "evidence": ev})

@app.route("/gov/list", methods=["GET"])
def gov_list():
    status = request.args.get("status")
    limit  = int(request.args.get("limit", 50))
    return jsonify({"status": "success",
                    "mutations": _mgov().list_mutations(status, limit)})

@app.route("/gov/stats", methods=["GET"])
def gov_stats():
    return jsonify({"status": "success", "stats": _mgov().get_stats()})


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# v4.1 ENDPOINTS — CIEC Advanced (6 new modules, 28 new endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_td3():
    if not hasattr(app, "_td3"):
        from .td3_controller import TD3IndustrialController
        app._td3 = TD3IndustrialController(state_dim=256)
    return app._td3

def _get_ast_parser():
    if not hasattr(app, "_ast_parser"):
        from .ast_parser import IEC61131ASTParser
        app._ast_parser = IEC61131ASTParser()
    return app._ast_parser

def _get_ext_physics():
    if not hasattr(app, "_ext_physics"):
        from .extended_physics import ExtendedPhysicsTwin, FaultConfig
        app._ext_physics = ExtendedPhysicsTwin()
        app._FaultConfig = FaultConfig
    return app._ext_physics

def _get_vmware():
    if not hasattr(app, "_vmware"):
        from .vmware_orchestrator import VMwareOrchestrator
        app._vmware = VMwareOrchestrator()
    return app._vmware

def _get_formal():
    if not hasattr(app, "_formal"):
        from .formal_verification import FormalVerificationEngine
        app._formal = FormalVerificationEngine()
    return app._formal

def _get_byzantine():
    if not hasattr(app, "_byzantine"):
        from .byzantine_aggregator import ByzantineFederatedAggregator
        app._byzantine = ByzantineFederatedAggregator(f_tolerance=1)
    return app._byzantine

def _get_governance():
    if not hasattr(app, "_governance"):
        from .mutation_governance import MutationGovernanceEngine
        app._governance = MutationGovernanceEngine()
    return app._governance


# ── TD3 Controller ────────────────────────────────────────────────────────────

@app.route("/td3/act", methods=["POST"])
def td3_act():
    """POST /td3/act — select bounded PLC parameter-shift action"""
    data  = request.get_json() or {}
    state = data.get("state", [0.0] * 256)
    det   = data.get("deterministic", False)
    noise = data.get("exploration_noise", 0.05)
    try:
        ctrl = _get_td3()
        action, info = ctrl.select_action(state, deterministic=det,
                                          exploration_noise=noise)
        return jsonify({"status": "ok", "action": action, "info": info})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/td3/observe", methods=["POST"])
def td3_observe():
    """POST /td3/observe — push transition into replay buffer"""
    data = request.get_json() or {}
    try:
        ctrl = _get_td3()
        ctrl.observe(
            state      = data.get("state", []),
            action     = data.get("action", []),
            reward     = float(data.get("reward", 0.0)),
            next_state = data.get("next_state", []),
            done       = bool(data.get("done", False)),
            cost       = float(data.get("cost", 0.0)),
        )
        return jsonify({"status": "ok",
                        "buffer_size": len(ctrl.buffer)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/td3/update", methods=["POST"])
def td3_update():
    """POST /td3/update — perform one TD3 training step"""
    try:
        ctrl   = _get_td3()
        result = ctrl.update()
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/td3/reward", methods=["POST"])
def td3_reward():
    """POST /td3/reward — compute CIEC reward from KPI scores"""
    data = request.get_json() or {}
    try:
        from .td3_controller import TD3IndustrialController
        r = TD3IndustrialController.compute_reward(
            stability_score    = float(data.get("stability_score",    0.8)),
            efficiency_score   = float(data.get("efficiency_score",   0.7)),
            actuator_cycles    = float(data.get("actuator_cycles",    0.1)),
            boundary_violation = float(data.get("boundary_violation", 0.0)),
            oscillation        = float(data.get("oscillation",        0.05)),
        )
        return jsonify({"status": "ok", "reward": round(r, 6)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/td3/stats", methods=["GET"])
def td3_stats():
    """GET /td3/stats — TD3 training statistics"""
    try:
        return jsonify({"status": "ok", "stats": _get_td3().get_stats()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ── IEC 61131-3 AST Parser ────────────────────────────────────────────────────

@app.route("/ast/parse", methods=["POST"])
def ast_parse():
    """POST /ast/parse — full IEC 61131-3 ST parse → AST+CFG+DDG+SDG"""
    data   = request.get_json() or {}
    source = data.get("source", "")
    if not source:
        return jsonify({"status": "error", "error": "source required"}), 400
    try:
        result = _get_ast_parser().parse(source, data.get("program_name", "PROG"))
        return jsonify({"status": "ok", "result": result.to_dict()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/ast/patterns", methods=["POST"])
def ast_patterns():
    """POST /ast/patterns — detect PID blocks, interlocks, dead code"""
    data   = request.get_json() or {}
    source = data.get("source", "")
    if not source:
        return jsonify({"status": "error", "error": "source required"}), 400
    try:
        result = _get_ast_parser().parse(source)
        return jsonify({
            "status":     "ok",
            "pid_blocks": [{"name": p.name, "kp": p.kp, "ki": p.ki, "kd": p.kd}
                           for p in result.pid_blocks],
            "interlocks": [{"name": i.name, "condition": i.condition, "safety": i.safety}
                           for i in result.interlocks],
            "dead_code":  [{"description": d.description, "line": d.line}
                           for d in result.dead_code],
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/ast/graphs", methods=["POST"])
def ast_graphs():
    """POST /ast/graphs — return CFG/DDG/SDG graph summaries"""
    data   = request.get_json() or {}
    source = data.get("source", "")
    if not source:
        return jsonify({"status": "error", "error": "source required"}), 400
    try:
        result = _get_ast_parser().parse(source)
        return jsonify({
            "status":    "ok",
            "cfg_nodes": len(result.cfg),
            "ddg_edges": len(result.ddg_edges),
            "sdg_edges": len(result.sdg_edges),
            "var_count": result.var_count,
            "stmt_count":result.stmt_count,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/ast/stats", methods=["GET"])
def ast_stats():
    """GET /ast/stats — parser cache and usage stats"""
    try:
        return jsonify({"status": "ok", "stats": _get_ast_parser().get_stats()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ── Extended Physics Twin ─────────────────────────────────────────────────────

@app.route("/physics/step", methods=["POST"])
def physics_step():
    """POST /physics/step — advance all physics blocks by dt"""
    data = request.get_json() or {}
    try:
        twin   = _get_ext_physics()
        inputs = data.get("inputs", {})
        dt     = float(data.get("dt", 0.1))
        method = data.get("method", "rk4")
        ss     = twin.step(inputs, dt, method)
        return jsonify({
            "status":          "ok",
            "step":            ss.step,
            "t":               round(ss.t, 4),
            "state":           {k: round(v, 6) for k, v in ss.state.items()},
            "hard_violation":  ss.hard_violation,
            "fault_active":    ss.fault_active,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/physics/episode", methods=["POST"])
def physics_episode():
    """POST /physics/episode — run full episode with optional fault injection"""
    data = request.get_json() or {}
    try:
        twin = _get_ext_physics()
        FaultConfig = app._FaultConfig if hasattr(app, "_FaultConfig") else None
        faults = []
        for fc in data.get("faults", []):
            if FaultConfig:
                faults.append(FaultConfig(
                    category   = fc.get("category", "sensor_bias"),
                    target     = fc.get("target", "Q_in"),
                    magnitude  = float(fc.get("magnitude", 1.0)),
                    onset_step = int(fc.get("onset_step", 0)),
                    duration   = int(fc.get("duration", -1)),
                ))
        result = twin.run_episode(
            n_steps = int(data.get("n_steps", 100)),
            dt      = float(data.get("dt", 0.1)),
            faults  = faults,
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/physics/evaluate-mutation", methods=["POST"])
def physics_evaluate_mutation():
    """POST /physics/evaluate-mutation — Monte Carlo mutation evaluation"""
    data = request.get_json() or {}
    try:
        twin = _get_ext_physics()
        promoted, metrics = twin.evaluate_mutation(
            param_deltas      = data.get("param_deltas", {}),
            n_runs            = int(data.get("n_runs", 5)),
            fault_categories  = data.get("fault_categories"),
        )
        return jsonify({"status": "ok", "promoted": promoted, "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/physics/stats", methods=["GET"])
def physics_stats():
    """GET /physics/stats — physics twin statistics"""
    try:
        return jsonify({"status": "ok", "stats": _get_ext_physics().get_stats()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ── VMware Orchestrator ───────────────────────────────────────────────────────

@app.route("/vmware/vms", methods=["GET"])
def vmware_list():
    """GET /vmware/vms — list all VMs in inventory"""
    try:
        return jsonify({"status": "ok", "vms": _get_vmware().list_vms()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/vmware/snapshot", methods=["POST"])
def vmware_snapshot():
    """POST /vmware/snapshot — create VM snapshot"""
    data = request.get_json() or {}
    try:
        result = _get_vmware().create_snapshot(
            data.get("vm_id", "VM-C"),
            data.get("snap_name", "auto_snap"),
            data.get("actor", "kiswarm"),
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/vmware/clone", methods=["POST"])
def vmware_clone():
    """POST /vmware/clone — clone VM for mutation testing"""
    data = request.get_json() or {}
    try:
        result = _get_vmware().clone_vm(
            data.get("src_vm_id", "VM-C"),
            data.get("clone_name"),
            data.get("isolate_network", True),
            data.get("actor", "kiswarm"),
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/vmware/mutation/begin", methods=["POST"])
def vmware_mutation_begin():
    """POST /vmware/mutation/begin — start VM mutation cycle"""
    data = request.get_json() or {}
    try:
        mid = _get_vmware().begin_mutation(
            data.get("source_vm", "VM-C"),
            data.get("param_deltas", {}),
            data.get("actor", "kiswarm"),
        )
        return jsonify({"status": "ok", "mutation_id": mid})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/vmware/mutation/promote", methods=["POST"])
def vmware_mutation_promote():
    """POST /vmware/mutation/promote — promote mutation to production (requires approval)"""
    data = request.get_json() or {}
    try:
        result = _get_vmware().promote_mutation(
            data.get("mutation_id", ""),
            data.get("approval_code", ""),
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/vmware/audit", methods=["GET"])
def vmware_audit():
    """GET /vmware/audit — VM operation audit log"""
    try:
        limit = int(request.args.get("limit", 50))
        return jsonify({"status": "ok",
                        "audit": _get_vmware().get_audit_log(limit)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/vmware/stats", methods=["GET"])
def vmware_stats():
    """GET /vmware/stats — VMware orchestrator statistics"""
    try:
        return jsonify({"status": "ok", "stats": _get_vmware().get_stats()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ── Formal Verification ───────────────────────────────────────────────────────

@app.route("/formal/lyapunov", methods=["POST"])
def formal_lyapunov():
    """POST /formal/lyapunov — Lyapunov stability check on system matrix A"""
    data = request.get_json() or {}
    A    = data.get("A")
    if not A:
        return jsonify({"status": "error", "error": "System matrix A required"}), 400
    try:
        result = _get_formal().verify_linearized(
            A           = A,
            mutation_id = data.get("mutation_id", "api_call"),
        )
        return jsonify({"status": "ok", "result": result.to_dict()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/formal/barrier", methods=["POST"])
def formal_barrier():
    """POST /formal/barrier — barrier certificate verification (sampling-based)"""
    data = request.get_json() or {}
    try:
        safe_set = [tuple(pair) for pair in data.get("safe_set", [[-1, 1], [-1, 1]])]
        decay    = float(data.get("decay", 0.1))
        # Simple quadratic barrier: B(x) = 1 - sum(xi^2/ri^2)
        def B(x):
            return 1.0 - sum((x[i] / (safe_set[i][1] or 1.0))**2
                             for i in range(min(len(x), len(safe_set))))
        # Simple stable system: dx/dt = -decay * x
        def f(x):
            return [-decay * xi for xi in x]
        result = _get_formal().verify_barrier(
            B           = B,
            f           = f,
            safe_set    = safe_set,
            n_samples   = int(data.get("n_samples", 200)),
            mutation_id = data.get("mutation_id", "api_call"),
        )
        return jsonify({"status": "ok", "result": result.to_dict()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/formal/ledger", methods=["GET"])
def formal_ledger():
    """GET /formal/ledger — mutation verification ledger"""
    try:
        limit = int(request.args.get("limit", 50))
        engine = _get_formal()
        return jsonify({
            "status":  "ok",
            "entries": engine.ledger.get_all(limit),
            "intact":  engine.ledger.verify_integrity(),
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/formal/stats", methods=["GET"])
def formal_stats():
    """GET /formal/stats — formal verification statistics"""
    try:
        return jsonify({"status": "ok", "stats": _get_formal().get_stats()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ── Byzantine Federated Aggregator ────────────────────────────────────────────

@app.route("/federated/register", methods=["POST"])
def federated_register():
    """POST /federated/register — register a site in the federated mesh"""
    data = request.get_json() or {}
    try:
        result = _get_byzantine().register_site(
            data.get("site_id", f"site_{int(time.time())}"),
            data.get("metadata", {}),
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/federated/aggregate", methods=["POST"])
def federated_aggregate():
    """POST /federated/aggregate — Byzantine-tolerant gradient aggregation"""
    data = request.get_json() or {}
    try:
        from .byzantine_aggregator import SiteUpdate
        updates = []
        for u in data.get("updates", []):
            updates.append(SiteUpdate(
                site_id     = u.get("site_id", "unknown"),
                gradient    = u.get("gradient", []),
                param_dim   = len(u.get("gradient", [])),
                step        = int(u.get("step", 0)),
                performance = float(u.get("performance", 0.0)),
                n_samples   = int(u.get("n_samples", 1)),
            ))
        result = _get_byzantine().aggregate(updates, data.get("method"))
        return jsonify({"status": "ok", "result": result.to_dict()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/federated/params", methods=["GET"])
def federated_params():
    """GET /federated/params — export current global parameters"""
    try:
        return jsonify({"status": "ok",
                        "params": _get_byzantine().export_global_params()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/federated/anomalies", methods=["GET"])
def federated_anomalies():
    """GET /federated/anomalies — Byzantine anomaly log"""
    try:
        limit = int(request.args.get("limit", 50))
        return jsonify({"status": "ok",
                        "anomalies": _get_byzantine().get_anomaly_log(limit)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/federated/leaderboard", methods=["GET"])
def federated_leaderboard():
    """GET /federated/leaderboard — site trust score leaderboard"""
    try:
        return jsonify({"status": "ok",
                        "leaderboard": _get_byzantine().get_site_leaderboard()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/federated/stats", methods=["GET"])
def federated_stats():
    """GET /federated/stats — federated aggregator statistics"""
    try:
        return jsonify({"status": "ok", "stats": _get_byzantine().get_stats()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ── Mutation Governance Pipeline ──────────────────────────────────────────────

@app.route("/governance/begin", methods=["POST"])
def governance_begin():
    """POST /governance/begin — start mutation governance pipeline"""
    data = request.get_json() or {}
    try:
        mid = _get_governance().begin_mutation(
            plc_program  = data.get("plc_program", "UNKNOWN"),
            param_deltas = data.get("param_deltas", {}),
        )
        return jsonify({"status": "ok", "mutation_id": mid,
                        "next_step": 1, "total_steps": 11})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/governance/step", methods=["POST"])
def governance_step():
    """POST /governance/step — execute next pipeline step"""
    data = request.get_json() or {}
    try:
        result = _get_governance().run_step(
            mutation_id = data.get("mutation_id", ""),
            step_id     = int(data.get("step_id", 1)),
            context     = data.get("context", {}),
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/governance/approve", methods=["POST"])
def governance_approve():
    """POST /governance/approve — Step 8 human approval gate (Baron Marco Paolo Ialongo only)"""
    data = request.get_json() or {}
    try:
        result = _get_governance().approve(
            mutation_id   = data.get("mutation_id", ""),
            approval_code = data.get("approval_code", ""),
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/governance/release", methods=["POST"])
def governance_release():
    """POST /governance/release — Step 11 production key release"""
    data = request.get_json() or {}
    try:
        result = _get_governance().release_production_key(
            data.get("mutation_id", "")
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/governance/mutation/<mutation_id>", methods=["GET"])
def governance_mutation(mutation_id):
    """GET /governance/mutation/<id> — get mutation pipeline record"""
    try:
        m = _get_governance().get_mutation(mutation_id)
        if not m:
            return jsonify({"status": "error", "error": "Not found"}), 404
        return jsonify({"status": "ok", "mutation": m})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/governance/list", methods=["GET"])
def governance_list():
    """GET /governance/list — list mutations with optional status filter"""
    try:
        status = request.args.get("status")
        limit  = int(request.args.get("limit", 50))
        return jsonify({"status": "ok",
                        "mutations": _get_governance().list_mutations(status, limit)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/governance/stats", methods=["GET"])
def governance_stats():
    """GET /governance/stats — governance engine statistics"""
    try:
        return jsonify({"status": "ok", "stats": _get_governance().get_stats()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT TIME HELPER
# ─────────────────────────────────────────────────────────────────────────────
import time as _time_mod
time = _time_mod


# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(_):
    return jsonify({
        "status":  "error",
        "error":   "Endpoint not found",
        "version": "4.1",
        "modules": {
            "v2.1 Sentinel Intelligence (6 modules, 17 endpoints)": [
                "POST /sentinel/extract", "POST /sentinel/debate",
                "GET  /sentinel/search",  "GET  /sentinel/status",
                "POST /firewall/scan",
                "GET  /decay/scan", "GET  /decay/record/<hash_id>", "POST /decay/revalidate",
                "GET  /ledger/status", "GET  /ledger/verify", "GET  /ledger/proof/<hash_id>",
                "POST /conflict/analyze", "POST /conflict/quick",
                "GET  /tracker/leaderboard", "GET  /tracker/model/<n>",
                "POST /tracker/validate", "POST /guard/assess",
            ],
            "v3.0 Industrial AI (4 modules, 13 endpoints)": [
                "POST /fuzzy/classify",  "POST /fuzzy/update",
                "POST /fuzzy/tune",      "GET  /fuzzy/stats",
                "POST /rl/act",          "POST /rl/learn",   "GET  /rl/stats",
                "POST /twin/evaluate",   "GET  /twin/stats",
                "POST /mesh/share",      "POST /mesh/register",
                "GET  /mesh/leaderboard","GET  /mesh/stats",
            ],
            "v4.0 CIEC Cognitive Industrial Core (6 modules, 28 endpoints)": [
                "POST /plc/parse",             "GET  /plc/stats",
                "POST /scada/push",            "POST /scada/ingest-history",
                "GET  /scada/state",           "GET  /scada/anomalies",
                "GET  /scada/stats",
                "POST /ciec-twin/run",         "POST /ciec-twin/evaluate",
                "GET  /ciec-twin/stats",
                "POST /constraints/validate",  "POST /constraints/check-state",
                "GET  /constraints/list",      "GET  /constraints/violations",
                "GET  /constraints/stats",
                "POST /kg/add-pid",            "POST /kg/add-failure",
                "POST /kg/find-similar",       "POST /kg/find-by-symptoms",
                "GET  /kg/recurring-patterns", "GET  /kg/export-bundle",
                "POST /kg/import-bundle",      "GET  /kg/nodes",
                "GET  /kg/stats",
                "POST /ciec-rl/act",           "POST /ciec-rl/observe",
                "POST /ciec-rl/update",        "GET  /ciec-rl/stats",
            ],
            "v4.1 Advanced CIEC (7 modules, 28 endpoints)": [
                "POST /td3/act",               "POST /td3/observe",
                "POST /td3/update",            "POST /td3/reward",
                "GET  /td3/stats",
                "POST /ast/parse",             "POST /ast/patterns",
                "POST /ast/graphs",            "GET  /ast/stats",
                "POST /physics/step",          "POST /physics/episode",
                "POST /physics/evaluate-mutation","GET  /physics/stats",
                "GET  /vmware/vms",            "POST /vmware/snapshot",
                "POST /vmware/clone",          "POST /vmware/mutation/begin",
                "POST /vmware/mutation/promote","GET  /vmware/audit",
                "GET  /vmware/stats",
                "POST /formal/lyapunov",       "POST /formal/barrier",
                "GET  /formal/ledger",         "GET  /formal/stats",
                "POST /federated/register",    "POST /federated/aggregate",
                "GET  /federated/params",      "GET  /federated/anomalies",
                "GET  /federated/leaderboard", "GET  /federated/stats",
                "POST /governance/begin",      "POST /governance/step",
                "POST /governance/approve",    "POST /governance/release",
                "GET  /governance/mutation/<id>","GET  /governance/list",
                "GET  /governance/stats",
            ],
        },
        "total_endpoints": 87,
        "system_health":   "GET /health",
    }), 404


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  KISWARM v4.1 — Advanced CIEC with TD3/AST/Formal/VMware   ║")
    logger.info("║  Port: 11436  |  Modules: 23  |  Endpoints: 87            ║")
    logger.info("║  v2.1→v4.0: 59 endpoints  +  v4.1: TD3·AST·FV·VMware     ║")
    logger.info("║  TD3-RL · IEC61131-AST · Physics · VMware · Formal · Gov  ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    app.run(host="127.0.0.1", port=11436, debug=False, threaded=True)
