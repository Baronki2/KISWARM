"""
KISWARM v2.2 — SENTINEL API SERVER
REST interface for the Sentinel Bridge AKE system.
Port: 11436 (alongside Ollama:11434, Tool Proxy:11435)

Endpoints:
  POST /sentinel/extract         — Trigger AKE for a query
  POST /sentinel/debate          — Trigger Swarm Debate for conflicting sources
  GET  /sentinel/search          — Search existing swarm knowledge
  GET  /sentinel/status          — Sentinel system status

  --- v2.2 Advanced Modules ---
  POST /firewall/scan            — Module 6: Scan content for adversarial patterns
  GET  /decay/scan               — Module 2: Run decay scan, return revalidation list
  GET  /decay/record/<hash_id>   — Module 2: Confidence for a single entry
  GET  /ledger/status            — Module 4: Ledger summary + root hash
  GET  /ledger/verify            — Module 4: Full Merkle tamper detection
  GET  /ledger/proof/<hash_id>   — Module 4: Merkle inclusion proof
  POST /conflict/analyze         — Module 1: Semantic conflict detection on content list
  POST /conflict/quick           — Module 1: Quick two-text contradiction check
  GET  /tracker/leaderboard      — Module 3: Model reliability leaderboard
  GET  /tracker/model/<name>     — Module 3: Per-model performance stats
  POST /guard/assess             — Module 5: Retrieval trust assessment
  GET  /health                   — Health check

Author: KISWARM Project (Baron Marco Paolo Ialongo)
Version: 2.2
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

_start  = datetime.datetime.now()
_stats  = {
    "extractions": 0, "debates": 0, "searches": 0,
    "firewall_scans": 0, "firewall_blocked": 0,
    "decay_scans": 0, "ledger_verifications": 0,
    "guard_assessments": 0,
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


# ═══════════════════════════════════════════════════════════════
# CORE ENDPOINTS (v2.1)
# ═══════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    uptime = (datetime.datetime.now() - _start).total_seconds()
    return jsonify({
        "status":   "active",
        "service":  "KISWARM-SENTINEL-BRIDGE",
        "version":  "2.2-EMS",
        "port":     11436,
        "uptime":   round(uptime, 1),
        "stats":    _stats,
        "modules":  ["AKE", "SemanticConflict", "KnowledgeDecay",
                     "ModelTracker", "CryptoLedger", "RetrievalGuard", "PromptFirewall"],
        "timestamp": datetime.datetime.now().isoformat(),
    })


@app.route("/sentinel/extract", methods=["POST"])
def extract():
    data  = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"status": "error", "error": "query is required"}), 400

    # Firewall check on query
    fw = _firewall.scan_query(query)
    if fw.blocked:
        _stats["firewall_blocked"] += 1
        return jsonify({
            "status": "blocked",
            "reason": "Query failed adversarial firewall",
            "threat_types": fw.threat_types,
        }), 403

    force     = bool(data.get("force", False))
    threshold = float(data.get("threshold", 0.85))
    _bridge.ckm.threshold = threshold

    try:
        result = run_async(_bridge.run(query, force=force))
        _stats["extractions"] += 1

        # Register in decay engine if successful
        if result.get("status") == "success" and result.get("hash_id"):
            source_names = [s.get("source", "") for s in result.get("source_list", [])]
            category = _decay.infer_category(source_names, query)
            _decay.register(result["hash_id"], query, result.get("confidence", 0.5), category)

        return jsonify(result)
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/sentinel/debate", methods=["POST"])
def debate():
    data = request.get_json() or {}
    for field_name in ["query", "content_a", "content_b"]:
        if not data.get(field_name):
            return jsonify({"status": "error", "error": f"{field_name} is required"}), 400

    # Firewall both payloads
    for key in ["content_a", "content_b"]:
        fw = _firewall.scan(data[key], source=data.get(f"source_{key[-1]}", "unknown"))
        if fw.blocked:
            _stats["firewall_blocked"] += 1
            return jsonify({
                "status": "blocked",
                "reason": f"{key} failed adversarial firewall",
                "threat_types": fw.threat_types,
            }), 403

    try:
        verdict = run_async(_debate.debate(
            query=data["query"],
            content_a=data["content_a"],
            content_b=data["content_b"],
            source_a_name=data.get("source_a", "Source A"),
            source_b_name=data.get("source_b", "Source B"),
        ))
        _stats["debates"] += 1

        # Record in model tracker
        if verdict.vote_tally:
            import uuid
            debate_id = str(uuid.uuid4())[:8]
            _tracker.record_debate(
                debate_id=debate_id,
                query=data["query"],
                votes={},   # votes not surfaced from verdict yet — placeholder
                winner_stance=max(verdict.vote_tally, key=verdict.vote_tally.get),
            )

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
        return jsonify({"status": "error", "error": "q parameter required"}), 400
    try:
        results = _bridge.memory.search(query, top_k=top_k)
        _stats["searches"] += 1
        return jsonify({"status": "success", "query": query, "results": results, "total": len(results)})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/sentinel/status")
def status():
    uptime = (datetime.datetime.now() - _start).total_seconds()
    return jsonify({
        "system":    "KISWARM-SENTINEL-v2.2-EMS",
        "status":    "operational",
        "uptime":    round(uptime, 1),
        "stats":     _stats,
        "threshold": _bridge.ckm.threshold,
        "scouts":    [s.__class__.__name__ for s in _bridge.scouts],
        "qdrant":    _bridge.memory.client is not None,
        "ledger":    {"entries": _ledger.size, "root": _ledger.root[:16] + "…"},
        "timestamp": datetime.datetime.now().isoformat(),
    })


# ═══════════════════════════════════════════════════════════════
# MODULE 6 — ADVERSARIAL PROMPT FIREWALL
# ═══════════════════════════════════════════════════════════════

@app.route("/firewall/scan", methods=["POST"])
def firewall_scan():
    """Scan content for adversarial patterns before injection."""
    data    = request.get_json() or {}
    content = data.get("content", "")
    source  = data.get("source", "unknown")
    query   = data.get("query", "")

    if not content and not query:
        return jsonify({"status": "error", "error": "content or query required"}), 400

    report = _firewall.scan(content, source=source, query=query)
    _stats["firewall_scans"] += 1
    if report.blocked:
        _stats["firewall_blocked"] += 1

    return jsonify({
        "status":        "blocked" if report.blocked else "clean",
        "blocked":       report.blocked,
        "threat_level":  report.threat_level,
        "threat_score":  report.threat_score,
        "threat_types":  report.threat_types,
        "match_count":   len(report.matches),
        "matches": [{
            "type":       m.threat_type.value,
            "description": m.description,
            "severity":   m.severity,
        } for m in report.matches],
        "statistical":   report.statistical,
        "recommendation": report.recommendation,
        "content_hash":  report.content_hash,
    })


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — KNOWLEDGE DECAY ENGINE
# ═══════════════════════════════════════════════════════════════

@app.route("/decay/scan")
def decay_scan():
    """Scan all registered knowledge entries for decay. Returns revalidation list."""
    report = _decay.scan()
    _stats["decay_scans"] += 1
    return jsonify({
        "scanned":             report.scanned,
        "healthy":             len(report.healthy),
        "needs_revalidation":  report.needs_revalidation,
        "retired":             report.retired,
        "average_confidence":  report.average_confidence,
        "oldest_entry_hours":  report.oldest_entry_hours,
        "timestamp":           report.timestamp,
    })


@app.route("/decay/record/<hash_id>")
def decay_record(hash_id: str):
    """Get current decayed confidence for a knowledge entry."""
    conf  = _decay.get_confidence(hash_id)
    query = _decay.get_query(hash_id)
    if query is None:
        return jsonify({"status": "error", "error": "hash_id not found"}), 404
    return jsonify({
        "hash_id":    hash_id,
        "query":      query,
        "confidence": conf,
        "needs_revalidation": conf < _decay.REVALIDATE_THRESHOLD,
    })


@app.route("/decay/revalidate", methods=["POST"])
def decay_revalidate():
    """Mark an entry as revalidated with updated confidence."""
    data       = request.get_json() or {}
    hash_id    = data.get("hash_id", "")
    new_conf   = float(data.get("confidence", 0.8))
    if not hash_id:
        return jsonify({"status": "error", "error": "hash_id required"}), 400
    _decay.mark_revalidated(hash_id, new_conf)
    return jsonify({"status": "success", "hash_id": hash_id, "new_confidence": new_conf})


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — CRYPTOGRAPHIC KNOWLEDGE LEDGER
# ═══════════════════════════════════════════════════════════════

@app.route("/ledger/status")
def ledger_status():
    """Get ledger summary including Merkle root hash."""
    return jsonify({**_ledger.summary(), "status": "ok"})


@app.route("/ledger/verify")
def ledger_verify():
    """Full Merkle tamper detection scan."""
    report = _ledger.verify_integrity()
    _stats["ledger_verifications"] += 1
    return jsonify({
        "valid":            report.valid,
        "total_entries":    report.total_entries,
        "tampered_entries": report.tampered_entries,
        "root_match":       report.root_match,
        "current_root":     report.current_root[:20] + "…",
        "is_clean":         report.is_clean,
        "timestamp":        report.timestamp,
    })


@app.route("/ledger/proof/<hash_id>")
def ledger_proof(hash_id: str):
    """Get Merkle inclusion proof for a specific knowledge entry."""
    entry = _ledger.get_entry(hash_id)
    if not entry:
        return jsonify({"status": "error", "error": "hash_id not in ledger"}), 404
    proof = _ledger.get_proof(entry.index)
    return jsonify({"status": "ok", **proof})


# ═══════════════════════════════════════════════════════════════
# MODULE 1 — SEMANTIC CONFLICT DETECTION
# ═══════════════════════════════════════════════════════════════

@app.route("/conflict/analyze", methods=["POST"])
def conflict_analyze():
    """Detect semantic contradictions across a list of content strings."""
    data  = request.get_json() or {}
    items = data.get("items", [])  # list of {"source": "...", "content": "..."}

    if len(items) < 2:
        return jsonify({"status": "error", "error": "At least 2 items required"}), 400

    # Build mock packets
    packets = [
        type("Packet", (), {"source": i.get("source", f"S{n}"), "content": i.get("content", "")})()
        for n, i in enumerate(items)
    ]

    report = _conflict.analyze(packets)
    return jsonify({
        "total_pairs":       report.total_pairs,
        "conflict_count":    report.conflict_count,
        "max_severity":      report.max_severity,
        "resolution_needed": report.resolution_needed,
        "clusters":          report.clusters,
        "conflicts": [{
            "source_a":    c.source_a,
            "source_b":    c.source_b,
            "similarity":  c.similarity,
            "severity":    c.severity,
            "cluster_id":  c.cluster_id,
        } for c in report.conflict_pairs],
    })


@app.route("/conflict/quick", methods=["POST"])
def conflict_quick():
    """Quick two-text contradiction check."""
    data = request.get_json() or {}
    a    = data.get("text_a", "")
    b    = data.get("text_b", "")
    if not a or not b:
        return jsonify({"status": "error", "error": "text_a and text_b required"}), 400
    sim, severity = _conflict.quick_check(a, b)
    return jsonify({
        "similarity": round(sim, 4),
        "severity":   severity,
        "contradicts": severity not in ("OK",),
    })


# ═══════════════════════════════════════════════════════════════
# MODULE 3 — MODEL PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════

@app.route("/tracker/leaderboard")
def tracker_leaderboard():
    """Model reliability leaderboard ranked by ELO + validation accuracy."""
    top_k = int(request.args.get("top_k", 10))
    board = _tracker.get_leaderboard(top_k)
    return jsonify({
        "leaderboard": [{
            "rank":        e.rank,
            "model":       e.model,
            "elo":         e.elo,
            "reliability": e.reliability,
            "win_rate":    e.win_rate,
            "debates":     e.debates,
            "validated":   e.validated_correct,
        } for e in board]
    })


@app.route("/tracker/model/<path:model_name>")
def tracker_model(model_name: str):
    """Per-model performance statistics."""
    stats = _tracker.get_model_stats(model_name)
    if not stats:
        return jsonify({"status": "error", "error": f"Model '{model_name}' not tracked yet"}), 404
    return jsonify({"status": "ok", "model": model_name, "stats": stats})


@app.route("/tracker/validate", methods=["POST"])
def tracker_validate():
    """Post-hoc validate a debate outcome (was the winner correct?)."""
    data      = request.get_json() or {}
    debate_id = data.get("debate_id", "")
    correct   = bool(data.get("correct", True))
    validator = data.get("validator", "unknown")
    if not debate_id:
        return jsonify({"status": "error", "error": "debate_id required"}), 400
    _tracker.validate_debate(debate_id, correct, validator)
    return jsonify({"status": "success", "debate_id": debate_id, "correct": correct})


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — DIFFERENTIAL RETRIEVAL GUARD
# ═══════════════════════════════════════════════════════════════

@app.route("/guard/assess", methods=["POST"])
def guard_assess():
    """Full retrieval trust assessment: drift + divergence + ledger + decay."""
    data     = request.get_json() or {}
    hash_id  = data.get("hash_id", "")
    query    = data.get("query", "")
    content  = data.get("retrieved_content", "")
    original = data.get("original_content")
    fresh    = data.get("fresh_content")

    if not hash_id or not content:
        return jsonify({"status": "error", "error": "hash_id and retrieved_content required"}), 400

    report = _guard.assess(
        hash_id=hash_id, query=query,
        retrieved_content=content,
        original_content=original,
        fresh_content=fresh,
    )
    _stats["guard_assessments"] += 1

    return jsonify({
        "hash_id":          report.hash_id,
        "trust_level":      report.trust_level,
        "trust_score":      report.trust_score,
        "ledger_valid":     report.ledger_valid,
        "decay_confidence": report.decay_confidence,
        "recommendation":   report.recommendation,
        "flags":            report.flags,
        "drift": {
            "detected":   report.drift.drift_detected if report.drift else False,
            "severity":   report.drift.drift_severity if report.drift else "N/A",
            "similarity": report.drift.similarity_to_original if report.drift else None,
        } if report.drift else None,
        "divergence": {
            "detected": report.divergence.divergence_detected if report.divergence else False,
            "level":    report.divergence.divergence_level if report.divergence else "N/A",
        } if report.divergence else None,
        "timestamp": report.timestamp,
    })


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return jsonify({
        "status": "error",
        "error":  "Endpoint not found",
        "v2_2_endpoints": [
            "POST /sentinel/extract", "POST /sentinel/debate",
            "GET  /sentinel/search", "GET  /sentinel/status",
            "POST /firewall/scan",
            "GET  /decay/scan", "GET  /decay/record/<hash_id>",
            "GET  /ledger/status", "GET  /ledger/verify", "GET  /ledger/proof/<hash_id>",
            "POST /conflict/analyze", "POST /conflict/quick",
            "GET  /tracker/leaderboard", "GET  /tracker/model/<name>",
            "POST /guard/assess",
            "GET  /health",
        ],
    }), 404


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  KISWARM v2.2-EMS — SENTINEL BRIDGE API             ║")
    logger.info("║  Port: 11436  |  Modules: 6  |  Endpoints: 17+     ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    app.run(host="127.0.0.1", port=11436, debug=False, threaded=True)
REST interface for the Sentinel Bridge AKE system.
Port: 11436 (alongside Ollama:11434, Tool Proxy:11435)

Endpoints:
  POST /sentinel/extract   — Trigger AKE for a query
  POST /sentinel/debate    — Trigger Swarm Debate for conflicting sources
  GET  /sentinel/search    — Search existing swarm knowledge
  GET  /sentinel/status    — Sentinel system status
  GET  /health             — Health check

Author: KISWARM Project (Baron Marco Paolo Ialongo)
Version: 2.1-EMS
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

# Add python/ to path for sentinel imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.sentinel_bridge import SentinelBridge
from sentinel.swarm_debate import SwarmDebateEngine

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

# Singletons
_bridge = SentinelBridge()
_debate = SwarmDebateEngine()
_start  = datetime.datetime.now()
_stats  = {"extractions": 0, "debates": 0, "searches": 0}


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine from a sync Flask handler."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    uptime = (datetime.datetime.now() - _start).total_seconds()
    return jsonify({
        "status":   "active",
        "service":  "KISWARM-SENTINEL-BRIDGE",
        "version":  "2.1-EMS",
        "port":     11436,
        "uptime":   round(uptime, 1),
        "stats":    _stats,
        "timestamp": datetime.datetime.now().isoformat(),
    })


@app.route("/sentinel/extract", methods=["POST"])
def extract():
    """
    Trigger Autonomous Knowledge Extraction for a query.

    Body: { "query": "...", "force": false, "threshold": 0.85 }
    """
    data  = request.get_json() or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"status": "error", "error": "query is required"}), 400

    force     = bool(data.get("force", False))
    threshold = float(data.get("threshold", 0.85))

    logger.info("AKE request: query='%s' force=%s", query, force)
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
    """
    Trigger a Swarm Debate to resolve conflicting intelligence.

    Body: {
      "query": "...",
      "content_a": "...", "source_a": "Source A",
      "content_b": "...", "source_b": "Source B"
    }
    """
    data = request.get_json() or {}
    required = ["query", "content_a", "content_b"]
    for field_name in required:
        if not data.get(field_name):
            return jsonify({"status": "error", "error": f"{field_name} is required"}), 400

    try:
        verdict = run_async(_debate.debate(
            query=data["query"],
            content_a=data["content_a"],
            content_b=data["content_b"],
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
    """
    Search existing swarm knowledge memory.

    Query params: ?q=<query>&top_k=5
    """
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 5))

    if not query:
        return jsonify({"status": "error", "error": "q parameter required"}), 400

    try:
        results = _bridge.memory.search(query, top_k=top_k)
        _stats["searches"] += 1
        return jsonify({
            "status":  "success",
            "query":   query,
            "results": results,
            "total":   len(results),
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/sentinel/status")
def status():
    uptime = (datetime.datetime.now() - _start).total_seconds()
    return jsonify({
        "system":    "KISWARM-SENTINEL-v2.1-EMS",
        "status":    "operational",
        "uptime":    round(uptime, 1),
        "stats":     _stats,
        "threshold": _bridge.ckm.threshold,
        "scouts":    [s.__class__.__name__ for s in _bridge.scouts],
        "qdrant":    _bridge.memory.client is not None,
        "encoder":   _bridge.memory.encoder is not None,
        "timestamp": datetime.datetime.now().isoformat(),
    })


@app.errorhandler(404)
def not_found(_):
    return jsonify({
        "status": "error",
        "error":  "Endpoint not found",
        "endpoints": [
            "POST /sentinel/extract",
            "POST /sentinel/debate",
            "GET  /sentinel/search?q=<query>",
            "GET  /sentinel/status",
            "GET  /health",
        ],
    }), 404


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  KISWARM v2.1 — SENTINEL BRIDGE API             ║")
    logger.info("║  Port: 11436  |  EMS Classification: Active     ║")
    logger.info("╚══════════════════════════════════════════════════╝")
    app.run(host="127.0.0.1", port=11436, debug=False, threaded=True)
