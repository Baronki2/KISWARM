"""
KISWARM v2.1 — SENTINEL API SERVER
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
