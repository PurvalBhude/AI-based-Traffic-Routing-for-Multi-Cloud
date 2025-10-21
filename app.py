# app.py
import os
import logging
import time
import threading
from datetime import datetime
from flask import Flask, jsonify, request

# Logging — stream only (Vercel captures stdout)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Simple safe metrics tracker (in-memory; serverless ephemeral) ---
class MetricsTracker:
    def __init__(self):
        self.request_count = 0
        self.start_time = time.time()
        self.response_times = []
        self.error_count = 0
        self.server_id = os.getenv('SERVER_ID', f'backend-{int(time.time())}')
        self.lock = threading.Lock()

    def record_request(self, response_time=None, is_error=False):
        with self.lock:
            self.request_count += 1
            if response_time is not None:
                self.response_times.append(response_time)
            if is_error:
                self.error_count += 1

    def get_metrics(self):
        with self.lock:
            uptime = time.time() - self.start_time
            avg_response_time = (
                sum(self.response_times) / len(self.response_times)
                if self.response_times else 0.0
            )
            success_rate = (
                ((self.request_count - self.error_count) / self.request_count * 100)
                if self.request_count > 0 else 100.0
            )
            return {
                'server_id': self.server_id,
                'total_requests': self.request_count,
                'success_rate': round(success_rate, 2),
                'avg_response_time_ms': round(avg_response_time, 2),
                'error_count': self.error_count,
                'uptime_seconds': round(uptime, 2),
                'cpu_percent': 0.0,      # psutil removed — not reliable in serverless
                'memory_percent': 0.0,   # psutil removed — not reliable in serverless
                'timestamp': datetime.utcnow().isoformat()
            }

# Initialize tracker
metrics_tracker = MetricsTracker()

# --- Small request timing/logging hooks ---
MAX_BODY_LOG = 1000

@app.before_request
def before_request_logging():
    request._received_at = time.time()
    try:
        body = request.get_data(as_text=True)
    except Exception:
        body = "<could not read body>"

    if body and len(body) > MAX_BODY_LOG:
        body_snippet = body[:MAX_BODY_LOG] + "...(truncated)"
    else:
        body_snippet = body

    logger.info(f"--> REQUEST {request.remote_addr} {request.method} {request.path} Body={body_snippet}")

@app.after_request
def after_request_logging(response):
    start = getattr(request, "_received_at", None)
    response_time_ms = None
    if start:
        response_time_ms = (time.time() - start) * 1000.0

    try:
        resp_body = response.get_data(as_text=True)
    except Exception:
        resp_body = "<could not read response body>"

    if resp_body and len(resp_body) > MAX_BODY_LOG:
        resp_snippet = resp_body[:MAX_BODY_LOG] + "...(truncated)"
    else:
        resp_snippet = resp_body

    logger.info(
        f"<-- RESPONSE {request.remote_addr} {request.method} {request.path} "
        f"Status={response.status} RespTimeMs={round(response_time_ms,2) if response_time_ms is not None else 'N/A'} "
        f"Body={resp_snippet}"
    )

    if metrics_tracker:
        metrics_tracker.record_request(response_time_ms)

    return response

# --- Routes ---
@app.route("/")
def index():
    return (
        "<html><head><title>Public Backend</title></head>"
        "<body>"
        f"<h1>Public Backend - {metrics_tracker.server_id}</h1>"
        "<p>Visit <a href=\"/metrics\">/metrics</a> or <a href=\"/health\">/health</a></p>"
        "</body></html>",
        200
    )

@app.route("/health")
def health_check():
    try:
        data = {
            "status": "healthy",
            "server_id": metrics_tracker.server_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics_tracker.get_metrics()
        }
        return jsonify(data), 200
    except Exception as e:
        if metrics_tracker:
            metrics_tracker.record_request(is_error=True)
        logger.exception("Health check error")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/healthz")
def healthz():
    return "OK", 200

@app.route("/metrics")
def metrics():
    try:
        return jsonify(metrics_tracker.get_metrics()), 200
    except Exception as e:
        logger.exception("Error getting metrics")
        return jsonify({"error": str(e)}), 500

# Example simple API endpoint
@app.route("/api/status")
def api_status():
    try:
        return jsonify({
            'server_id': metrics_tracker.server_id,
            'status': 'running',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics_tracker.get_metrics()
        }), 200
    except Exception as e:
        logger.exception("api/status error")
        return jsonify({'error': str(e)}), 500
