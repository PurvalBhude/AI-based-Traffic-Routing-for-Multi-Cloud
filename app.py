import os
import logging
import time
import psutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from werkzeug.middleware.proxy_fix import ProxyFix
import argparse
import threading
import json
import sys
import traceback

LOG_FILE = "server.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",  # open to everyone (not secure for production)
    ping_timeout=60,
    ping_interval=25
)

# Active socket connections
active_users = set()

# --- Utility: safe psutil reads ---
def safe_psutil():
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
    except Exception:
        cpu = 0.0
        mem = 0.0
    return cpu, mem

# --- Metrics tracker (fixed constructor) ---
class MetricsTracker:
    def __init__(self):
        self.request_count = 0
        self.start_time = time.time()
        self.response_times = []
        self.error_count = 0
        self.server_id = os.getenv('SERVER_ID', f'server-{int(time.time())}')
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
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            success_rate = ((self.request_count - self.error_count) / self.request_count * 100) if self.request_count > 0 else 100
            cpu, mem = safe_psutil()
            return {
                'server_id': self.server_id,
                'total_requests': self.request_count,
                'success_rate': round(success_rate, 2),
                'avg_response_time_ms': round(avg_response_time, 2),
                'error_count': self.error_count,
                'uptime_seconds': round(uptime, 2),
                'cpu_percent': cpu,
                'memory_percent': mem,
                'timestamp': datetime.utcnow().isoformat()
            }

# Prepare SERVER_ID before tracker creation (keeps IDs stable per port)
def setup_server_id(port):
    os.environ['SERVER_ID'] = f'backend-{port}'

# --- We'll initialize metrics tracker later in main(), after SERVER_ID set ---
metrics_tracker = None  # will be set in main()

# --- Request / Response logging middleware ---
MAX_BODY_LOG = 1000  # chars

@app.before_request
def log_request_info():
    try:
        # start timer on request
        request._received_at = time.time()

        # Grab small body safely
        try:
            body = request.get_data(as_text=True)
        except Exception:
            body = "<could not read body>"

        if body and len(body) > MAX_BODY_LOG:
            body_snippet = body[:MAX_BODY_LOG] + "...(truncated)"
        else:
            body_snippet = body

        logger.info(
            f"--> REQUEST {request.remote_addr} {request.method} {request.path} "
            f"Headers={dict(request.headers)} Body={body_snippet}"
        )
    except Exception:
        logger.error("Error logging request info:\n" + traceback.format_exc())

@app.after_request
def log_response_info(response: Response):
    try:
        # Calculate response time if start set
        start = getattr(request, "_received_at", None)
        response_time_ms = None
        if start:
            response_time_ms = (time.time() - start) * 1000.0

        # Safe response body reading (may be bytes)
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
            f"Status={response.status} Headers={dict(response.headers)} Body={resp_snippet} "
            f"RespTimeMs={round(response_time_ms,2) if response_time_ms is not None else 'N/A'}"
        )

        # record metrics
        if metrics_tracker:
            metrics_tracker.record_request(response_time_ms)

    except Exception:
        logger.error("Error logging response info:\n" + traceback.format_exc())

    return response

# --- Basic routes ---
@app.route('/')
def index():
    # simple HTML page
    return f"""
    <html>
      <head><title>Public Backend</title></head>
      <body>
        <h1>Public Backend On Vercel - {os.getenv('SERVER_ID', 'unknown')}</h1>
        <p>Visit <a href="/metrics">/metrics</a> or use sockets.</p>
        <p>To stop the server (unprotected): POST /shutdown</p>
      </body>
    </html>
    """, 200

@app.route('/health')
def health_check():
    try:
        data = {
            "status": "healthy",
            "server_id": metrics_tracker.server_id if metrics_tracker else "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics_tracker.get_metrics() if metrics_tracker else {}
        }
        return jsonify(data), 200
    except Exception as e:
        if metrics_tracker:
            metrics_tracker.record_request(is_error=True)
        logger.error("Health check error: " + str(e))
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/healthz')
def healthz():
    return "OK", 200

@app.route('/metrics')
def metrics():
    try:
        return jsonify(metrics_tracker.get_metrics()), 200
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

# --- Unprotected shutdown endpoint (stops the server when called) ---
@app.route('/shutdown', methods=['POST'])
def shutdown():
    """
    Immediately attempt to stop the running server.
    WARNING: This endpoint is unprotected (anyone can call it). Use only for testing.
    """
    logger.warning("Shutdown requested via /shutdown endpoint. Stopping server...")
    # Record as a request
    if metrics_tracker:
        metrics_tracker.record_request()

    # Try to stop Flask-SocketIO cleanly
    try:
        # socketio.stop() is supported by Flask-SocketIO to stop the server loop
        socketio.stop()
    except Exception as e:
        logger.info("socketio.stop() not available or failed: " + str(e))
        # Fallback to werkzeug shutdown (works for the development server)
        func = request.environ.get('werkzeug.server.shutdown')
        if func:
            func()
        else:
            # If we cannot cleanly shutdown, exit process
            logger.info("Falling back to os._exit(0)")
            os._exit(0)

    return jsonify({"status": "shutting_down"}), 200

@app.route('/api/status')
def api_status():
    try:
        return jsonify({
            'server_id': metrics_tracker.server_id,
            'status': 'running',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics_tracker.get_metrics()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- SocketIO events logging ---
@socketio.on('connect')
def handle_connect():
    try:
        sid = request.sid
        active_users.add(sid)
        logger.info(f"Socket CONNECT: sid={sid} remote={request.remote_addr}")
        emit('system_message', {'message': f'Connected to {metrics_tracker.server_id}'})
    except Exception:
        logger.error("Error in connect handler:\n" + traceback.format_exc())

@socketio.on('disconnect')
def handle_disconnect():
    try:
        sid = request.sid
        active_users.discard(sid)
        logger.info(f"Socket DISCONNECT: sid={sid}")
    except Exception:
        logger.error("Error in disconnect handler:\n" + traceback.format_exc())

@socketio.on('flask-chat-event')
def handle_chat_message(data):
    start_time = time.time()
    try:
        if not isinstance(data, dict):
            raise ValueError("Invalid message format")

        user_name = str(data.get('user_name', '')).strip()
        message = str(data.get('message', '')).strip()

        if not user_name or not message:
            raise ValueError("Username and message are required")

        # Prevent naive XSS (basic escaping)
        user_name = user_name.replace('<', '&lt;').replace('>', '&gt;')
        message = message.replace('<', '&lt;').replace('>', '&gt;')

        response_data = {
            'user_name': user_name,
            'message': message,
            'server_id': metrics_tracker.server_id,
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"Socket MESSAGE from {user_name}: {message}")
        emit('flask-chat-response', response_data, broadcast=True)

        # metrics
        response_time = (time.time() - start_time) * 1000
        if metrics_tracker:
            metrics_tracker.record_request(response_time)

    except Exception as e:
        logger.error(f'Error handling socket message: {str(e)}')
        if metrics_tracker:
            metrics_tracker.record_request(is_error=True)
        emit('error', {'message': 'Error processing message'}, room=request.sid)

# --- Main entrypoint ---
def main(portnumber=5000):
    global metrics_tracker

    # ensure SERVER_ID defined early
    setup_server_id(portnumber)
    metrics_tracker = MetricsTracker()
    logger.info(f"Starting server with ID: {metrics_tracker.server_id} on port {portnumber}")

    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', portnumber))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    # Start SocketIO (development server). For production, use appropriate WSGI server.
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug,
        allow_unsafe_werkzeug=debug
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Public Flask+SocketIO server with verbose logging and unprotected shutdown.")
    parser.add_argument("--port", type=int, default=5000, help="port number")
    args = parser.parse_args()

    try:
        main(args.port)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Stopping.")
        try:
            socketio.stop()
        except Exception:
            pass
        os._exit(0)
