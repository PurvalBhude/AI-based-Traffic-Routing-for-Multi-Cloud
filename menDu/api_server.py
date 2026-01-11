from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time
import json
from typing import Dict, Any

# Global variables to store metrics
ai_metrics = {
    'training_iterations': 0,
    'buffer_size': 0,
    'current_reward': 0.0,
    'model_accuracy': 0.0,
    'recent_rewards': [],
    'recent_losses': [],
    'endpoint_weights': {},
    'last_update': time.time()
}

app = Flask(__name__)
CORS(app)  # Enable CORS for dashboard access

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'menDu AI Service',
        'timestamp': time.time()
    })

@app.route('/api/metrics/training', methods=['GET'])
def get_training_metrics():
    """Get AI training metrics"""
    return jsonify({
        'training_iterations': ai_metrics['training_iterations'],
        'buffer_size': ai_metrics['buffer_size'],
        'current_reward': ai_metrics['current_reward'],
        'model_accuracy': ai_metrics['model_accuracy'],
        'recent_rewards': ai_metrics['recent_rewards'][-20:],  # Last 20 rewards
        'recent_losses': ai_metrics['recent_losses'][-20:],    # Last 20 losses
        'last_update': ai_metrics['last_update']
    })

@app.route('/api/metrics/weights', methods=['GET'])
def get_endpoint_weights():
    """Get current endpoint weights from AI model"""
    return jsonify({
        'endpoint_weights': ai_metrics['endpoint_weights'],
        'total_requests': ai_metrics.get('total_requests', 0),
        'load_balance_efficiency': ai_metrics.get('load_balance_efficiency', 0),
        'last_update': ai_metrics['last_update']
    })

@app.route('/api/metrics/all', methods=['GET'])
def get_all_metrics():
    """Get all AI service metrics"""
    return jsonify({
        'training': {
            'iterations': ai_metrics['training_iterations'],
            'buffer_size': ai_metrics['buffer_size'],
            'current_reward': ai_metrics['current_reward'],
            'model_accuracy': ai_metrics['model_accuracy'],
            'recent_rewards': ai_metrics['recent_rewards'][-20:],
            'recent_losses': ai_metrics['recent_losses'][-20:]
        },
        'load_balancing': {
            'endpoint_weights': ai_metrics['endpoint_weights'],
            'total_requests': ai_metrics.get('total_requests', 0),
            'efficiency': ai_metrics.get('load_balance_efficiency', 0)
        },
        'status': {
            'last_update': ai_metrics['last_update'],
            'uptime': time.time() - ai_metrics.get('start_time', time.time())
        }
    })

def update_metrics(training_iterations: int, buffer_size: int, reward: float, 
                   accuracy: float, weights: Dict[str, int], total_requests: int = 0):
    """Update AI metrics (called from main AI service)"""
    global ai_metrics
    
    ai_metrics['training_iterations'] = training_iterations
    ai_metrics['buffer_size'] = buffer_size
    ai_metrics['current_reward'] = reward
    ai_metrics['model_accuracy'] = accuracy
    ai_metrics['endpoint_weights'] = weights
    ai_metrics['total_requests'] = total_requests
    ai_metrics['last_update'] = time.time()
    
    # Keep recent history
    ai_metrics['recent_rewards'].append(reward)
    ai_metrics['recent_losses'].append(1.0 - reward)  # Loss is inverse of reward
    
    # Keep only last 100 values
    if len(ai_metrics['recent_rewards']) > 100:
        ai_metrics['recent_rewards'] = ai_metrics['recent_rewards'][-100:]
        ai_metrics['recent_losses'] = ai_metrics['recent_losses'][-100:]

def run_api_server(host='0.0.0.0', port=5000):
    """Run the Flask API server"""
    ai_metrics['start_time'] = time.time()
    print(f"Starting AI Service API server on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run_api_server()
