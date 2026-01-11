# --- AWS & Networking Configuration ---
# IPs of your 3 demo backend servers
# These would be the IPs of EC2 instances running your application (e.g., NGINX)
BACKEND_SERVER_IPS = {
    "10.0.195.3:5000": 1, # Your deployed backend server 1
    "10.0.195.3:5001": 1, # Your deployed backend server 2
    "10.0.195.3:5002": 1, # Your deployed backend server 3
}

# The Private IP address of the Prometheus EC2 instance
# need to put the ip adress here temp change 
#PURVAL LOOK HERE---------------------------------------------------------------------------
PROMETHEUS_IP = "prometheus" # e.g., "172.31.0.100"
PROMETHEUS_PORT = 9090

# The address the AI Service will bind to. Use 0.0.0.0 to be reachable
# within the VPC by other EC2 instances.
AI_SERVICE_BIND_IP = "0.0.0.0"
XDS_SERVER_PORT = 50051

# --- Prometheus Queries ---
# {envoy_cluster_name="my_service"} must match the cluster name in envoy.yaml
# We combine multiple metrics into our state vector.
PROMETHEUS_QUERIES = {
    # P95 Latency in milliseconds
    "latency": 'histogram_quantile(0.95, sum(rate(envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name="my_service"}[1m])) by (le, envoy_upstream_address)) * 1000',

    
    # Server-side (5xx) error rate as a percentage
    "error_rate": 'sum(rate(envoy_cluster_my_service_upstream_rq_total{envoy_response_code_class="5"}[1m])) by (envoy_upstream_address) / sum(rate(envoy_cluster_my_service_upstream_rq_total[1m])) by (envoy_upstream_address) * 100',
    
    # The number of requests currently buffered (Cluster-level metric, no per-host fix needed)
    "saturation": 'envoy_cluster_upstream_rq_pending_active{envoy_cluster_name="my_service"}',
    
    # Current throughput in requests per second (RPS)
    "throughput": 'sum(rate(envoy_cluster_upstream_rq_total{envoy_cluster_name="my_service"}[1m])) by (envoy_upstream_address)'

}
# --- CFR-RL Algorithm Configuration ---
ENDPOINT_ADDRESSES = BACKEND_SERVER_IPS
NUM_ENDPOINTS = 3
NUM_METRICS = 4

# State: A flattened vector of (latency, error_rate) for each endpoint
STATE_SIZE = NUM_ENDPOINTS * NUM_METRICS 
ACTION_SIZE = NUM_ENDPOINTS

# CFR-RL will identify the top K endpoints as "critical" for re-weighting
K_CRITICAL_ENDPOINTS = 2  # e.g., always optimize the top 2 trouble-makers

# # Path to the pre-trained TF model
MODEL_PATH = './cfr_rl/pre_trained_model/model.ckpt'
LEARNING_RATE = 0.001 # For training

# --- Application Configuration ---
OPTIMIZATION_INTERVAL_SECONDS = 5


# --- REAL-TIME LEARNING CONFIGURATION ---
EXPERIENCE_BUFFER_SIZE = 10000  # How many past experiences to remember
TRAINING_BATCH_SIZE = 16      # How many experiences to learn from at once
MIN_BUFFER_SIZE_FOR_TRAINING = 10 # How many experiences to collect before starting to learn
MIN_BATCH_SIZE = 3

# The training thread will run more frequently than the action/inference loop
TRAINING_INTERVAL_SECONDS = 2