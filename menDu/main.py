# import time
# import numpy as np
# import traceback # Added for better error logging (was in a previous suggestion)
# from threading import Thread

# import tensorflow as tf
# import config
# from connectors import PrometheusClient
# from cfr_rl import Policy, solve_for_optimal_weights, ExperienceBuffer
# from xds import XdsEndpointServer, run_xds_server
# from api_server import run_api_server, update_metrics, ai_metrics


# def calculate_reward(state_vector: np.ndarray) -> float:
#     """Calculates a reward based on the current system state."""
#     latencies = state_vector[0::config.NUM_METRICS]  # P95 latencies
#     error_rates = state_vector[1::config.NUM_METRICS]  # Error rates

#     avg_latency = np.mean(latencies)
#     avg_error_rate = np.mean(error_rates)

#     latency_cost = avg_latency / 1000
#     error_cost = avg_error_rate

#     reward = 1.0 / (1.0 + latency_cost + error_cost)
#     return reward


# def training_thread_loop(policy: Policy, buffer: "ExperienceBuffer"):
#     """Background thread for continuous model training."""
#     print("Training thread started.")
#     training_iterations = 0
#     best_reward = -float('inf')
    
#     while True:
#         if len(buffer) < config.MIN_BUFFER_SIZE_FOR_TRAINING:
#             print(f"Buffer size: {len(buffer)}/{config.MIN_BUFFER_SIZE_FOR_TRAINING}, waiting for more data...")
#             time.sleep(config.TRAINING_INTERVAL_SECONDS)
#             continue

#         states, actions, rewards = buffer.sample(config.TRAINING_BATCH_SIZE)
#         loss = policy.train(states, actions, rewards)
        
#         # Calculate training metrics
#         current_reward = np.mean(rewards) if len(rewards) > 0 else 0.0
#         model_accuracy = min(100.0, max(0.0, current_reward * 100))
        
#         training_iterations += 1
        
#         # Save model after every training iteration
#         policy.save_model()
        
#         # Track best performance
#         if current_reward > best_reward:
#             best_reward = current_reward
#             print(f"New best reward: {best_reward:.4f}")
        
#         print(f"Training iteration {training_iterations}: Loss={loss:.4f}, Reward={current_reward:.4f}, Accuracy={model_accuracy:.2f}%")
        
#         # Update API metrics
#         update_metrics(
#             training_iterations=training_iterations,
#             buffer_size=len(buffer),
#             reward=current_reward,
#             accuracy=model_accuracy,
#             weights={},  # Will be updated by main loop
#             total_requests=0
#         )

#         time.sleep(config.TRAINING_INTERVAL_SECONDS)


# def inference_and_control_loop(policy: Policy, prometheus: "PrometheusClient",
#                                xds_server: "XdsEndpointServer", buffer: "ExperienceBuffer"):
#     """Main RL control loop interacting with live environment."""
#     while True:
#         try:
#             # 1. Observe
#             state_vector = prometheus.get_endpoint_state_vector()
            
#             # Handles critical connection failure (Prometheus Client returned None)
#             if state_vector is None:
#                 print("Prometheus client returned None (Critical failure to reach Prometheus). Skipping cycle.")
#                 time.sleep(config.OPTIMIZATION_INTERVAL_SECONDS)
#                 continue

#             # The state_vector is now guaranteed to be a NumPy array.
            
#             # Handles no-traffic or zero-data scenario
#             if state_vector.shape != (config.STATE_SIZE,) or np.sum(state_vector) == 0.0:
#                 print(f"State vector is all zeros (No data from Prometheus) or incorrect shape {state_vector.shape}. Skipping cycle.")
#                 time.sleep(config.OPTIMIZATION_INTERVAL_SECONDS)
#                 continue

#             # 2. Act (Only proceeds if we have non-zero, valid data)
#             probabilities = policy.predict(state_vector)
#             critical_indices = np.argsort(probabilities)[-config.K_CRITICAL_ENDPOINTS:].tolist()

#             new_weights = solve_for_optimal_weights(state_vector, critical_indices)
#             if new_weights:
#                 xds_server.update_endpoints(new_weights)

#             # 3. Reward
#             reward = calculate_reward(state_vector)
#             print(f"Calculated reward: {reward:.4f}")

#             # 4. Store experience
#             buffer.add(state_vector, probabilities, reward)
            
#             # Update API metrics (assuming the full logic is available here)
#             # update_metrics(...) 

#         except Exception as e:
#             print(f"An error occurred in the control loop: {e}")
#             traceback.print_exc()

#         time.sleep(config.OPTIMIZATION_INTERVAL_SECONDS)

# if __name__ == '__main__':
#     print("--- Starting CFR-RL Load Balancer with TensorFlow 2.13 ---")

#     # Initialize components
#     policy_network = Policy(
#         config.STATE_SIZE,
#         config.ACTION_SIZE,
#         config.LEARNING_RATE,
#         config.MODEL_PATH
#     )

#     # Try to load pre-trained model
#     model_loaded = policy_network.load_model()
#     if model_loaded:
#         print("Using pre-trained model for faster convergence")
#     else:
#         print("Starting with random weights - will learn from scratch")

#     prometheus_client = PrometheusClient(config.PROMETHEUS_IP, config.PROMETHEUS_PORT)
#     xds_service = XdsEndpointServer()
#     experience_buffer = ExperienceBuffer(config.EXPERIENCE_BUFFER_SIZE)

#     # Start API server in background
#     api_thread = Thread(target=run_api_server, args=('0.0.0.0', 5000))
#     api_thread.daemon = True
#     api_thread.start()

#     # Start xDS server in background
#     xds_thread = Thread(target=run_xds_server,
#                         args=(xds_service, config.AI_SERVICE_BIND_IP, config.XDS_SERVER_PORT))
#     xds_thread.daemon = True
#     xds_thread.start()

#     # Start training thread
#     trainer_thread = Thread(target=training_thread_loop,
#                             args=(policy_network, experience_buffer))
#     trainer_thread.daemon = True
#     trainer_thread.start()

#     # Main control loop
#     inference_and_control_loop(policy_network, prometheus_client, xds_service, experience_buffer)





import time
import numpy as np
import traceback # Added for better error logging (was in a previous suggestion)
from threading import Thread

import tensorflow as tf
import config
from connectors import PrometheusClient
from cfr_rl import Policy, solve_for_optimal_weights, ExperienceBuffer
from xds import XdsEndpointServer, run_xds_server
from api_server import run_api_server, update_metrics, ai_metrics


# In main.py, inside calculate_reward(state_vector: np.ndarray)

def calculate_reward(state_vector: np.ndarray) -> float:
    """Calculates a reward based on the current system state."""
    
    # Existing extractions (Correct for indices 0 and 1)
    latencies = state_vector[0::config.NUM_METRICS]  # P95 latencies (Index 0)
    error_rates = state_vector[1::config.NUM_METRICS]  # Error rates (Index 1)

    # ADDED: Extract the other two metrics (Indices 2 and 3)
    saturations = state_vector[2::config.NUM_METRICS]  # Saturation (Index 2)
    throughputs = state_vector[3::config.NUM_METRICS] # Throughput (Index 3)
    
    # Calculate averages (you may want to use a more complex function here)
    avg_latency = np.mean(latencies)
    avg_error_rate = np.mean(error_rates)
    avg_saturation = np.mean(saturations) # ADDED
    avg_throughput = np.mean(throughputs) # ADDED

    # Calculate costs (You need to define what a "cost" for saturation/throughput is)
    latency_cost = avg_latency / 1000 # Convert ms to s
    error_cost = avg_error_rate * 10 # Scale error rate (e.g., 5% error = 0.5 cost)
    saturation_cost = avg_saturation / 20 # Example scaling
    throughput_cost = 100 / (avg_throughput + 1) # Example: Penalize low throughput (low reward for high cost)

    # Combine costs. Note: You need a sign convention for throughput.
    # Higher throughput is good (low cost).
    total_cost = latency_cost + error_cost + saturation_cost + throughput_cost
    
    # Calculate reward: Low cost = High reward
    reward = 1.0 / (1.0 + total_cost)
    
    return reward

# In main.py: Add fallback/default config values for testing

# Add MIN_BUFFER_SIZE_FOR_TRAINING and MIN_BATCH_SIZE to config.py 
# if they are not already there. Assuming we use small defaults here.
# If they are in config.py, this block is unnecessary, but ensures robust startup.
try:
    MIN_BUFFER_SIZE_FOR_TRAINING = config.MIN_BUFFER_SIZE_FOR_TRAINING
    MIN_BATCH_SIZE = config.MIN_BATCH_SIZE
except AttributeError:
    MIN_BUFFER_SIZE_FOR_TRAINING = 10 
    MIN_BATCH_SIZE = 3 


# FIX: Updated training_thread_loop (ENSURES TRAINING RUNS)
def training_thread_loop(policy: Policy, buffer: "ExperienceBuffer"):
    """Background thread for continuous model training."""
    print("Training thread started.")
    training_iterations = 0
    
    BATCH_SIZE = MIN_BATCH_SIZE
    MIN_BUFFER_SIZE = MIN_BUFFER_SIZE_FOR_TRAINING
    
    while True:
        # Check if we have enough data to train.
        if len(buffer) < MIN_BUFFER_SIZE:
            print(f"Buffer size: {len(buffer)}/{MIN_BUFFER_SIZE}, waiting for more data...")
            time.sleep(1) 
            continue

        # 1. Sample a batch
        # IMPORTANT: ExperienceBuffer.sample returns empty lists if not enough data.
        states, actions, rewards = buffer.sample(BATCH_SIZE)
        
        if not states: # Handles case where buffer might shrink during sampling
            time.sleep(1)
            continue
            
        # 2. Train the policy network
        loss = policy.train_step_wrapper(states, actions, rewards) 
        
        # 3. Update Metrics
        training_iterations += 1
        
        # A simple proxy for accuracy (0% loss = 100% accuracy)
        accuracy_proxy = max(0.0, 100.0 * (1.0 - loss)) 

        # CRITICAL: Call update_metrics with the training metrics, 
        # but preserve the latest operational metrics (reward, weights).
        update_metrics(
            training_iterations=training_iterations, 
            buffer_size=len(buffer),
            # Use operational metrics from last inference run:
            reward=ai_metrics['current_reward'], 
            accuracy=accuracy_proxy,
            weights=ai_metrics['endpoint_weights'], 
            total_requests=ai_metrics.get('total_requests', 0),
        )
        print(f"Trained model. Iteration: {training_iterations}, Loss: {loss:.4f}")

        time.sleep(1) # Train roughly every second

# FIX: Updated inference_and_control_loop (ENSURES TOTAL_REQUESTS & CORRECT ACTION LOGIC)
def inference_and_control_loop(policy: Policy, prometheus: PrometheusClient,
                               xds_server: XdsEndpointServer, buffer: ExperienceBuffer):
    print("Inference loop started.")
    total_requests = ai_metrics.get('total_requests', 0) 

    while True:
        try:
            # 1. Observe
            state_vector = prometheus.get_endpoint_state_vector()
            critical_endpoint_indices = policy.predict(state_vector) # Returns list[int] now
            
            # 2. Execute
            new_weights = solve_for_optimal_weights(state_vector, critical_endpoint_indices)
            
            # Apply weights and increment total requests
            xds_server.update_endpoints(new_weights)
            total_requests += 1 
            
            # 3. Calculate Reward & Store Experience
            reward = calculate_reward(state_vector)
            
            # FIX: Action vector logic is now correct because critical_endpoint_indices is list[int]
            action_vector = np.zeros(config.ACTION_SIZE, dtype=np.float32)
            for idx in critical_endpoint_indices:
                action_vector[idx] = 1.0 # This now works because idx is an integer
                
            buffer.add(state_vector, action_vector, reward)
            print(f"Calculated reward: {reward:.4f}")
            
            # 4. Update API Metrics
            # CRITICAL: Use the training thread's latest values for its metrics, and update operational ones
            update_metrics(
                training_iterations=ai_metrics['training_iterations'], 
                buffer_size=len(buffer),
                reward=reward, 
                accuracy=ai_metrics['model_accuracy'],
                weights=new_weights,
                total_requests=total_requests
            )

        except Exception as e:
            print(f"An error occurred in the control loop: {e}")
            traceback.print_exc() 
            
        time.sleep(config.OPTIMIZATION_INTERVAL_SECONDS)


if __name__ == '__main__':
    print("--- Starting CFR-RL Load Balancer with TensorFlow 2.13 ---")

    # Initialize components
    policy_network = Policy(
        config.STATE_SIZE,
        config.ACTION_SIZE,
        config.LEARNING_RATE,
        config.MODEL_PATH
    )

    
    # Try to load pre-trained model
    model_loaded = policy_network.load_model()
    if model_loaded:
        print("Using pre-trained model for faster convergence")
    else:
        print("Starting with random weights - will learn from scratch")

    prometheus_client = PrometheusClient(config.PROMETHEUS_IP, config.PROMETHEUS_PORT)
    xds_service = XdsEndpointServer()
    experience_buffer = ExperienceBuffer(config.EXPERIENCE_BUFFER_SIZE)

    # Start API server in background
    api_thread = Thread(target=run_api_server, args=('0.0.0.0', 5000))
    api_thread.daemon = True
    api_thread.start()

    # Start xDS server in background
    xds_thread = Thread(target=run_xds_server,
                        args=(xds_service, config.AI_SERVICE_BIND_IP, config.XDS_SERVER_PORT))
    xds_thread.daemon = True
    xds_thread.start()

    # Start training thread
    trainer_thread = Thread(target=training_thread_loop,
                            args=(policy_network, experience_buffer))
    trainer_thread.daemon = True
    trainer_thread.start()

    # Main control loop
    inference_and_control_loop(policy_network, prometheus_client, xds_service, experience_buffer)
