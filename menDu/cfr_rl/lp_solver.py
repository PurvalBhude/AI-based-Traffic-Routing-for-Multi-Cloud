# import pulp
# import numpy as np
# import config

# def solve_for_optimal_weights(state_vector: np.ndarray, critical_endpoint_indices: list) -> dict:
#     """
#     Formulates and solves an LP problem to find the optimal weights for
#     the identified critical endpoints. Non-critical endpoints get a fixed baseline weight.
    
#     Args:
#         state_vector: The current state of all endpoints.
#         critical_endpoint_indices: List of indices for the critical endpoints.

#     Returns: A dictionary mapping {'ip:port': weight} for all endpoints.
#     """
#     prob = pulp.LpProblem("OptimalWeightDistribution", pulp.LpMinimize)
    
#     # --- Decision Variables ---
#     # We define weight variables only for the critical endpoints
#     weights = pulp.LpVariable.dicts(
#         "weight", critical_endpoint_indices, lowBound=1, cat='Integer'
#     )
#     # This variable represents the maximum weighted "cost" (e.g., latency)
#     U = pulp.LpVariable('U', lowBound=0) 
    
#     # --- Objective Function ---
#     # Minimize the maximum "cost" across all critical endpoints.
#     # This promotes fairness and prevents overloading a single healthy server.
#     prob += U

#     # --- Constraints ---
#     # Unpack the latencies from the state vector for clarity
#     latencies = state_vector[0::config.NUM_METRICS] # Grabs every Nth element starting at 0
    
#     # Constraint 1: The objective `U` must be greater than or equal to the cost of each endpoint.
#     # cost = latency * weight. By minimizing U, we are minimizing the highest cost.
#     for i in critical_endpoint_indices:
#         prob += latencies[i] * weights[i] <= U

#     # Constraint 2: The sum of weights for critical endpoints must be a defined portion of total weight.
#     # We'll allocate 80% of the total traffic share (e.g., 80 of 100) to the critical nodes.
#     total_critical_weight = 80
#     prob += pulp.lpSum(weights[i] for i in critical_endpoint_indices) == total_critical_weight
    
#     # Solve the problem
#     prob.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 suppresses solver output

#     # --- Process Results ---
#     final_weights = {}
    
#     # Baseline weight for non-critical endpoints
#     num_non_critical = config.NUM_ENDPOINTS - len(critical_endpoint_indices)
#     non_critical_weight_share = 100 - total_critical_weight
#     baseline_weight = non_critical_weight_share // num_non_critical if num_non_critical > 0 else 0

#     for i in range(config.NUM_ENDPOINTS):
#         addr = config.ENDPOINT_ADDRESSES[i]
#         if i in critical_endpoint_indices:
#             final_weights[addr] = int(pulp.value(weights[i]))
#         else:
#             final_weights[addr] = baseline_weight

#     # Ensure total is 100 due to integer division
#     total = sum(final_weights.values())
#     if total < 100 and final_weights:
#         # Add remainder to the best-performing endpoint
#         best_endpoint_idx = np.argmin(latencies)
#         final_weights[config.ENDPOINT_ADDRESSES[best_endpoint_idx]] += 100 - total
    
#     print(f"LP Solver determined new weights: {final_weights}")
#     return final_weights
# ---------------------------------------------------------------------------------------------
# EAERLIER JUST WE ARE USING THE 1 MATRIX LETENCY I HAVE USED 3 MORE HERE
# import pulp
# import numpy as np
# import config

# # Define weights for each metric to compute a combined cost
# # You can tune these based on priority
# METRIC_WEIGHTS = {
#     "latency": 0.4,      # importance of latency
#     "error_rate": 0.3,   # importance of error rate
#     "saturation": 0.2,   # importance of saturation
#     "throughput": 0.1    # importance of throughput
# }

# def solve_for_optimal_weights(state_vector: np.ndarray, critical_endpoint_indices: list) -> dict:
#     """
#     Formulates and solves an LP problem to find the optimal weights for
#     the identified critical endpoints using multiple metrics.
#     Non-critical endpoints get a fixed baseline weight.
    
#     Args:
#         state_vector: The current state of all endpoints (flattened, metrics per endpoint).
#         critical_endpoint_indices: List of indices for the critical endpoints.

#     Returns: A dictionary mapping {'ip:port': weight} for all endpoints.
#     """
#     prob = pulp.LpProblem("OptimalWeightDistribution", pulp.LpMinimize)
    
#     # --- Decision Variables ---
#     weights = pulp.LpVariable.dicts(
#         "weight", critical_endpoint_indices, lowBound=1, cat='Integer'
#     )
#     U = pulp.LpVariable('U', lowBound=0)
    
#     # --- Compute combined cost for each endpoint ---
#     combined_costs = []
#     num_metrics = config.NUM_METRICS
#     for i in range(config.NUM_ENDPOINTS):
#         start_idx = i * num_metrics
#         latency = state_vector[start_idx]
#         error_rate = state_vector[start_idx + 1]
#         saturation = state_vector[start_idx + 2]
#         throughput = state_vector[start_idx + 3]
#         # For throughput, higher is better → we invert it to represent “cost”
#         throughput_cost = 1.0 / (throughput + 1e-6)  # avoid div by zero
#         cost = (METRIC_WEIGHTS["latency"] * latency +
#                 METRIC_WEIGHTS["error_rate"] * error_rate +
#                 METRIC_WEIGHTS["saturation"] * saturation +
#                 METRIC_WEIGHTS["throughput"] * throughput_cost)
#         combined_costs.append(cost)
    
#     # --- Objective ---
#     prob += U
    
#     # Constraint: max cost across critical endpoints
#     for i in critical_endpoint_indices:
#         prob += combined_costs[i] * weights[i] <= U
    
#     # Constraint: total weight for critical endpoints
#     total_critical_weight = 80
#     prob += pulp.lpSum(weights[i] for i in critical_endpoint_indices) == total_critical_weight
    
#     # --- Solve LP ---
#     prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
#     # --- Process results ---
#     final_weights = {}
#     num_non_critical = config.NUM_ENDPOINTS - len(critical_endpoint_indices)
#     non_critical_weight_share = 100 - total_critical_weight
#     baseline_weight = non_critical_weight_share // num_non_critical if num_non_critical > 0 else 0
    
#     for i in range(config.NUM_ENDPOINTS):
#         addr = config.ENDPOINT_ADDRESSES[i]
#         if i in critical_endpoint_indices:
#             final_weights[addr] = int(pulp.value(weights[i]))
#         else:
#             final_weights[addr] = baseline_weight
    
#     # Adjust remainder to ensure sum = 100
#     total = sum(final_weights.values())
#     if total < 100 and final_weights:
#         best_endpoint_idx = np.argmin(combined_costs)
#         final_weights[config.ENDPOINT_ADDRESSES[best_endpoint_idx]] += 100 - total
    
#     print(f"LP Solver determined new weights: {final_weights}")
#     return final_weights

import numpy as np
import config
from typing import Dict, List

# This replaces the entire commented pulp-based solve_for_optimal_weights function.
def solve_for_optimal_weights(state_vector: np.ndarray, critical_endpoint_indices: List[int]) -> Dict[str, int]:
    """
    Mocks the LP solver to return a set of random, valid weights for all endpoints.
    
    Returns: A dictionary mapping {'ip:port': weight} for all endpoints.
    """
    
    # Generate random weights for all endpoints that sum close to 100 (load balancer standard)
    random_weights = np.random.dirichlet(np.ones(config.NUM_ENDPOINTS), size=1)[0] * 100
    
    final_weights = {}
    addresses = list(config.ENDPOINT_ADDRESSES.keys())
    
    # Assign weights to addresses
    for i in range(config.NUM_ENDPOINTS):
        address = addresses[i]
        # Weights must be positive integers, ensure a minimum of 1
        final_weights[address] = max(1, int(random_weights[i]))
        
    print(f"Mock: Generated new weights: {final_weights} using critical indices: {critical_endpoint_indices}")
    return final_weights