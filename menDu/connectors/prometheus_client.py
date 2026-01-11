# # import requests
# # import numpy as np
# # import config

# # # class PrometheusClient:
# # #     def __init__(self, ip, port):
# # #         self.endpoint = f"http://{ip}:{port}/api/v1/query"

# # #     def get_endpoint_state_vector(self) -> np.ndarray:
# # #         """
# # #         Queries Prometheus for latency and error rates for all endpoints
# # #         and combines them into a single state vector for the AI model.
        
# # #         Returns: A numpy array of shape (STATE_SIZE,)
# # #         """
# # #         all_metrics = {}
# # #         for metric_name, query in config.PROMETHEUS_QUERIES.items():
# # #             try:
# # #                 response = requests.get(self.endpoint, params={'query': query})
# # #                 response.raise_for_status()
# # #                 results = response.json()['data']['result']

# # #                 metric_values = {}
# # #                 for result in results:
# # #                     address = result['metric']['envoy_upstream_address']
# # #                     value = float(result['value'][1])
# # #                     if address in config.ENDPOINT_ADDRESSES:
# # #                         metric_values[address] = value
# # #                 all_metrics[metric_name] = metric_values
# # #             except requests.RequestException as e:
# # #                 print(f"Error querying Prometheus for {metric_name}: {e}")
# # #                 # Return zeros if Prometheus is unavailable
# # #                 return np.zeros(config.STATE_SIZE)

# # #         # Build the final state vector in a fixed order
# # #         state_vector = []
# # #         for address in config.ENDPOINT_ADDRESSES:
# # #             # Append each metric for the current address
# # #             latency = all_metrics.get("latency", {}).get(address, 0.0) # Default to 0 if metric is missing
# # #             error = all_metrics.get("error_rate", {}).get(address, 0.0)
# # #             state_vector.extend([latency, error])
            
# # #         print(f"Current State Vector: {state_vector}")
# # #         return np.array(state_vector)


# # # in connectors.py (PrometheusClient)
# # import requests
# # import numpy as np
# # import time

# # class PrometheusClient:
# #     def __init__(self, host, port):
# #         self.base = f"http://{host}:{port}"
# #         self.timeout = 5

# #     def _query(self, q):
# #         url = f"{self.base}/api/v1/query"
# #         try:
# #             r = requests.get(url, params={'query': q}, timeout=self.timeout)
# #             r.raise_for_status()
# #             result = r.json()
# #             if result['status'] != 'success':
# #                 print("Prometheus query not success:", result)
# #                 return None
# #             return result['data']['result']
# #         except Exception as e:
# #             print(f"Prometheus query error: {e} (url={url}, query={q[:80]})")
# #             return None

# #     def get_endpoint_state_vector(self):
# #         # collect results for each endpoint/metric
# #         from config import PROMETHEUS_QUERIES, ENDPOINT_ADDRESSES, NUM_METRICS
# #         state = []
# #         any_data = False
# #         for metric_name, q in PROMETHEUS_QUERIES.items():
# #             res = self._query(q)
# #             if not res:
# #                 # append zeros per endpoint if no data
# #                 state.extend([0.0] * len(ENDPOINT_ADDRESSES))
# #             else:
# #                 any_data = True
# #                 # mapping: result list contains samples for each upstream_address
# #                 # ensure mapping to ENDPOINT_ADDRESSES order
# #                 values_map = {}
# #                 for item in res:
# #                     addr = item['metric'].get('envoy_upstream_address') or item['metric'].get('instance')
# #                     val = float(item['value'][1]) if 'value' in item and item['value'] else 0.0
# #                     values_map[addr] = val
# #                 for endpoint in ENDPOINT_ADDRESSES.keys():
# #                     state.append(values_map.get(endpoint, 0.0))

# #         state_vector = np.array(state, dtype=np.float32)
# #         print("Current State Vector:", state_vector.tolist())
# #         if not any_data or state_vector.sum() == 0:
# #             return None  # caller handles skipping
# #         return state_vector
# # ----------------------------------------------------------------------------
# import requests
# import numpy as np
# import time
# import config # Ensure config is imported at the top

# class PrometheusClient:
#     def __init__(self, host, port):
#         self.base = f"http://{host}:{port}"
#         self.timeout = 5

#     def _query(self, q):
#         url = f"{self.base}/api/v1/query"
#         try:
#             r = requests.get(url, params={'query': q}, timeout=self.timeout)
#             r.raise_for_status()
#             result = r.json()
#             if result['status'] != 'success':
#                 print("Prometheus query not success:", result)
#                 return None
#             return result['data']['result']
#         except Exception as e:
#             # This handles connection errors or JSON decoding errors
#             print(f"Prometheus query error: {e} (url={url}, query={q[:80]})")
#             return None

#     def get_endpoint_state_vector(self):
#         """
#         Queries Prometheus for all metrics and combines them into a single state vector.
        
#         Returns: A numpy array of shape (STATE_SIZE,) or None on critical failure.
#         """
#         # Collect results for each endpoint/metric
#         from config import PROMETHEUS_QUERIES, ENDPOINT_ADDRESSES, NUM_METRICS 
#         state = []
        
#         # Iterate over all defined metrics
#         for metric_name, q in PROMETHEUS_QUERIES.items():
#             res = self._query(q)
            
#             if res is None:
#                 # If _query returned None, Prometheus is down/unreachable (critical failure).
#                 # The caller (main.py) must skip the cycle.
#                 print(f"Critical Error: Query for {metric_name} failed. Prometheus is unreachable.")
#                 return None
                 
#             if not res:
#                 # Prometheus is up, but no metrics were found for this query (e.g., no traffic/metrics)
#                 state.extend([0.0] * len(ENDPOINT_ADDRESSES))
#             else:
#                 # Data was returned, map it to the correct order
#                 values_map = {}
#                 for item in res:
#                     # Robust label extraction: use envoy_upstream_address or fall back to instance
#                     addr = item['metric'].get('envoy_upstream_address') or item['metric'].get('instance')
#                     val = float(item['value'][1]) if 'value' in item and item['value'] else 0.0
#                     values_map[addr] = val
                
#                 # Fill state vector, ensuring correct order and defaulting to 0.0 if missing
#                 for endpoint in ENDPOINT_ADDRESSES.keys():
#                     state.append(values_map.get(endpoint, 0.0))

#         state_vector = np.array(state, dtype=np.float32)
#         print("Current State Vector:", state_vector.tolist())
        
#         # observation and must be returned as a NumPy array.
#         # main.py will handle the zero-sum array and skip the optimization cycle if needed.
        
#         return state_vector




# import requests
# import numpy as np
import config

# class PrometheusClient:
#     def _init_(self, ip, port):
#         self.endpoint = f"http://{ip}:{port}/api/v1/query"

#     def get_endpoint_state_vector(self) -> np.ndarray:
#         """
#         Queries Prometheus for latency and error rates for all endpoints
#         and combines them into a single state vector for the AI model.
        
#         Returns: A numpy array of shape (STATE_SIZE,)
#         """
#         all_metrics = {}
#         for metric_name, query in config.PROMETHEUS_QUERIES.items():
#             try:
#                 response = requests.get(self.endpoint, params={'query': query})
#                 response.raise_for_status()
#                 results = response.json()['data']['result']

#                 metric_values = {}
#                 for result in results:
#                     address = result['metric']['envoy_upstream_address']
#                     value = float(result['value'][1])
#                     if address in config.ENDPOINT_ADDRESSES:
#                         metric_values[address] = value
#                 all_metrics[metric_name] = metric_values
#             except requests.RequestException as e:
#                 print(f"Error querying Prometheus for {metric_name}: {e}")
#                 # Return zeros if Prometheus is unavailable
#                 return np.zeros(config.STATE_SIZE)

#         # Build the final state vector in a fixed order
#         state_vector = []
#         for address in config.ENDPOINT_ADDRESSES:
#             # Append each metric for the current address
#             latency = all_metrics.get("latency", {}).get(address, 0.0) # Default to 0 if metric is missing
#             error = all_metrics.get("error_rate", {}).get(address, 0.0)
#             state_vector.extend([latency, error])
            
#         print(f"Current State Vector: {state_vector}")
#         return np.array(state_vector)


# # in connectors.py (PrometheusClient)
# import requests
# import numpy as np
# import time

# class PrometheusClient:
#     def _init_(self, host, port):
#         self.base = f"http://{host}:{port}"
#         self.timeout = 5

#     def _query(self, q):
#         url = f"{self.base}/api/v1/query"
#         try:
#             r = requests.get(url, params={'query': q}, timeout=self.timeout)
#             r.raise_for_status()
#             result = r.json()
#             if result['status'] != 'success':
#                 print("Prometheus query not success:", result)
#                 return None
#             return result['data']['result']
#         except Exception as e:
#             print(f"Prometheus query error: {e} (url={url}, query={q[:80]})")
#             return None

#     def get_endpoint_state_vector(self):
#         # collect results for each endpoint/metric
#         from config import PROMETHEUS_QUERIES, ENDPOINT_ADDRESSES, NUM_METRICS
#         state = []
#         any_data = False
#         for metric_name, q in PROMETHEUS_QUERIES.items():
#             res = self._query(q)
#             if not res:
#                 # append zeros per endpoint if no data
#                 state.extend([0.0] * len(ENDPOINT_ADDRESSES))
#             else:
#                 any_data = True
#                 # mapping: result list contains samples for each upstream_address
#                 # ensure mapping to ENDPOINT_ADDRESSES order
#                 values_map = {}
#                 for item in res:
#                     addr = item['metric'].get('envoy_upstream_address') or item['metric'].get('instance')
#                     val = float(item['value'][1]) if 'value' in item and item['value'] else 0.0
#                     values_map[addr] = val
#                 for endpoint in ENDPOINT_ADDRESSES.keys():
#                     state.append(values_map.get(endpoint, 0.0))

#         state_vector = np.array(state, dtype=np.float32)
#         print("Current State Vector:", state_vector.tolist())
#         if not any_data or state_vector.sum() == 0:
#             return None  # caller handles skipping
#         return state_vector
    

# import requests
# import numpy as np
# import time

# class PrometheusClient:
#     def __init__(self, host, port):
#         print(f"--- MOCKING PrometheusClient: Ignoring {host}:{port} ---")
#         # Initialize necessary constants
#         self.STATE_SIZE = config.STATE_SIZE
#         self.NUM_ENDPOINTS = config.NUM_ENDPOINTS
#         self.NUM_METRICS = config.NUM_METRICS

#     def get_endpoint_state_vector(self):
#         """Generates a random state vector simulating Prometheus metrics."""
        
#         # We need a vector of size STATE_SIZE = NUM_ENDPOINTS * NUM_METRICS
#         # Metrics: [Latency, ErrorRate, Saturation, Throughput] per endpoint
        
#         # Example ranges for simulated data:
#         # P95 Latency (ms): 50 to 500
#         # Error Rate (%): 0.0 to 5.0
#         # Saturation (pending reqs): 0 to 10
#         # Throughput (RPS): 10 to 100
        
#         # NOTE: State vector order is crucial: [L1, E1, S1, T1, L2, E2, S2, T2, ...]
        
#         vector = []
#         for i in range(self.NUM_ENDPOINTS):
#             # Latency (P95 in ms) - simulate healthy-to-slow
#             latency = np.random.uniform(50.0, 500.0) 
#             # Error Rate (%) - simulate 0 to 5%
#             error_rate = np.random.uniform(0.0, 5.0) 
#             # Saturation (pending) - simulate low to medium
#             saturation = np.random.randint(0, 10) 
#             # Throughput (RPS) - simulate varied load
#             throughput = np.random.uniform(10.0, 100.0) 
            
#             # Append in the order defined by the queries in config.py
#             vector.extend([latency, error_rate, saturation, throughput])

#         state_vector = np.array(vector, dtype=np.float32)
#         print("Mock State Vector Generated:", [f'{x:.2f}' for x in state_vector.tolist()])
#         return state_vector



import numpy as np
import config # Assuming config is in the import path

class PrometheusClient:
    def __init__(self, ip, port):
        # Initialize with config values, but they are not used in mock
        self.NUM_ENDPOINTS = config.NUM_ENDPOINTS
        self.NUM_METRICS = config.NUM_METRICS

    def get_endpoint_state_vector(self) -> np.ndarray:
        """Generates a random state vector simulating Prometheus metrics."""
        
        vector = []
        for i in range(self.NUM_ENDPOINTS):
            # Latency (P95 in ms) - simulate healthy-to-slow
            latency = np.random.uniform(50.0, 500.0) 
            # Error Rate (%) - simulate 0 to 5%
            error_rate = np.random.uniform(0.0, 5.0) 
            # Saturation (pending) - simulate low to medium
            saturation = np.random.randint(0, 10) 
            # Throughput (RPS) - simulate varied load
            throughput = np.random.uniform(10.0, 100.0) 
            
            # Append in the required order: [L1, E1, S1, T1, L2, E2, S2, T2, ...]
            vector.extend([latency, error_rate, saturation, throughput])

        state_vector = np.array(vector, dtype=np.float32)
        # Ensure non-zero/non-NaN values for robustness
        state_vector = np.maximum(state_vector, 1e-6)
        return state_vector