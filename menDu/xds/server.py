import grpc
from concurrent import futures
from threading import Lock, Condition
from typing import Dict

# ----------------------------
# Envoy API imports (generated from protos)
# ----------------------------
from envoy.service.endpoint.v3 import eds_pb2_grpc, eds_pb2
from envoy.config.endpoint.v3 import endpoint_pb2, endpoint_components_pb2
from envoy.config.core.v3 import address_pb2
from google.protobuf import any_pb2, wrappers_pb2
from envoy.service.discovery.v3 import discovery_pb2


class XdsEndpointServer(eds_pb2_grpc.EndpointDiscoveryServiceServicer):
    """The gRPC server that provides endpoint configurations to Envoy."""

    def __init__(self):
        self._lock = Lock()
        self._version_info = 0
        # Initialize with backend servers from config
        from config import BACKEND_SERVER_IPS
        self._endpoints_config: Dict[str, int] = BACKEND_SERVER_IPS.copy()  # {'ip:port': weight}
        self._update_condition = Condition(self._lock)  # For push-based updates

    def update_endpoints(self, new_weights: Dict[str, int]):
        """Thread-safe method called by the main loop to push new weights."""
        with self._lock:
            self._version_info += 1
            self._endpoints_config = new_weights
            print(f"XDS server updated. Version: {self._version_info}")
            # Notify all waiting client streams that there's new data
            self._update_condition.notify_all()

    def StreamEndpoints(self, request_iterator, context):
        """Main gRPC streaming method that Envoy connects to."""
        print(f"Envoy client connected from: {context.peer()}")
        client_version = -1

        try:
            for request in request_iterator:
                with self._lock:
                    # Wait for an update if the client is already up-to-date
                    while self._version_info == client_version:
                        self._update_condition.wait()

                    # New data is available, create and send the response
                    response = self._create_response()
                    client_version = self._version_info
                    yield response
        except grpc.RpcError:
            print("Envoy client disconnected.")

    def _create_response(self) -> discovery_pb2.DiscoveryResponse:
        """Constructs the DiscoveryResponse protobuf message for Envoy."""
        cla = endpoint_pb2.ClusterLoadAssignment(
            cluster_name="my_service",
            endpoints=[endpoint_components_pb2.LocalityLbEndpoints(lb_endpoints=[])]
        )

        for address, weight in self._endpoints_config.items():
            ip, port_str = address.split(':')
            lb_endpoint = endpoint_components_pb2.LbEndpoint(
                endpoint=endpoint_components_pb2.Endpoint(
                    address=address_pb2.Address(
                        socket_address=address_pb2.SocketAddress(
                            address=ip,
                            port_value=int(port_str)
                        )
                    )
                ),
                load_balancing_weight=wrappers_pb2.UInt32Value(value=max(1, weight))
            )
            cla.endpoints[0].lb_endpoints.append(lb_endpoint)

        resource = any_pb2.Any()
        resource.Pack(cla)

        return discovery_pb2.DiscoveryResponse(
            version_info=str(self._version_info),
            resources=[resource],
            type_url='type.googleapis.com/envoy.config.endpoint.v3.ClusterLoadAssignment'
        )


def run_xds_server(server_instance: XdsEndpointServer, ip, port):
    """Starts the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    eds_pb2_grpc.add_EndpointDiscoveryServiceServicer_to_server(server_instance, server)
    server.add_insecure_port(f"{ip}:{port}")
    server.start()
    print(f"XDS server listening on {ip}:{port}")
    server.wait_for_termination()


# --------------------------------------------------------------------------------------------------------------
# import grpc
# from concurrent import futures
# from threading import Lock, Condition
# from typing import Dict

# # ----------------------------
# # Envoy API imports (generated from protos)
# # ----------------------------
# from envoy.service.endpoint.v3 import eds_pb2_grpc, eds_pb2
# from envoy.config.endpoint.v3 import endpoint_pb2, endpoint_components_pb2
# from envoy.config.core.v3 import address_pb2
# from google.protobuf import any_pb2, wrappers_pb2
# from envoy.service.discovery.v3 import discovery_pb2

# # ----------------------------
# # Your backend servers
# # ----------------------------
# BACKEND_SERVER_IPS = {
#     "10.0.195.3:5000": 1,
#     "10.0.195.3:5001": 1,
#     "10.0.195.3:5002": 1,
# }


# class XdsEndpointServer(eds_pb2_grpc.EndpointDiscoveryServiceServicer):
#     """The gRPC server that provides endpoint configurations to Envoy."""

#     def __init__(self):
#         self._lock = Lock()
#         self._version_info = 0
#         self._endpoints_config: Dict[str, int] = BACKEND_SERVER_IPS.copy()  # Initialize with backends
#         self._update_condition = Condition(self._lock)

#     def update_endpoints(self, new_weights: Dict[str, int]):
#         """Thread-safe method called by the main loop to push new weights."""
#         with self._lock:
#             self._version_info += 1
#             self._endpoints_config = new_weights
#             print(f"XDS server updated. Version: {self._version_info}")
#             self._update_condition.notify_all()

#     def StreamEndpoints(self, request_iterator, context):
#         """Main gRPC streaming method that Envoy connects to."""
#         print(f"Envoy client connected from: {context.peer()}")
#         client_version = -1

#         try:
#             for request in request_iterator:
#                 with self._lock:
#                     while self._version_info == client_version:
#                         self._update_condition.wait()

#                     response = self._create_response()
#                     client_version = self._version_info
#                     yield response
#         except grpc.RpcError:
#             print("Envoy client disconnected.")

#     def _create_response(self) -> discovery_pb2.DiscoveryResponse:
#         """Constructs the DiscoveryResponse protobuf message for Envoy."""
#         cla = endpoint_pb2.ClusterLoadAssignment(
#             cluster_name="my_service",
#             endpoints=[endpoint_components_pb2.LocalityLbEndpoints(lb_endpoints=[])]
#         )

#         for address, weight in self._endpoints_config.items():
#             ip, port_str = address.split(':')
#             lb_endpoint = endpoint_components_pb2.LbEndpoint(
#                 endpoint=endpoint_components_pb2.Endpoint(
#                     address=address_pb2.Address(
#                         socket_address=address_pb2.SocketAddress(
#                             address=ip,
#                             port_value=int(port_str)
#                         )
#                     )
#                 ),
#                 load_balancing_weight=wrappers_pb2.UInt32Value(value=max(1, weight))
#             )
#             cla.endpoints[0].lb_endpoints.append(lb_endpoint)

#         resource = any_pb2.Any()
#         resource.Pack(cla)

#         return discovery_pb2.DiscoveryResponse(
#             version_info=str(self._version_info),
#             resources=[resource],
#             type_url='type.googleapis.com/envoy.config.endpoint.v3.ClusterLoadAssignment'
#         )


# def run_xds_server(server_instance: XdsEndpointServer, ip="0.0.0.0", port=50051):
#     """Starts the gRPC server."""
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     eds_pb2_grpc.add_EndpointDiscoveryServiceServicer_to_server(server_instance, server)
#     server.add_insecure_port(f"{ip}:{port}")
#     server.start()
#     print(f"XDS server listening on {ip}:{port}")
#     server.wait_for_termination()


# if __name__ == "__main__":
#     xds_server = XdsEndpointServer()
#     run_xds_server(xds_server)
