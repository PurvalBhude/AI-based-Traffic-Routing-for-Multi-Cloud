# Use official Envoy proxy image
FROM envoyproxy/envoy:v1.31-latest

# Copy your configuration into the container
COPY envoy.yaml /etc/envoy/envoy.yaml

# Expose ports: 80 (client traffic) and 9901 (admin + Prometheus scraping)
EXPOSE 80 9901

# Run Envoy with your config
CMD ["envoy", "-c", "/etc/envoy/envoy.yaml", "--service-node", "node_1", "--service-cluster", "envoy_cluster"]
