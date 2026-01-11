# Use the official Prometheus image as base
FROM prom/prometheus:v2.53.0

# Set working directory
WORKDIR /etc/prometheus

# Copy your Prometheus config into the image
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Expose Prometheus web port
EXPOSE 9090

# Start Prometheus with your config
CMD ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus"]
