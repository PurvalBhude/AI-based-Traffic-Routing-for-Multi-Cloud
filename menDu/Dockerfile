# ----------------------------
# Dockerfile for AI Service with Envoy Protos (Fixed)
# ----------------------------
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ----------------------------
# Install dependencies
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
        protobuf-compiler \
        python3-dev \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Copy source
COPY . .

# ----------------------------
# Install Python dependencies
# ----------------------------
RUN pip install tensorflow==2.13.0 "protobuf<=3.20.3" \
                grpcio grpcio-tools pulp numpy pandas requests prometheus-client tqdm flask googleapis-common-protos flask-cors

# ----------------------------
# Clone and compile Envoy + XDS protos
# ----------------------------
RUN git clone https://github.com/envoyproxy/data-plane-api.git envoy_protos && \
    git clone https://github.com/cncf/xds.git xds_protos && \
    git clone https://github.com/bufbuild/protoc-gen-validate.git validate_protos && \
    git clone https://github.com/googleapis/googleapis.git googleapis_protos && \
    find ./envoy_protos ./xds_protos ./validate_protos ./googleapis_protos -name "*.proto" > proto_files.txt && \
    while read proto; do \
        python -m grpc_tools.protoc \
            -I./envoy_protos \
            -I./xds_protos \
            -I./validate_protos \
            -I./googleapis_protos \
            --python_out=. \
            --grpc_python_out=. \
            "$proto"; \
    done < proto_files.txt && \
    # Ensure ALL generated directories are valid Python packages
    find . -type d -exec touch {}/__init__.py \; && \
    # Cleanup to reduce image size
    rm -rf envoy_protos xds_protos validate_protos googleapis_protos proto_files.txt

# ----------------------------
# Environment path fix (important)
# ----------------------------
ENV PYTHONPATH=/app:$PYTHONPATH

# ----------------------------
# Final setup
# ----------------------------
EXPOSE 50051
EXPOSE 5000

# Create directory for model checkpoints
RUN mkdir -p /app/cfr_rl/pre_trained_model

# ----------------------------
# Default command
# ----------------------------
CMD ["python", "main.py"]
