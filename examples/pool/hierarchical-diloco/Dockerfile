# The NGC vLLM image (PyTorch, transformers with OlmoeForCausalLM) plus torchft, built from source:
# torchft ships only an sdist on PyPI, with a Rust core (the lighthouse and the manager server).
#   docker build -t pool/diloco-torchft:0.3.1 examples/pool/hierarchical-diloco
FROM nvcr.io/nvidia/vllm:26.04-py3
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev curl && rm -rf /var/lib/apt/lists/* \
 && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
 && . /root/.cargo/env && pip install --no-cache-dir maturin \
 && pip install --no-cache-dir --no-deps --no-build-isolation torchft==0.3.1 \
 && pip install --no-cache-dir "opentelemetry-exporter-otlp-proto-http>=1.39.0" "opentelemetry-sdk>=1.39.0" "opentelemetry-api>=1.39.0" \
 && rm -rf /root/.cargo/registry /root/.rustup /root/.cargo
RUN pip install --no-cache-dir pyarrow  # make_data.py reads parquet
RUN python -c "import torchft, shutil; print('torchft', torchft.__file__); assert shutil.which('torchft_lighthouse')"
