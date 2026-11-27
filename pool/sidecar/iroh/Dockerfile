# The iroh-blobs sidecar of a pool agent (pool/peers.py, IrohSidecar). Built per architecture:
#   docker build -t pool-iroh pool/sidecar/iroh
# The agent runs it with the host's network (it talks to the agent's peer server on loopback):
#   docker run -d --network host pool-iroh --control 127.0.0.1:8767 --agent http://127.0.0.1:8766
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src src
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/pool-iroh /usr/local/bin/pool-iroh
USER 65534:65534
ENTRYPOINT ["/usr/local/bin/pool-iroh"]
