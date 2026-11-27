# The compute pool in a container (docs/pool.md, "Docker"): one image for both halves.
#
#   pool server   the controller: docker run ... <image> server --public-url http://<lan address>:8765
#   pool agent    a worker:       docker run ... <image> agent   (what install.sh runs by default)
#
# The agent drives the host's Docker through the mounted /var/run/docker.sock: job containers are the
# host's, siblings of this one, with the same sandbox as before (no network, read-only root, every
# capability dropped, an unprivileged uid, only the GPUs they were given). The entrypoint drops root
# for the agent's own process: it runs as POOL_UID (default 10001) with the socket's group.
# Multi-arch (amd64, arm64: x86 machines, DGX Spark, Jetson). Build from the repository root:
#     docker build -f deploy/pool/Dockerfile -t pool .
FROM python:3.12-slim
ARG DOCKER_CLI_VERSION=27.5.1

# The Docker CLI (static, client only; 27.x speaks to daemons from Docker 20.10 to 29). No apt: the
# image is Python and this one binary.
RUN python -c "import io, platform, tarfile, urllib.request; \
url = 'https://download.docker.com/linux/static/stable/%s/docker-${DOCKER_CLI_VERSION}.tgz' % platform.machine(); \
tar = tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url, timeout=300).read())); \
member = tar.getmember('docker/docker'); member.name = 'docker'; tar.extract(member, '/usr/local/bin')" \
 && docker --version
# The server's dependencies (the agent and the CLI use only the standard library).
RUN pip install --no-cache-dir fastapi==0.165.0 uvicorn==0.54.0 "psycopg[binary]>=3.2"

COPY pool /opt/pool/pool
COPY deploy/pool/entrypoint.sh /usr/local/bin/pool-entrypoint
RUN chmod -R a+rX /opt/pool && chmod 0755 /usr/local/bin/pool-entrypoint \
 && ln -s /usr/local/bin/pool-entrypoint /usr/local/bin/pool-agent \
 && ln -s /usr/local/bin/pool-entrypoint /usr/local/bin/pool \
 && PYTHONPATH=/opt/pool python -c "import pool.server, pool.agent, pool.cli"

# /opt/pool holds the agent's code: install.sh mounts the server's own agent package over it, so a
# machine runs the agent its server ships, whatever this image's build.
ENV PYTHONPATH=/opt/pool PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 POOL_IN_CONTAINER=1 \
    POOL_AGENT_CONFIG=/etc/pool/agent.json POOL_STATE=/var/lib/pool POOL_DATA=/data
ENTRYPOINT ["pool-entrypoint"]
CMD ["agent"]
