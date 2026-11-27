"""pool: run containers on machines you and your friends own (issue #45).

M1, the core:
- a server (pool.server) with SQLite and a local artifact folder;
- an agent (pool.agent) on every machine, which leases jobs and runs them in hardened containers;
- a CLI (pool.cli): login, machine add, run, submit, ps, logs, get, cancel.

The agent and the CLI use only the standard library, so a new machine needs nothing but Python 3.10+
and Docker. The server needs FastAPI and uvicorn.
"""

__version__ = "0.1.0"
