"""Measure the link between two loopback addresses: python3 linktest.py [MB] (127.0.0.2 -> 127.0.0.3, and 127.0.0.1 -> 127.0.0.1)."""

import socket
import sys
import threading
import time


def measure(src: str, dst: str, megabytes: int) -> float:
    server = socket.socket()
    server.bind((dst, 0))
    server.listen(1)
    port = server.getsockname()[1]

    def sink():
        conn, _ = server.accept()
        while conn.recv(1 << 20):
            pass
        conn.close()
    thread = threading.Thread(target=sink)
    thread.start()
    client = socket.socket()
    client.bind((src, 0))
    client.connect((dst, port))
    block = b"x" * (1 << 20)
    start = time.perf_counter()
    for _ in range(megabytes):
        client.sendall(block)
    client.shutdown(socket.SHUT_WR)
    thread.join()
    seconds = time.perf_counter() - start
    client.close()
    server.close()
    return megabytes * 8 * 1.048576 / seconds


if __name__ == "__main__":
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"127.0.0.2 -> 127.0.0.3: {measure('127.0.0.2', '127.0.0.3', mb):.0f} Mbit/s")
    print(f"127.0.0.3 -> 127.0.0.2: {measure('127.0.0.3', '127.0.0.2', mb):.0f} Mbit/s")
    print(f"127.0.0.1 -> 127.0.0.1: {measure('127.0.0.1', '127.0.0.1', mb * 10):.0f} Mbit/s (unshaped)")
