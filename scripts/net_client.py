#!/usr/bin/env python3
"""Host-side client for the MarkOS network acceptance test: connect to the
forwarded appliance port, send MARKOS-PING, require MARKOS-PONG back."""
import socket
import sys


def main(host: str, port: int) -> int:
    s = socket.create_connection((host, port), timeout=8)
    s.settimeout(8)
    s.sendall(b"MARKOS-PING")
    data = b""
    try:
        while b"MARKOS-PONG" not in data and len(data) < 256:
            chunk = s.recv(256)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        pass
    s.close()
    if b"MARKOS-PONG" in data:
        print("CLIENT PASS:", data)
        return 0
    print("CLIENT FAIL:", data)
    return 1


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    sys.exit(main(host, port))
