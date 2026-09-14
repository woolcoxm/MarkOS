#!/usr/bin/env python3
"""MARKOS control-protocol acceptance client.

Connects to the appliance TCP port and walks the v0 command set:
auth-negative HELLO -> auth HELLO -> STATUS -> LOAD -> RUN -> PING.
Exits 0 only if every response matches the protocol contract."""
import socket
import sys


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    token = sys.argv[3] if len(sys.argv) > 3 else ""
    s = socket.create_connection((host, port), timeout=15)
    f = s.makefile("rwb")

    def cmd(line, expect_prefix):
        f.write(line.encode() + b"\n")
        f.flush()
        resp = f.readline().strip().decode()
        ok = resp.startswith(expect_prefix)
        print(("OK  " if ok else "BAD ") + f"{line} -> {resp}")
        return ok, resp

    ok = True
    if token:
        # Auth must reject a wrong token before anything else works.
        o, r = cmd("HELLO not-the-token", "ERR auth")
        ok &= o
        o, r = cmd("STATUS", "ERR auth")
        ok &= o
    o, r = cmd(f"HELLO {token}".rstrip(), "MARKOS/1 READY")
    ok &= o
    o, r = cmd("STATUS", "OK state=")
    ok &= o
    o, r = cmd("STATS", "OK stats uptime_ms=")
    ok &= o and "cores=" in r and "served=" in r
    o, r = cmd("LOAD", "OK loaded bytes=")
    ok &= o and "tensors=" in r
    o, r = cmd("RUN", "OK run")
    ok &= o and "exact=1" in r
    o, r = cmd("PING", "PONG")
    ok &= o

    s.close()
    print("CONTROL PASS" if ok else "CONTROL FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
