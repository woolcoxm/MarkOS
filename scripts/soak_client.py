#!/usr/bin/env python3
"""MARKOS soak gate: many sequential control transactions over one TCP
connection (RUN + ECHO each round). RX-ring wraparound and per-connection
state bugs die here: the default round count pushes far past the number of
RX buffers the kernel posts."""
import socket
import sys


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    s = socket.create_connection((host, port), timeout=20)
    f = s.makefile("rwb")

    def cmd(line):
        f.write(line.encode() + b"\n")
        f.flush()
        return f.readline().strip().decode()

    r = cmd("HELLO")
    assert r.startswith("MARKOS/1 READY"), f"hello: {r}"
    r = cmd("LOAD")
    assert r.startswith("OK loaded"), f"load: {r}"

    ok = 0
    for i in range(rounds):
        r = cmd("RUN")
        assert r.startswith("OK run") and "exact=1" in r, f"round {i} run: {r}"
        r = cmd(f"ECHO n={i}")
        assert r == f"ECHO n={i}", f"round {i} echo: {r!r}"
        ok += 1
        if (i + 1) % 10 == 0:
            print(f"soak: {i + 1}/{rounds} rounds OK")

    r = cmd("STATUS")
    assert r.startswith("OK state=loaded"), f"status: {r}"
    print(f"SOAK PASS ({ok}/{rounds} rounds, single connection)")
    s.close()


if __name__ == "__main__":
    main()
