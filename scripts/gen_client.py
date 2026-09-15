#!/usr/bin/env python3
"""Phase 9 acceptance: GEN over the TCP control protocol.

HELLO -> LOAD -> GEN <steps> <prompt>; asserts every streamed TOK line
(step order + token ids), the GEN_END id list, and the detokenized text,
against the numpy reference output (forward_ref.py --gen).
Exit 0 = pass."""
import socket
import sys


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    ref = sys.argv[3] if len(sys.argv) > 3 else ""

    exp_ids = None
    exp_text = []
    for line in open(ref):
        if line.startswith("GEN ids="):
            exp_ids = [int(x) for x in line.split("ids=")[1].split()[0].split(",")]
        if line.startswith("STEP") and "text=" in line:
            exp_text.append(line.split("text=")[1].strip())
    if exp_ids is None:
        sys.exit("reference file has no GEN line")

    s = socket.create_connection((host, port), timeout=900)
    f = s.makefile("rwb")

    def cmd(line):
        f.write(line.encode() + b"\n")
        f.flush()
        return f.readline().strip().decode()

    ok = True
    r = cmd("HELLO")
    print(("OK  " if r.startswith("MARKOS/1 READY") else "BAD ") + r)
    ok &= r.startswith("MARKOS/1 READY")
    r = cmd("LOAD")
    print(("OK  " if r.startswith("OK loaded") else "BAD ") + r)
    ok &= r.startswith("OK loaded")

    f.write(b"GEN 2 hello world\n")
    f.flush()
    steps, gen_ids, gen_end, text = [], None, None, []
    while True:
        line = f.readline()
        if not line:
            print("connection closed before GEN_END")
            sys.exit(1)
        line = line.strip().decode()
        if line.startswith("TOK "):
            fields = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
            steps.append((int(fields["g"]), int(fields["id"])))
            if "text" in fields:
                text.append(fields["text"])
            print(f"TOK {fields}")
        elif line.startswith("GEN_END"):
            gen_ids = [int(x) for x in
                       line.split("ids=")[1].split()[0].split(",")]
            print(f"GEN_END {gen_ids}")
            break
        elif line.startswith("ERR"):
            print("ERR line:", line)
            sys.exit(1)

    ok &= steps == list(enumerate(exp_ids))
    ok &= gen_ids == exp_ids
    if exp_text:
        ok &= text == exp_text

    print("GEN-NET PASS" if ok else "GEN-NET FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
