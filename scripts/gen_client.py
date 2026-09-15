#!/usr/bin/env python3
"""Phase 9/10/11 acceptance client.

HELLO -> LOAD -> GEN <steps> <prompt>; asserts every streamed TOK line
(step order + token ids) and GEN_END against the numpy reference
(forward_ref.py --gen). Then STATS: asserts the kernel's gen accounting
matches the streamed token count and cross-checks the kernel-reported
decode time against the client's own wall clock (Phase 10).

--soak <runs> repeats the GEN run <runs> times and checks per-run latency
stability (drift) from the kernel's STATS counters (Phase 11).
Exit 0 = pass."""
import socket
import sys
import time


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    ref = sys.argv[3] if len(sys.argv) > 3 else ""
    runs = 1
    if "--soak" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--soak") + 1])

    exp_ids = None
    for line in open(ref):
        if line.startswith("GEN ids="):
            exp_ids = [int(x) for x in line.split("ids=")[1].split()[0].split(",")]
    if exp_ids is None:
        sys.exit("reference file has no GEN line")
    steps = len(exp_ids)

    s = socket.create_connection((host, port), timeout=950)
    f = s.makefile("rwb")

    def cmd(line):
        f.write(line.encode() + b"\n")
        f.flush()
        return f.readline().strip().decode()

    def parse_stats(line):
        d = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    d[k] = int(v)
                except ValueError:
                    d[k] = v
        return d

    ok = True
    r = cmd("HELLO")
    print(("OK  " if r.startswith("MARKOS/1 READY") else "BAD ") + r)
    ok &= r.startswith("MARKOS/1 READY")
    r = cmd("LOAD")
    print(("OK  " if r.startswith("OK loaded") else "BAD ") + r)
    ok &= r.startswith("OK loaded")

    run_means = []
    for run in range(runs):
        f.write(b"GEN 2 hello world\n")
        f.flush()
        wall0 = time.time()
        step_rows, gen_ids = [], None
        while True:
            line = f.readline()
            if not line:
                print("connection closed before GEN_END")
                sys.exit(1)
            line = line.strip().decode()
            if line.startswith("TOK "):
                fields = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
                step_rows.append((int(fields["g"]), int(fields["id"])))
                if "text" in fields:
                    print(f"TOK {fields}")
            elif line.startswith("GEN_END"):
                gen_ids = [int(x) for x in
                           line.split("ids=")[1].split()[0].split(",")]
                break
            elif line.startswith("ERR"):
                print("ERR line:", line)
                sys.exit(1)
        wall_ms = int((time.time() - wall0) * 1000)
        print(f"run {run}: GEN_END {gen_ids} wall_ms={wall_ms}")

        ok &= step_rows == list(enumerate(exp_ids))
        ok &= gen_ids == exp_ids

        # Phase 10: the kernel's own accounting must agree with the client's
        # wall clock (run_ms covers prefill + decode, as wall does) and with
        # the streamed token count.
        st = parse_stats(cmd("STATS"))
        ok &= st.get("gen_tokens", 0) == steps * (run + 1)
        run_ms = st.get("run_ms", 0)
        drift_ok = run_ms > 0 and abs(wall_ms - run_ms) <= max(2000, run_ms // 2)
        ok &= drift_ok
        print(
            f"run {run}: gen_tokens={st.get('gen_tokens')} tps_milli={st.get('tps_milli')} "
            f"mean_gen_ms={st.get('mean_gen_ms')} min={st.get('gen_min_ms')} "
            f"max={st.get('gen_max_ms')} run_ms={run_ms} wall_ms={wall_ms} "
            f"{'OK' if drift_ok else 'DRIFT'}"
        )
        run_means.append(st.get("mean_gen_ms", 0))

    ok &= all(m > 0 for m in run_means)
    if runs > 1:
        # Phase 11 drift: slowest run's mean must stay within 3x the fastest.
        lo, hi = min(run_means), max(run_means)
        drift = hi <= 3 * max(lo, 1)
        ok &= drift
        print(f"soak drift: min_mean={lo} max_mean={hi} -> {'OK' if drift else 'DRIFT'}")

    print("GEN-NET PASS" if ok else "GEN-NET FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
