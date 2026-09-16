#!/usr/bin/env python3
"""Phase 9/10/11 acceptance client.

HELLO -> LOAD -> GEN <steps> <prompt>; asserts every streamed TOK line
(step order + token ids) and GEN_END against the numpy reference
(forward_ref.py --gen). Then STATS: asserts the kernel's gen accounting
matches the streamed token count and cross-checks the kernel-reported
decode time against the client's own wall clock (Phase 10).

--soak <runs> repeats the full GEN run <runs> times over ONE connection
(HELLO+LOAD once, then back-to-back GEN sessions — the stricter leak
probe: the engine's statics cycle IN_GEN repeatedly on a single session)
and checks per-run latency stability (drift) from the kernel's STATS
counters (Phase 11).

Transport note: reads are deadline-bounded with a STATS kick. The kernel
answers STATS between decode steps, so a kick is both a liveness probe
and a nudge for the dev transport (QEMU user-mode networking), which can
stall a reply's delivery across long decode pauses.
Exit 0 = pass."""
import select
import socket
import sys
import time


class Reader:
    """Deadline-bounded line reader over the raw socket."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def line(self, deadline_s=120.0):
        end = time.time() + deadline_s
        while True:
            nl = self.buf.find(b"\n")
            if nl >= 0:
                out = self.buf[:nl]
                self.buf = self.buf[nl + 1:]
                return out.decode().strip()
            if time.time() > end:
                return None
            r, _, _ = select.select([self.sock], [], [], 1.0)
            if r:
                chunk = self.sock.recv(4096)
                if not chunk:
                    # EOF is recoverable at a run boundary (the appliance
                    # accepts a fresh handshake): report it, do not die.
                    return "EOF"
                self.buf += chunk


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


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    ref = sys.argv[3] if len(sys.argv) > 3 else ""
    runs = 1
    if "--soak" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--soak") + 1])

    structural = "--structural" in sys.argv
    exp_ids = None
    for line in open(ref):
        if line.startswith("GEN ids="):
            exp_ids = [int(x) for x in line.split("ids=")[1].split()[0].split(",")]
    if not structural and exp_ids is None:
        sys.exit("reference file has no GEN line")
    steps = len(exp_ids)

    s = socket.create_connection((host, port), timeout=950)
    rd = Reader(s)

    def send(line):
        try:
            s.sendall(line.encode() + b"\n")
        except OSError:
            reconnect("send failed")

    def cmd(line, deadline=950):
        send(line)
        r = rd.line(deadline)
        if r == "EOF":
            reconnect("connection closed")
            send(line)
            r = rd.line(deadline)
        return r

    def parse_kv(line):
        d = {}
        for p in line.split()[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                try:
                    d[k] = int(v)
                except ValueError:
                    d[k] = v
        return d

    ok = True
    run_means = []
    prev_run_ms = 0

    r = cmd("HELLO")
    print(("OK  " if r is not None and r.startswith("MARKOS/1 READY") else "BAD ") + str(r))
    ok &= r is not None and r.startswith("MARKOS/1 READY")
    r = cmd("LOAD")
    print(("OK  " if r is not None and r.startswith("OK loaded") else "BAD ") + str(r))
    ok &= r is not None and r.startswith("OK loaded")

    def reconnect(reason):
        """The dev transport (QEMU user-mode hostfwd) has been observed to
        kill a long-lived flow outright. The appliance is fine — its
        counters persist and it accepts a fresh handshake — so tear down
        the dead socket and re-establish, then resume the soak."""
        nonlocal s, rd
        print(f"transport lost ({reason}); reconnecting", flush=True)
        try:
            s.close()
        except OSError:
            pass
        for attempt in range(10):
            try:
                s = socket.create_connection((host, port), timeout=950)
                rd2 = Reader(s)
                s.sendall(b"HELLO\n")
                r2 = rd2.line(10)
                if r2 is not None and r2.startswith("MARKOS/1 READY"):
                    rd = rd2
                    print("reconnected", flush=True)
                    return
            except OSError as e:
                print(f"reconnect attempt {attempt + 1}: {e}")
            try:
                s.close()
            except OSError:
                pass
            time.sleep(3)
        sys.exit("appliance never answered HELLO after transport loss")

    for run in range(runs):
        if run > 0:
            r = cmd("HELLO")
            if r is None or r == "EOF":
                reconnect("no banner")
                r = "MARKOS/1 READY"
            print(("OK  " if r.startswith("MARKOS/1 READY") else "BAD ") + str(r))
            ok &= r.startswith("MARKOS/1 READY")

        send("GEN 2 hello world")
        wall0 = time.time()
        step_rows, gen_ids = [], None
        kicks = 0
        while True:
            line = rd.line(120)
            if line == "EOF":
                reconnect("connection closed mid-run")
                send("GEN 2 hello world")
                kicks = 0
                continue
            if line is None:
                if kicks >= 3:
                    reconnect("GEN stream stalled")
                    send("GEN 2 hello world")
                    kicks = 0
                    continue
                kicks += 1
                print(f"stall in GEN stream; STATS kick {kicks}", flush=True)
                send("STATS")
                continue
            if line.startswith("#"):
                continue  # keepalive comment
            if line.startswith("TOK "):
                fields = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
                step_rows.append((int(fields["g"]), int(fields["id"])))
                if "text" in fields:
                    print(f"TOK {fields}")
            elif line.startswith("OK stats"):
                # Reply to a kick: live proof the kernel is generating.
                print("kick reply:", line)
            elif line.startswith("GEN_END"):
                gen_ids = [int(x) for x in
                           line.split("ids=")[1].split()[0].split(",")]
                break
            elif line.startswith("ERR busy"):
                # A transport replay re-triggered GEN while one was still
                # in flight (or our resend raced it): the in-flight run's
                # stream is what this run should consume.
                print("kernel busy; consuming in-flight run", flush=True)
                continue
            elif line.startswith("ERR"):
                print("ERR line:", line)
                sys.exit(1)
        wall_ms = int((time.time() - wall0) * 1000)
        print(f"run {run}: GEN_END {gen_ids} wall_ms={wall_ms}")

        if structural:
            # Quantized-activation serving: ids legitimately diverge from
            # the f32/numpy reference; assert structural plausibility.
            vocab = 151936
            ok &= all(0 <= g < vocab for g in gen_ids)
            ok &= len(gen_ids) == len(exp_ids)
        else:
            ok &= step_rows == list(enumerate(exp_ids))
            ok &= gen_ids == exp_ids

        # Phase 10: the kernel's own accounting must agree with the client's
        # wall clock (run_ms covers prefill + decode, as wall does) and with
        # the streamed token count.
        st = parse_stats(cmd("STATS"))
        ok &= st.get("gen_tokens", 0) == steps * (run + 1)
        # run_ms accumulates across runs; the per-run delta is what the
        # client's wall clock covers.
        run_ms_total = st.get("run_ms", 0)
        run_ms = run_ms_total - prev_run_ms
        prev_run_ms = run_ms_total
        drift_ok = run_ms > 0 and abs(wall_ms - run_ms) <= max(2000, run_ms // 2)
        ok &= drift_ok
        print(
            f"run {run}: gen_tokens={st.get('gen_tokens')} tps_milli={st.get('tps_milli')} "
            f"mean_gen_ms={st.get('mean_gen_ms')} min={st.get('gen_min_ms')} "
            f"max={st.get('gen_max_ms')} run_ms={run_ms} wall_ms={wall_ms} "
            f"{'OK' if drift_ok else 'DRIFT'}"
        )
        run_means.append(st.get("mean_gen_ms", 0))

    # All runs complete: one clean close (kernel FIN -> LISTEN).
    s.close()

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
