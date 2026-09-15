#!/usr/bin/env python3
"""One-shot: stage markers with uptime timestamps in gen_stream."""

src = open("kernel/src/control.rs").read()

old = """    let mut ids = [0u32; engine::MAX_TOKENS];
    let n_tok = match engine::tokenize(meta, prompt, &mut ids) {
        Ok(n) if n > 0 => n,
        _ => {
            tcp::stream(b"ERR tokenize\\n");
            return;
        }
    };"""
new = """    let mut ids = [0u32; engine::MAX_TOKENS];
    let n_tok = match engine::tokenize(meta, prompt, &mut ids) {
        Ok(n) if n > 0 => n,
        _ => {
            tcp::stream(b"ERR tokenize\\n");
            return;
        }
    };
    uart::locked_write(format_args!(
        "g2: tok t={} n_tok={n_tok} ids={} {}\\n",
        timer::uptime_ms(),
        ids[0],
        if n_tok > 1 { ids[1] } else { 0 }
    ));"""
assert old in src, "tokenize anchor missing"
src = src.replace(old, new)

old2 = """        for l in 0..geo.n_layers as usize {
            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {
                tcp::stream(b"ERR layer\\n");
                return;
            }
            if l % 8 == 7 {
                tcp::stream(b"# k\\n");
            }
        }
    }

    // Greedy steps: norm -> argmax -> stream -> feed back."""
new2 = """        for l in 0..geo.n_layers as usize {
            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {
                tcp::stream(b"ERR layer\\n");
                return;
            }
            if l % 8 == 7 {
                tcp::stream(b"# k\\n");
            }
        }
        uart::locked_write(format_args!(
            "g2: prefill pos={pos} t={}\\n",
            timer::uptime_ms()
        ));
    }

    // Greedy steps: norm -> argmax -> stream -> feed back."""
assert old2 in src, "prefill anchor missing"
src = src.replace(old2, new2)

old3 = """        gen_ids[g] = tok;
        let mut line = [0u8; 96];"""
new3 = """        gen_ids[g] = tok;
        uart::locked_write(format_args!(
            "g2: argmax g={g} tok={tok} t={}\\n",
            timer::uptime_ms()
        ));
        let mut line = [0u8; 96];"""
assert old3 in src, "argmax anchor missing"
src = src.replace(old3, new3)

open("kernel/src/control.rs", "w").write(src)
print("stage markers added")

c = open("scripts/gen_client.py").read()
oldc = """    s = socket.create_connection((host, port), timeout=950)
    f = s.makefile("rwb")

    def cmd(line):
        f.write(line.encode() + b"\\n")
        f.flush()
        return f.readline().strip().decode()

    ok = True
    r = cmd("HELLO")
    print(("OK  " if r.startswith("MARKOS/1 READY") else "BAD ") + r)
    ok &= r.startswith("MARKOS/1 READY")
    r = cmd("LOAD")
    print(("OK  " if r.startswith("OK loaded") else "BAD ") + r)
    ok &= r.startswith("OK loaded")

    run_means = []
    for run in range(runs):"""
newc = """    ok = True
    run_means = []
    for run in range(runs):
        # Fresh connection per run: the kernel returns to LISTEN on FIN and
        # the run boundary then exercises a clean handshake each time.
        s = socket.create_connection((host, port), timeout=950)
        f = s.makefile("rwb")

        def cmd(line):
            f.write(line.encode() + b"\\n")
            f.flush()
            return f.readline().strip().decode()

        r = cmd("HELLO")
        print(("OK  " if r.startswith("MARKOS/1 READY") else "BAD ") + r)
        ok &= r.startswith("MARKOS/1 READY")
        if run == 0:
            r = cmd("LOAD")
            print(("OK  " if r.startswith("OK loaded") else "BAD ") + r)
            ok &= r.startswith("OK loaded")"""
assert oldc in c, "client head anchor missing"
c = c.replace(oldc, newc)

oldc2 = """    print("GEN-NET PASS" if ok else "GEN-NET FAIL")
    sys.exit(0 if ok else 1)"""
newc2 = """        f.close()
        s.close()

    print("GEN-NET PASS" if ok else "GEN-NET FAIL")
    sys.exit(0 if ok else 1)"""
assert oldc2 in c, "client tail anchor missing"
c = c.replace(oldc2, newc2)
open("scripts/gen_client.py", "w").write(c)
print("client reconnect-per-run added")
