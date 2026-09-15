#!/usr/bin/env python3
"""Compare kernel forward-pass output against the numpy reference.

Lines must have identical keys and appear in the same order. Within a
line, fields are name=value; comma-separated values compare element-wise.
Integers compare exactly; floats with tolerance (f32 kernel vs f64
reference). Exit 0 = pass."""
import re
import sys

ATOL = 2e-4
RTOL = 5e-3

INT_RE = re.compile(r"^-?\d+$")


def rows(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("PASS:"):
            out.append(line.split())
    return out


def fields(tokens):
    """-> list of (name_or_None, [values])."""
    out = []
    for tok in tokens:
        if "=" in tok:
            name, rest = tok.split("=", 1)
        else:
            name, rest = None, tok
        out.append((name, rest.split(",")))
    return out


def main():
    exp, got = rows(sys.argv[1]), rows(sys.argv[2])
    if [r[0] for r in exp] != [r[0] for r in got]:
        print("key mismatch:")
        print("  expected:", [r[0] for r in exp])
        print("  got:     ", [r[0] for r in got])
        sys.exit(1)
    worst = 0.0
    for e, g in zip(exp, got):
        # Pure-text lines (e.g. TOKS): exact match wins immediately.
        if " ".join(e[1:]) == " ".join(g[1:]):
            continue
        fe, fg = fields(e[1:]), fields(g[1:])
        if len(fe) != len(fg):
            print(f"{e[0]}: field count {len(fe)} vs {len(fg)}")
            sys.exit(1)
        for (ne, ve), (ng, vg) in zip(fe, fg):
            if ne != ng:
                print(f"{e[0]}: field name {ne} vs {ng}")
                sys.exit(1)
            if len(ve) != len(vg):
                print(f"{e[0]}.{ne}: value count {len(ve)} vs {len(vg)}")
                sys.exit(1)
            for a, b in zip(ve, vg):
                if INT_RE.match(a):
                    if a != b:
                        print(f"{e[0]}.{ne}: {a} != {b} (int)")
                        sys.exit(1)
                    continue
                fa, fb = float(a), float(b)
                d = abs(fa - fb)
                tol = ATOL + RTOL * abs(fa)
                if d > tol:
                    print(f"{e[0]}.{ne}: {a} vs {b} (diff {d:.3e} > {tol:.3e})")
                    sys.exit(1)
                worst = max(worst, d)
    print(f"forward check OK (worst abs diff {worst:.3e})")


if __name__ == "__main__":
    main()
