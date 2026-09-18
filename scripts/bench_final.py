#!/usr/bin/env python3
"""Final validation: first-request latency, decode, prefill, multi-turn TTFT."""
import json
import time
import urllib.request

BASE = "http://10.0.0.69:8080"


def chat(model, messages, max_tokens=128, temp=0.7, stream=False):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temp, "stream": stream}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    gen = 0
    if stream:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                if first is None:
                    first = time.time() - t0
                d = json.loads(line[6:])
                if d["choices"][0].get("delta", {}).get("content"):
                    gen += 1
        return first, time.time() - t0, gen
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return None, time.time() - t0, d


def decode_bench(model, runs=3):
    out = []
    for _ in range(runs):
        _, el, d = chat(model, [{"role": "user", "content": "Count from 1 to 100."}])
        g = d["usage"]["completion_tokens"]
        out.append(g / el)
    return out


def stream_bench(model, runs=2, tokens=128):
    res = []
    for _ in range(runs):
        first, total, gen = chat(model, [{"role": "user", "content": "Write a story about a robot."}],
                                 max_tokens=tokens, stream=True)
        res.append((first, gen / (total - first)))
    return res


def prefill_bench(model):
    long_prompt = "Summarize: " + ("The quick brown fox jumps over the lazy dog. " * 40)
    _, el, d = chat(model, [{"role": "user", "content": long_prompt}], max_tokens=8)
    p = d["usage"]["prompt_tokens"]
    g = d["usage"]["completion_tokens"]
    gen_s = g / 20.0  # rough decode time at ~20 t/s, subtracted from total
    return p, el, p / max(el - gen_s, 0.001)


def multiturn_ttft(model):
    hist = [{"role": "system", "content": "You are a helpful assistant. Answer briefly."}]
    hist += [{"role": "user", "content": "What is the capital of France?"},
             {"role": "assistant", "content": "Paris."}]
    ttfts = []
    for q in ["What is the capital of Germany?", "And of Italy?", "And of Japan?"]:
        hist.append({"role": "user", "content": q})
        first, total, gen = chat(model, hist, max_tokens=32, stream=True)
        hist.append({"role": "assistant", "content": "ok"})
        ttfts.append(first)
    return ttfts


for model in ["qwen25-05b", "qwen3-0.6b"]:
    print(f"=== {model} ===")
    # FIRST request after boot/warmup (streaming TTFT shows server-side warmth)
    first, total, gen = chat(model, [{"role": "user", "content": "hello"}], max_tokens=16, stream=True)
    print(f"  first-request after boot: ttft={first:.2f}s")
    dec = decode_bench(model)
    print(f"  decode (non-stream): " + " ".join(f"{x:.1f}" for x in dec) + f"  median {sorted(dec)[1]:.1f} t/s")
    for first, tps in stream_bench(model):
        print(f"  decode (stream): ttft={first:.2f}s decode={tps:.1f} t/s")
    p, el, pps = prefill_bench(model)
    print(f"  prefill: {p} tok in {el:.2f}s total -> ~{pps:.0f} tok/s prompt processing")
    tt = multiturn_ttft(model)
    print(f"  multi-turn TTFTs: " + " ".join(f"{x:.2f}" for x in tt))
