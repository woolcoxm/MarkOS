#!/usr/bin/env python3
"""MarkOS hardware benchmark + eval suite.

Measures decode speed (tokens/sec), prompt-processing speed, and output
quality across both serving tiers (CPU and NPU) on the real appliance.

Usage: python3 bench.py [--host 10.0.0.69]
"""
import argparse
import json
import time
import urllib.request
import urllib.error
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="10.0.0.69")
args = parser.parse_args()
BASE = f"http://{args.host}:8080"


def chat(model, messages, max_tokens=64, temperature=0.7):
    """One completion, returns (text, gen_tokens, prompt_tokens, elapsed_s)."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return f"ERROR {e.code}: {e.read().decode()[:200]}", 0, 0, time.time() - t0
    elapsed = time.time() - t0
    text = d["choices"][0]["message"]["content"]
    return (text,
            d["usage"]["completion_tokens"],
            d["usage"]["prompt_tokens"],
            elapsed)


def bench_decode(model, n_runs=3, max_tokens=64):
    """Decode speed: fixed short prompt, measure tokens/sec."""
    results = []
    for i in range(n_runs):
        text, gen, prompt, elapsed = chat(
            model, [{"role": "user", "content": "Count from 1 to 100."}],
            max_tokens=max_tokens)
        tps = gen / elapsed if elapsed > 0 and gen > 0 else 0
        results.append((gen, prompt, elapsed, tps, text[:40]))
        print(f"  run {i+1}: {gen} tok in {elapsed:.1f}s = {tps:.1f} t/s")
    best = max(r[3] for r in results)
    median = sorted(r[3] for r in results)[len(results) // 2]
    return best, median, results


def bench_prefill(model, prompt_tokens_approx=200):
    """Prompt processing: long prompt, short generation.

    Reports prompt-processing throughput as prompt_tok / (total - gen_time),
    where gen_time is estimated from the measured decode rate. (The older
    prompt/total formula under-reported prefill badly on fast engines:
    at 20 t/s decode, 8 generation tokens alone outweigh a 0.1 s prefill.)
    """
    long_prompt = "Explain the history of computing from the abacus through "
    "transistors, integrated circuits, microprocessors, personal computers, "
    "the internet, mobile devices, and artificial intelligence. Cover key "
    "innovators, breakthrough moments, and the social impact of each era. "
    "Include specific dates and technical details where possible. "
    "Discuss how each advancement built upon previous ones."
    text, gen, prompt, elapsed = chat(model, [{"role": "user", "content": long_prompt}],
                                      max_tokens=8)
    tps = gen / elapsed if elapsed > 0 else 0
    gen_s = gen / max(tps, 0.001)
    prefill_s = max(elapsed - gen_s, 0.001)
    prefill_tps = prompt / prefill_s
    print(f"  {prompt} prompt tok + {gen} gen in {elapsed:.1f}s "
          f"(total {tps:.1f} t/s, prefill ~{prefill_tps:.0f} t/s)")
    return prompt, gen, elapsed, prefill_tps


def eval_quality(model, tests):
    """Quality eval: run each test, check for expected substrings."""
    passed = 0
    for name, prompt, expected, max_tokens in tests:
        text, gen, _, elapsed = chat(model, [{"role": "user", "content": prompt}],
                                     max_tokens=max_tokens, temperature=0.1)
        ok = any(e.lower() in text.lower() for e in expected)
        status = "PASS" if ok else "FAIL"
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}: {status} — {repr(text[:80])}")
        if ok:
            passed += 1
    return passed, len(tests)


EVAL_TESTS = [
    ("arithmetic-1", "What is 7 × 8?", ["56"], 16),
    ("arithmetic-2", "What is 15 + 27?", ["42"], 16),
    ("factual-1", "What is the capital of France?", ["Paris"], 16),
    ("factual-2", "What planet is closest to the sun?", ["Mercury"], 16),
    ("reasoning-1", "If I have 3 apples and buy 5 more, how many do I have?", ["8", "eight"], 24),
    ("instruction-1", "Say the word 'hello' and nothing else.", ["hello"], 12),
    ("reasoning-2", "What comes next: 2, 4, 6, 8, ...?", ["10", "ten"], 12),
    ("factual-3", "How many days are in a week?", ["7", "seven", "seven"], 12),
]


def main():
    print("=" * 70)
    print("MarkOS Hardware Benchmark + Quality Eval")
    print(f"Target: http://{args.host}:8080")
    print("=" * 70)

    # Check health
    try:
        with urllib.request.urlopen(f"{BASE}/healthz", timeout=300) as r:
            health = json.loads(r.read())
            print(f"\nEngine: {health['version']}, uptime {health['uptime_s']}s")
    except Exception as e:
        print(f"ERROR: engine not reachable: {e}")
        sys.exit(1)

    # Discover models
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=300) as r:
        models = json.loads(r.read())
        model_ids = [m["id"] for m in models["data"]]
    print(f"Models: {', '.join(model_ids)}")

    results = {}

    for model in model_ids:
        print(f"\n{'─' * 50}")
        print(f"Model: {model}")
        print(f"{'─' * 50}")

        # Decode benchmark
        print(f"\nDecode speed ({model}):")
        best, median, runs = bench_decode(model, n_runs=3)
        results[model] = {"decode_best_tps": round(best, 1),
                          "decode_median_tps": round(median, 1)}

        # Prefill benchmark
        print(f"\nPrompt processing ({model}):")
        prompt_tok, gen_tok, elapsed, prefill_tps = bench_prefill(model)
        results[model]["prefill_tps"] = round(prefill_tps, 0)
        results[model]["prompt_tokens"] = prompt_tok

        # Quality eval
        print(f"\nQuality eval ({model}):")
        passed, total = eval_quality(model, EVAL_TESTS)
        results[model]["eval_passed"] = passed
        results[model]["eval_total"] = total
        results[model]["eval_pct"] = f"{passed}/{total}"

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'Decode t/s':<12} {'Prefill t/s':<12} {'Eval':<8}")
    print(f"{'─' * 25} {'─' * 12} {'─' * 12} {'─' * 8}")
    for model, r in results.items():
        print(f"{model:<25} {r['decode_median_tps']:<12} "
              f"{r.get('prefill_tps', '—'):<12} {r['eval_pct']:<8}")

    print(f"\nFull results JSON:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
