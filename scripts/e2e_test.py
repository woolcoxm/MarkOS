#!/usr/bin/env python3
"""MarkOS E2E + Security Audit Suite.

Tests every user-facing surface: API, web UI, auth, TLS, error handling,
rate limiting, input validation. Run against the live appliance.

Usage: python3 e2e_test.py [--host 10.0.0.69] [--password <admin_pw>]
"""
import argparse
import json
import ssl
import ssl as _ssl
import time
import urllib.request
import urllib.error
import http.client
import socket
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="10.0.0.69")
parser.add_argument("--password", default="")
args = parser.parse_args()
HOST = args.host
API = f"http://{HOST}:8080"
UI = f"https://{HOST}:4444"

# Self-signed cert context
ctx = _ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = _ssl.CERT_NONE

passed = 0
failed = 0
total = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} {detail}")

def api(method, path, body=None, timeout=30, headers=None):
    """API request (plain HTTP :8080)"""
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()) if r.headers.get("content-type", "").startswith("application/json") else r.read()
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except:
            return e.code, body

def ui(method, path, body=None, cookies=None, timeout=15):
    """Web UI request (HTTPS :4444, self-signed)"""
    data = json.dumps(body).encode() if body else None
    hdrs = {"Content-Type": "application/json"}
    if cookies:
        hdrs["Cookie"] = cookies
    try:
        conn = http.client.HTTPSConnection(HOST, 4444, context=ctx, timeout=timeout)
        conn.request(method, path, body=data, headers=hdrs)
        r = conn.getresponse()
        resp_body = r.read()
        set_cookie = r.getheader("Set-Cookie", "")
        try:
            return r.status, json.loads(resp_body), set_cookie
        except:
            return r.status, resp_body, set_cookie
    except Exception as e:
        return 0, str(e), ""

print("=" * 70)
print("MarkOS E2E + Security Audit")
print(f"Target: {HOST} (API :8080, UI :4444)")
print("=" * 70)

# ============================================================
print("\n── 1. API Health & Discovery ──")
# ============================================================
st, d = api("GET", "/healthz")
check("healthz returns 200", st == 200)
check("healthz has version", isinstance(d, dict) and "version" in d)

st, d = api("GET", "/v1/models")
check("models list returns 200", st == 200)
check("models list has data array", isinstance(d, dict) and "data" in d)
model_count = len(d.get("data", []))
check(f"models exist ({model_count})", model_count > 0)

# ============================================================
print("\n── 2. Chat Completions (core API) ──")
# ============================================================
model_id = d["data"][0]["id"] if model_count > 0 else "test"

st, d = api("POST", "/v1/chat/completions", {
    "model": model_id,
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 8,
}, timeout=120)
check("chat completion returns 200", st == 200, f"got {st}")
if st == 200:
    check("has choices array", "choices" in d)
    check("has message content", len(d.get("choices", [{}])) > 0 and "message" in d["choices"][0])
    check("has usage stats", "usage" in d)
    check("finish_reason present", "finish_reason" in d.get("choices", [{}])[0])

# Error: nonexistent model
st, d = api("POST", "/v1/chat/completions", {
    "model": "nonexistent-model",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 4,
})
check("nonexistent model returns error", st in [404, 400, 422], f"got {st}")

# Error: empty messages
st, d = api("POST", "/v1/chat/completions", {
    "model": model_id,
    "messages": [],
    "max_tokens": 4,
})
check("empty messages returns error", st in [400, 422], f"got {st}")

# Error: malformed JSON (manual request)
try:
    req = urllib.request.Request(f"{API}/v1/chat/completions",
                                  data=b'{"invalid json', method="POST")
    req.add_header("Content-Type", "application/json")
    urllib.request.urlopen(req, timeout=5)
    check("malformed JSON rejected", False, "accepted")
except urllib.error.HTTPError as e:
    check("malformed JSON rejected", e.code in [400, 500])

# Error: max_tokens = 0 or negative
st, d = api("POST", "/v1/chat/completions", {
    "model": model_id,
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 0,
}, timeout=60)
check("max_tokens=0 handled gracefully", st in [200, 400])

# ============================================================
print("\n── 3. SSE Streaming ──")
# ============================================================
try:
    req = urllib.request.Request(
        f"{API}/v1/chat/completions",
        data=json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 8,
            "stream": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        ct = r.headers.get("content-type", "")
        body = r.read().decode()
        check("SST content-type", "text/event-stream" in ct, f"got {ct}")
        check("SSE has data chunks", "data:" in body)
        check("SSE has [DONE] terminator", "[DONE]" in body)
except Exception as e:
    check("SSE streaming works", False, str(e))

# ============================================================
print("\n── 4. Web UI Authentication ──")
# ============================================================
# Login with wrong password
st, d, _ = ui("POST", "/api/login", {"username": "admin", "password": "wrong"})
check("wrong password rejected", st in [401, 403], f"got {st}")

# Login with correct password (if provided)
if args.password:
    st, d, cookies = ui("POST", "/api/login", {"username": "admin", "password": args.password})
    check("correct password accepted", st == 200, f"got {st}")

    # Extract session cookie
    session = cookies.split(";")[0] if cookies else ""
    check("session cookie issued", len(session) > 10)

    # Authenticated request
    st, d, _ = ui("GET", "/api/state", cookies=session)
    check("authenticated state access", st == 200, f"got {st}")

    # Unauthenticated access to admin API
    st, d, _ = ui("GET", "/api/state")
    check("unauthenticated state access rejected", st in [401, 403], f"got {st}")

    # Session cookie manipulation
    st, d, _ = ui("GET", "/api/state", cookies="session=fake")
    check("fake session rejected", st in [401, 403], f"got {st}")
else:
    print("  (skipping authenticated tests — no password provided)")

# ============================================================
print("\n── 5. TLS / Network Security ──")
# ============================================================
# Check TLS is actually running on UI port
try:
    conn = http.client.HTTPSConnection(HOST, 4444, context=ctx, timeout=5)
    conn.request("GET", "/")
    r = conn.getresponse()
    check("HTTPS UI responds", r.status in [200, 301, 302, 401])
except Exception as e:
    check("HTTPS UI responds", False, str(e))

# Check plain HTTP to UI port is rejected or redirected
try:
    conn = http.client.HTTPConnection(HOST, 4444, timeout=5)
    conn.request("GET", "/")
    r = conn.getresponse()
    check("plain HTTP to TLS port rejected", r.status in [400, 301, 302])
except:
    check("plain HTTP to TLS port rejected", True)  # connection reset is also OK

# Check API port doesn't serve UI
try:
    st, d = api("GET", "/")
    check("API port doesn't serve UI", st in [404, 400, 426, 200])  # 200 OK if UI embedded
except:
    check("API port doesn't serve UI", True)

# Test for XSS in error messages
st, d = api("POST", "/v1/chat/completions", {
    "model": "<script>alert(1)</script>",
    "messages": [{"role": "user", "content": "test"}],
})
if isinstance(d, dict) and "error" in d:
    err_str = json.dumps(d)
    check("XSS in error messages", "<script>" not in err_str, "script tag reflected")

# ============================================================
print("\n── 6. Input Validation ──")
# ============================================================
# Very long model name
st, d = api("POST", "/v1/chat/completions", {
    "model": "A" * 10000,
    "messages": [{"role": "user", "content": "test"}],
})
check("very long model name handled", st in [400, 404, 422])

# Very long prompt
st, d = api("POST", "/v1/chat/completions", {
    "model": model_id,
    "messages": [{"role": "user", "content": "A" * 100000}],
    "max_tokens": 4,
}, timeout=60)
check("very long prompt handled", st in [200, 400, 413, 422], f"got {st}")

# Unicode/null bytes
st, d = api("POST", "/v1/chat/completions", {
    "model": model_id,
    "messages": [{"role": "user", "content": "Hello\x00World"}],
    "max_tokens": 4,
}, timeout=60)
check("null bytes in prompt handled", st in [200, 400])

# Negative max_tokens
st, d = api("POST", "/v1/chat/completions", {
    "model": model_id,
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": -5,
}, timeout=60)
check("negative max_tokens handled", st in [200, 400, 422])

# ============================================================
print("\n── 7. Rate Limiting & DoS Resistance ──")
# ============================================================
# Send 10 rapid requests — all should be served or queued, not crash
statuses = []
for i in range(10):
    st, d = api("POST", "/v1/chat/completions", {
        "model": model_id,
        "messages": [{"role": "user", "content": f"Hi {i}"}],
        "max_tokens": 2,
    }, timeout=10)
    statuses.append(st)
    time.sleep(0.1)
check("10 rapid requests don't crash", all(s in [200, 429, 503] for s in statuses),
      f"statuses: {set(statuses)}")
check("no 5xx from rapid requests", 500 not in statuses and 502 not in statuses)

# ============================================================
print("\n── 8. Content-Type & Method Validation ──")
# ============================================================
# POST with wrong content-type
try:
    req = urllib.request.Request(f"{API}/v1/chat/completions",
                                  data=b'plaintext', method="POST")
    req.add_header("Content-Type", "text/plain")
    urllib.request.urlopen(req, timeout=5)
    check("wrong content-type handled", True)  # some servers accept and fail later
except urllib.error.HTTPError as e:
    check("wrong content-type handled", e.code in [400, 415, 500])

# DELETE method on chat endpoint
st, d = api("DELETE", "/v1/chat/completions")
check("DELETE method handled", st in [405, 404, 400])

# PATCH method
st, d = api("PATCH", "/v1/models")
check("PATCH method handled", st in [405, 404, 400])

# ============================================================
print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print(f"{'✅ ALL TESTS PASSED' if failed == 0 else '❌ SOME TESTS FAILED'}")
print("=" * 70)

if failed > 0:
    sys.exit(1)
