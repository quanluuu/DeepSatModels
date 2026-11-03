"""
Smoke test for a running inference API (local uvicorn or the docker image), standard library only. Exit 0 = contract OK.
    python scripts/smoke_api.py http://localhost:8000 [--fixture tests/fixtures/window.npz] [--wait 120]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid


def get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.load(r)


def post_file(url, path, timeout=60):
    """multipart/form-data upload of one file as field 'file'; returns (status, json body)"""
    boundary = uuid.uuid4().hex
    with open(path, 'rb') as f:
        body = (b'--%s\r\nContent-Disposition: form-data; name="file"; filename="window.npz"\r\n'
                b'Content-Type: application/octet-stream\r\n\r\n' % boundary.encode()) + f.read() + b'\r\n--%s--\r\n' % boundary.encode()
    req = urllib.request.Request(url, body, {'Content-Type': 'multipart/form-data; boundary=' + boundary})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def wait_health(base, wait_s):
    deadline = time.time() + wait_s
    while True:
        try:
            return get_json(base + '/health')
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if time.time() > deadline:
                raise
            time.sleep(2)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('base', help='API base URL, e.g. http://localhost:8000')
    parser.add_argument('--fixture', default='tests/fixtures/window.npz')
    parser.add_argument('--wait', type=float, default=120, help='seconds to wait for /health (model loading)')
    args = parser.parse_args(argv)
    base = args.base.rstrip('/')
    failures = []

    def check(ok, what):
        print('%s %s' % ('PASS' if ok else 'FAIL', what))
        if not ok:
            failures.append(what)

    try:
        status, health = wait_health(base, args.wait)
    except Exception as e:
        print('FAIL /health unreachable after %g s: %s' % (args.wait, e))
        return 1
    check(status == 200 and health.get('status') == 'ok' and health.get('model_version'), '/health ok with model_version: %s' % health)

    status, body = post_file(base + '/predict', args.fixture)
    pred = body.get('pred') or []
    check(status == 200, '/predict returns 200 (got %d)' % status)
    check(body.get('status') in ('OK', 'UNCERTAIN'), '/predict status OK or UNCERTAIN (got %s)' % body.get('status'))
    check(len(pred) == 24 and all(len(r) == 24 for r in pred), 'pred is 24x24')
    check(all(0 <= v <= 18 for r in pred for v in r), 'pred values in 0..18')
    check(body.get('model_version') == health.get('model_version'), 'model_version matches /health')
    print('confidence %s, latency_ms %s' % (body.get('confidence'), body.get('latency_ms')))

    status, body = post_file(base + '/predict', __file__)     # this script is not an .npz: must be rejected before the model
    check(status == 422, 'invalid file rejected with 422 (got %d)' % status)

    print('smoke test %s' % ('FAILED: %d check(s)' % len(failures) if failures else 'passed'))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
