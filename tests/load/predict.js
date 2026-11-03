// Load test for POST /predict (course session 9B): 5 virtual users for 2 minutes on the fixture window.
//   k6 run --summary-export docs/report/k6-summary.json tests/load/predict.js
//   BASE_URL defaults to http://localhost:8000
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const fixture = open('../fixtures/window.npz', 'b');

const uncertain = new Rate('ai_uncertain_rate');   // AI not confident: the user must review the map
const fallback = new Rate('ai_fallback_rate');     // timeout or model error: no prediction returned
const serverLatency = new Trend('api_latency_ms'); // latency_ms reported by the API (queue + inference)

export const options = {
  vus: 5,
  duration: '2m',
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    http_req_duration: ['p(95)<5000'],   // SPEC: p95 < 5 s
    http_req_failed: ['rate<0.01'],      // SPEC: error rate < 1%
    checks: ['rate>0.99'],
  },
};

export default function () {
  const res = http.post(`${BASE_URL}/predict`, { file: http.file(fixture, 'window.npz', 'application/octet-stream') });
  const body = res.status === 200 ? res.json() : {};
  check(res, {
    'status is 200': (r) => r.status === 200,
    'has model_version': () => body.model_version !== undefined,
    'status is OK, UNCERTAIN or FALLBACK': () => ['OK', 'UNCERTAIN', 'FALLBACK'].includes(body.status),
  });
  uncertain.add(body.status === 'UNCERTAIN');
  fallback.add(body.status === 'FALLBACK');
  if (body.latency_ms !== undefined) serverLatency.add(body.latency_ms);
  sleep(1);
}
