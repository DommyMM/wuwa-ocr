# Railway Observability and Cost Runbook

Current production Railway project, public-safe view:

| Field | Value |
|---|---|
| Project | `wuwa-backend` |
| Environment | `production` |
| OCR service | `WuWa OCR` / `https://ocr.wuwa.build` |
| LB service | `DB Server` / `https://api.wuwa.build` |
| Postgres | `WuWaBuilds DB` |

Do not commit Railway project, environment, service, deployment, or instance
IDs. Use `railway status` locally to resolve them when needed.

CLI checked on 2026-06-26:

- `railway 5.20.0`
- agent tooling healthy: skills installed and MCP configured
- CLI reported an available update to `5.23.1`
- `railway status --json` returned `Problem processing request`; plain
  `railway status` worked and should be used as a fallback.

## CLI Surface

Useful commands exposed by the installed CLI:

- `status`, `project`, `service`, `deployment`
- `logs` for deploy, build, HTTP, and network flow logs
- `metrics` for CPU, memory, network, volume, and HTTP summaries
- `variable` for service variables
- `domain`, `private-network`, `tcp-proxy`, `waf`, `cdn`
- `bucket`, `volume`, `connect`, `ssh`
- `docs`, `mcp`, `skills`, `agent`, `setup`

Always bound log reads with `--lines`, `--since`, or `--until`; unbounded
`railway logs` streams forever.

## Health Snapshot

```powershell
$env:RAILWAY_CALLER='skill:use-railway@1.3.0'
$env:RAILWAY_AGENT_SESSION='railway-wuwabuilds-ops'
railway status
railway deployment list --environment production --service "WuWa OCR" --limit 10 --json
railway deployment list --environment production --service "DB Server" --limit 10 --json
```

2026-06-26 snapshot:

- `WuWa OCR`: online in `US West`; latest deployment status `SUCCESS`, repo
  `DommyMM/wuwa-ocr`.
- `DB Server`: online at `api.wuwa.build`; latest HTTP logs sampled from the
  active deployment.
- `WuWaBuilds DB (Postgres)`: online.
- `kurobot` and `Kurobot DB`: offline in this project.

The latest OCR deployment reverted ThreadPoolExecutor back to
ProcessPoolExecutor. The deployment commit message records the production
latency regression: median OCR latency rose from about `687 ms` to about
`3855 ms` under ThreadPoolExecutor, then the process pool was restored.

## Latency Queries

OCR Railway HTTP logs:

```powershell
railway logs --environment production --service "WuWa OCR" --http --lines 500 --json
railway logs --environment production --service "WuWa OCR" --http --filter "@totalDuration:>=1000" --lines 50 --json
railway metrics --environment production --service "WuWa OCR" --since 6h --json
```

OCR app-level timings are emitted by `server.py` as:

```text
import: Completed wall=<ms> body=<ms> decode=<ms> crop=<ms> recognition=<ms> bytes=<n> lang=<code> slow=<region timings>
```

Useful filtered query:

```powershell
railway logs --environment production --service "WuWa OCR" --since 1h --lines 400 --filter "Completed wall" --json
```

2026-06-26 OCR samples:

| Source | Window / sample | Result |
|---|---:|---|
| Railway metrics | 6h, 207 HTTP requests | `0` errors, p50/p90/p95/p99 `27 ms` |
| HTTP logs | 10 latest requests | 9 health checks, 1 `/api/ocr`; p50 `5 ms`, p95 `1140 ms`, all `200` |
| OCR completion logs | 7 imports, 21:50:29Z-22:33:37Z | wall avg `933.81 ms`, p50 `973.62 ms`, p95/max `1133.97 ms` |
| OCR completion logs | same sample | recognition avg `785.34 ms`, p50 `800.39 ms`, p95 `834.37 ms` |
| Latest import | `2026-06-26T22:33:37Z` | wall `1133.97 ms`, body `281.03 ms`, decode `32.57 ms`, crop `0.35 ms`, recognition `818.4 ms` |

LB HTTP logs:

```powershell
railway logs --environment production --service "DB Server" --http --lines 500 --json
railway logs --environment production --service "DB Server" --http --status ">=400" --lines 100 --json
railway logs --environment production --service "DB Server" --http --filter "@totalDuration:>=100" --lines 50 --json
```

2026-06-26 LB sample:

| Source | Window / sample | Result |
|---|---:|---|
| HTTP logs | 156 latest requests, 22:33:41Z-22:48:17Z | `0` errors, all `200` |
| HTTP logs | same sample | p50 `41 ms`, p90 `60 ms`, p95 `64 ms`, p99 `118 ms` |
| Slow filter | `@totalDuration:>=100` | 3 requests, worst `131 ms` on `/leaderboard/1205` |
| Top sampled paths | same sample | `/profile` 19, `/leaderboard/1107` 17, `/leaderboard/1409` 14, `/leaderboard/1603` 12, `/leaderboard/1507` 12 |

## Error Queries

```powershell
railway logs --environment production --service "WuWa OCR" --http --status ">=400" --lines 100 --json
railway logs --environment production --service "DB Server" --http --status ">=400" --lines 100 --json
railway logs --environment production --service "WuWa OCR" --filter "@level:error OR @level:warn" --lines 50 --json
railway logs --environment production --service "DB Server" --filter "@level:error OR @level:warn" --lines 50 --json
```

2026-06-26 notes:

- No sampled OCR or LB HTTP errors.
- No sampled DB Server runtime warnings/errors.
- OCR runtime `@level:error` included Uvicorn startup/shutdown `INFO:` lines.
  Treat those as Railway log-level classification noise unless there is a real
  traceback, non-2xx HTTP status, crash loop, or deployment failure alongside it.

## Resource Metrics and Cost

```powershell
railway metrics --environment production --service "WuWa OCR" --since 6h --json
railway metrics --environment production --service "DB Server" --since 6h --json
railway metrics --environment production --service "WuWaBuilds DB" --since 6h --json
```

2026-06-26 OCR 6-hour metrics:

| Metric | Value |
|---|---:|
| CPU avg / max / limit | `0.0306` / `0.8060` / `8.0` vCPU |
| Memory avg / max / limit | `2025.35` / `2467.97` / `8192` MB |
| HTTP total / error rate | `207` / `0%` |
| Public ingress avg / max | `0.1155` / `3.9692` MB |
| Public egress avg / max | `0.0025` / `0.0434` MB |

Railway's CLI metrics are useful for cost drivers, but the Railway dashboard
Usage page is the bill of record for exact current charges. The installed CLI
does not expose a `billing` or exact project-cost command.

Pricing references checked on 2026-06-26:

- Hobby plan includes `$5` of usage and charges `$5/mo`.
- CPU: `$0.000463` per vCPU minute.
- RAM: `$0.000231` per GB minute.
- Public network egress: `$0.05` per GB.
- Volumes: `$0.0000007` per GB minute.

Approximate OCR compute cost for the sampled 6h window, using average CPU and
RAM only and excluding subscription, network, DB, volumes, and other services:

```text
CPU  ~= 0.0305666 vCPU * 360 min * $0.000463 = $0.0051
RAM  ~= 2025.35 MB / 1024 * 360 min * $0.000231 = $0.1644
Total ~= $0.17 for the sampled 6h OCR compute window
```

Do not treat that estimate as total project cost. Pull the dashboard Usage page
for exact current usage/costs, then compare it with these service metrics to
identify drivers.

Official docs:

- `https://docs.railway.com/reference/pricing`
- `https://docs.railway.com/guides/project-usage`
- `https://docs.railway.com/cli/logs`
- `https://docs.railway.com/cli/metrics`
