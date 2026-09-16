# Railway observability

The OCR service, the leaderboard API and Postgres share one Railway project, so this runbook covers all three.

| field | value |
| --- | --- |
| Project | `wuwa-backend` |
| Environment | `production` |
| OCR service | `WuWa OCR`, `https://ocr.wuwa.build` |
| LB service | `DB Server`, `https://api.wuwa.build` |
| Postgres | `WuWaBuilds DB` |

Never commit project, environment, service, deployment or instance IDs. `railway status` resolves them locally, and plain `railway status` works where `--json` has failed.

## Logs

Always bound a log read with `--lines`, `--since` or `--until`, since an unbounded `railway logs` streams forever. Logs span only the current deployment, so a before-and-after across a deploy has to be captured before deploying.

```powershell
railway status
railway deployment list --environment production --service "WuWa OCR" --limit 10 --json
railway logs --environment production --service "WuWa OCR" --http --lines 500 --json
railway logs --environment production --service "WuWa OCR" --http --filter "@totalDuration:>=1000" --lines 50 --json
railway logs --environment production --service "DB Server" --http --status ">=400" --lines 100 --json
```

`server.py` writes one JSON line per event through `log_events.log_event`, and each message starts with the event name, so a plain filter on the name finds it:

| event | carries |
| --- | --- |
| `ocr_import_completed` | `wall_ms`, `ocr_wall_ms`, `hash_ms`, `r2_result`, `r2_ms`, `slow_regions`, `unsupported_language`, `scan_id` |
| `ocr_import_rejected` | integrity `reason`, `hash_prefix`, `media_type` |
| `ocr_import_failed` | `error_code` for an unexpected 500 |
| `ocr_image_storage_completed` | final result of an upload that outlived recognition, including `cancelled` |
| `echo_bed_observed` | Phase B score above its logging floor |
| `ocr_issue_report_stored` | issue report `reason` and image storage outcome |

```powershell
railway logs --environment production --service "WuWa OCR" --since 1h --lines 400 --filter "ocr_import_completed" --json
railway logs --environment production --service "WuWa OCR" --filter "@level:error OR @level:warn" --lines 50 --json
```

Railway classifies Uvicorn's startup and shutdown `INFO:` lines as errors. Treat them as noise unless a traceback, non-2xx status, crash loop or failed deployment comes with them.

## Metrics and cost

```powershell
railway metrics --environment production --service "WuWa OCR" --since 6h --json
railway metrics --environment production --service "DB Server" --since 6h --json
railway metrics --environment production --service "WuWaBuilds DB" --since 6h --json
```

The OCR bill is RAM-bound: CPU sits near idle between imports while every worker holds its memory. Estimate a window as average vCPU x minutes x the CPU rate plus average GB x minutes x the RAM rate from [Railway pricing](https://docs.railway.com/reference/pricing). The dashboard Usage page is the bill of record, since the CLI has no billing command.

## Per-worker memory

`railway metrics` reports the service total. The per-worker split needs `/proc`, and the container has no `ps`. `railway ssh` needs a registered key first (`railway ssh keys github` or `railway ssh keys add`).

```
railway ssh 'for d in /proc/[0-9]*; do pid=${d#/proc/}; cmd=$(tr "\0" " " < $d/cmdline 2>/dev/null | cut -c1-50); case "$cmd" in *python*) echo "=== PID $pid | $cmd"; grep -E "^(Rss|Pss|Private_Dirty|Shared_Dirty):" $d/smaps_rollup | tr "\n" " "; echo;; esac; done'
```

Sum `Pss`, which divides shared pages among their sharers. `Private_Dirty` is what a worker holds alone, and most of it is retained runtime allocation beyond the template load. Measure here before sizing `OCR_WORKERS`, since a Windows dev-box RSS understated production by more than half.
