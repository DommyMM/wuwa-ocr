# Observability

Read-only views of the `WuWa OCR` service (`ocr.wuwa.build`) in Railway project `wuwa-backend`, environment `production`, and of the R2 bucket it writes. Ids stay out of this doc, `railway status` resolves them. Log reads follow the same Railway rules as the leaderboard service: bound every read, `--lines` caps at 5000 per page, and logs reach back only to the current deployment.

## Logs

```bash
railway logs -e production -s "WuWa OCR" --since 1h --lines 400 --filter "ocr_import_completed" --json
railway logs -e production -s "WuWa OCR" --filter "@level:error OR @level:warn" --lines 50 --json
railway logs -e production -s "WuWa OCR" --http --filter "@totalDuration:>=1000" --lines 50 --json
```

`server.py` writes one JSON line per event through `log_events.log_event`, each message starting with the event name:

| event | carries |
| --- | --- |
| `ocr_import_completed` | `wall_ms`, `ocr_wall_ms`, `hash_ms`, `r2_result`, `r2_ms`, `slow_regions`, `unsupported_language`, `scan_id` |
| `ocr_import_rejected` | integrity `reason`, `hash_prefix`, `media_type` |
| `ocr_import_failed` | `error_code` for an unexpected 500 |
| `ocr_image_storage_completed` | final result of an upload that outlived recognition, including `cancelled` |
| `echo_bed_observed` | Phase B score above its logging floor |
| `ocr_issue_report_stored` | issue report `reason` and image storage outcome |

Railway classifies Uvicorn's startup and shutdown `INFO:` lines as errors. They are noise unless a traceback, non-2xx status, crash loop or failed deployment comes with them.

## Memory and cost

```bash
railway metrics -e production -s "WuWa OCR" --since 6h --json
```

- The bill is RAM-bound: CPU idles between imports while every worker holds its memory. The dashboard Usage page is the bill of record, since the CLI has no billing command
- A deploy hour double-counts memory, since the old and new containers overlap

`railway metrics` reports the service total. The per-worker split needs `/proc`, and the container has no `ps`. `railway ssh` needs a registered key first (`railway ssh keys github`).

```
railway ssh 'for d in /proc/[0-9]*; do pid=${d#/proc/}; cmd=$(tr "\0" " " < $d/cmdline 2>/dev/null | cut -c1-50); case "$cmd" in *python*) echo "=== PID $pid | $cmd"; grep -E "^(Rss|Pss|Private_Dirty|Shared_Dirty):" $d/smaps_rollup | tr "\n" " "; echo;; esac; done'
```

Sum `Pss`, which divides shared pages among their sharers. Measure here before sizing `OCR_WORKERS`, since a Windows dev-box RSS understated production by more than half.

## R2

Bucket `wuwabuilds` holds source screenshots under content-addressed keys and `db-backups/` with the leaderboard database's daily `pg_dump`. `sync_r2.py` mirrors the screenshots into the workspace's `r2-backup/`. The bucket sits in a different Cloudflare account than wrangler's default, so set `CLOUDFLARE_ACCOUNT_ID` before any wrangler R2 command. Listing works through `GET /client/v4/accounts/<account>/r2/buckets/wuwabuilds/objects?prefix=` with the wrangler login's OAuth token.
