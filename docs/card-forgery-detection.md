# Card Forgery Detection — status and open work

Written 2026-08-19 after two confirmed cheating cases. Two *different* attacks
turned up, and they need different defences. This is the working note for
picking the work back up.

Owners: `backend/` (Phase C detector), `lb/` (submit hardening).

---

## Problem 1 — forged card images

**What happened.** An account uploaded cards whose substat rows had been
re-typed in an image editor. Two builds: Hiyuki (8 of 25 rows redrawn) and
Aemeath (14 of 25). Every substat value was legal, because
`calc.ValidateEchoPanels` already enforces the exact roll-value table — the
forger picked values off that table. Stored CV matched the forged pixels
exactly (2 x sum(CR) + sum(CD) + 4-cost main).

**Why nothing caught it.** Phase A (`chrome_score`) only asks whether the image
is a KuroBot card; it is, so `validate_image_integrity` returned `ok`. Phase B
(`echo_bed_score`) is the pasted-cell detector and is exactly the right idea,
but it is observe-only because wrapped substat labels ("Resonance Liberation
DMG Bonus") break its background-gradient assumption. **The forged rows were
those wraps.** Phase B was blinded by the very feature the cheat used.

**What we added.** `forensics_card_render.py`, Phase C. Sort the 25 substat
label cells by brightness, split at the largest gap leaving >= 3 rows per side,
and compare error level either side of the split. Re-rendered text is bright
*and* has a low error level, because it never went through the original
quantization. Genuine cards show no such pairing however you split them.

Validated on 2649 cards stratified >= 300 per upload month across the whole
corpus (500/month for 2026-05..08):

| statistic | genuine max | genuine p99.9 | the two forgeries |
| --- | --- | --- | --- |
| `tone_sd` | 18.48 | 5.95 | 10.45 / 11.17 |
| `gap` | 41.27 | 4.67 | 8.10 / 12.18 |
| `ela_delta` | 0.40 | 0.09 | 5.04 / 5.29 |
| **`combined` = gap x ela_delta** | **2.00** | 0.14 | **40.8 / 64.4** |

End to end through the tool: **0 of 2649 genuine cards flagged, both forgeries
flagged**, 20x margin at `combined >= 5`.

**Use `combined` only.** `gap` and `tone_sd` are not safe thresholds at corpus
scale — cards whose substat rows carry highlight/selection bands reach gap=41
and tone_sd=18 while being perfectly genuine. `ela_delta` is what separates
"different background" from "different layer".

Cost: the tone/gap pass is 2.2 ms on an already-decoded frame; `ela_delta` needs
one re-encode (~20 ms) and only runs when `gap >= 5`, which fires on 3 of 2649
genuine cards (0.11%). `validate_image_integrity` already receives a decoded
ndarray, so Phase C decodes nothing extra.

### The JPEG encoder signature is a weak flag, not evidence

The forged files carry `DQT[0] mean 9.25`, 4:4:4 chroma and a JFIF/APP0 header,
while 2026-08 uploads are `23.08` / 4:2:0 / no APP0. That looked decisive against
a recent-only sample and **it is not** — the corpus has two encoder eras:

| era | signature |
| --- | --- |
| before 2026-07 (frontend canvas-recompressed) | `dqt 9.25` + APP0 dominates (452/499 in June); ~10% 4:4:4 |
| from 2026-07 (original input bytes stored) | `dqt 23.08`, 4:2:0, no APP0, plus PNG |

So the forgeries' signature is simply the *v1* signature appearing in the v2 era
(0 of 500 sampled August cards carry it). A stale cached client would look
identical. `encoder_signature()` is exposed for triage and logging; never reject
on it.

---

## Problem 2 — forged submit payloads

**What happened.** A second account posted two builds with no source image at all;
one stored `sequence = 67` and `weapon_rank = 9`, both out of legal range and
the only such rows in 34,413 builds.

**How.** `POST /build` is a public endpoint. `NEXT_PUBLIC_LB_URL` means the
browser calls the Cloudflare gateway directly, and the gateway injects
`X-Internal-Key` — which authenticates *the gateway to the Go service*, not the
caller. There is no account, session or token on the submit path, and
`watermark.uid` is self-declared. An Origin/CORS allowlist cannot help: CORS is
browser-enforced, and the Go service runs `cors.AllowAll()`.

**Impact was narrower than it looked.** `CalculateForBuild` standardizes both
fields away (`weaponRankFor(weaponID)`, and `seqInp.Sequence` is overridden with
each track's own level), so board damage was never inflated. The illegal values
polluted the promoted columns and the rendered build, and served as the
fingerprint that the payload was hand-crafted.

**Fixed.** `buildStateScalarViolations` in `lb/internal/api/handlers.go` now
returns 422 for out-of-range `sequence` (0-6), `characterLevel` / `weaponLevel`
(1-90) and `weaponRank` (1-5). Documented in `lb/docs/api-behaviors.md`.

---

## Open work

Ordered by value, not effort.

1. **Corpus-wide Phase C sweep.** Nothing else tells us whether these two are
   alone. Needs the full bucket locally:
   ```
   py sync_r2.py --run                                     # 128 workers, incremental
   py forensics_card_render.py ../r2-backup --out ../forensics/card_render
   ```
   Then review `review_queue.json` by hand. **Do not auto-delete** — see the
   standing rule in `image-integrity.md`; two labelled positives is not enough
   to justify a production rejection rule.

2. **`requireInternalAPIKey` fails open.** `if trustedKey == "" { return next }`
   in `lb/internal/api/auth.go`. If `INTERNAL_API_KEY` is ever unset on the Go
   service, the whole API becomes publicly writable with no signal. Fail closed,
   or refuse to boot.

3. **Bind submit to a completed scan.** The real fix for problem 2. Bind on
   `scanId`, *not* `sourceImageKey` — the key is optimistic and returned while
   the R2 upload may still be `pending`, so requiring it would fail legitimate
   submits whose upload was slow. Today `scan_id` is a bare `uuid.uuid4()` used
   for log correlation and never persisted (`server.py`), so this means adding a
   short-TTL single-use record the LB service can redeem. Cheap to adopt:
   August image attachment was 2898/2898 legitimate builds.

4. **Promote Phase B + C to gating.** Phase C resolves the wrapped-label false
   positives that have kept `echo_bed_score` observe-only. Use `bed_score` or
   `gap >= 5` as the trigger and require `combined >= 5` to act. Land it as a
   review queue first, not a user-facing rejection.

5. **Rate limiting.** There is none in the Go service; it is edge-only and keyed
   on IP. Add per-UID and per-board limits so a script cannot grind submissions
   under rotating UIDs.

6. **Quarantine instead of hard delete.** `cmd/blacklist` hard-deletes builds,
   echoes and profile. Takedowns should be reversible and reviewable — a
   quarantine table plus a moderation audit log.

### Known limits, worth being honest about

- **The blacklist is evadable.** `uid` is self-declared, so a blacklisted player
  resubmits under any other UID. Enforcement here is retrospective — detect and
  remove — not preventive. Do not treat `build_blacklist` as a wall.
- **CV monitoring is weak triage, not a detector.** The forged Hiyuki scored
  225.2 against a 2925-build distribution with p99 = 221.6 and max = 234.8: top
  1%, but below the legitimate maximum.
- Screenshots are self-reported by nature. The trust anchor is that every build
  traces to an immutable content-addressed image that passed integrity — which
  is why problem 2 (a build with no image at all) matters more than it first
  appeared.
